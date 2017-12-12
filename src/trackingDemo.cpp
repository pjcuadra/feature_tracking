/*
 * MIT License
 *
 * Copyright (c) 2017 Pedro Cuadra
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */
#include <dirent.h>
#include <iostream>
#include <sys/stat.h>

// External
#include <opencv2/opencv.hpp>

// Internal
#include <trackers/Tracker.hpp>
#include <util/Debug.hpp>
#include <util/Mosaic.hpp>
#include <util/CustomSerializer.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::detail;
using namespace std;

#define PREV_IDX 0
#define CURR_IDX 1

static String keys = "{help h usage ? |      | Print this message    }"
                     "{v              |      | Verbose               }"
                     "{out            |      | Output Image Path     }"
                     "{indir          |      | Input Directory Path  }"
                     "{show           |      | Display images        }"
                     "{all            |      | Display images        }"
                     "{finder         |      | Feature Finder        }"
                     "{extract        |      | Extract Features      }"
                     "{match          |      | Match Features        }";

const string featuresFile("features.yml");

// Global constants
static const double scaleFactor = 0.3;
static const float	match_conf = 0.66f;
static const int imageHeight = 5472;
static const int imageWidth = 3648;

static const double focalLength = 4419.441;
static const double principalPointX = 2708.765;
static const double principalPointY = 1775.895;
static const Size scaled(imageWidth * scaleFactor, imageHeight * scaleFactor);
static const char * image1Window = "Image 1";
static const char * image2Window = "Image 2";

// Global variables
static Ptr<CommandLineParser> parser;
static vector<string> inputImagesPaths;
static vector<Mat> inputImages;
static bool enableGui;
static int key = 0;
static Ptr<FeaturesFinder> finder;
static vector<ImageFeatures> features;
static vector<MatchesInfo> pairwiseMatches;
static FileStorage fs;
static vector<CameraParams>	estimatedCamerasParams;
static CameraParams specifiedCameraParams;
static vector<Mat> homography;
static vector<vector<Mat>> calculatedRotation;
static vector<vector<Mat>> calculatedTranslation;
static Ptr<WarperCreator> warperCreator = makePtr<cv::CompressedRectilinearWarper>();
static Ptr<RotationWarper> warper = warperCreator->create(1.0f);
static vector<Point2f> imageCorners;
static Mat fullImage;

void parserInputImagesFiles() {
  string fileName;
  struct dirent *ent;
  DIR *dir;

  if (!parser->has("indir")) {
    DEBUG_STREAM( "No input file specified" );
    exit(-1);
  }

  // Add all directory's files to the vector
  if ((dir = opendir(parser->get<string>("indir").c_str())) != NULL) {
    /* print all the files and directories within directory */
    while ((ent = readdir(dir)) != NULL) {
      // DEBUG_STREAM( "file: " << ent->d_name );
      if (ent->d_type != DT_REG) {
        continue;
      }

      inputImagesPaths.push_back(parser->get<string>("indir") + "/" +
                                ent->d_name);
    }

    sort(inputImagesPaths.begin(), inputImagesPaths.end());
    closedir(dir);
  } else {
    /* could not open directory */
    perror("Could Not open specified directory");
    exit(EXIT_FAILURE);
  }

}

void versionPrinting() {
  DEBUG_STREAM( "OpenCV Version: " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION);

  DEBUG_STREAM("C++ Standard: ");
  if (__cplusplus == 201103L) {
    DEBUG_STREAM( "C++11" );
  } else if (__cplusplus == 19971L) {
    DEBUG_STREAM( "C++98" );
  } else {
    DEBUG_STREAM( "pre-standard C++" );
  }
}

bool waitAndContinue() {
  key = 0;
  if (enableGui) {
    while (true) {
      key = waitKey(0);

      switch (key) {
      case 27: // ESC
        return false;
      case 13: // Enter
        return true;
      default:
        continue;
      }
    }
  }

  return true;
}

void readImage(Mat &image, int i) {
  assert(i < inputImagesPaths.size());

  image = imread(inputImagesPaths[i]);
  assert(!image.empty());
  resize(image, image, scaled);
}

void readImages(Mat images[2], int i) {
  assert(i < inputImagesPaths.size() - 1);

  if (images[PREV_IDX].empty()){
    images[PREV_IDX] = imread(inputImagesPaths[i]);
    assert(!images[PREV_IDX].empty());
    resize(images[PREV_IDX], images[PREV_IDX], scaled);
  } else {
    images[CURR_IDX].copyTo(images[PREV_IDX]);
  }

  images[CURR_IDX] = imread(inputImagesPaths[i + 1]);
  assert(!images[CURR_IDX].empty());
  resize(images[CURR_IDX], images[CURR_IDX], scaled);
}

void printFeaturesStats(int i) {
  DEBUG_STREAM(inputImagesPaths[i] << " # features " << features[i].keypoints.size());
}

void printMatchesStats(int i, MatchesInfo info) {
  assert(i < inputImagesPaths.size() - 1);

  DEBUG_STREAM(inputImagesPaths[i] << " -> " << inputImagesPaths[i + 1] << " # Matches " << info.matches.size());
}

void findFeatures(Mat images[2], int i) {
  assert(i < inputImagesPaths.size() - 1);

  // For the first run
  if (features[i].keypoints.empty()){
    (*finder)(images[PREV_IDX],	features[i]);
    features[i].img_idx = i;
    printFeaturesStats(i);
  }

  (*finder)(images[CURR_IDX],	features[i + 1]);
  features[i + 1].img_idx = i + 1;
  printFeaturesStats(i + 1);
}

void parserFinder() {
  string finderName("");

  if (!parser->has("finder")) {
    LOG("Unspecified finder using SURF");
    finder = makePtr<SurfFeaturesFinder>();
    return;
  }

  finderName = parser->get<string>("finder");

  if (finderName == "surf") {
    finder = makePtr<SurfFeaturesFinder>();
    return;
  }

  if (finderName == "orb") {
    finder = makePtr<OrbFeaturesFinder>();
    return;
  }

  LOG("Unsupported finder using SURF");
  finder = makePtr<SurfFeaturesFinder>();
}

inline bool checkFileExists(const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

void parseFeatures() {
  Mat images[2];
  vector<ImageFeaturesSerializer> serFeatures(features.size());
  FileStorage fs;

  for (int i = 0; i < features.size(); i++) {
    serFeatures[i] = ImageFeaturesSerializer(features[i]);
  }

  if (parser->has("extract") || parser->has("match") || !checkFileExists(featuresFile)) {

    // Extract all features
    for (int i = 0; i < inputImagesPaths.size() - 1; i++) {
      readImages(images, i);
      TRACE_LINE(__FILE__, __LINE__);

      // Get the features
      findFeatures(images, i);
    }

    fs = FileStorage(featuresFile, FileStorage::WRITE);
    fs << "features" << serFeatures;
    fs.release();
    return;
  }

  fs = FileStorage(featuresFile, FileStorage::READ);
  fs["features"] >> serFeatures;
  fs.release();
}

MatchesInfo findMatchInfo(int src, int dst) {
  auto it = find_if(pairwiseMatches.begin(),
                  pairwiseMatches.end(),
                  [&src, &dst](const MatchesInfo& obj) {
                    return (obj.src_img_idx == src) && (obj.dst_img_idx == dst);
                  });

  assert(it != pairwiseMatches.end());

  return pairwiseMatches[distance(pairwiseMatches.begin(), it)];
}

void warpImages(Mat images[2], int i) {
  Mat warpedImage;
  Mat fullImage;

  // Apply homography to the image
  warpPerspective(images[CURR_IDX], warpedImage, homography[i], images[CURR_IDX].size());
  addWeighted(images[PREV_IDX], 0.5, warpedImage, 0.5, 1, fullImage);

  imshow("Warped", fullImage);
  imwrite("/home/pjcuadra/Downloads/test.jpg", fullImage);
}

void drawGui() {
  Mat images[2];
  Mat imagesWithFeatures[2];
  Mat matchesImage;
  stringstream ss;
  MatchesInfo currMatch;

  namedWindow(image1Window, WINDOW_GUI_EXPANDED);
  namedWindow(image2Window, WINDOW_GUI_EXPANDED);
  namedWindow("Matches", WINDOW_GUI_EXPANDED);
  namedWindow("Warped", WINDOW_GUI_EXPANDED);

  // Extract all features
  for (int i = 0; i < inputImagesPaths.size() - 1; i++) {
    readImages(images, i);
    TRACE_LINE(__FILE__, __LINE__);

    drawKeypoints(images[PREV_IDX],
                  features[i].keypoints,
                  imagesWithFeatures[PREV_IDX],
                  Scalar::all(-1),
                  DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    drawKeypoints(images[CURR_IDX],
                  features[i + 1].keypoints,
                  imagesWithFeatures[CURR_IDX],
                  Scalar::all(-1),
                  DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    imshow(image1Window, imagesWithFeatures[PREV_IDX]);
    imshow(image2Window, imagesWithFeatures[CURR_IDX]);

    ss.str("");
    ss << inputImagesPaths[i] << " - " << features[i].keypoints.size() << " Features";
    displayStatusBar(image1Window, ss.str());

    ss.str("");
    ss << inputImagesPaths[i + 1] << " - " << features[i + 1].keypoints.size() << " Features";
    displayStatusBar(image2Window, ss.str());

    currMatch = findMatchInfo(i, i + 1);

    printMatchesStats(i, currMatch);

    drawMatches(images[PREV_IDX],
                features[i].keypoints,
                images[CURR_IDX],
                features[i + 1].keypoints,
                currMatch.matches,
                matchesImage);

    imshow("Matches", matchesImage);
    ss.str("");
    ss << inputImagesPaths[i] << " -> " << inputImagesPaths[i + 1] << " - " << currMatch.matches.size() << " Matches";
    displayStatusBar("Matches", ss.str());

    LOG("Warping");
    warpImages(images, i);

    if (!waitAndContinue()) {
      break;
    }
  }

  // Wait for the ESC key to be pressed
  while (key != 27) {
    key = waitKey(0);
  }
}

void estimateCameraParams() {
  HomographyBasedEstimator estimator;

  if	(!estimator(features,	pairwiseMatches,	estimatedCamerasParams)) {
      DEBUG_STREAM("Homography	estimation	failed.");
      return;
  }

  for (int i = 0; i < inputImagesPaths.size(); i++) {
    DEBUG_STREAM( "Camera Params Image - " << inputImagesPaths[i]);
    DEBUG_STREAM( "  K = " << estimatedCamerasParams[i].K());
    DEBUG_STREAM( "  R = " << estimatedCamerasParams[i].R);
    DEBUG_STREAM( "  t = " << estimatedCamerasParams[i].t);

    estimatedCamerasParams[i].R.convertTo(estimatedCamerasParams[i].R, CV_32F);
  }
}

void adjustCameraParams() {
  static Ptr<BundleAdjusterBase> adjuster = makePtr<BundleAdjusterReproj>();
  static float confThresh	=	0.03f;

  adjuster->setConfThresh(confThresh);

  if	(!(*adjuster)(features,	pairwiseMatches,	estimatedCamerasParams)) {
      LOG("Adjusting camera parameters	failed.");
      return;
  }

  for (int i = 0; i < inputImagesPaths.size(); i++) {
    DEBUG_STREAM( "Camera Params Image - " << inputImagesPaths[i]);
    DEBUG_STREAM( "  K = " << estimatedCamerasParams[i].K());
    DEBUG_STREAM( "  R = " << estimatedCamerasParams[i].R);
    DEBUG_STREAM( "  t = " << estimatedCamerasParams[i].t);

    estimatedCamerasParams[i].R.convertTo(estimatedCamerasParams[i].R, CV_32F);
  }
}

void calcHomographyMatrix() {
  MatchesInfo currInfo;
  vector<Point2f> srcPoints, dstPoints;
  Mat H;

  for (int i = 0; i < inputImagesPaths.size() - 1; i++) {
    currInfo = findMatchInfo(i, i + 1);

    srcPoints.clear();
    dstPoints.clear();

    for (int o = 0; o < currInfo.matches.size(); o++) {
      Point2f src, dst;
      dst = features[currInfo.src_img_idx].keypoints[currInfo.matches[o].queryIdx].pt;
      src = features[currInfo.dst_img_idx].keypoints[currInfo.matches[o].trainIdx].pt;

      srcPoints.push_back(src);
      dstPoints.push_back(dst);
    }

    H = findHomography(srcPoints, dstPoints, currInfo.inliers_mask, RANSAC);

    DEBUG_STREAM("Homography of " << inputImagesPaths[i + 1] << " -> " << inputImagesPaths[i]);
    DEBUG_STREAM(" H = " << H);

    H.copyTo(homography[i]);

  }
}

void compareProjectedPoints() {
  MatchesInfo currInfo;
  vector<Point2f> inPoints;
  vector<Point2f> outPoints;

  for (int i = 0; i < inputImagesPaths.size() - 1; i++) {
    currInfo = findMatchInfo(i, i + 1);

    inPoints.clear();

    for (int p = 0; p < currInfo.matches.size(); p++) {
      inPoints.push_back(features[currInfo.src_img_idx].keypoints[currInfo.matches[p].queryIdx].pt);
    }

    perspectiveTransform(inPoints, outPoints, homography[i]);

    for (int p = 0; p < currInfo.matches.size(); p++) {
      DEBUG_STREAM("Original Image Point : " << features[currInfo.dst_img_idx].keypoints[currInfo.matches[p].trainIdx].pt);
      DEBUG_STREAM("Projected Image Point (before): " << inPoints[p]);
      DEBUG_STREAM("Projected Image Point : " << outPoints[p]);
    }

  }
}

void decomoposeHMatrix() {
  Mat K;

  specifiedCameraParams.K().copyTo(K);

  DEBUG_STREAM("Specified Camera Matrix ");
  DEBUG_STREAM(" K = " << K);

  for (int i = 0; i < inputImagesPaths.size() - 1; i++) {
    decomposeHomographyMat(homography[i],
                           K,
                           calculatedRotation[i],
                           calculatedTranslation[i],
                           noArray());

    DEBUG_STREAM("Rotation of " << inputImagesPaths[i + 1] << " -> " << inputImagesPaths[i]);
    for (int o = 0; o < calculatedRotation[i].size(); o++) {
      DEBUG_STREAM(" R[" << o << "] = " << calculatedRotation[i][o]);
    }

    DEBUG_STREAM("Translation of " << inputImagesPaths[i + 1] << " -> " << inputImagesPaths[i]);
    for (int o = 0; o < calculatedTranslation[i].size(); o++) {
      DEBUG_STREAM(" t[" << o << "]= " << calculatedTranslation[i][o]);
    }

    decomposeHomographyMat(homography[i],
                           estimatedCamerasParams[i].K(),
                           calculatedRotation[i],
                           calculatedTranslation[i],
                           noArray());

    DEBUG_STREAM("Rotation K() of " << inputImagesPaths[i + 1] << " -> " << inputImagesPaths[i]);
    for (int o = 0; o < calculatedRotation[i].size(); o++) {
      DEBUG_STREAM(" R[" << o << "] = " << calculatedRotation[i][o]);
    }

    DEBUG_STREAM("Translation K() of " << inputImagesPaths[i + 1] << " -> " << inputImagesPaths[i]);
    for (int o = 0; o < calculatedTranslation[i].size(); o++) {
      DEBUG_STREAM(" t[" << o << "]= " << calculatedTranslation[i][o]);
    }

  }
}

void matchFeatures() {
  BestOf2NearestMatcher	matcher(false,	match_conf);
  vector<MatchesInfoSerializer> serMatches;
  FileStorage fs;

  if (parser->has("match") || !checkFileExists(featuresFile)) {
    matcher(features,	pairwiseMatches);
    matcher.collectGarbage();

    serMatches = vector<MatchesInfoSerializer>(pairwiseMatches.size());

    for (int i = 0; i < pairwiseMatches.size(); i++) {
      serMatches[i] = MatchesInfoSerializer(pairwiseMatches[i]);
    }

    fs = FileStorage(featuresFile, FileStorage::APPEND);
    fs << "matches" << serMatches;
    fs.release();
    return;
  }

  fs = FileStorage(featuresFile, FileStorage::READ);
  fs["matches"] >> serMatches;
  fs.release();

  for (int i = 0; i < serMatches.size(); i++) {
    pairwiseMatches.push_back(*serMatches[i].matches);
  }

}

int main(int argc, char **argv) {
  parser = makePtr<CommandLineParser>(argc, argv, keys);

  Debug::setEnable(parser->has("v"));
  enableGui = parser->has("show");

  // From the data source
  specifiedCameraParams.ppy = 2708.765 * scaleFactor;
  specifiedCameraParams.ppx = 1775.895 * scaleFactor;
  specifiedCameraParams.focal = 4419.441 * scaleFactor;

  versionPrinting();
  parserFinder();

  TRACE_LINE(__FILE__, __LINE__);
  parserInputImagesFiles();

  LOG("Initializing Vectors");
  features = vector<ImageFeatures>(inputImagesPaths.size());
  homography = vector<Mat>(inputImagesPaths.size() - 1);
  calculatedRotation = vector<vector<Mat>>(inputImagesPaths.size() - 1);
  calculatedTranslation = vector<vector<Mat>>(inputImagesPaths.size() - 1);

  imageCorners.push_back(Point2f(0, 0));
  imageCorners.push_back(Point2f(scaled.width, 0));
  imageCorners.push_back(Point2f(scaled.width, scaled.height));
  imageCorners.push_back(Point2f(0, scaled.height));

  LOG("Detecting Features");
  parseFeatures();

  LOG("Matching Features");
  matchFeatures();

  LOG("Calculate Homography Matrix");
  calcHomographyMatrix();

  LOG("Estimate Camera Parameters");
  estimateCameraParams();

//  LOG("Adjust Camera Parameters");
//  adjustCameraParams();

  LOG("Decompose Rotational and Translational Matrix");
  decomoposeHMatrix();

  LOG("Testing projection");
  compareProjectedPoints();

  if (enableGui) {
    drawGui();
  }

  return 0;

}



