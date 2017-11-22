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
                     "{extract        |      | Extract Features      }";


// Global constants
static const int imageHeight = 5472;
static const int imageWidth = 3648;
static const double scaleFactor = 0.3;
static const float	match_conf = 0.66f;

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

  if (features[i].keypoints.empty()){
    (*finder)(images[PREV_IDX],	features[i]);
    printFeaturesStats(i);
  }

  (*finder)(images[CURR_IDX],	features[i + 1]);
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
  const string featuresFile("features.yml");

//  if (parser->has("extract") || !checkFileExists(featuresFile)) {

    // Extract all features
    for (int i = 0; i < inputImagesPaths.size() - 1; i++) {
      readImages(images, i);
      TRACE_LINE(__FILE__, __LINE__);

      // Get the features
      findFeatures(images, i);
    }

    FileStorage fs = FileStorage(featuresFile, FileStorage::WRITE);
    fs.write("features", features);
    fs.release();
    return;
//  }

//  FileStorage fs = FileStorage(featuresFile, FileStorage::READ);
//  fs.read("features", features);
//  fs.release();

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

void drawGui() {
  Mat images[2];
  Mat imagesWithFeatures[2];
  Mat matchesImage;
  stringstream ss;
  MatchesInfo currMatch;
  Ptr<Mosaic> mosaic;
  Mat warpedImage;

  namedWindow(image1Window, WINDOW_GUI_EXPANDED);
  namedWindow(image2Window, WINDOW_GUI_EXPANDED);
  namedWindow("Matches", WINDOW_GUI_EXPANDED);

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
    images[PREV_IDX].copyTo(warpedImage);
    mosaic = makePtr<Mosaic>("Warped");
//    mosaic->setReference(images[PREV_IDX]);
//    mosaic->pushImage(images[CURR_IDX], homography[i]);
//    mosaic->show();
    warpPerspective(warpedImage, warpedImage,  homography[i], warpedImage.size(), INTER_LINEAR /*| WARP_INVERSE_MAP*/);

    warpedImage += images[CURR_IDX];

    imshow("Warped",  warpedImage);

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
  }
}

void calcHomographyMatrix() {
  MatchesInfo currInfo;

  for (int i = 0; i < inputImagesPaths.size() - 1; i++) {
    currInfo = findMatchInfo(i, i + 1);
    currInfo.H.copyTo(homography[i]); // BestOf2NearestMatche calculates Homography for us using RANSAC
    DEBUG_STREAM("Homography of " << inputImagesPaths[i + 1] << " -> " << inputImagesPaths[i]);
    DEBUG_STREAM(" H = " << homography[i]);
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
      DEBUG_STREAM(" R[" << o << "] = " << calculatedRotation[i][0]);
    }

    DEBUG_STREAM("Translation of " << inputImagesPaths[i + 1] << " -> " << inputImagesPaths[i]);
    for (int o = 0; o < calculatedTranslation[i].size(); o++) {
      DEBUG_STREAM(" t[" << o << "]= " << calculatedTranslation[i][0]);
    }

  }
}

int main(int argc, char **argv) {
  BestOf2NearestMatcher	matcher(false,	match_conf);

  parser = makePtr<CommandLineParser>(argc, argv, keys);

  Debug::setEnable(parser->has("v"));
  enableGui = parser->has("show");

  // From the data source
  specifiedCameraParams.ppx = 2708.765 * scaleFactor;
  specifiedCameraParams.ppy = 1775.895 * scaleFactor;
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

  LOG("Detecting Features");
  parseFeatures();

  LOG("Matching Features");
  matcher(features,	pairwiseMatches);
  matcher.collectGarbage();

  LOG("Calculate Homography Matrix");
  calcHomographyMatrix();

  LOG("Estimate Camera Parameters");
  estimateCameraParams();

  LOG("Decompose Rotational and Translational Matrix");
  decomoposeHMatrix();

  if (enableGui) {
    drawGui();
  }

  return 0;

}



