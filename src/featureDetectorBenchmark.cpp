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

// External
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// Internal
#include <Debug.hpp>
#include <Timing.hpp>
#include <detectors/AdaptativeThresholdDetect.hpp>
#include <detectors/CannyDetect.hpp>
#include <detectors/FastDetect.hpp>
#include <detectors/FindContourDetect.hpp>
#include <detectors/HarrisCornerDetect.hpp>
#include <detectors/HarrisLaplaceDetect.hpp>
#include <detectors/HoughDetect.hpp>
#include <detectors/KMeanDetect.hpp>
#include <detectors/LucidDetect.hpp>
#include <detectors/MSDDetectorDetect.hpp>
#include <detectors/OtsuThresholdDetect.hpp>
#include <detectors/RoadDetect.hpp>
#include <detectors/SegmentationDetect.hpp>
#include <detectors/SiftDetect.hpp>
#include <detectors/SimpleBlobDetect.hpp>
#include <detectors/StarDetectorDetect.hpp>
#include <detectors/SurfDetect.hpp>
#include <detectors/ThresholdDetect.hpp>
#include <detectors/VggDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

static String keys =
    "{help h usage ? |      | Print this message    }"
    "{v              |      | Verbose               }"
    "{in             |      | Input Image Path      }"
    "{out            |      | Output Image Path     }"
    "{indir          |      | Input Directory Path  }"
    "{outdir         |      | Output Directory Path }"
    "{show           |      | Display images        }"
    "{all            |      | All Detectors Enable  }" +
    AdaptativeThresholdDetect::options + CannyDetect::options +
    FastDetect::options + FindContourDetect::options +
    HarrisCornerDetect::options + HarrisLaplaceDetect::options +
    HoughDetect::options + KMeanDetect::options + LucidDetect::options +
    MSDDetectorDetect::options + OtsuThresholdDetect::options +
    RoadDetect::options + SegmentationDetect::options + SiftDetect::options +
    SimpleBlobDetect::options + StarDetectorDetect::options +
    SurfDetect::options + ThresholdDetect::options + VggDetect::options;

int main(int argc, char **argv) {
  Mat inputImage, inputImageColor;
  int k = 0;

  keys += AdaptativeThresholdDetect::options;
  CommandLineParser parser(argc, argv, keys);
  string inputImagePath = parser.get<string>("in");
  vector<FeatureDetect *> algGrayScalePool, algColorPool;
  vector<string> lInputImagePath, lOutputImagePath;
  DIR *dir;
  struct dirent *ent;
  bool enableGui = parser.has("show") && !parser.has("indir");

  Debug::setEnable(parser.has("v"));
  FeatureDetect::enableLog(parser.has("v"));

  if (parser.has("v")) {
    cout << "OpenCV Version: " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION
         << endl;

    cout << "C++ Standard: ";
    if (__cplusplus == 201103L)
      cout << "C++11" << endl;
    else if (__cplusplus == 19971L)
      cout << "C++98" << endl;
    else
      cout << "pre-standard C++" << endl;
  }

  if (!(parser.has("in") || parser.has("indir"))) {
    cout << "No input file specified" << endl;
    return EXIT_FAILURE;
  }

  if (parser.has("in")) {
    lInputImagePath.push_back(parser.get<string>("in"));

    if (parser.has("out")) {
      lOutputImagePath.push_back(parser.get<string>("out"));
    }
  }

  if (parser.has("indir")) {
    if ((dir = opendir(parser.get<string>("indir").c_str())) != NULL) {
      /* print all the files and directories within directory */
      while ((ent = readdir(dir)) != NULL) {
        // cout << "file: " << ent->d_name << endl;
        if (ent->d_type != DT_REG) {
          continue;
        }

        lInputImagePath.push_back(parser.get<string>("indir") + "/" +
                                  ent->d_name);

        if (!parser.has("outdir")) {
          continue;
        }

        lOutputImagePath.push_back(parser.get<string>("outdir") + "/" +
                                   ent->d_name);
      }
      closedir(dir);
    } else {
      /* could not open directory */
      perror("");
      return EXIT_FAILURE;
    }
  }

  // Add all algorithms to the pool (COLOR ONLY)
  algColorPool.push_back(new LucidDetect(parser));
  algColorPool.push_back(new KMeanDetect(parser));

  // Add all algorithms to the pool (GRAY SCALE ONLY)
  algGrayScalePool.push_back(new FindContourDetect(parser));
  algGrayScalePool.push_back(new MSDDetectorDetect(parser));
  algGrayScalePool.push_back(new SurfDetect(parser));
  algGrayScalePool.push_back(new SiftDetect(parser));
  algGrayScalePool.push_back(new StarDetectorDetect(parser));
  algGrayScalePool.push_back(new VggDetect(parser));
  algGrayScalePool.push_back(new HarrisCornerDetect(parser));
  algGrayScalePool.push_back(new SimpleBlobDetect(parser));
  algGrayScalePool.push_back(new CannyDetect(parser));
  algGrayScalePool.push_back(new ThresholdDetect(parser));
  algGrayScalePool.push_back(new SegmentationDetect(parser));
  algGrayScalePool.push_back(new RoadDetect(parser));
  algGrayScalePool.push_back(new AdaptativeThresholdDetect(parser));
  algGrayScalePool.push_back(new OtsuThresholdDetect(parser));
  algGrayScalePool.push_back(new FastDetect(parser));
  algGrayScalePool.push_back(new HarrisLaplaceDetect(parser));
  algGrayScalePool.push_back(new HoughDetect(parser));

  for (int idx = 0; idx < lInputImagePath.size(); idx++) {

    Debug::printMessage("Reading - " + lInputImagePath[idx]);

    inputImage = imread(lInputImagePath[idx], CV_LOAD_IMAGE_COLOR);

    if (inputImage.empty()) {
      cout << "Oopps! Couldn't read the inputImage!" << endl;
      continue;
    }

    TRACE_LINE(__FILE__, __LINE__);

    if (enableGui) {
      namedWindow("Original Color", WINDOW_GUI_EXPANDED);
      imshow("Original Color", inputImage);
    }

    TRACE_LINE(__FILE__, __LINE__);

    // Run all color detections
    for (int i = 0; i < algColorPool.size(); i++) {
      if (!algColorPool[i]->getEnable()) {
        continue;
      }

      algColorPool[i]->detect(inputImage);

      if (idx < lOutputImagePath.size()) {
        algColorPool[i]->writeImage(lOutputImagePath[idx]);
        Debug::printMessage("Writting - " + lOutputImagePath[idx]);
      }
    }

    TRACE_LINE(__FILE__, __LINE__);

    // Convert image to gray scale
    cvtColor(inputImage, inputImage, COLOR_RGB2GRAY);
    if (enableGui) {
      namedWindow("Original", WINDOW_GUI_EXPANDED);
      imshow("Original", inputImage);
    }

    TRACE_LINE(__FILE__, __LINE__);

    // Run all grary scale detection
    for (int i = 0; i < algGrayScalePool.size(); i++) {
      if (!algGrayScalePool[i]->getEnable()) {
        continue;
      }

      algGrayScalePool[i]->detect(inputImage);

      if (idx < lOutputImagePath.size()) {
        algGrayScalePool[i]->writeImage(lOutputImagePath[idx]);
        Debug::printMessage("Writting - " + lOutputImagePath[idx]);
      }
    }

    TRACE_LINE(__FILE__, __LINE__);
  }

  for (int i = 0; i < algColorPool.size(); i++) {
    algColorPool[i]->printStats();

    if (parser.has("outdir")) {
      algColorPool[i]->dumpStatsToFile(parser.get<string>("outdir") +
                                       "/stats.txt");
    }
  }

  for (int i = 0; i < algGrayScalePool.size(); i++) {
    algGrayScalePool[i]->printStats();

    if (parser.has("outdir")) {
      algGrayScalePool[i]->dumpStatsToFile(parser.get<string>("outdir") +
                                           "/stats.txt");
    }
  }

  cout << "Benchmark Finished" << endl;

  // Wait for the ESC key to be pressed
  if (enableGui) {
    while (k != 27) {
      k = waitKey(0);
    }
  }
}
