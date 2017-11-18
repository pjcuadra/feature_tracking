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

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

// Internal
#include <trackers/Tracker.hpp>
#include <util/Debug.hpp>
#include <util/Mosaic.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

static String keys = "{help h usage ? |      | Print this message    }"
                     "{v              |      | Verbose               }"
                     "{out            |      | Output Image Path     }"
                     "{indir          |      | Input Directory Path  }"
                     "{show           |      | Display images        }"
                     "{all            |      | Display images        }";

static const int imageWidth = 5472;
static const int imageHeight = 3648;

vector<Mat> inputImages;
bool enableGui;
int key = 0;

void parserInputImages(CommandLineParser parser);
void versionPrinting(CommandLineParser parser);
bool waitAndContinue();

int main(int argc, char **argv) {
  CommandLineParser parser(argc, argv, keys);
  Ptr<SURF> surf = SURF::create(2000);
  Ptr<Feature2D> detector = surf;
  Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_L2, true);
  vector<Point2f> points[2];
  Tracker tracker(parser, "SURF Tracker", detector, matcher);
  Mat tmpImage;
  Mat homography;
  Mosaic mosaic;

  Debug::setEnable(parser.has("v"));
  enableGui = parser.has("show");

  versionPrinting(parser);

  surf->setExtended(true);
  tracker.setMatchingThreshold(0.06);

  parserInputImages(parser);

  mosaic.setReference(inputImages[0]);

  for (int i = 0; i < inputImages.size() - 1; i++) {
    tracker.track(inputImages[i], inputImages[i + 1]);
    tracker.show();

    TRACE_LINE(__FILE__, __LINE__);

    tracker.matchesToPoints(points[i + 1], points[i]);

    TRACE_LINE(__FILE__, __LINE__);

    homography = findHomography(points[i + 1], points[i], RANSAC);

    TRACE_LINE(__FILE__, __LINE__);

    mosaic.setImage(inputImages[i + 1], homography);

    TRACE_LINE(__FILE__, __LINE__);
    mosaic.show();

    TRACE_LINE(__FILE__, __LINE__);

    if (!waitAndContinue()) {
      break;
    }
  }

  // Wait for the ESC key to be pressed
  if (enableGui) {
    while (key != 27) {
      key = waitKey(0);
    }
  }
}

void parserInputImages(CommandLineParser parser) {
  vector<string> lInputImagePath;
  string fileName;
  Mat tmpImage;
  struct dirent *ent;
  DIR *dir;

  if (!parser.has("indir")) {
    cout << "No input file specified" << endl;
    exit(-1);
  }

  // Add all directory's files to the vector
  if ((dir = opendir(parser.get<string>("indir").c_str())) != NULL) {
    /* print all the files and directories within directory */
    while ((ent = readdir(dir)) != NULL) {
      // cout << "file: " << ent->d_name << endl;
      if (ent->d_type != DT_REG) {
        continue;
      }

      lInputImagePath.push_back(parser.get<string>("indir") + "/" +
                                ent->d_name);
    }

    sort(lInputImagePath.begin(), lInputImagePath.end());
    closedir(dir);
  } else {
    /* could not open directory */
    perror("Could Not open specified directory");
    exit(EXIT_FAILURE);
  }

  for (int idx = 0; idx < lInputImagePath.size(); idx++) {
    Debug::printMessage("Reading - " + lInputImagePath[idx]);

    tmpImage = imread(lInputImagePath[idx], CV_LOAD_IMAGE_COLOR);

    if (tmpImage.empty()) {
      cout << "Oopps! Couldn't read the currInputImage!" << endl;
      continue;
    }

    inputImages.push_back(tmpImage);
  }
}

void versionPrinting(CommandLineParser parser) {
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
