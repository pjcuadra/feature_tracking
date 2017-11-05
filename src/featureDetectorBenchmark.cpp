/**
* @file FeatureDetect.hpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief Feature Detection Benchmark
*
*/
#include <iostream>

// External
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Internal
#include <Timing.hpp>
#include <detectors/SurfDetect.hpp>
#include <detectors/SiftDetect.hpp>
#include <detectors/MSDDetectorDetect.hpp>
#include <detectors/StarDetectorDetect.hpp>
#include <detectors/VggDetect.hpp>
#include <detectors/HarrisCornerDetect.hpp>
#include <detectors/LucidDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

static const String keys =
       "{help h usage ? |      | Print this message   }"
       "{v              |      | Print Version        }"
       "{in             |      | Input Image Path     }"
       "{show           |      | Display images       }"
       "{all            |      | All Detectors Enable }"
       SURF_OPTIONS
       SIFT_OPTIONS
       MSDDETECTORDETECT_OPTIONS
       VGGDETECT_OPTIONS
       HARRISCORNERDETECT_OPTIONS
       LUCID_OPTIONS
       STARDETECTOR_OPTIONS
       ;


int main( int argc, char** argv ) {
  Mat inputImage, inputImageColor;
  int k = 0;
  CommandLineParser parser(argc, argv, keys);
  string inputImagePath = parser.get<string>("in");
  MSDDetectorDetect msdDetectorDetect = MSDDetectorDetect(parser);
  SurfDetect surfDetect = SurfDetect(parser);
  SiftDetect siftDetect = SiftDetect(parser);
  StarDetectorDetect starDetectorDetect = StarDetectorDetect(parser);
  VggDetect vggDetect = VggDetect(parser);
  HarrisCornerDetect harrisCornerDetect = HarrisCornerDetect(parser);
  LucidDetect lucidDetect = LucidDetect(parser);

  if (parser.has("v")) {
    cout << "OpenCV Version: " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << endl;

    cout << "C++ Standard: ";
    if( __cplusplus == 201103L ) cout << "C++11" << endl;
    else if( __cplusplus == 19971L ) cout << "C++98" << endl;
    else cout << "pre-standard C++" << endl;
  }

  inputImage = imread(inputImagePath, CV_LOAD_IMAGE_GRAYSCALE);
  inputImageColor = imread(inputImagePath, CV_LOAD_IMAGE_COLOR);

  if (inputImage.empty()) {
    cout << "Oopps! Couldn't read the inputImage!" << endl;
    return 0;
  }

  if (parser.has("show")) {
    namedWindow("Original", WINDOW_GUI_EXPANDED);
    imshow("Original", inputImage);
    namedWindow("Original Color", WINDOW_GUI_EXPANDED);
    imshow("Original Color", inputImageColor);
  }

  // Run all the dection algorithm
  surfDetect.detect(inputImage);
  siftDetect.detect(inputImage);
  harrisCornerDetect.detect(inputImage);
  vggDetect.detect(inputImage);
  starDetectorDetect.detect(inputImage);
  msdDetectorDetect.detect(inputImage);
  lucidDetect.detect(inputImageColor);

  cout << "Benchmark Finished" << endl;

  // Wait for the ESC key to be pressed
  if (parser.has("show")) {
    while (k != 27) {
      k = waitKey(0);

    }
  }
}
