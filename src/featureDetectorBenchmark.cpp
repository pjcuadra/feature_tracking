/**
* @file FeatureDetect.hpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief Feature Detection Benchmark
*
*/
#include <iostream>
#include <dirent.h>

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
#include <Debug.hpp>
#include <detectors/SurfDetect.hpp>
#include <detectors/SiftDetect.hpp>
#include <detectors/MSDDetectorDetect.hpp>
#include <detectors/StarDetectorDetect.hpp>
#include <detectors/VggDetect.hpp>
#include <detectors/HarrisCornerDetect.hpp>
#include <detectors/LucidDetect.hpp>
#include <detectors/SimpleBlobDetect.hpp>
#include <detectors/CannyDetect.hpp>
#include <detectors/ThresholdDetect.hpp>
#include <detectors/AdaptativeThresholdDetect.hpp>
#include <detectors/SegmentationDetect.hpp>
#include <detectors/FindContourDetect.hpp>
#include <detectors/RoadDetect.hpp>
#include <detectors/OtsuThresholdDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

static const String keys =
       "{help h usage ? |      | Print this message   }"
       "{v              |      | Print Version        }"
       "{in             |      | Input Image Path     }"
       "{indir          |      | Input Directory Path }"
       "{show           |      | Display images       }"
       "{all            |      | All Detectors Enable }"
       SURF_OPTIONS
       SIFT_OPTIONS
       MSDDETECTORDETECT_OPTIONS
       VGGDETECT_OPTIONS
       HARRISCORNERDETECT_OPTIONS
       LUCID_OPTIONS
       STARDETECTOR_OPTIONS
       SIMPLEBLOB_OPTIONS
       CANNY_OPTIONS
       THRESHOLD_OPTIONS
       SEGMENTATION_OPTIONS
       FINDCONTOUR_OPTIONS
       ROADDETECT_OPTIONS
       ADAPTATIVTHRESHOLD_OPTIONS
       OTSUTHRESHOLD_OPTIONS
       ;


int main( int argc, char** argv ) {
  Mat inputImage, inputImageColor;
  int k = 0;
  CommandLineParser parser(argc, argv, keys);
  string inputImagePath = parser.get<string>("in");
  vector<FeatureDetect*> algGrayScalePool, algColorPool;
  list<string> lInputImagePath;
  DIR *dir;
  struct dirent *ent;

  Debug::setEnable(true);
  FeatureDetect::enableLog(true);

  if (parser.has("v")) {
    cout << "OpenCV Version: " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << endl;

    cout << "C++ Standard: ";
    if( __cplusplus == 201103L ) cout << "C++11" << endl;
    else if( __cplusplus == 19971L ) cout << "C++98" << endl;
    else cout << "pre-standard C++" << endl;
  }

  if (parser.has("in")) {
    lInputImagePath.push_back(parser.get<string>("in"));
  }

  if (parser.has("indir")) {
    if ((dir = opendir(parser.get<string>("indir").c_str())) != NULL) {
      /* print all the files and directories within directory */
      while ((ent = readdir(dir)) != NULL) {
        cout << "file: " << ent->d_name << endl;
      }
      closedir (dir);
    } else {
      /* could not open directory */
      perror ("");
      return EXIT_FAILURE;
    }
  }

  // Add all algorithms to the pool (COLOR ONLY)
  algColorPool.push_back(new LucidDetect(parser));

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

  inputImage = imread(inputImagePath, CV_LOAD_IMAGE_COLOR);

  if (inputImage.empty()) {
    cout << "Oopps! Couldn't read the inputImage!" << endl;
    return 0;
  }

  TRACE_LINE(__FILE__, __LINE__);

  if (parser.has("show") && !parser.has("indir")) {
    namedWindow("Original Color", WINDOW_GUI_EXPANDED);
    imshow("Original Color", inputImage);
  }

  TRACE_LINE(__FILE__, __LINE__);

  // Run all color detections
  for (int i = 0; i < algColorPool.size(); i++) {
    algColorPool[i]->detect(inputImage);
    algColorPool[i]->printStats();
  }

  TRACE_LINE(__FILE__, __LINE__);

  // Convert image to gray scale
  cvtColor(inputImage, inputImage, COLOR_RGB2GRAY);
  if (parser.has("show") && !parser.has("indir")) {
    namedWindow("Original", WINDOW_GUI_EXPANDED);
    imshow("Original", inputImage);
  }

  TRACE_LINE(__FILE__, __LINE__);

  // Run all grary scale detection
  for (int i = 0; i < algGrayScalePool.size(); i++) {
    algGrayScalePool[i]->detect(inputImage);
    algGrayScalePool[i]->printStats();
  }

  TRACE_LINE(__FILE__, __LINE__);

  cout << "Benchmark Finished" << endl;

  // Wait for the ESC key to be pressed
  if (parser.has("show")) {
    while (k != 27) {
      k = waitKey(0);

    }
  }
}
