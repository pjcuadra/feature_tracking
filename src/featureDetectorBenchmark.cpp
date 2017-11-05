
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

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

static const String keys =
       "{help h usage ? |      | Print this message   }"
       "{v              |      | Print Version        }"
       "{in             |      | Input Image Path     }"
       "{show           |      | Display images       }"
       SURF_OPTIONS
       SIFT_OPTIONS
       MSDDETECTORDETECT_OPTIONS
       VGGDETECT_OPTIONS
       HARRISCORNERDETECT_OPTIONS
       ;

void harrisCornerDetect(CommandLineParser parser, Mat inputImage);

int main( int argc, char** argv ) {
  Mat inputImage;
  int k = 0;
  CommandLineParser parser(argc, argv, keys);
  string inputImagePath = parser.get<string>("in");
  MSDDetectorDetect msdDetectorDetect = MSDDetectorDetect(parser);
  SurfDetect surfDetect = SurfDetect(parser);
  SiftDetect siftDetect = SiftDetect(parser);
  StarDetectorDetect starDetectorDetect = StarDetectorDetect(parser);
  VggDetect vggDetect = VggDetect(parser);
  HarrisCornerDetect harrisCornerDetect = HarrisCornerDetect(parser);

  if (!parser.has("in")) {
    cout << "No input file provided" << endl;
    return 0;
  }

  if (parser.has("v")) {
    cout << "OpenCV Version: " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << endl;

    cout << "C++ Standard: ";
    if( __cplusplus == 201103L ) cout << "C++11" << endl;
    else if( __cplusplus == 19971L ) cout << "C++98" << endl;
    else cout << "pre-standard C++" << endl;
  }

  inputImage = imread(inputImagePath, CV_LOAD_IMAGE_GRAYSCALE);

  if (inputImage.empty()) {
    cout << "Upps! Couldn't read the inputImage!" << endl;
  }

  if (parser.has("show")) {
    namedWindow("Original", WINDOW_GUI_EXPANDED);
    imshow("Original", inputImage);
  }

  surfDetect.detect(inputImage);
  siftDetect.detect(inputImage);
  harrisCornerDetect.detect(inputImage);
  vggDetect.detect(inputImage);
  starDetectorDetect.detect(inputImage);
  msdDetectorDetect.detect(inputImage);

  cout << "Benchmark Finished" << endl;

  // Wait for the ESC key to be pressed
  if (parser.has("show")) {
    while (k != 27) {
      k = waitKey(0);

    }
  }
}

void harrisCornerDetect(CommandLineParser parser, Mat inputImage) {
  Mat outputHarris, outputHarrisNorm, outputHarrisNormScaled;
  int blockSize = parser.get<int>("harris_bz");
  int apertureSize = parser.get<int>("harris_ap");
  int harrisThreshold = parser.get<int>("harris_th");
  double kh = parser.get<double>("harris_k");

  Timing timing;

  // Benchmark Corner Harris
  cout << "Running Corner Harris" << endl;

  outputHarris = Mat::zeros( inputImage.size(), CV_32FC1 );

  cout << "  Block Size: " << blockSize << endl;
  cout << "  Aperture Size: " << apertureSize << endl;
  cout << "  K: " << kh << endl;
  cout << "  Drawing Threshold: " << harrisThreshold << endl;

  timing.start();
  cornerHarris(inputImage,
    outputHarris,
    blockSize,
    apertureSize,
    kh,
    BorderTypes::BORDER_DEFAULT);
  timing.end();
  cout << "  ";
  timing.print();

  if (!parser.has("show")) {
    return;
  }

  /// Normalizing
  normalize(outputHarris,
    outputHarrisNorm,
    0,
    255,
    NORM_MINMAX,
    CV_32FC1,
    Mat());
  convertScaleAbs(outputHarrisNorm,
    outputHarrisNormScaled);

  inputImage.copyTo(outputHarris);

  /// Drawing a circle around corners
  for(int j = 0; j < outputHarrisNorm.rows; j++) {
    for(int i = 0; i < outputHarrisNorm.cols; i++) {
      if(outputHarrisNorm.at<float>(j, i) > harrisThreshold) {
        circle(outputHarris,
          Point(i, j),
          1,
          Scalar(255, 0, 0, 0));
      }
    }
  }

  namedWindow("Harris", WINDOW_GUI_EXPANDED);
  imshow("Harris", outputHarris);
  namedWindow("Harris - Keypoints", WINDOW_GUI_EXPANDED);
  imshow("Harris - Keypoints", outputHarrisNormScaled);

}
