#include <iostream>
#include <ctime>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Timing.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

static const String keys =
       "{help h usage ? |      | Print this message   }"
       "{in             |      | Input image          }"
       "{show           |      | Display images       }"
       "{surf_h         | 400  | Display images       }"
       "{sift           |      | Include SIFT         }"
       "{harris_th      | 50   | Harris Threshold     }"
       "{harris_k       | 0.04 | Harris K             }"
       "{harris_bz      | 50   | Harris Block Size    }"
       "{harris_ap      | 31   | Harris Aperture Size }"
       ;

int main( int argc, char** argv ) {

  Mat inputImage;
  Mat outputSurf, outputSift, outputHarris, outputHarrisNorm, outputHarrisNormScaled;
  Ptr<SURF> surf;
  Ptr<SIFT> sift;
  vector<KeyPoint> keyPoints;
  int k = 0;
  CommandLineParser parser(argc, argv, keys);
  string inputImagePath = parser.get<string>("in");
  Timing timing;
  int surfHessianTh = parser.get<int>("surf_h");

  // Harris detector
  int blockSize = parser.get<int>("harris_bz");
  int apertureSize = parser.get<int>("harris_ap");
  int harrisThreshold = parser.get<int>("harris_th");
  double kh = parser.get<double>("harris_k");

  cout << "OpenCV Version: " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << endl;

  cout << "C++ Standard: ";
  if( __cplusplus == 201103L ) cout << "C++11" << endl;
  else if( __cplusplus == 19971L ) cout << "C++98" << endl;
  else cout << "pre-standard C++" << endl;

  inputImage = imread(inputImagePath, CV_LOAD_IMAGE_GRAYSCALE);

  if (inputImage.empty()) {
    cout << "Upps! Couldn't read the inputImage!" << endl;
  }

  if (parser.has("show")) {
    namedWindow("Original", WINDOW_GUI_EXPANDED);
    imshow("Original", inputImage);
  }

  // Benchmark SURF
  surf = SURF::create(surfHessianTh);
  cout << "Running SURF " << endl;
  cout << "  Hessian Threshold " << surfHessianTh << endl;
  timing.start();
  surf->detect(inputImage, keyPoints);
  timing.end();
  cout << "  ";
  timing.print();

  if (parser.has("show")) {
    drawKeypoints(inputImage,
      keyPoints,
      outputSurf,
      Scalar::all(-1),
      DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    namedWindow("SURF", WINDOW_GUI_EXPANDED);
    imshow("SURF", outputSurf);
  }

  if (parser.has("sift")) {
    // Benchmark SIFT
    sift = SIFT::create();
    cout << "Running SIFT " << endl;
    timing.start();
    sift->detect(inputImage, keyPoints);
    timing.end();
    cout << "  ";
    timing.print();

    if (parser.has("show")) {
      drawKeypoints(inputImage,
        keyPoints,
        outputSift,
        Scalar::all(-1),
        DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

      namedWindow("SIFT", WINDOW_GUI_EXPANDED);
      imshow("SIFT", outputSift);
    }
  }

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

  if (parser.has("show")) {

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

    outputHarris = inputImage;

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

  cout << "Benchmark Finished" << endl;

  // Wait for the ESC key to be pressed
  if (parser.has("show")) {
    while (k != 27) {
      k = waitKey(0);

    }
  }

}
