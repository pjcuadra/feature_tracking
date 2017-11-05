/**
* @file FeatureDetect.cpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief <brief>
*
*/

#include <detectors/FeatureDetect.hpp>

using namespace cv::xfeatures2d;
using namespace std;

FeatureDetect::FeatureDetect(CommandLineParser parser, string name) {
  this->showEnable = parser.has("show");
  this->name = name;
}

void FeatureDetect::runDetect(Mat inputImage) {
  detector->detect(inputImage, this->keyPoints, this->descriptors);
  inputImage.copyTo(this->inputImage);
}

void FeatureDetect::detect(Mat inputImage) {
  Timing timing;

  cout << "Running " << this->name << endl;
  timing.start();
  this->runDetect(inputImage);
  timing.end();
  cout << "  ";
  timing.print();

  this->show();
}

void FeatureDetect::show() {
  Mat output;

  if (!showEnable) {
    return;
  }

  drawKeypoints(this->inputImage,
    keyPoints,
    output,
    Scalar::all(-1),
    DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  namedWindow(this->name, WINDOW_GUI_EXPANDED);
  imshow(this->name, output);
}
