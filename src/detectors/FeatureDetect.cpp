/**
* @file FeatureDetect.cpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief Feature Detect clas implementation
*
*/
#include <detectors/FeatureDetect.hpp>

using namespace cv::xfeatures2d;
using namespace std;

FeatureDetect::FeatureDetect(CommandLineParser parser, string name) {
  this->showEnable = parser.has("show");
  this->name = name;
  this->allEnable = parser.has("all");
}

FeatureDetect::FeatureDetect(CommandLineParser parser,
  string name,
  string enableFlag) : FeatureDetect(parser, name) {
  this->enable = parser.has(enableFlag);
}

void FeatureDetect::_runDetect(Mat inputImage) {
  detector->detect(inputImage, this->keyPoints);
  inputImage.copyTo(this->inputImage);
}

void FeatureDetect::_detect(Mat inputImage) {
  Timing timing;

  cout << "Running " << this->name << endl;
  timing.start();
  this->runDetect(inputImage);
  timing.end();
  cout << "  ";
  timing.print();

  this->show();
}

void FeatureDetect::_show() {
  Mat output;

  drawKeypoints(this->inputImage,
    keyPoints,
    output,
    Scalar::all(-1),
    DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  namedWindow(this->name, WINDOW_GUI_EXPANDED);
  imshow(this->name, output);
}

void FeatureDetect::printLog(string message) {
  if (!this->debug) {
    return;
  }

  cout << "  [log] "  << this->name << ": " << message << endl;
}

void FeatureDetect::_runCompute(Mat inputImage) {
  detector->compute(inputImage, this->keyPoints, this->descriptors);
  inputImage.copyTo(this->inputImage);
}

void FeatureDetect::runDetect(Mat inputImage) {
  if (!(this->enable || this->allEnable)) {
      return;
  }

  printLog("Running FeatureDetect::runDetect");

  this->_runDetect(inputImage);
}

void FeatureDetect::detect(Mat inputImage) {
  if (!(this->enable || this->allEnable)) {
      return;
  }

  printLog("Running FeatureDetect::detect");

  this->_detect(inputImage);
}

void FeatureDetect::show() {
  if (!(this->enable || this->allEnable)) {
      return;
  }

  if (!this->showEnable) {
      return;
  }

  printLog("Running FeatureDetect::show");

  this->_show();
}

void FeatureDetect::runCompute(Mat inputImage) {
  if (!(this->enable || this->allEnable)) {
      return;
  }

  printLog("Running FeatureDetect::runCompute");

  this->_runCompute(inputImage);
}

string FeatureDetect::getName() {
  return this->name;
}
