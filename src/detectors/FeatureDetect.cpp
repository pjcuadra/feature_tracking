/**
* @file FeatureDetect.cpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief Feature Detect clas implementation
*
*/

#include <fstream>

#include <detectors/FeatureDetect.hpp>

#include <Debug.hpp>

using namespace cv::xfeatures2d;
using namespace std;

bool FeatureDetect::debug = false;

FeatureDetect::FeatureDetect(CommandLineParser parser, string name)
    : timingStats(name + " - Timing", "s"),
      keyPointsStats(name + " - Keypoints", "") {
  this->showEnable = parser.has("show") && !parser.has("indir");
  this->name = name;
  this->allEnable = parser.has("all");
}

FeatureDetect::FeatureDetect(CommandLineParser parser, string name,
                             string enableFlag)
    : FeatureDetect(parser, name) {
  this->enable = parser.has(enableFlag);
}

void FeatureDetect::_detect(Mat inputImage) {
  printLog(this->name);
  inputImage.copyTo(this->inputImage);
  this->runDetect();
  this->show();
}

void FeatureDetect::printLog(string message) {
  if (!this->debug) {
    return;
  }

  cout << "  [log] " << this->name << ": " << message << endl;
}

void FeatureDetect::_runCompute() {
  detector->compute(this->inputImage, this->keyPoints, this->descriptors);
}

void FeatureDetect::_runDetect() {
  TRACE_LINE(__FILE__, __LINE__);
  detector->detect(this->inputImage, this->keyPoints);
}

void FeatureDetect::runDetect() {
  Timing timing;
  if (!(this->enable || this->allEnable)) {
    return;
  }

  printLog("Running FeatureDetect::runDetect");

  timing.start();
  this->_runDetect();
  timing.end();

  this->collectStats(timing.getDelta());

  if (keyPoints.size()) {
    this->keyPointsStats.push_back(keyPoints.size());
  }
}

void FeatureDetect::detect(Mat inputImage) {
  if (!(this->enable || this->allEnable)) {
    return;
  }

  printLog("Running FeatureDetect::detect");

  this->_detect(inputImage);
}

void FeatureDetect::updateOutputImage() {
  drawKeypoints(this->inputImage, keyPoints, this->outputImage, Scalar::all(-1),
                DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}

void FeatureDetect::drawOutput() {
  printLog("Running FeatureDetect::drawOutput");
  updateOutputImage();

  imshow(this->name, this->outputImage);
}

void FeatureDetect::show() {
  if (!(this->enable || this->allEnable)) {
    return;
  }

  if (!this->showEnable) {
    updateOutputImage();
    return;
  }

  printLog("Running FeatureDetect::show");

  if (paramsString.str().size()) {
    cout << this->name << " - Params:" << endl;
    cout << paramsString.str() << endl;
  }

  namedWindow(this->name, WINDOW_GUI_EXPANDED);

  createControls();
  drawOutput();
}

void FeatureDetect::runCompute() {
  if (!(this->enable || this->allEnable)) {
    return;
  }

  printLog("Running FeatureDetect::runCompute");

  this->_runCompute();
}

string FeatureDetect::getName() { return this->name; }

// TODO: Refactor this to be a separate class or use an existing one
void FeatureDetect::generateStatsString() {}

void FeatureDetect::printStats() {
  if (!(this->enable || this->allEnable)) {
    return;
  }

  if (paramsString.str().size()) {
    cout << this->name << " - Params:" << endl;
    cout << paramsString.str();
  }

  cout << timingStats.str();
  cout << keyPointsStats.str();
  generateStatsString();
  cout << statsString.str();
}

void FeatureDetect::dumpStatsToFile(string path) {
  if (!(this->enable || this->allEnable)) {
    return;
  }
  ofstream outFile(path, std::ofstream::app);

  if (paramsString.str().size()) {
    outFile << this->name << " - Params:" << endl;
    outFile << paramsString.str();
  }

  outFile << timingStats.str();
  outFile << keyPointsStats.str();
  generateStatsString();
  outFile << statsString.str();
}

void FeatureDetect::collectStats(double delta) {
  this->timingStats.push_back(delta);
}

void FeatureDetect::createControls() {}

void FeatureDetect::enableLog(bool enable) { FeatureDetect::debug = enable; }

void FeatureDetect::writeImage(string path) {
  if (!(this->enable || this->allEnable)) {
    return;
  }
  imwrite(path, this->outputImage);
}

bool FeatureDetect::getEnable() { return this->enable || this->allEnable; }
