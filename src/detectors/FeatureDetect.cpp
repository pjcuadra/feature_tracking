/**
* @file FeatureDetect.cpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief Feature Detect clas implementation
*
*/
#include <detectors/FeatureDetect.hpp>

#include <Debug.hpp>

using namespace cv::xfeatures2d;
using namespace std;

bool FeatureDetect::debug = false;

FeatureDetect::FeatureDetect(CommandLineParser parser, string name) {
  this->showEnable = parser.has("show") && !parser.has("indir");
  this->name = name;
  this->allEnable = parser.has("all");
}

FeatureDetect::FeatureDetect(CommandLineParser parser,
  string name,
  string enableFlag) : FeatureDetect(parser, name) {
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

  cout << "  [log] "  << this->name << ": " << message << endl;
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
}

void FeatureDetect::detect(Mat inputImage) {
  if (!(this->enable || this->allEnable)) {
      return;
  }

  printLog("Running FeatureDetect::detect");

  this->_detect(inputImage);
}

void FeatureDetect::updateOutputImage() {
  drawKeypoints(this->inputImage,
    keyPoints,
    this->outputImage,
    Scalar::all(-1),
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

string FeatureDetect::getName() {
  return this->name;
}

// TODO: Refactor this to be a separate class or use an existing one
void FeatureDetect::printStats() {
  double sum = accumulate(timeDeltas.begin(), timeDeltas.end(), 0.0);
  double mean = 0;
  vector<double> diff(timeDeltas.size());
  double stdDev = 0, sq_sum = 0;
  double max = 0;
  double min = 0;

  if (!(this->enable || this->allEnable)) {
      return;
  }

  printLog("Running FeatureDetect::printStats");

  mean = sum / timeDeltas.size();
  max = *max_element(timeDeltas.begin(), timeDeltas.end());
  min = *min_element(timeDeltas.begin(), timeDeltas.end());

  transform(timeDeltas.begin(), timeDeltas.end(), diff.begin(), [mean](double x) { return x - mean; });

  sq_sum = inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  stdDev = sqrt(sq_sum / timeDeltas.size());

  cout << this->name << " - Stats: " << endl;
  cout << "  " << "Data Set size: " << timeDeltas.size() << endl;
  cout << "  " << "Mean: " << mean << endl;
  cout << "  " << "Std. Dev.: " << stdDev << endl;
  cout << "  " << "Range: [" << max << ", " << min << "]"<< endl;
}

void FeatureDetect::collectStats(double delta) {
  this->timeDeltas.push_back(delta);
}

void FeatureDetect::createControls() {

}

void FeatureDetect::enableLog(bool enable) {
  FeatureDetect::debug = enable;
}
