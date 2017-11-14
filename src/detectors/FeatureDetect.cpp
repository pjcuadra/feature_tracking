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
#include <fstream>

#include <detectors/FeatureDetect.hpp>

#include <Debug.hpp>

using namespace cv::xfeatures2d;
using namespace std;

bool FeatureDetect::debug = false;

/**
 * Feature Detection Wrapper Class
 */
FeatureDetect::FeatureDetect(CommandLineParser parser, string name)
    : timingStats(name + " - Timing", "s"),
      keyPointsStats(name + " - Keypoints", "") {
  this->showEnable = parser.has("show") && !parser.has("indir");
  this->name = name;
  this->allEnable = parser.has("all");
}

/**
 * Feature Detection Wrapper Class
 */
FeatureDetect::FeatureDetect(CommandLineParser parser, string name,
                             string enableFlag)
    : FeatureDetect(parser, name) {
  this->enable = parser.has(enableFlag);
}

/**
 * Run Compute method of the detector
 */
void FeatureDetect::runCompute() {
  detector->compute(this->inputImage, this->keyPoints, this->descriptors);
}

/**
 * Wrapper run computation of the keypoint descriptors
 */
void FeatureDetect::_runCompute() {
  if (!(this->enable || this->allEnable)) {
    return;
  }

  printLog("Running FeatureDetect::_runCompute");

  this->runCompute();
}

/**
 * Run feature detection algorithm
 */
void FeatureDetect::runDetect() {
  TRACE_LINE(__FILE__, __LINE__);
  detector->detect(this->inputImage, this->keyPoints);
}

/**
 * Wrapper run detection of the keypoint
 */
void FeatureDetect::_runDetect() {
  Timing timing;
  if (!(this->enable || this->allEnable)) {
    return;
  }

  printLog("Running FeatureDetect::_runDetect");

  timing.start();
  this->runDetect();
  timing.end();

  this->collectStats(timing.getDelta());

  if (keyPoints.size()) {
    this->keyPointsStats.push_back(keyPoints.size());
  }
}

/**
 * Apply detection algorithm
 */
void FeatureDetect::detect(Mat inputImage) {
  if (!(this->enable || this->allEnable)) {
    return;
  }

  printLog("Running FeatureDetect::detect");
  inputImage.copyTo(this->inputImage);
  this->_runDetect();
  this->show();
}

/**
 * Re-apply the detection algorithm to the stored input image
 */
void FeatureDetect::redetect() {
  this->_runDetect();
  this->drawOutput();
}

/**
 * Update the output image
 */
void FeatureDetect::updateOutputImage() {
  drawKeypoints(this->inputImage, keyPoints, this->outputImage, Scalar::all(-1),
                DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}

/**
 * Draw ouput image to the GUI
 */
void FeatureDetect::drawOutput() {
  printLog("Running FeatureDetect::drawOutput");
  updateOutputImage();

  imshow(this->name, this->outputImage);
}

/**
 * Show the GUI
 */
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

/**
 * Get feature detector name
 */
string FeatureDetect::getName() { return this->name; }

/**
 * Print statistics
 */
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
  cout << statsString.str();
}

/**
 * Dump stats to file
 */
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
  outFile << statsString.str();
}

/**
 * Collect timing stats
 */
void FeatureDetect::collectStats(double delta) {
  this->timingStats.push_back(delta);
}

/**
 * Create the window's controls
 */
void FeatureDetect::createControls() {}

/**
 * Enable logging
 */
void FeatureDetect::enableLog(bool enable) { FeatureDetect::debug = enable; }

/**
 * Write image to file
 */
void FeatureDetect::writeImage(string path) {
  if (!(this->enable || this->allEnable)) {
    return;
  }
  imwrite(path, this->outputImage);
}

/**
 * Get feature detection enable state
 */
bool FeatureDetect::getEnable() { return this->enable || this->allEnable; }

/**
 * Print Log Message
 */
void FeatureDetect::printLog(string message) {
  if (!this->debug) {
    return;
  }

  cout << "  [log] " << this->name << ": " << message << endl;
}
