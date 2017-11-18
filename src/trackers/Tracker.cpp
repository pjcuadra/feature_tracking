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
#include <trackers/Tracker.hpp>

#include <util/Debug.hpp>

using namespace cv::xfeatures2d;
using namespace std;

static const double gth = 0.3;

Tracker::Tracker(CommandLineParser parser, string name) {
  this->showEnable = parser.has("show");
  this->name = name;

  this->goodTh = gth;
}

Tracker::Tracker(CommandLineParser parser, string name, Ptr<Feature2D> detector)
    : detector(detector) {
  this->showEnable = parser.has("show");
  this->name = name;

  this->goodTh = gth;
}

Tracker::Tracker(CommandLineParser parser, string name, Ptr<Feature2D> detector,
                 Ptr<DescriptorMatcher> matcher)
    : detector(detector), matcher(matcher) {
  this->showEnable = parser.has("show");
  this->name = name;

  this->goodTh = gth;
}

void Tracker::runExtract() {
  STACK_TRACE(__FUNCTION__);
  for (int i = 0; i < 2; i++) {
    this->detector->detectAndCompute(this->inputImage[i], noArray(),
                                     this->keypoints[i], this->descriptors[i]);
  }
}

void Tracker::_runExtract() {
  STACK_TRACE(__FUNCTION__);
  assert(!detector.empty());

  this->runExtract();
}

void Tracker::runTrack() {
  STACK_TRACE(__FUNCTION__);

  maxDist = 0;
  minDist = 100000;

  TRACE_LINE(__FILE__, __LINE__);
  cout << this->descriptors[0].size() << endl;
  cout << this->descriptors[1].size() << endl;

  this->matcher->match(this->descriptors[0], this->descriptors[1],
                       this->matches);

  std::sort(matches.begin(), matches.end(),
            [](DMatch a, DMatch b) { return a.distance < b.distance; });

  TRACE_LINE(__FILE__, __LINE__);

  minDist = matches[0].distance;
  maxDist = matches[matches.size() - 1].distance;
}

void Tracker::_runTrack() {
  STACK_TRACE(__FUNCTION__);
  assert(!matcher.empty());

  this->runTrack();
}

void Tracker::runFilter() {
  STACK_TRACE(__FUNCTION__);

  double normDistance;
  vector<DMatch> goodMatches;

  for (int i = 0; i < this->descriptors[0].rows; i++) {
    normDistance = (matches[i].distance - minDist) / (maxDist - minDist);

    // Since it's pre-ordered
    if (normDistance > this->goodTh) {
      break;
    }

    goodMatches.push_back(matches[i]);
  }

  matches.clear();

  matches = goodMatches;
}

void Tracker::matchesToKeypoints(vector<KeyPoint> &kp1, vector<KeyPoint> &kp2) {
  kp1.clear();
  kp2.clear();

  for (int i = 0; i < this->matches.size(); i++) {
    kp2.push_back(this->keypoints[0][matches[i].queryIdx]);
    kp1.push_back(this->keypoints[1][matches[i].trainIdx]);
  }
}

void Tracker::matchesToPoints(vector<Point2f> &p1, vector<Point2f> &p2) {
  vector<KeyPoint> kp1;
  vector<KeyPoint> kp2;
  p1.clear();
  p2.clear();

  matchesToKeypoints(kp1, kp2);

  cout << kp1.size() << endl;
  cout << kp2.size() << endl;

  KeyPoint::convert(kp1, p1);
  KeyPoint::convert(kp2, p2);

  cout << p1.size() << endl;
  cout << p2.size() << endl;
}

void Tracker::_runFilter() {
  STACK_TRACE(__FUNCTION__);
  assert(!matcher.empty());

  this->runFilter();
}

void Tracker::track(Mat img1, Mat img2) {
  STACK_TRACE(__FUNCTION__);

  this->inputImage[0] = img1;
  this->inputImage[1] = img2;

  this->_runExtract();
  this->_runTrack();
  this->_runFilter();
}

void Tracker::track(Mat img1, Mat img2, int k) {
  STACK_TRACE(__FUNCTION__);
  vector<DMatch> tmp;

  this->inputImage[0] = img1;
  this->inputImage[1] = img2;

  this->_runExtract();
  this->_runTrack();
  this->_runFilter();

  tmp = matches;
  matches.clear();

  if (k > tmp.size()) {
    k = tmp.size();
  }

  for (int i = 0; i < k; i++) {
    matches.push_back(tmp[i]);
  }
}

void Tracker::updateOutputImage() {
  STACK_TRACE(__FUNCTION__);

  assert(this->inputImage[0].rows != 0 && this->inputImage[0].cols != 0);
  assert(this->inputImage[1].rows != 0 && this->inputImage[1].cols != 0);
  assert(this->keypoints[0].size() != 0);
  assert(this->keypoints[1].size() != 0);
  assert(matches.size() != 0);

  drawMatches(this->inputImage[0], this->keypoints[0], this->inputImage[1],
              this->keypoints[1], matches, this->outputImage[2],
              Scalar::all(-1), Scalar::all(-1), vector<char>(),
              DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS |
                  DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  drawKeypoints(this->inputImage[0], this->keypoints[0], this->outputImage[0],
                Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  drawKeypoints(this->inputImage[1], this->keypoints[1], this->outputImage[1],
                Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}

/**
 * Show the GUI
 */
void Tracker::show() {
  STACK_TRACE(__FUNCTION__);

  if (!this->showEnable) {
    return;
  }

  updateOutputImage();

  namedWindow(this->name + " - Image - 1", WINDOW_GUI_EXPANDED);
  namedWindow(this->name + " - Image - 2", WINDOW_GUI_EXPANDED);
  namedWindow(this->name, WINDOW_GUI_EXPANDED);
  imshow(this->name + " - Image - 1", this->outputImage[0]);
  imshow(this->name + " - Image - 2", this->outputImage[1]);
  imshow(this->name, this->outputImage[2]);
}

void Tracker::setMatchingThreshold(double threshold) {
  this->goodTh = threshold;
}
