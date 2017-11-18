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
#ifndef TRACKER_H
#define TRACKER_H

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

class Tracker {
public:
  Tracker(CommandLineParser parser, string name, Ptr<Feature2D> detector,
          Ptr<DescriptorMatcher> matcher);
  Tracker(CommandLineParser parser, string name, Ptr<Feature2D> detector);
  Tracker(CommandLineParser parser, string name);

  void track(Mat img1, Mat img2);
  void track(Mat img1, Mat img2, int k);

  void matchesToKeypoints(vector<KeyPoint> &kp1, vector<KeyPoint> &kp2);
  void matchesToPoints(vector<Point2f> &p1, vector<Point2f> &p2);

  void setMatchingThreshold(double threshold);

  void show();

protected:
  vector<KeyPoint> keypoints[2];
  Mat descriptors[2];
  Mat inputImage[2];

  virtual void runExtract();

  virtual void runTrack();

  virtual void runFilter();

  virtual void updateOutputImage();

private:
  bool showEnable;
  string name;
  Ptr<Feature2D> detector;
  Ptr<DescriptorMatcher> matcher;
  vector<DMatch> matches;
  Mat outputImage[3];
  double goodTh;
  double minDist;
  double maxDist;

  void _runExtract();

  void _runTrack();

  void _runFilter();
};

#endif /* TRACKER_H */
