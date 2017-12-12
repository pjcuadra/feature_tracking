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
#ifndef TEST_UTIL_H
#define TEST_UTIL_H

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <util/Debug.hpp>

using namespace std;
using namespace cv;
using namespace cv::detail;

namespace testing {

template <class T> T getRandPrimitive() { return (T)rand(); }
}

Size getRandSize() {
  return Size(testing::getRandPrimitive<int>(),
              testing::getRandPrimitive<int>());
}

Point getRandPoint() {
  return Point(testing::getRandPrimitive<int>() % 1000,
               testing::getRandPrimitive<int>() % 1000);
}

KeyPoint &getRandKeyPoint() {
  KeyPoint *kp = new KeyPoint(getRandPoint(),
                              1000.0f * testing::getRandPrimitive<float>() /
                                  static_cast<float>(RAND_MAX));
  return *kp;
}

Mat &getRandMat() {
  int x0 = testing::getRandPrimitive<int>() % 100;
  int x1 = testing::getRandPrimitive<int>() % 100;
  Mat * mat = new Mat(x0, x1, CV_8S);
  double low = -500.0;
  double high = +500.0;
  randu(*mat, Scalar(low), Scalar(high));

  return *mat;
}

ImageFeatures &getRandImageFeatures() {
  Mat desc;
  ImageFeatures *feat = new ImageFeatures();
  int featuresSize = testing::getRandPrimitive<int>() % 100;

  feat->img_idx = testing::getRandPrimitive<int>();
  feat->img_size = getRandSize();

  for (int i = 0; i < featuresSize; i++) {
    feat->keypoints.push_back(getRandKeyPoint());
  }

  getRandMat().copyTo(feat->descriptors);

  return *feat;
}

#endif /* TEST_UTIL_H */
