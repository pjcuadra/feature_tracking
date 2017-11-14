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
#ifndef HOUGHDETECT_H
#define HOUGHDETECT_H

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define HOUGH_OPTIONS "{hough           |      | Hough Transform Enable  }"

class HoughDetect : public FeatureDetect {
public:
  /** Comand line parser options */
  static const String options;

  /**
   * Hough transform lines detection
   * @param parser Comand Line Parser
   */
  HoughDetect(CommandLineParser parser)
      : FeatureDetect(parser, "Hough", "hough") {
    this->ratio = 3;
    this->blur_size = 14;
    this->low_th = 27;
  }

protected:
  /**
   * Run feature detection algorithm
   */
  virtual void runDetect() {

    blur(this->inputImage, this->tmpImage,
         Size(this->blur_size, this->blur_size));

    Canny(this->tmpImage, this->tmpImage, low_th, low_th * ratio, 3);

    HoughLinesP(this->tmpImage, lines, 1, CV_PI / 180, 100, 50, 500);
  }

  /**
   * Update the output image
   */
  virtual void updateOutputImage() {
    cvtColor(this->inputImage, this->outputImage, COLOR_GRAY2BGR);

    keyPointsStats.push_back(lines.size());

    for (size_t i = 0; i < lines.size(); i++) {
      Vec4i l = lines[i];
      line(this->outputImage, Point(l[0], l[1]), Point(l[2], l[3]),
           Scalar(0, 0, 255), 3, CV_AA);
    }
  }

  /**
   * Trackbar on change event callback
   * @param state current trackbar possition
   * @param ptr pointer to the user data
   */
  static void onChange(int state, void *ptr) {
    HoughDetect *that = (HoughDetect *)ptr;

    that->redetect();
  }

  /**
   * Create the window's controls
   */
  virtual void createControls() {
    this->ratio = 3;
    this->blur_size = 14;
    this->low_th = 5;

    /// Create a Trackbar for user to enter threshold
    createTrackbar("Min Threshold:", this->name, &low_th, 100, onChange, this);
    createTrackbar("Ratio", this->name, &ratio, 50, onChange, this);
    setTrackbarMin("Ratio", this->name, 1);
    createTrackbar("Blur size", this->name, &blur_size, 30, onChange, this);
    setTrackbarMin("Blur size", this->name, 3);
  }

private:
  /** Obtained feature lines vector */
  vector<Vec4i> lines;
  /** Temporal Image storage */
  Mat tmpImage;
  /** Canny edge detector lower threshold */
  int low_th;
  /** Blur kernel size */
  int blur_size;
  /** Canny edge detector Lower/Upper thresholds ration */
  int ratio;
};

const String HoughDetect::options = "{hough | | Hough Transform Enable }";

#endif /* HOUGHDETECT_H */
