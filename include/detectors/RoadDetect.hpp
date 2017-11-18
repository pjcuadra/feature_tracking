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
#ifndef ROADDETECT_H
#define ROADDETECT_H

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <detectors/FeatureDetect.hpp>
#include <util/Debug.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

class RoadDetect : public FeatureDetect {
public:
  /** Comand line parser options */
  static const String options;

  /**
   * Road detection
   * @param parser Comand Line Parser
   */
  RoadDetect(CommandLineParser parser)
      : FeatureDetect(parser, "RoadDetect", "roaddetect") {}

  /**
   * Set fill contours flag
   * @param enable Enable value
   */
  void setFill(bool enable) { this->fill = enable; }

  /**
   * Get fill contour flag
   * @return fill contour flag value
   */
  bool getFill() { return this->fill; }

protected:
  /**
   * Run feature detection algorithm
   */
  virtual void runDetect() {
    TRACE_LINE(__FILE__, __LINE__);

    contours0.clear();
    this->hierarchy.clear();
    blur(this->inputImage, this->tmpImage, Size(100, 100));
    threshold(this->tmpImage, this->tmpImage, 0, 255,
              THRESH_BINARY | THRESH_OTSU);
    findContours(this->tmpImage, this->contours0, this->hierarchy, RETR_TREE,
                 CHAIN_APPROX_SIMPLE);

    this->createColorsVector();
  }

  /**
   * Update the output image
   */
  virtual void updateOutputImage() {
    RotatedRect ellipse;
    float exentricity;
    stringstream message;

    TRACE_LINE(__FILE__, __LINE__);

    this->inputImage.copyTo(this->outputImage);

    cvtColor(this->outputImage, this->outputImage, CV_GRAY2RGB);

    for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
      if (this->contours0[idx].size() < 5) {
        continue;
      }

      ellipse = fitEllipse(this->contours0[idx]);
      exentricity = ellipse.size.width / ellipse.size.height;

      if (exentricity > (float)this->exentricityMax / 100.0f) {
        continue;
      }

      if (exentricity < (float)this->exentricityMin / 100.0f) {
        continue;
      }

      message.str("");
      message << "Excentricity: " << exentricity;

      Debug::addPoint(__FILE__, __LINE__, message.str());

      if (getFill()) {
        TRACE_LINE(__FILE__, __LINE__);
        drawContours(this->outputImage, this->contours0, idx, colorsVec[idx],
                     FILLED, 8, hierarchy);
      } else {
        TRACE_LINE(__FILE__, __LINE__);
        drawContours(this->outputImage, this->contours0, idx, colorsVec[idx],
                     10);
      }
    }
  }

  /**
   * Trackbar on change event callback
   * @param pos current trackbar possition
   * @param ptr pointer to the user data
   */
  static void onChange(int pos, void *ptr) {
    RoadDetect *that = (RoadDetect *)ptr;

    that->redetect();
  }

  static void onClick(int state, void *ptr) {
    RoadDetect *that = (RoadDetect *)ptr;

    that->setFill(state != 0);
    that->drawOutput();
  }

  /**
   * Create the window's controls
   */
  virtual void createControls() {
    exentricityMin = 0;
    exentricityMax = 100;

    /// Create a Trackbar for user to enter threshold
    createTrackbar("Excentricity Min", this->name, &exentricityMin, 100,
                   onChange, this);
    createTrackbar("Excentricity Max", this->name, &exentricityMax, 100,
                   onChange, this);

    createButton("Fill Regions", onClick, this, QT_CHECKBOX, true);
  }

private:
  /** Temporal Image storage */
  Mat tmpImage;
  bool fill;
  int exentricityMin;
  int exentricityMax;
  vector<vector<Point>> contours0;
  vector<Vec4i> hierarchy;
  vector<Scalar> colorsVec;

  /**
   * Create the contours colors vector
   */
  void createColorsVector() {
    colorsVec.clear();

    for (int idx = 0; idx < contours0.size(); idx++) {
      colorsVec.push_back(Scalar(rand() & 255, rand() & 255, rand() & 255));
    }
  }
};

const String RoadDetect::options = "{roaddetect | | Road Detect Enable }";

#endif /* ROADDETECT_H */
