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
#ifndef FINDCONTOURDETECT_H
#define FINDCONTOURDETECT_H

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

class FindContourDetect : public FeatureDetect {
public:
  /** Comand line parser options */
  static const String options;

  /**
   * Find contour feature detection
   * @param parser Comand Line Parser
   */
  FindContourDetect(CommandLineParser parser)
      : FeatureDetect(parser, "FindContour", "findcontour") {}

protected:
  void setFill(bool enable) { this->fill = enable; }

  bool getFill() { return this->fill; }

  /**
   * Run feature detection algorithm
   */
  virtual void runDetect() {
    // Pixels classification
    blur(this->inputImage, this->tmpImage, Size(100, 100));
    threshold(this->tmpImage, this->tmpImage, 0, 255,
              THRESH_BINARY | THRESH_OTSU);

    // Segmantation
    findContours(this->tmpImage, contours0, hierarchy, RETR_TREE,
                 CHAIN_APPROX_SIMPLE);

    this->createColorsVector();
  }

  /**
   * Update the output image
   */
  virtual void updateOutputImage() {
    cvtColor(this->inputImage, this->outputImage, CV_GRAY2RGB);

    for (int idx = 0; idx < contours0.size(); idx++) {
      if (getFill()) {
        drawContours(this->outputImage, contours0, idx, colorsVec[idx], FILLED,
                     8, hierarchy);
      } else {
        drawContours(this->outputImage, contours0, idx, colorsVec[idx], 20);
      }
    }
  }

  /**
   * Button on click event callback
   * @param state Button state
   * @param ptr   User data pointer
   */
  static void onClick(int state, void *ptr) {
    FindContourDetect *that = (FindContourDetect *)ptr;

    that->setFill(state != 0);
    that->drawOutput();
  }

  /**
   * Create the window's controls
   */
  virtual void createControls() {
    createButton("Fill Regions", onClick, this, QT_CHECKBOX, true);
  }

private:
  /** Contours vector */
  vector<vector<Point>> contours0;
  /** Contours hierarchy vector */
  vector<Vec4i> hierarchy;
  /** Temporal Image storage */
  Mat tmpImage;
  /** Contours colors vector */
  vector<Scalar> colorsVec;
  /** Contours filling flag */
  bool fill = false;

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

const String FindContourDetect::options =
    "{findcontour | | Find Contour Enable }";

#endif /* FINDCONTOURDETECT_H */
