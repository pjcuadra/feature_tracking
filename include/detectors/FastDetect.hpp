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
#ifndef FASTDETECT_H
#define FASTDETECT_H

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

class FastDetect : public FeatureDetect {
public:
  /** Comand line parser options */
  static const String options;

  /**
   * FAST feature detection
   * @param parser Comand Line Parser
   */
  FastDetect(CommandLineParser parser) : FeatureDetect(parser, "FAST", "fast") {
    fastTh = parser.get<int>("fast_th");
    nonmax = parser.get<bool>("fast_nm");

    this->fastDetector = FastFeatureDetector::create(
        fastTh, nonmax, FastFeatureDetector::TYPE_9_16);
    paramsString << "  FAST Threshold: " << fastTh << endl;
    paramsString << "  FAST Non Max Supression: " << nonmax << endl;
  }

protected:
  /**
   * Set the Non-Max Suppression flag
   * @param value Value to set to the flag
   */
  void setNonMaxSupression(bool value) { this->nonmax = value; }

  /**
   * Run feature detection algorithm
   */
  virtual void runDetect() {
    fastDetector->setThreshold(fastTh);
    fastDetector->setNonmaxSuppression(nonmax);
    this->fastDetector->detect(this->inputImage, this->keyPoints);
  }

  /**
   * Trackbar on change event callback
   * @param state current trackbar possition
   * @param ptr pointer to the user data
   */
  static void onChange(int state, void *ptr) {
    FastDetect *that = (FastDetect *)ptr;
    that->redetect();
  }

  /**
   * Button on click event callback
   * @param state Button state
   * @param ptr   User data pointer
   */
  static void onClick(int state, void *ptr) {
    FastDetect *that = (FastDetect *)ptr;
    that->setNonMaxSupression(state != 0);
    that->redetect();
  }

  /**
   * Create the window's controls
   */
  virtual void createControls() {
    /// Create a Trackbar for user to enter threshold
    createTrackbar("FAST Threshold", this->name, &fastTh, 255, onChange, this);
    createButton("Non Max Suppresion", onClick, this, QT_CHECKBOX, true);
  }

private:
  /** FAST Threshold */
  int fastTh;
  /** Non-Max Suppression flag */
  bool nonmax;
  /** FAST Detector */
  Ptr<FastFeatureDetector> fastDetector;
};

const String FastDetect::options = "{fast    |      | FAST Enable        }"
                                   "{fast_th | 10   | FAST Theshold      }"
                                   "{fast_nm | true | NonMax Suppression }";

#endif /* FASTDETECT_H */
