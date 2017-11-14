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
#ifndef OTSUTHRESHOLDDETECT_H
#define OTSUTHRESHOLDDETECT_H

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

class OtsuThresholdDetect : public FeatureDetect {
public:
  /** Comand line parser options */
  static const String options;

  /**
   * Otsu Threshold feature detection
   * @param parser Comand Line Parser
   */
  OtsuThresholdDetect(CommandLineParser parser)
      : FeatureDetect(parser, "Otsu Threshold", "otsu") {}

protected:
  /**
   * Run feature detection algorithm
   */
  virtual void runDetect() {
    blur(this->inputImage, this->tmpImage, Size(100, 100));
    threshold(this->tmpImage, this->tmpImage, 0, 255,
              THRESH_BINARY | THRESH_OTSU);
  }

  /**
   * Update the output image
   */
  virtual void updateOutputImage() { this->tmpImage.copyTo(this->outputImage); }

private:
  /** Temporal Image storage */
  Mat tmpImage;
};

const String OtsuThresholdDetect::options = "{otsu | | Otsu Threshold Enable }";

#endif /* OTSUTHRESHOLDDETECT_H */
