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
#ifndef CANNYDETECT_H
#define CANNYDETECT_H

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

class CannyDetect : public FeatureDetect {
public:
  /** Comand line parser options */
  static const String options;

  /**
   * Adaptative Threshold feature detection
   * @param parser Comand Line Parser
   */
  CannyDetect(CommandLineParser parser)
      : FeatureDetect(parser, "Canny", "canny") {
    this->low_th = parser.get<int>("canny_low_th");
  }

protected:
  /**
   * Run feature detection algorithm
   */
  virtual void runDetect() {
    this->inputImage.copyTo(this->tmpImage);

    for (int i = 0; i < this->cascade_blur; i++) {
      /// Reduce noise with a kernel 3x3
      blur(this->tmpImage, this->tmpImage,
           Size(this->blur_size, this->blur_size));
    }

    /// Canny detector
    Canny(this->tmpImage, this->tmpImage, this->low_th,
          this->low_th * this->ratio, 3);
  }

  /**
   * Update the output image
   */
  virtual void updateOutputImage() { this->tmpImage.copyTo(this->outputImage); }

  /**
   * Trackbar on change event callback
   * @param state current trackbar possition
   * @param ptr pointer to the user data
   */
  static void onChange(int state, void *ptr) {
    CannyDetect *that = (CannyDetect *)ptr;

    that->redetect();
  }

  /**
   * Create the window's controls
   */
  virtual void createControls() {
    this->ratio = 3;
    this->blur_size = 14;
    this->low_th = 5;
    this->cascade_blur = 1;

    /// Create a Trackbar for user to enter threshold
    createTrackbar("Min Threshold:", this->name, &low_th, 100, onChange, this);
    createTrackbar("Ratio", this->name, &ratio, 50, onChange, this);
    setTrackbarMin("Ratio", this->name, 1);
    createTrackbar("Blur size", this->name, &blur_size, 30, onChange, this);
    setTrackbarMin("Blur size", this->name, 3);
    createTrackbar("Cascade Blur", this->name, &cascade_blur, 30, onChange,
                   this);
  }

private:
  /** Temporal Image storage */
  Mat tmpImage;
  /** Lower threshold */
  int low_th;
  /** Blur Kernel size */
  int blur_size;
  /** Lower/Upper thresholds ration */
  int ratio;
  /** Number of cascaded blur blocks */
  int cascade_blur;
};

const String CannyDetect::options =
    "{canny        |     | Canny Enable          }"
    "{canny_low_th | 100 | Canny Lower Threshold }";

#endif /* CANNYDETECT_H */
