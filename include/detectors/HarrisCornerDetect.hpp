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
#ifndef HARRISCORNERDETECT_H
#define HARRISCORNERDETECT_H

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

class HarrisCornerDetect : public FeatureDetect {
public:
  /** Comand line parser options */
  static const String options;

  /**
   * Harris corner detection
   * @param parser Comand Line Parser
   */
  HarrisCornerDetect(CommandLineParser parser)
      : FeatureDetect(parser, "Harris Corner", "harris") {
    this->blockSize = parser.get<int>("harris_bz");
    this->apertureSize = parser.get<int>("harris_ap");
    // this->harrisThreshold = parser.get<int>("harris_th");
    this->kh = parser.get<double>("harris_k");
    paramsString << "  Block Size: " << blockSize << endl;
    paramsString << "  Aperture Size: " << apertureSize << endl;
    paramsString << "  K: " << kh << endl;
    // paramsString << "  Drawing Threshold: " << harrisThreshold << endl;
  }

protected:
  /**
   * Run feature detection algorithm
   */
  virtual void runDetect() {
    cornerHarris(this->inputImage, this->tmpImage, this->blockSize,
                 this->apertureSize, this->kh, BorderTypes::BORDER_DEFAULT);
  }

  /**
   * Update the output image
   */
  virtual void updateOutputImage() {
    /// Normalizing
    normalize(this->tmpImage, this->tmpImage, 0, 255, NORM_MINMAX, CV_32FC1,
              Mat());
    convertScaleAbs(this->tmpImage, this->tmpImage);

    // inputImage.copyTo(outputHarris);

    /// Drawing a circle around corners
    // for (int j = 0; j < outputHarrisNorm.rows; j++) {
    //   for (int i = 0; i < outputHarrisNorm.cols; i++) {
    //     if (outputHarrisNorm.at<float>(j, i) > harrisThreshold) {
    //       circle(outputHarris, Point(i, j), 1, Scalar(255, 0, 0, 0));
    //     }
    //   }
    // }
    //
    // namedWindow("Harris", WINDOW_GUI_EXPANDED);
    // imshow("Harris", outputHarris);
    // namedWindow("Harris - Keypoints", WINDOW_GUI_EXPANDED);
    // imshow("Harris - Keypoints", outputHarrisNormScaled);

    this->tmpImage.copyTo(this->outputImage);
  }

private:
  /** Block Size */
  int blockSize;
  /** Aperture Size */
  int apertureSize;
  /** Harris Threshold */
  // int harrisThreshold;
  /** Kernel Size */
  double kh;
  /** Temporal Image storage */
  Mat tmpImage;
};

const String HarrisCornerDetect::options =
    "{harris         |      | Harris Enable        }"
    // "{harris_th      | 50   | Harris Threshold     }"
    "{harris_k       | 0.04 | Harris K             }"
    "{harris_bz      | 50   | Harris Block Size    }"
    "{harris_ap      | 31   | Harris Aperture Size }";

#endif /* HARRISCORNERDETECT_H */
