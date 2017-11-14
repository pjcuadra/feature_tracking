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
#ifndef KMEANSDETECT_H
#define KMEANSDETECT_H

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Debug.hpp>
#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

class KMeanDetect : public FeatureDetect {
public:
  /** Comand line parser options */
  static const String options;

  /**
   * KMean clustering feature detection
   * @param parser Comand Line Parser
   */
  KMeanDetect(CommandLineParser parser)
      : FeatureDetect(parser, "K-means", "kmeans") {}

protected:
  /**
   * Run feature detection algorithm
   */
  virtual void runDetect() {
    Mat channeslBgr[3];
    int count = this->inputImage.cols * this->inputImage.rows;
    int labelIndex = 0;
    Mat samples(count, 3, CV_32F);
    int currSample = 0;

    blur(this->inputImage, this->tmpImage, Size(100, 100));

    this->tmpImage.convertTo(this->tmpImage, CV_32F);

    split(this->tmpImage, channeslBgr);

    // Convert BGR to Mat(count, 3)
    for (int y = 0; y < this->inputImage.rows; y++) {
      for (int x = 0; x < this->inputImage.cols; x++) {
        currSample = y * this->inputImage.cols + x;

        samples.at<float>(currSample, 0) = channeslBgr[0].at<float>(y, x);
        samples.at<float>(currSample, 1) = channeslBgr[1].at<float>(y, x);
        samples.at<float>(currSample, 2) = channeslBgr[2].at<float>(y, x);
      }
    }

    TRACE_LINE(__FILE__, __LINE__);

    kmeans(samples, clusterNum, this->label,
           TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3,
           KMEANS_PP_CENTERS, centers);

    TRACE_LINE(__FILE__, __LINE__);

    this->inputImage.copyTo(this->tmpImage);
    centers.convertTo(centers, CV_8UC1);

    TRACE_LINE(__FILE__, __LINE__);

    // Color pixels using labels
    for (int x = 0; x < this->tmpImage.cols; x++) {
      for (int y = 0; y < this->tmpImage.rows; y++) {
        currSample = y * this->tmpImage.cols + x;
        labelIndex = this->label.at<uint>(Point(currSample, 0));
        this->tmpImage.at<Vec3b>(y, x) = centers.at<Vec3b>(labelIndex);
      }
    }

    TRACE_LINE(__FILE__, __LINE__);
  }

  /**
   * Update the output image
   */
  virtual void updateOutputImage() { this->tmpImage.copyTo(this->outputImage); }

private:
  /** Temporal Image storage */
  Mat tmpImage;
  /** Obtained centers matrix */
  Mat centers;
  /** Obtained pixels labels */
  Mat label;
  /** Clusters number */
  static const int clusterNum = 3;
};

const String KMeanDetect::options = "{kmeans | | K-means Enable }";

#endif /* KMEANSDETECT_H */
