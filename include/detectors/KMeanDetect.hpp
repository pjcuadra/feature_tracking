/**
* @file FeatureDetect.hpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief StarDetector Feature Detector Class
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

#define KMEANS_OPTIONS "{kmeans      |      | K-means Enable          }"

class KMeanDetect : public FeatureDetect {
public:
  KMeanDetect(CommandLineParser parser)
      : FeatureDetect(parser, "K-means", "kmeans") {}

protected:
  static const int clusterNum = 3;

  virtual void _runDetect() {
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

  virtual void updateOutputImage() { this->tmpImage.copyTo(this->outputImage); }

private:
  Mat tmpImage;
  Mat centers;
  Mat label;
};

#endif /* KMEANSDETECT_H */
