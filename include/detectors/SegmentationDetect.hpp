/**
* @file FeatureDetect.hpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief StarDetector Feature Detector Class
*
*/
#ifndef SEGMENTATIONDETECT_H
#define SEGMENTATIONDETECT_H

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define SEGMENTATION_OPTIONS                                                   \
  "{segment         |      | Segmentation Enable          }"

class SegmentationDetect : public FeatureDetect {
public:
  SegmentationDetect(CommandLineParser parser)
      : FeatureDetect(parser, "Segmentation", "segment") {
    SimpleBlobDetector::Params params;

    // params.filterByArea = false;
    // params.filterByCircularity = false;
    // params.filterByConvexity = false;
    // params.minThreshold = 0;
    // params.filterByColor = true;
    // params.maxThreshold = 255;
    // params.thresholdStep = 50;

    params.filterByInertia = false;
    params.filterByConvexity = false;
    params.filterByCircularity = false;
    params.filterByArea = false;
    params.filterByColor = true;
    params.minThreshold = 0;
    params.maxThreshold = 100.0f;
    params.thresholdStep = 50.0f;
    params.blobColor = 255;

    this->detector = SimpleBlobDetector::create(params);
  }

protected:
  virtual void _runDetect() {
    blur(this->inputImage, this->tmpImage, Size(100, 100));
    threshold(this->tmpImage, this->tmpImage, 0, 255,
              THRESH_BINARY | THRESH_OTSU);
    this->detector->detect(this->tmpImage, this->keyPoints);
  }

  virtual void updateOutputImage() {
    cvtColor(this->tmpImage, this->outputImage, CV_GRAY2RGB);

    drawKeypoints(this->outputImage, this->keyPoints, this->outputImage,
                  Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  }

private:
  Mat tmpImage;
};

#endif /* SEGMENTATIONDETECT_H */
