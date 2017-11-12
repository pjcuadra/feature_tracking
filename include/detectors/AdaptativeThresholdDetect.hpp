/**
* @file FeatureDetect.hpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief StarDetector Feature Detector Class
*
*/
#ifndef ADAPTATIVTHESHOLDDETECT_H
#define ADAPTATIVTHESHOLDDETECT_H

#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define ADAPTATIVTHRESHOLD_OPTIONS "{athreshold         |      | Adaptative Threshold Enable          }"

class AdaptativeThresholdDetect : public FeatureDetect {
public:
  AdaptativeThresholdDetect(CommandLineParser parser) :
  FeatureDetect(parser, "Adaptative Threshold", "athreshold") {
  }

protected:

  virtual void _runDetect() {
    blur(this->inputImage, tmpImage, Size(100, 100));
    adaptiveThreshold(tmpImage,
      tmpImage,
      255,
      ADAPTIVE_THRESH_MEAN_C,
      THRESH_BINARY,
      3,
      1);
  }

  void updateOutputImage() {
    this->tmpImage.copyTo(this->outputImage);
  }

private:
  Mat tmpImage;
};

#endif /* ADAPTATIVTHESHOLDDETECT_H */
