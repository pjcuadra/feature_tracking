/**
* @file FeatureDetect.hpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief StarDetector Feature Detector Class
*
*/
#ifndef OTSUTHRESHOLDDETECT_H
#define OTSUTHRESHOLDDETECT_H

#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define OTSUTHRESHOLD_OPTIONS "{otsu         |      | Otsu Threshold Enable          }"
class OtsuThresholdDetect : public FeatureDetect {
public:

  OtsuThresholdDetect(CommandLineParser parser) :
  FeatureDetect(parser, "Otsu Threshold", "otsu") {
  }

protected:
  virtual void _runDetect() {
    blur(this->inputImage, this->tmpImage, Size(100, 100));
    threshold(this->tmpImage,
      this->tmpImage,
      0,
      255,
      THRESH_BINARY | THRESH_OTSU);
  }

  void updateOutputImage() {
    this->tmpImage.copyTo(this->outputImage);
  }

private:
  Mat tmpImage;
};

#endif /* OTSUTHRESHOLDDETECT_H */
