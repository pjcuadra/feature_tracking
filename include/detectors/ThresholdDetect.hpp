/**
* @file FeatureDetect.hpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief StarDetector Feature Detector Class
*
*/
#ifndef THESHOLDDETECT_H
#define THESHOLDDETECT_H

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define THRESHOLD_OPTIONS                                                      \
  "{threshold         |      | Threshold Enable          }"
class ThresholdDetect : public FeatureDetect {
public:
  ThresholdDetect(CommandLineParser parser)
      : FeatureDetect(parser, "Threshold", "threshold") {}

protected:
  virtual void _runDetect() {
    blur(this->inputImage, this->tmpImage, Size(100, 100));
    threshold(this->tmpImage, this->tmpImage, this->th, 255, THRESH_BINARY);
  }

  virtual void updateOutputImage() { this->tmpImage.copyTo(this->outputImage); }

  /**
   * @function CannyThreshold
   * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
   */
  static void onChange(int, void *ptr) {
    ThresholdDetect *that = (ThresholdDetect *)ptr;

    that->_runDetect();
    that->drawOutput();
  }

  virtual void createControls() {
    /// Create a Trackbar for user to enter threshold
    createTrackbar("Min Threshold", this->name, &th, 255, onChange, this);
  }

private:
  Mat tmpImage;
  int th;
};

#endif /* THESHOLDDETECT_H */
