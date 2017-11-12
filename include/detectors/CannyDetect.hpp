/**
* @file FeatureDetect.hpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief StarDetector Feature Detector Class
*
*/
#ifndef CANNYDETECT_H
#define CANNYDETECT_H

#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define CANNY_OPTIONS "{canny         |      | Canny Enable          }" \
                      "{canny_low_th  | 100  | Canny Lower Threshold }" \
                      "{canny_upp_th  | 200  | Canny Upper Threshold }"

class CannyDetect : public FeatureDetect {
public:
  CannyDetect(CommandLineParser parser) :
  FeatureDetect(parser, "Canny", "canny") {
    this->low_th = parser.get<int>("canny_low_th");
  }

protected:
  virtual void _runDetect() {
    this->inputImage.copyTo(this->tmpImage);

    for (int i = 0; i < this->cascade_blur; i++) {
      /// Reduce noise with a kernel 3x3
      blur(this->tmpImage,
        this->tmpImage,
        Size(this->blur_size, this->blur_size));
    }

    /// Canny detector
    Canny(this->tmpImage,
      this->tmpImage,
      this->low_th,
      this->low_th*this->ratio, 3);
  }

  virtual void updateOutputImage() {
    this->tmpImage.copyTo(this->outputImage);
  }

  /**
   * @function CannyThreshold
   * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
   */
  static void onChange(int, void* ptr) {
    CannyDetect * that = (CannyDetect *) ptr;
    that->runDetect();
    that->drawOutput();
   }


  virtual void createControls() {
    this->ratio = 3;
    this->blur_size = 14;
    this->low_th = 5;
    this->cascade_blur = 1;

    /// Create a Trackbar for user to enter threshold
    createTrackbar("Min Threshold:",
      this->name,
      &low_th,
      100,
      onChange,
      this);
    createTrackbar("Ratio", this->name, &ratio, 50, onChange, this);
    setTrackbarMin("Ratio", this->name, 1);
    createTrackbar("Blur size",
      this->name,
      &blur_size,
      30,
      onChange,
      this);
    setTrackbarMin("Blur size", this->name, 3);
    createTrackbar("Cascade Blur",
      this->name,
      &cascade_blur,
      30,
      onChange,
      this);
  }


private:
  Mat tmpImage;
  int low_th;
  int blur_size;
  int ratio;
  int cascade_blur;
};




#endif /* CANNYDETECT_H */
