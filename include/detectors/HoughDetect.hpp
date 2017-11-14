/**
* @file FeatureDetect.hpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief SURF Feature Detector Class
*
*/
#ifndef HOUGHDETECT_H
#define HOUGHDETECT_H

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define HOUGH_OPTIONS "{hough           |      | Hough Transform Enable  }"

class HoughDetect : public FeatureDetect {
public:
  HoughDetect(CommandLineParser parser)
      : FeatureDetect(parser, "Hough", "hough") {
    this->ratio = 3;
    this->blur_size = 14;
    this->low_th = 27;
  }

protected:
  virtual void _runDetect() {

    blur(this->inputImage, this->tmpImage,
         Size(this->blur_size, this->blur_size));

    Canny(this->tmpImage, this->tmpImage, low_th, low_th * ratio, 3);

    HoughLinesP(this->tmpImage, lines, 1, CV_PI / 180, 100, 50, 500);
  }

  virtual void updateOutputImage() {
    cvtColor(this->inputImage, this->outputImage, COLOR_GRAY2BGR);

    keyPointsStats.push_back(lines.size());

    for (size_t i = 0; i < lines.size(); i++) {
      Vec4i l = lines[i];
      line(this->outputImage, Point(l[0], l[1]), Point(l[2], l[3]),
           Scalar(0, 0, 255), 3, CV_AA);
    }
  }

  /**
   * @function CannyThreshold
   * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
   */
  static void onChange(int, void *ptr) {
    HoughDetect *that = (HoughDetect *)ptr;

    that->runDetect();
    that->drawOutput();
  }

  virtual void createControls() {
    this->ratio = 3;
    this->blur_size = 14;
    this->low_th = 5;

    /// Create a Trackbar for user to enter threshold
    createTrackbar("Min Threshold:", this->name, &low_th, 100, onChange, this);
    createTrackbar("Ratio", this->name, &ratio, 50, onChange, this);
    setTrackbarMin("Ratio", this->name, 1);
    createTrackbar("Blur size", this->name, &blur_size, 30, onChange, this);
    setTrackbarMin("Blur size", this->name, 3);
  }

private:
  vector<Vec4i> lines;
  Mat tmpImage;
  int low_th;
  int blur_size;
  int ratio;
};

#endif /* HOUGHDETECT_H */
