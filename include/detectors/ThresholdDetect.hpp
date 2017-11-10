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

#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define THRESHOLD_OPTIONS "{threshold         |      | Threshold Enable          }"
class ThresholdDetect : public FeatureDetect {
public:
  Mat inImg;
  int th;

  ThresholdDetect(CommandLineParser parser) :
  FeatureDetect(parser, "Threshold", "threshold") {
  }

protected:
  virtual void _runDetect(Mat inputImage) {
    inputImage.copyTo(inImg);
    // blur(inImg, inImg, Size(50,50));
    // Canny(inImg, canny_out, low_th, low_th*3, 7);
  }

  /**
   * @function CannyThreshold
   * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
   */
  static void onChange(int, void* ptr) {
    Mat out;
    Mat blured;
    Timing timing;

    ThresholdDetect * that = (ThresholdDetect *) ptr;
    that->inImg.copyTo(blured);

    timing.start();
    blur(blured, blured, Size(100, 100));
    threshold(blured, out, that->th, 255, THRESH_BINARY);
    timing.end();
    cout << "  ";
    timing.print();

    imshow(that->getName(), out);
   }


  virtual void _show() {
    namedWindow(this->name, WINDOW_GUI_EXPANDED);

    /// Create a Trackbar for user to enter threshold
    createTrackbar("Min Threshold",
      this->name,
      &th,
      255,
      onChange,
      this);

    /// Show the image
    onChange(0, this);
  }


private:
};

#endif /* THESHOLDDETECT_H */
