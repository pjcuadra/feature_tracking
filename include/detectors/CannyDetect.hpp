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
  Mat canny_out;
  Mat inImg;
  int low_th;
  int blur_size;
  int ratio;
  int cascade_blur;

  CannyDetect(CommandLineParser parser) :
  FeatureDetect(parser, "Canny", "canny") {
    this->low_th = parser.get<int>("canny_low_th");
    this->upp_th = parser.get<int>("canny_upp_th");
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
  static void CannyThreshold(int, void* ptr) {
    Mat canny_out;
    Mat blured;
    Timing timing;

    CannyDetect * that = (CannyDetect *) ptr;

    that->inImg.copyTo(blured);

    timing.start();
    for (int i = 0; i < that->cascade_blur; i++) {
      /// Reduce noise with a kernel 3x3
      blur(blured, blured, Size(that->blur_size, that->blur_size));
    }

    /// Canny detector
    Canny(blured, canny_out, that->low_th, that->low_th*that->ratio, 3);
    timing.end();
    cout << "  ";
    timing.print();

    imshow(that->getName(), canny_out);
    imshow("Canny Blur", blured);

   }


  virtual void _show() {
    namedWindow(this->name, WINDOW_GUI_EXPANDED);
    namedWindow("Canny Blur", WINDOW_GUI_EXPANDED);

    ratio = 3;
    blur_size = 14;
    low_th = 5;
    cascade_blur = 1;

    /// Create a Trackbar for user to enter threshold
    createTrackbar("Min Threshold:",
      this->name,
      &low_th,
      100,
      CannyThreshold,
      this);
    createTrackbar("Ratio", this->name, &ratio, 50, CannyThreshold, this);
    setTrackbarMin("Ratio", this->name, 1);
    createTrackbar("Blur size",
      this->name,
      &blur_size,
      30,
      CannyThreshold,
      this);
    setTrackbarMin("Blur size", this->name, 3);
    createTrackbar("Cascade Blur",
      this->name,
      &cascade_blur,
      30,
      CannyThreshold,
      this);

    /// Show the image
    CannyThreshold(0, this);
  }


private:

  int upp_th;
};




#endif /* CANNYDETECT_H */
