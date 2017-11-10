/**
* @file FeatureDetect.hpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief StarDetector Feature Detector Class
*
*/
#ifndef FINDCONTOURDETECT_H
#define FINDCONTOURDETECT_H

#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define FINDCONTOUR_OPTIONS "{findcontour      |      | Find Contour Enable          }"

class FindContourDetect : public FeatureDetect {
public:
  Mat inImg;
  int th;
  bool fill;

  FindContourDetect(CommandLineParser parser) :
  FeatureDetect(parser, "FindContour", "findcontour") {
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
  static void onChange(int pos, void* ptr) {
    Mat out;
    Mat blured;
    Timing timing;
    vector<vector<Point> > contours0;
    vector<Vec4i> hierarchy;

    FindContourDetect * that = (FindContourDetect *) ptr;
    that->inImg.copyTo(blured);

    timing.start();
    blur(blured, blured, Size(100, 100));
    threshold(blured, out, that->th, 255, THRESH_BINARY);
    findContours(out, contours0, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    timing.end();
    cout << "  ";
    timing.print();

    cvtColor(that->inImg, out, CV_GRAY2RGB);

    for(int idx = 0 ; idx < contours0.size(); idx++ )
    {
        Scalar color( rand()&255, rand()&255, rand()&255 );
        if (that->fill) {
          drawContours(out, contours0, idx, color, FILLED, 8, hierarchy);
        } else {
          drawContours(out, contours0, idx, color, 10);
        }
    }


    imshow(that->getName(), out);
   }

  static void onClick(int state, void* ptr) {
    FindContourDetect * that = (FindContourDetect *) ptr;

    that->fill = (state != 0);

    onChange(0, ptr);

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

    createButton("Fill Regions", onClick, this, QT_CHECKBOX, true);

    /// Show the image
    onChange(0, this);
  }


private:
};

#endif /* FINDCONTOURDETECT_H */
