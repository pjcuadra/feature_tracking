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

#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define SEGMENTATION_OPTIONS "{segment         |      | Segmentation Enable          }"

class SegmentationDetect : public FeatureDetect {
public:
  Mat inImg;
  int th;

  SegmentationDetect(CommandLineParser parser) :
  FeatureDetect(parser, "Segmentation", "segment") {
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

    SegmentationDetect * that = (SegmentationDetect *) ptr;
    that->inImg.copyTo(blured);

    timing.start();
    blur(blured, blured, Size(100, 100));
    threshold(blured, out, that->th, 255, THRESH_BINARY);
    that->detector->detect(out, that->keyPoints);

    timing.end();
    cout << "  ";
    timing.print();

    drawKeypoints(out,
      that->keyPoints,
      out,
      Scalar::all(-1),
      DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

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

#endif /* SEGMENTATIONDETECT_H */
