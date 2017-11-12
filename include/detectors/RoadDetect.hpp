/**
* @file FeatureDetect.hpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief StarDetector Feature Detector Class
*
*/
#ifndef ROADDETECT_H
#define ROADDETECT_H

#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <detectors/FeatureDetect.hpp>
#include <Debug.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define ROADDETECT_OPTIONS "{roaddetect      |      | Road Detect Enable          }"

class RoadDetect : public FeatureDetect {
public:
  RoadDetect(CommandLineParser parser) :
  FeatureDetect(parser, "RoadDetect", "roaddetect") {
  }

  void setFill(bool enable) {
    this->fill = enable;
  }

  bool getFill() {
    return this->fill;
  }

protected:

  virtual void _runDetect() {
    TRACE_LINE(__FILE__, __LINE__);

    contours0.clear();
    this->hierarchy.clear();
    blur(this->inputImage, this->tmpImage, Size(100, 100));
    threshold(this->tmpImage,
      this->tmpImage,
      0,
      255,
      THRESH_BINARY | THRESH_OTSU);
    findContours(this->tmpImage,
      this->contours0,
      this->hierarchy,
      RETR_TREE,
      CHAIN_APPROX_SIMPLE);

    this->createColorsVector();
  }

  virtual void updateOutputImage() {
    RotatedRect ellipse;
    float exentricity;
    stringstream message;

    TRACE_LINE(__FILE__, __LINE__);

    this->inputImage.copyTo(this->outputImage);

    cvtColor(this->outputImage, this->outputImage, CV_GRAY2RGB);

    for(int idx = 0 ; idx >= 0; idx = hierarchy[idx][0]) {
      if (this->contours0[idx].size() < 5) {
        continue;
      }

      ellipse =  fitEllipse(this->contours0[idx]);
      exentricity = ellipse.size.width/ellipse.size.height;

      if (exentricity > (float)this->exentricityMax/100.0f) {
        continue;
      }

      if (exentricity < (float)this->exentricityMin/100.0f) {
        continue;
      }

      message.str("");
      message << "Excentricity: " << exentricity;

      Debug::addPoint(__FILE__, __LINE__, message.str());

      if (getFill()) {
        TRACE_LINE(__FILE__, __LINE__);
        drawContours(this->outputImage,
          this->contours0,
          idx,
          colorsVec[idx],
          FILLED,
          8,
          hierarchy);
      } else {
        TRACE_LINE(__FILE__, __LINE__);
        drawContours(this->outputImage,
          this->contours0,
          idx,
          colorsVec[idx],
          10);
      }
    }
  }

  /**
   * @function CannyThreshold
   * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
   */
  static void onChange(int pos, void* ptr) {
    RoadDetect * that = (RoadDetect *) ptr;

    that->_runDetect();
    that->drawOutput();
   }

  static void onClick(int state, void* ptr) {
    RoadDetect * that = (RoadDetect *) ptr;

    that->setFill(state != 0);
    that->drawOutput();
  }

  void createColorsVector() {
    colorsVec.clear();

    for(int idx = 0 ; idx < contours0.size(); idx++ ) {
      colorsVec.push_back(Scalar( rand()&255, rand()&255, rand()&255));
    }
  }

  virtual void createControls() {
    exentricityMin = 0;
    exentricityMax = 100;

    /// Create a Trackbar for user to enter threshold
    createTrackbar("Excentricity Min",
      this->name,
      &exentricityMin,
      100,
      onChange,
      this);
    createTrackbar("Excentricity Max",
      this->name,
      &exentricityMax,
      100,
      onChange,
      this);

    createButton("Fill Regions",
      onClick,
      this,
      QT_CHECKBOX,
      true);
  }


private:
  Mat tmpImage;
  bool fill;
  int exentricityMin;
  int exentricityMax;
  vector<vector<Point> > contours0;
  vector<Vec4i> hierarchy;
  vector<Scalar> colorsVec;
};

#endif /* ROADDETECT_H */
