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

  FindContourDetect(CommandLineParser parser) :
  FeatureDetect(parser, "FindContour", "findcontour") {
  }



protected:

  void setFill(bool enable) {
    this->fill = enable;
  }

  bool getFill() {
    return this->fill;
  }

  void createColorsVector() {
    colorsVec.clear();

    for(int idx = 0 ; idx < contours0.size(); idx++ ) {
      colorsVec.push_back(Scalar( rand()&255, rand()&255, rand()&255));
    }
  }

  virtual void _runDetect() {
    blur(this->inputImage, this->tmpImg, Size(100, 100));
    threshold(this->tmpImg, this->tmpImg, 0, 255, THRESH_BINARY | THRESH_OTSU);
    findContours(this->tmpImg,
      contours0,
      hierarchy,
      RETR_TREE,
      CHAIN_APPROX_SIMPLE);

    this->createColorsVector();
  }

  virtual void updateOutputImage() {
    cvtColor(this->inputImage, this->outputImage, CV_GRAY2RGB);

    for(int idx = 0 ; idx < contours0.size(); idx++ ) {
        if (getFill()) {
          drawContours(this->outputImage,
            contours0,
            idx,
            colorsVec[idx],
            FILLED,
            8,
            hierarchy);
        } else {
          drawContours(this->outputImage,
            contours0,
            idx,
            colorsVec[idx],
            10);
        }
    }
  }

  static void onClick(int state, void* ptr) {
    FindContourDetect * that = (FindContourDetect *) ptr;

    that->setFill(state != 0);
    that->drawOutput();
  }

  virtual void createControls() {
    createButton("Fill Regions", onClick, this, QT_CHECKBOX, true);
  }

private:
  vector<vector<Point> > contours0;
  vector<Vec4i> hierarchy;
  Mat tmpImg;
  vector<Scalar> colorsVec;
  bool fill;

};

#endif /* FINDCONTOURDETECT_H */
