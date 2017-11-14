/**
 * @file FeatureDetect.hpp
 * @author Pedro Cuadra
 * @date 5 Nov 2017
 * @copyright 2017 Pedro Cuadra
 * @brief SURF Feature Detector Class
 *
 */
#ifndef FASTDETECT_H
#define FASTDETECT_H

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define FAST_OPTIONS                                                           \
  "{fast           |      | FAST Enable  }"                                    \
  "{fast_th        | 10   | FAST Theshold  }"                                  \
  "{fast_nm        | true | NonMax Suppression }"

class FastDetect : public FeatureDetect {
public:
  FastDetect(CommandLineParser parser) : FeatureDetect(parser, "FAST", "fast") {
    fastTh = parser.get<int>("fast_th");
    nonmax = parser.get<bool>("fast_nm");

    this->fastDetector = FastFeatureDetector::create(
        fastTh, nonmax, FastFeatureDetector::TYPE_9_16);
    paramsString << "  FAST Threshold: " << fastTh << endl;
    paramsString << "  FAST Non Max Supression: " << nonmax << endl;
  }

protected:
  void setNonMaxSupression(bool value) { this->nonmax = value; }

  virtual void _runDetect() {
    fastDetector->setThreshold(fastTh);
    fastDetector->setNonmaxSuppression(nonmax);
    this->fastDetector->detect(this->inputImage, this->keyPoints);
  }

  /**
   * @function CannyThreshold
   * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
   */
  static void onChange(int, void *ptr) {
    FastDetect *that = (FastDetect *)ptr;

    that->_runDetect();
    that->drawOutput();
  }

  static void onClick(int state, void *ptr) {
    FastDetect *that = (FastDetect *)ptr;

    that->setNonMaxSupression(state != 0);
    that->drawOutput();
  }

  virtual void createControls() {
    /// Create a Trackbar for user to enter threshold
    createTrackbar("FAST Threshold", this->name, &fastTh, 255, onChange, this);
    createButton("Non Max Suppresion", onClick, this, QT_CHECKBOX, true);
  }

private:
  int fastTh;
  bool nonmax;
  Ptr<FastFeatureDetector> fastDetector;
};

#endif /* FASTDETECT_H */
