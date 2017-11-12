/**
* @file FeatureDetect.hpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief SURF Feature Detector Class
*
*/
#ifndef LUCIDDETECT_H
#define LUCIDDETECT_H

#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define LUCID_OPTIONS "{lucid         |      | LUCID Enable        }"

class LucidDetect : public FeatureDetect {
public:
  LucidDetect(CommandLineParser parser) :
  FeatureDetect(parser, "LUCID", "lucid") {
    this->detector = LUCID::create();
  }

protected:
  virtual void _runDetect() {
    FeatureDetect::runCompute();
  }
};

#endif /* LUCIDDETECT_H */
