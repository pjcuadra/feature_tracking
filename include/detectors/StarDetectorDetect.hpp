/**
* @file FeatureDetect.hpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief StarDetector Feature Detector Class
*
*/
#ifndef STARDETECTORDETECT_H
#define STARDETECTORDETECT_H

#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define STARDETECTOR_OPTIONS "{star         |      | StarDetector Enable        }"

class StarDetectorDetect : public FeatureDetect {
public:
  StarDetectorDetect(CommandLineParser parser) :
  FeatureDetect(parser, "StarDetector", "star") {
    this->detector = StarDetector::create();
  }
};

#endif /* STARDETECTORDETECT_H */
