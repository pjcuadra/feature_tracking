/**
* @file FeatureDetect.hpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief StarDetector Feature Detector Class
*
*/
#ifndef SIMPLEBLOBDETECT_H
#define SIMPLEBLOBDETECT_H

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define SIMPLEBLOB_OPTIONS "{sblob         |      | SimpleBlob Enable        }"

class SimpleBlobDetect : public FeatureDetect {
public:
  SimpleBlobDetect(CommandLineParser parser)
      : FeatureDetect(parser, "SimpleBlob", "sblob") {
    SimpleBlobDetector::Params params;

    // params.filterByArea = false;
    // params.filterByCircularity = false;
    // params.filterByConvexity = false;
    // params.filterByColor = true;
    // params.minThreshold = 0;
    // params.maxThreshold = 255;
    // params.thresholdStep = 50;

    params.filterByInertia = false;
    params.filterByConvexity = false;
    params.filterByCircularity = false;
    params.filterByArea = false;
    params.filterByColor = true;
    params.minThreshold = 0;
    params.maxThreshold = 200.0f;
    params.thresholdStep = 50.0f;

    this->detector = SimpleBlobDetector::create(params);
  }
};

#endif /* SIMPLEBLOBDETECT_H */
