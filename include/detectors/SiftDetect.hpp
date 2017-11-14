/**
* @file FeatureDetect.hpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief SIFT Feature Detector Class
*
*/
#ifndef SIFTDETECT_H
#define SIFTDETECT_H

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define SIFT_OPTIONS "{sift           |      | SIFT enable        }"

class SiftDetect : public FeatureDetect {
public:
  SiftDetect(CommandLineParser parser) : FeatureDetect(parser, "SIFT", "sift") {
    this->detector = SIFT::create();
    paramsString << "  Defaults";
  }
};

#endif /* SIFTDETECT_H */
