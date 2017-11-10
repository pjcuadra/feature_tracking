/**
* @file FeatureDetect.hpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief SURF Feature Detector Class
*
*/
#ifndef SURFDETECT_H
#define SURFDETECT_H

#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define SURF_OPTIONS "{surf           |      | StarDetector Enable  }"\
                     "{surf_h         | 400  | Display images       }"

class SurfDetect : public FeatureDetect {
public:
  SurfDetect(CommandLineParser parser) :
  FeatureDetect(parser, "SURF", "surf") {
    this->surfHessianTh = parser.get<int>("surf_h");
    this->detector = SURF::create(this->surfHessianTh);
  }

private:
  int surfHessianTh;
};

#endif /* SURFDETECT_H */
