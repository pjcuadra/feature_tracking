/**
* @file FeatureDetect.hpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief VGG Feature Detector Class
*
*/
#ifndef VGGDETECT_H
#define VGGDETECT_H

#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define VGGDETECT_OPTIONS "{vgg           |      | VGG Enable  }"\

class VggDetect : public FeatureDetect {
public:
  VggDetect(CommandLineParser parser) : FeatureDetect(parser, "VGG", "vgg") {
    this->detector = VGG::create();
  }

protected:
  virtual void _runDetect(Mat inputImage) {
    FeatureDetect::runCompute(inputImage);
  }
};

#endif /* VGGDETECT_H */
