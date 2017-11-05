#ifndef VGGDETECT_H
#define VGGDETECT_H

#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define VGGDETECT_OPTIONS ""

class VggDetect : public FeatureDetect {
public:
  VggDetect(CommandLineParser parser) : FeatureDetect(parser, "VGG") {
    this->detector = VGG::create();
  }

protected:
  virtual void runDetect(Mat inputImage) {
    detector->compute(inputImage, this->keyPoints, this->descriptors);
    inputImage.copyTo(this->inputImage);
  }
};

#endif /* VGGDETECT_H */
