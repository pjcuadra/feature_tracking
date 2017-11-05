#ifndef STARDETECTORDETECT_H
#define STARDETECTORDETECT_H

#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define STARDETECTOR_OPTIONS ""

class StarDetectorDetect : public FeatureDetect {
public:
  StarDetectorDetect(CommandLineParser parser) : FeatureDetect(parser, "StarDetector") {
    this->detector = StarDetector::create();
  }
};

#endif /* STARDETECTORDETECT_H */
