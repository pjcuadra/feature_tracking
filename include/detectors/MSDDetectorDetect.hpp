#ifndef MSDDETECTORDETECT_H
#define MSDDETECTORDETECT_H

#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define MSDDETECTORDETECT_OPTIONS ""

class MSDDetectorDetect : public FeatureDetect {
public:
  MSDDetectorDetect(CommandLineParser parser) : FeatureDetect(parser, "MSDDetectorDetect") {
    this->detector = MSDDetector::create();
  }
};

#endif /* MSDDETECTORDETECT_H */
