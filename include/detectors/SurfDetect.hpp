#ifndef SURFDETECT_H
#define SURFDETECT_H

#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define SURF_OPTIONS "{surf_h         | 400  | Display images       }"

class SurfDetect : public FeatureDetect {
public:
  SurfDetect(CommandLineParser parser) : FeatureDetect(parser, "SURF") {
    this->surfHessianTh = parser.get<int>("surf_h");
    this->detector = SURF::create(this->surfHessianTh);
  }

private:
  int surfHessianTh;
};

#endif /* SURFDETECT_H */
