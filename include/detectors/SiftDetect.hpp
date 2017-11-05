#ifndef SIFTDETECT_H
#define SIFTDETECT_H

#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define SIFT_OPTIONS "{sift           |      | Include SIFT         }"

class SiftDetect : public FeatureDetect {
public:
  SiftDetect(CommandLineParser parser) : FeatureDetect(parser, "SIFT") {
    this->detector = SIFT::create();
    this->siftEnabled = parser.has("sift");
  }

  void detect(Mat inputImage) {
    if (!this->siftEnabled) {
      return;
    }

    FeatureDetect::detect(inputImage);
  }

private:
  bool siftEnabled = false;
};

#endif /* SIFTDETECT_H */
