/**
* @file FeatureDetect.hpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief Harris Corner Feateure Detector
*
*/
#ifndef HARRISLAPLACEDETECT_H
#define HARRISLAPLACEDETECT_H

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define HARRISLAPLACEDETECT_OPTIONS                                            \
  "{harrislaplace         |      | Harris Laplace Enable        }"

class HarrisLaplaceDetect : public FeatureDetect {
public:
  HarrisLaplaceDetect(CommandLineParser parser)
      : FeatureDetect(parser, "Harris Laplace", "harrislaplace") {
    this->detector = HarrisLaplaceFeatureDetector::create();
  }

protected:
private:
};

#endif /* HARRISLAPLACEDETECT_H */
