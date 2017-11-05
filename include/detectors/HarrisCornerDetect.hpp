/**
* @file FeatureDetect.hpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief Harris Corner Feateure Detector
*
*/
#ifndef HARRISCORNERDETECT_H
#define HARRISCORNERDETECT_H

#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define HARRISCORNERDETECT_OPTIONS "{harris         |      | Harris Enable        }" \
                                   "{harris_th      | 50   | Harris Threshold     }" \
                                   "{harris_k       | 0.04 | Harris K             }" \
                                   "{harris_bz      | 50   | Harris Block Size    }" \
                                   "{harris_ap      | 31   | Harris Aperture Size }"

class HarrisCornerDetect : public FeatureDetect {
public:
  HarrisCornerDetect(CommandLineParser parser) : FeatureDetect(parser, "Harris Corner") {
    this->blockSize = parser.get<int>("harris_bz");
    this->apertureSize = parser.get<int>("harris_ap");
    this->harrisThreshold = parser.get<int>("harris_th");
    this->kh = parser.get<double>("harris_k");
    this->enable = parser.has("harris");
  }

  virtual void _detect(Mat inputImage) {
    Timing timing;

    inputImage.copyTo(this->inputImage);

    // Benchmark Corner Harris
    cout << "Running Corner Harris" << endl;

    outputHarris = Mat::zeros( inputImage.size(), CV_32FC1 );

    cout << "  Block Size: " << blockSize << endl;
    cout << "  Aperture Size: " << apertureSize << endl;
    cout << "  K: " << kh << endl;
    cout << "  Drawing Threshold: " << harrisThreshold << endl;

    timing.start();
    cornerHarris(this->inputImage,
      this->outputHarris,
      this->blockSize,
      this->apertureSize,
      this->kh,
      BorderTypes::BORDER_DEFAULT);
    timing.end();
    cout << "  ";
    timing.print();

    if (this->showEnable) {
      this->show();
    }
  }

protected:

  virtual void _show() {
    Mat outputHarrisNorm, outputHarrisNormScaled;

    /// Normalizing
    normalize(outputHarris,
      outputHarrisNorm,
      0,
      255,
      NORM_MINMAX,
      CV_32FC1,
      Mat());
    convertScaleAbs(outputHarrisNorm,
      outputHarrisNormScaled);

    inputImage.copyTo(outputHarris);

    /// Drawing a circle around corners
    for(int j = 0; j < outputHarrisNorm.rows; j++) {
      for(int i = 0; i < outputHarrisNorm.cols; i++) {
        if(outputHarrisNorm.at<float>(j, i) > harrisThreshold) {
          circle(outputHarris,
            Point(i, j),
            1,
            Scalar(255, 0, 0, 0));
        }
      }
    }

    namedWindow("Harris", WINDOW_GUI_EXPANDED);
    imshow("Harris", outputHarris);
    namedWindow("Harris - Keypoints", WINDOW_GUI_EXPANDED);
    imshow("Harris - Keypoints", outputHarrisNormScaled);
  }

private:
  Mat outputHarris;
  int blockSize;
  int apertureSize;
  int harrisThreshold;
  double kh;

};

#endif /* HARRISCORNERDETECT_H */
