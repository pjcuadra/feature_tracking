/**
* @file FeatureDetect.hpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief Feature Detector Class
*
*/
#ifndef FEATURE_DETECTOR_H
#define FEATURE_DETECTOR_H

#include <iostream>

#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>

#include <Timing.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

class FeatureDetect {
public:

  /**
   * @brief Constructor
   * @param [in] parser Command line parser
   * @param [in] name Name of the feature detector
   * @details <details>
   */
  FeatureDetect(CommandLineParser parser, string name);

  /**
   * @brief <brief>
   * @param [in] <name> <parameter_description>
   * @return <return_description>
   * @details <details>
   */
  virtual void detect(Mat inputImage);

protected:

  /** Detector */
  Ptr<Feature2D> detector;
  /** Detector */
  vector<KeyPoint> keyPoints;
  /** Detector */
  Mat descriptors;
  /** Detector */
  Mat inputImage;
  /** Detector */
  bool showEnable = false;
  bool enable = false;
  bool allEnable = false;


  virtual void runCompute(Mat inputImage);

  /**
   * @brief <brief>
   * @param [in] <name> <parameter_description>
   * @return <return_description>
   * @details <details>
   */
  virtual void runDetect(Mat inputImage);

  /**
   * @brief <brief>
   * @param [in] <name> <parameter_description>
   * @return <return_description>
   * @details <details>
   */
  virtual void show();

  virtual void _detect(Mat inputImage);

  virtual void _runCompute(Mat inputImage);

  /**
   * @brief <brief>
   * @param [in] <name> <parameter_description>
   * @return <return_description>
   * @details <details>
   */
  virtual void _runDetect(Mat inputImage);

  /**
   * @brief <brief>
   * @param [in] <name> <parameter_description>
   * @return <return_description>
   * @details <details>
   */
  virtual void _show();

private:
  /** Detector */
  string name;
  /** Detector */
  bool debug = true;

  virtual void printLog(string message);



};

#endif /* FEATURE_DETECTOR_H */
