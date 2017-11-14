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
#include <list>

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <Stats.hpp>
#include <Timing.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

class FeatureDetect {
public:
  /** Detector */
  static bool debug;

  /**
   * @brief Constructor
   * @param [in] parser Command line parser
   * @param [in] name Name of the feature detector
   * @details <details>
   */
  FeatureDetect(CommandLineParser parser, string name);
  FeatureDetect(CommandLineParser parser, string name, string enableFlag);

  /**
   * @brief <brief>
   * @param [in] <name> <parameter_description>
   * @return <return_description>
   * @details <details>
   */
  virtual void detect(Mat inputImage);

  void printStats();

  void collectStats(double delta);

  string getName();

  static void enableLog(bool enable);

  virtual void writeImage(string path);

  bool getEnable();

  void dumpStatsToFile(string path);

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
  /** Detector */
  string name;
  Mat outputImage;
  stringstream paramsString;
  stringstream statsString;
  Stats<int> keyPointsStats;

  virtual void runCompute();

  /**
   * @brief <brief>
   * @param [in] <name> <parameter_description>
   * @return <return_description>
   * @details <details>
   */
  virtual void runDetect();

  virtual void _detect(Mat inputImage);

  virtual void _runCompute();

  /**
   * @brief <brief>
   * @param [in] <name> <parameter_description>
   * @return <return_description>
   * @details <details>
   */
  virtual void _runDetect();

  /**
   * @brief <brief>
   * @param [in] <name> <parameter_description>
   * @return <return_description>
   * @details <details>
   */
  virtual void updateOutputImage();

  void printLog(string message);

  virtual void createControls();
  void drawOutput();

private:
  /**
   * @brief <brief>
   * @param [in] <name> <parameter_description>
   * @return <return_description>
   * @details <details>
   */
  virtual void show();
  void generateStatsString();

  Stats<double> timingStats;
};

#endif /* FEATURE_DETECTOR_H */
