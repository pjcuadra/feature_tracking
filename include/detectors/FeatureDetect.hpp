/*
 * MIT License
 *
 * Copyright (c) 2017 Pedro Cuadra
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
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

#include <util/Stats.hpp>
#include <util/Timing.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

class FeatureDetect {
public:
  /**
   * Feature Detection Wrapper Class
   * @param parser Command line parser
   * @param name   Name of the feature detector
   */
  FeatureDetect(CommandLineParser parser, string name);

  /**
   * Feature Detection Wrapper Class
   * @param parser     Command line parser
   * @param name       Name of the feature detector
   * @param enableFlag Enable comand line flag name
   */
  FeatureDetect(CommandLineParser parser, string name, string enableFlag);

  /**
   * Apply detection algorithm
   * @param inputImage Input image
   */
  void detect(Mat inputImage);

  /**
   * Compute the descriptors
   * @param inputImage Input image
   */
  void compute(Mat inputImage);

  /**
   * Print statistics
   */
  void printStats();

  /**
   * Collect timing stats
   * @param delta delta time
   */
  void collectStats(double delta);

  /**
   * Get feature detector name
   * @return detector name
   */
  string getName();

  /**
   * Write image to file
   * @param path Path of the file to be written
   */
  virtual void writeImage(string path);

  /**
   * Get feature detection enable state
   */
  bool getEnable();

  /**
   * Dump stats to file
   * @param path Path of the file to be written
   */
  void dumpStatsToFile(string path);

  /**
   * Get the obtained keypoints
   * @return a vector containing all the keypoints
   */
  vector<KeyPoint> getKeyPoints();

  /**
   * Get the obtained descriptors
   * @return obtained descriptors
   */
  Mat getDescriptors();

protected:
  /** Detector */
  Ptr<Feature2D> detector;
  /** Keypoints */
  vector<KeyPoint> keyPoints;
  /** Descriptos */
  Mat descriptors;
  /** Input Image */
  Mat inputImage;
  /** Enable showing of image and GUI */
  bool showEnable = false;
  /** Enable state of the detector */
  bool enable = false;
  /** All algorithms enable flag */
  bool allEnable = false;
  /** Name of the detector */
  string name;
  /** Output Image */
  Mat outputImage;
  /** Parameters string */
  stringstream paramsString;
  /** Statistics string */
  stringstream statsString;
  /** Keypoints statistics */
  Stats<int> keyPointsStats;

  /**
   * Re-apply the detection algorithm to the stored input image
   */
  void redetect();

  /**
   * Draw ouput image to the GUI
   */
  void drawOutput();

  /**
   * Run Compute method of the detector
   */
  virtual void runCompute();

  /**
   * Run feature detection algorithm
   */
  virtual void runDetect();

  /**
   * Create the window's controls
   */
  virtual void createControls();

  /**
   * Update the output image
   */
  virtual void updateOutputImage();

private:
  /** Timing stats collector */
  Stats<double> timingStats;

  /**
   * Show the GUI
   */
  void show();

  /**
   * Wrapper run computation of the keypoint descriptors
   *
   * @warning Do not overload since timig stats are automatically gathered thru
   * this interface
   */
  virtual void _runCompute();

  /**
   * Wrapper run detection of the keypoint
   *
   * @warning Do not overload since timig stats are automatically gathered thru
   * this interface
   */
  virtual void _runDetect();
};

#endif /* FEATURE_DETECTOR_H */
