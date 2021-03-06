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
#ifndef LUCIDDETECT_H
#define LUCIDDETECT_H

#include <detectors/FeatureDetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

class LucidDetect : public FeatureDetect {
public:
  /** Comand line parser options */
  static const String options;

  /**
   * LUCID feature detection
   * @param parser Comand Line Parser
   */
  LucidDetect(CommandLineParser parser)
      : FeatureDetect(parser, "LUCID", "lucid") {
    this->detector = LUCID::create();
  }

protected:
  /**
   * Run feature detection algorithm
   */
  virtual void runDetect() { FeatureDetect::runCompute(); }
};

const String LucidDetect::options = "{lucid | | LUCID Enable }";

#endif /* LUCIDDETECT_H */
