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
#ifndef SURFDETECT_H
#define SURFDETECT_H

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <detectors/FeatureDetect.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

class SurfDetect : public FeatureDetect {
public:
  /** Comand line parser options */
  static const String options;

  /**
   * SURF feature detection
   * @param parser Comand Line Parser
   */
  SurfDetect(CommandLineParser parser) : FeatureDetect(parser, "SURF", "surf") {
    this->surfHessianTh = parser.get<int>("surf_h");
    this->detector = SURF::create(this->surfHessianTh);
    paramsString << "  Hessian Threshold: " << this->surfHessianTh << endl;
  }

private:
  /** SURF Hessian Threshold */
  int surfHessianTh;
};

const String SurfDetect::options = "{surf   |     | SURF Enable    }"
                                   "{surf_h | 400 | Display images }";

#endif /* SURFDETECT_H */
