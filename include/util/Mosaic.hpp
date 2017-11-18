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
#ifndef MOSAIC_H
#define MOSAIC_H

#include <chrono>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <util/Debug.hpp>

using namespace std;
using namespace cv;

class Mosaic {
public:
  struct MosaicImage {
    Mat image;
    Mat m;
  };
  /**
   * Constructor
   * @param rows number of rows
   * @param cols number of columns
   */
  Mosaic(string name = "Mosaic");

  void setImage(Mat image, Mat M);
  void setImage(MosaicImage mosaicImage);
  void pushImage(Mat image, Mat M);
  void setReference(Mat image);

  /**
   * Display the window
   */
  void show();

private:
  /* Images storage */
  vector<MosaicImage> mosaicImages;
  /* Window name */
  string name;

  Mat fullImage;

  /**
   * Create the mosaic image
   * @return Mosaic image
   */
  void create();
};

#endif /* GRIDMOSAIC_H */