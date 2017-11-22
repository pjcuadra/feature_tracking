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
#ifndef GRIDMOSAIC_H
#define GRIDMOSAIC_H

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>

#include <chrono>
#include <iostream>

#include <util/Debug.hpp>

using namespace std;
using namespace cv;

class GridMosaic {
public:
  /**
   * Constructor
   * @param rows number of rows
   * @param cols number of columns
   */
  GridMosaic(int rows, int cols, string name = "Grid Mosaic");

  /**
   * Set an image in the corresponing row and column
   * @param row   row of the image
   * @param col   column of the image
   * @param image image
   */
  void setImage(int rows, int cols, Mat image);

  /**
   * Create the mosaic image
   * @return Mosaic image
   */
  Mat create();

  /**
   * Display the window
   */
  void show();

protected:
  /* Images storage */
  vector<vector<Mat>> images;
  /* Window name */
  string name;

  void checkRowCol(int row, int col);

private:
  /* Number of Rows */
  int rows;
  /* Number of columns */
  int cols;
};

#endif /* GRIDMOSAIC_H */
