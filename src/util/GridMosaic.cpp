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
#include <assert.h>
#include <opencv2/highgui.hpp>
#include <util/GridMosaic.hpp>

GridMosaic::GridMosaic(int rows, int cols, string name) {
  this->rows = rows;
  this->cols = cols;
  this->name = name;

  this->images.resize(rows, vector<Mat>(cols));

  namedWindow(this->name, WINDOW_GUI_EXPANDED);
}

/**
 * Set an image in the corresponing row and column
 */
void GridMosaic::setImage(int row, int col, Mat image) {
  STACK_TRACE(__FUNCTION__);
  checkRowCol(row, col);
  assert(!image.empty());

  this->images[row].at(col) = image;
}

/**
 * Display the window
 */
void GridMosaic::show() { imshow(this->name, this->create()); }

void GridMosaic::checkRowCol(int row, int col) {
  assert(row < this->rows);
  assert(col < this->cols);
}

Mat GridMosaic::create() {
  int width = images[0][0].cols;
  int height = images[0][0].rows;
  Mat fullImage(height * this->rows, width * this->cols, images[0][0].type());
  Mat currentImage;

  cout << "width: " << width << endl;
  cout << "height: " << height << endl;

  cout << "width: " << fullImage.cols << endl;
  cout << "height: " << fullImage.rows << endl;

  for (int i = 0; i < this->rows; i++) {
    for (int j = 0; j < this->cols; j++) {
      cout << Rect(i * width, j * height, width, height) << endl;
      currentImage = Mat(fullImage, Rect(j * width, i * height, width, height));
      TRACE_LINE(__FILE__, __LINE__);
      // cout << this->images[i][j];
      Mat::zeros(width, height, images[0][0].type());
      this->images[i][j].copyTo(currentImage);
    }
  }

  TRACE_LINE(__FILE__, __LINE__);

  return fullImage;
}
