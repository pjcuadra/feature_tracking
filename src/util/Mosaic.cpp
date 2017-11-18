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

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>

#include <util/Mosaic.hpp>

Mosaic::Mosaic(string name) {
  this->name = name;
  namedWindow(this->name, WINDOW_GUI_EXPANDED);
}

void Mosaic::setReference(Mat image) {
  Mat A;

  STACK_TRACE(__FUNCTION__);

  assert(!image.empty());

  struct MosaicImage mImage;

  image.copyTo(mImage.image);

  this->mosaicImages.push_back(mImage);
}

void Mosaic::setImage(Mat image, Mat M) {
  STACK_TRACE(__FUNCTION__);

  assert(!image.empty());
  assert(!M.empty());

  struct MosaicImage mImage;

  image.copyTo(mImage.image);
  M.copyTo(mImage.m);

  setImage(mImage);
}

void Mosaic::setImage(MosaicImage mosaicImage) {
  STACK_TRACE(__FUNCTION__);

  assert(!mosaicImage.image.empty());
  assert(!mosaicImage.m.empty());

  if (mosaicImage.m.rows != 3) {
    return;
  }

  this->mosaicImages.push_back(mosaicImage);
}

void Mosaic::pushImage(Mat image, Mat M) {
  struct MosaicImage mImage;

  STACK_TRACE(__FUNCTION__);

  image.copyTo(mImage.image);
  M.copyTo(mImage.m);

  // Concat transform matrix with last in the list
  mImage.m *= this->mosaicImages.back().m;

  setImage(mImage);
}

/**
 * Display the window
 */
void Mosaic::show() {
  create();

  TRACE_LINE(__FILE__, __LINE__);
  imshow(this->name, this->fullImage);
  TRACE_LINE(__FILE__, __LINE__);
}

void Mosaic::create() {
  Mat tmpImage;

  STACK_TRACE(__FUNCTION__);

  for (int i = 0; i < this->mosaicImages.size(); i++) {

    TRACE_LINE(__FILE__, __LINE__);

    // cvtColor(inputImages[i + 1], tmpImage, CV_32FC3);
    //

    LOG("Mosaic Image size:");
    DEBUG_STREAM(this->mosaicImages[i].image.size());

    TRACE_LINE(__FILE__, __LINE__);

    // cvtColor(tmpImage, tmpImage, CV_8U);
    // perspectiveTransform(inputImages[i + 1], tmpImage, homography);
    if (this->mosaicImages[i].m.empty()) {
      this->mosaicImages[i].image.copyTo(fullImage);
      continue;
    }

    TRACE_LINE(__FILE__, __LINE__);

    warpPerspective(this->mosaicImages[i].image, tmpImage,
                    this->mosaicImages[i].m,
                    this->mosaicImages[i].image.size());

    TRACE_LINE(__FILE__, __LINE__);

    LOG("Full Image size:");
    DEBUG_STREAM(fullImage.size());

    addWeighted(tmpImage, 0.5, fullImage, 0.5, 2.2, fullImage);

    TRACE_LINE(__FILE__, __LINE__);
  }
}
