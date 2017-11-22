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
  namedWindow("Warped", WINDOW_GUI_EXPANDED);
}

void Mosaic::setReference(Mat image) {
  Mat A;

  STACK_TRACE(__PRETTY_FUNCTION__);

  assert(!image.empty());

  struct MosaicImage mImage;

  image.copyTo(mImage.image);

  this->mosaicImages.push_back(mImage);
}

void Mosaic::setImage(Mat image, Mat M) {
  STACK_TRACE(__PRETTY_FUNCTION__);

  assert(!image.empty());
  assert(!M.empty());

  struct MosaicImage mImage;

  image.copyTo(mImage.image);
  M.copyTo(mImage.m);

  setImage(mImage);
}

void Mosaic::setImage(MosaicImage mosaicImage) {
  STACK_TRACE(__PRETTY_FUNCTION__);

  assert(!mosaicImage.image.empty());
  assert(!mosaicImage.m.empty());

  if (mosaicImage.m.rows != 3) {
    return;
  }

  this->mosaicImages.push_back(mosaicImage);
}

void Mosaic::pushImage(Mat image, Mat M) {
  struct MosaicImage mImage;
  vector<Mat> Rs;
  vector<Mat> Ts;

  STACK_TRACE(__PRETTY_FUNCTION__);

  // decomposeHomographyMat(M, cameraMatrix, Rs, Ts, noArray());

  image.copyTo(mImage.image);
  // Rs[0].copyTo(mImage.rotation);
  // Ts[0].copyTo(mImage.translate);
  M.copyTo(mImage.m);

  if (this->mosaicImages.empty()) {
    setImage(mImage);
    return;
  }

  if (this->mosaicImages.back().m.empty()) {
    setImage(mImage);
    return;
  }

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
  vector<Point2f> inCorners, outCorners;
  int shiftRight = 0, shiftDown = 0, outRight = 0, outDown = 0;
  Mat shiftTransform, accumulativeShift;
  Size newSize;
  Mat mask;
  vector<Point> maskPoints;

  STACK_TRACE(__PRETTY_FUNCTION__);

  for (int i = 0; i < this->mosaicImages.size(); i++) {

    TRACE_LINE(__FILE__, __LINE__);

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

    inCorners.clear();

    inCorners.push_back(Point2f(shiftRight, shiftDown));
    inCorners.push_back(
        Point2f(this->mosaicImages[i].image.cols + shiftRight, shiftDown));
    inCorners.push_back(Point2f(this->mosaicImages[i].image.cols + shiftRight,
                                this->mosaicImages[i].image.rows + shiftDown));
    inCorners.push_back(
        Point2f(shiftRight, this->mosaicImages[i].image.rows + shiftDown));

    perspectiveTransform(inCorners, outCorners, this->mosaicImages[i].m);
    DEBUG_STREAM(inCorners << "->" << outCorners);

    shiftRight = 0;
    shiftDown = 0;

    // Calc the shifting to the right/down
    for (int c = 0; c < outCorners.size(); c++) {
      if (outCorners[c].x < 0 && abs(outCorners[c].x) > shiftRight) {
        shiftRight = ceil(abs(outCorners[c].x));
      }
      if (outCorners[c].y < 0 && abs(outCorners[c].y) > shiftDown) {
        shiftDown = ceil((abs(outCorners[c].y)));
      }
    }

    LOG("Shift Right");
    DEBUG_STREAM(shiftRight);
    LOG("Shift Down");
    DEBUG_STREAM(shiftDown);

    TRACE_LINE(__FILE__, __LINE__);

    maskPoints.clear();

    for (int c = 0; c < outCorners.size(); c++) {
      outCorners[c].x = round(shiftRight + outCorners[c].x);
      outCorners[c].y = round(shiftDown + outCorners[c].y);

      maskPoints.push_back(outCorners[c]);
    }

    DEBUG_STREAM(maskPoints);

    TRACE_LINE(__FILE__, __LINE__);

    newSize = Size(fullImage.cols + shiftRight, fullImage.rows + shiftDown);

    mask = Mat(newSize.height, newSize.width, CV_8UC1);
    fillConvexPoly(mask, maskPoints, 255, 8, 0);

    TRACE_LINE(__FILE__, __LINE__);

    inCorners.clear();
    inCorners.push_back(Point2f(0, 0));
    inCorners.push_back(Point2f(0, fullImage.rows));
    inCorners.push_back(Point2f(fullImage.cols, 0));
    inCorners.push_back(Point2f(fullImage.cols, fullImage.rows));

    outCorners.clear();
    outCorners.push_back(Point2f(shiftRight, shiftDown));
    outCorners.push_back(Point2f(shiftRight, fullImage.rows + shiftDown));
    outCorners.push_back(Point2f(fullImage.cols + shiftRight, shiftDown));
    outCorners.push_back(
        Point2f(fullImage.cols + shiftRight, fullImage.rows + shiftDown));

    shiftTransform = findHomography(inCorners, outCorners, RANSAC);

    DEBUG_STREAM(shiftTransform);

    warpPerspective(fullImage, fullImage, shiftTransform, newSize);

    TRACE_LINE(__FILE__, __LINE__);

    LOG("Full Image size:");
    DEBUG_STREAM(fullImage.size());

    if (accumulativeShift.empty()) {
      accumulativeShift = shiftTransform;
    } else {
      accumulativeShift *= shiftTransform;
    }

    warpPerspective(this->mosaicImages[i].image, tmpImage,
                    accumulativeShift * this->mosaicImages[i].m, newSize);

    imshow("Warped", tmpImage);

    tmpImage.copyTo(fullImage, mask);

    TRACE_LINE(__FILE__, __LINE__);
  }
}
