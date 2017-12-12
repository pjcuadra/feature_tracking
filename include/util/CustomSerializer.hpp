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
#ifndef CUSTOM_SERIALIZER_H
#define CUSTOM_SERIALIZER_H

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <util/Debug.hpp>

using namespace std;
using namespace cv;
using namespace cv::detail;

class ImageFeaturesSerializer {
public:
  ImageFeaturesSerializer();

  explicit ImageFeaturesSerializer(ImageFeatures &features);

  ImageFeaturesSerializer &operator=(const ImageFeaturesSerializer &obj);

  void write(FileStorage &fs) const;

  void read(const FileNode &node);

public:
  ImageFeatures *features;
};

void write(FileStorage &fs, const std::string &,
           const ImageFeaturesSerializer &x);

void read(const FileNode &node, ImageFeaturesSerializer &x,
          const ImageFeaturesSerializer &default_value);

class MatchesInfoSerializer {
public:
  MatchesInfoSerializer();

  explicit MatchesInfoSerializer(MatchesInfo &features);

  MatchesInfoSerializer &operator=(const MatchesInfoSerializer &obj);

  void write(FileStorage &fs) const;

  void read(const FileNode &node);

public:
  MatchesInfo *matches;
};

void write(FileStorage &fs, const std::string &,
           const MatchesInfoSerializer &x);

void read(const FileNode &node, MatchesInfoSerializer &x,
          const MatchesInfoSerializer &default_value);

#endif /* CUSTOM_SERIALIZER_H */
