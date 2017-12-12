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
#include <util/CustomSerializer.hpp>

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
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <util/Debug.hpp>

using namespace std;
using namespace cv;
using namespace cv::detail;

ImageFeaturesSerializer::ImageFeaturesSerializer() {
  this->features = new ImageFeatures();
}

ImageFeaturesSerializer::ImageFeaturesSerializer(ImageFeatures &features) {
  this->features = &features;
}

ImageFeaturesSerializer &ImageFeaturesSerializer::
operator=(const ImageFeaturesSerializer &obj) {
  this->features = obj.features;
}

void ImageFeaturesSerializer::write(FileStorage &fs) const {
  Mat desc;
  fs << "{";

  // Image index
  fs << "img_idx" << this->features->img_idx;

  // Image size
  fs << "img_size"
     << "{"
     << "height" << this->features->img_size.height << "width"
     << this->features->img_size.width << "}";

  // Keypoints

  fs << "keypoints" << this->features->keypoints;

  desc = this->features->descriptors.getMat(cv::ACCESS_READ);

  fs << "descriptors" << desc;

  fs << "}";
}

void ImageFeaturesSerializer::read(const FileNode &node) {
  Mat desc;
  this->features->img_idx = (int)node["img_idx"];
  this->features->img_size.height = node["img_size"]["height"];
  this->features->img_size.width = node["img_size"]["width"];

  node["keypoints"] >> this->features->keypoints;

  node["descriptors"] >> desc;

  desc.copyTo(this->features->descriptors);
}

void write(FileStorage &fs, const std::string &,
           const ImageFeaturesSerializer &x) {
  x.write(fs);
}

void read(
    const FileNode &node, ImageFeaturesSerializer &x,
    const ImageFeaturesSerializer &default_value = ImageFeaturesSerializer()) {
  if (node.empty())
    x = default_value;
  else
    x.read(node);
}

MatchesInfoSerializer::MatchesInfoSerializer() {
  this->matches = new MatchesInfo();
}

MatchesInfoSerializer::MatchesInfoSerializer(MatchesInfo &features) {
  this->matches = &features;
}

MatchesInfoSerializer &MatchesInfoSerializer::operator=(const MatchesInfoSerializer &obj) {
  this->matches = obj.matches;
}

void MatchesInfoSerializer::write(FileStorage &fs) const {
  fs << "{";

  fs << "src_img_idx" << this->matches->src_img_idx;
  fs << "dst_img_idx" << this->matches->dst_img_idx;
  fs << "matches" << this->matches->matches;
  fs << "inliers_mask" << this->matches->inliers_mask;
  fs << "num_inliers" << this->matches->num_inliers;
  fs << "H" << this->matches->H;
  fs << "confidence" << this->matches->confidence;

  fs << "}";
}

void MatchesInfoSerializer::read(const FileNode &node) {
  node["src_img_idx"] >> this->matches->src_img_idx;
  node["dst_img_idx"] >> this->matches->dst_img_idx;
  node["matches"] >> this->matches->matches;
  node["inliers_mask"] >> this->matches->inliers_mask;
  node["num_inliers"] >> this->matches->num_inliers;
  node["H"] >> this->matches->H;
  node["confidence"] >> this->matches->confidence;
}

void write(FileStorage &fs, const std::string &,
           const MatchesInfoSerializer &x) {
  x.write(fs);
}

void read(
    const FileNode &node, MatchesInfoSerializer &x,
    const MatchesInfoSerializer &default_value = MatchesInfoSerializer()) {
  if (node.empty())
    x = default_value;
  else
    x.read(node);
}
