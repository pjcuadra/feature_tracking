#include <gtest/gtest.h>

#include <util/CustomSerializer.hpp>
#include <util/TestUtil.hpp>

static const string featuresFile = "testFile.yml";

using namespace testing;

static void checkFeature(ImageFeatures actual, ImageFeatures expected) {
  EXPECT_EQ(actual.img_idx, expected.img_idx);
  EXPECT_EQ(actual.img_size, expected.img_size);
  EXPECT_EQ(actual.descriptors.size(), expected.descriptors.size());
  EXPECT_EQ(actual.keypoints.size(), expected.keypoints.size());

  for (int i = 0; i < actual.keypoints.size(); i++) {
    EXPECT_EQ(actual.keypoints[i].pt, expected.keypoints[i].pt);
    EXPECT_EQ(actual.keypoints[i].size, expected.keypoints[i].size);
  }

  EXPECT_EQ(actual.descriptors.cols, expected.descriptors.cols);
  EXPECT_EQ(actual.descriptors.rows, expected.descriptors.rows);
}

TEST(features_serialization_ut, single_feature) {
  string filename = "single_" + featuresFile;
  ImageFeatures feature, readFeature;
  ImageFeaturesSerializer readData(readFeature);
  FileStorage fs = FileStorage(filename, FileStorage::WRITE);

  feature = getRandImageFeatures();

  fs << "feature" << ImageFeaturesSerializer(feature);
  fs.release();

  fs = FileStorage(filename, FileStorage::READ);

  fs["feature"] >> readData;
  fs.release();

  checkFeature(feature, readFeature);
}

TEST(features_serialization_ut, feature_list) {
  int listSize = getRandPrimitive<int>() % 20;
  string filename = "list_" + featuresFile;
  vector<ImageFeatures> feature(listSize), readFeature(listSize);
  vector<ImageFeaturesSerializer> writtenData(listSize), readData(listSize);
  FileStorage fs = FileStorage(filename, FileStorage::WRITE);

  for (int i = 0; i < listSize; i++) {
    feature[i] = getRandImageFeatures();
    writtenData[i] = ImageFeaturesSerializer(feature[i]);
    readData[i] = ImageFeaturesSerializer(readFeature[i]);
  }

  fs << "features" << writtenData;
  fs.release();

  fs = FileStorage(filename, FileStorage::READ);

  fs["features"] >> readData;
  fs.release();

  for (int i = 0; i < listSize; i++) {
    checkFeature(feature[i], readFeature[i]);
  }
}
