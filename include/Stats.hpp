/**
* @file FeatureDetect.hpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief Feature Detector Class
*
*/
#ifndef STATS_H
#define STATS_H

#include <chrono>
#include <iostream>

using namespace std;

template <typename T> class Stats {
public:
  Stats(string name, string units) {
    this->name = name;
    this->units = units;
  }

  void push_back(T newValue) { this->dataSet.push_back(newValue); }

  string str() {
    T sum = accumulate(dataSet.begin(), dataSet.end(), 0.0);
    double mean = 0;
    vector<T> diff(dataSet.size());
    double stdDev = 0, sq_sum = 0;
    T max = 0;
    T min = 0;

    ss.str("");

    if (dataSet.size() == 0) {
      return ss.str();
    }

    mean = sum / dataSet.size();
    max = *max_element(dataSet.begin(), dataSet.end());
    min = *min_element(dataSet.begin(), dataSet.end());

    transform(dataSet.begin(), dataSet.end(), diff.begin(),
              [mean](double x) { return x - mean; });

    sq_sum = inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    stdDev = sqrt(sq_sum / dataSet.size());

    ss << this->name << " - Stats: " << endl;
    ss << "  "
       << "Data Set size: " << dataSet.size() << endl;
    ss << "  "
       << "Mean: " << mean << this->units << endl;
    ss << "  "
       << "Std. Dev.: " << stdDev << endl;
    ss << "  "
       << "Range: [" << min << ", " << max << "]" << endl;

    return ss.str();
  }

  // ostream& operator<< (ostream& stream, const Stats<T>& value) {
  //   stream << value.string();
  //   return stream;
  // }

private:
  vector<T> dataSet;
  string name;
  string units;
  ostringstream ss;
};

#endif /* STATS_H */
