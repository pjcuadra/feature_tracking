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
#ifndef STATS_H
#define STATS_H

#include <chrono>
#include <iostream>

using namespace std;

template <typename T> class Stats {
public:
  /**
   * Statistics Collector
   * @param name  Statistics name
   * @param units Variable units
   */
  Stats(string name, string units) {
    this->name = name;
    this->units = units;
  }

  /**
   * Record a new value
   * @param newValue value to be recorded
   */
  void push_back(T newValue) { this->dataSet.push_back(newValue); }

  /**
   * Return a string statistics summary
   * @return statistics summary
   */
  string str() {
    ostringstream ss;
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

private:
  /** Data set */
  vector<T> dataSet;
  /** Name of the statistics */
  string name;
  /** Units of the data's values */
  string units;
};

#endif /* STATS_H */
