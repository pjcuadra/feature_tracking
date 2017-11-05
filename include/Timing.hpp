/**
* @file FeatureDetect.hpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief Feature Detector Class
*
*/
#ifndef TIMING_H
#define TIMING_H

#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

class Timing {
public:

  void start() {
    startTime = high_resolution_clock::now();
  }

  void end() {
    endTime = high_resolution_clock::now();
    elapsedTime = duration_cast<second> (endTime - startTime).count();
  }

  void print() {
    cout << "Elapsed time: " << elapsedTime << "s" << endl;
  }

private:
  typedef high_resolution_clock clock;
  typedef duration<double, std::ratio<1> > second;
  time_point<clock> startTime;
  time_point<clock> endTime;
  double elapsedTime;
};

#endif /* TIMING_H */
