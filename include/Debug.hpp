/**
* @file FeatureDetect.hpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief Feature Detector Class
*
*/
#ifndef DEBUG_H
#define DEBUG_H

#include <chrono>
#include <iostream>

using namespace std;

#define TRACE_LINE(file, line) Debug::addPoint(file, line)

class Debug {
private:
  static bool enable;

public:
  static void setEnable(bool enableVal);

  static void printMessage(string message);

  static void addPoint(string file, int line);

  static void addPoint(string file, int line, string message);
};

#endif /* DEBUG_H */
