/**
* @file FeatureDetect.cpp
* @author Pedro Cuadra
* @date 5 Nov 2017
* @copyright 2017 Pedro Cuadra
* @brief Feature Detect clas implementation
*
*/
#include <Debug.hpp>

#define DEBUG_PREPEND "[debug] "

bool Debug::enable = false;

void Debug::setEnable(bool enableVal) { enable = enableVal; }

void Debug::printMessage(string message) {
  if (!enable) {
    return;
  }

  cout << DEBUG_PREPEND << message << endl;
}

void Debug::addPoint(string file, int line) {
  if (!enable) {
    return;
  }

  cout << DEBUG_PREPEND << file << " (" << line << ")" << endl;
}

void Debug::addPoint(string file, int line, string message) {
  if (!enable) {
    return;
  }

  cout << DEBUG_PREPEND << file << " (" << line << ") - " << message << endl;
}
