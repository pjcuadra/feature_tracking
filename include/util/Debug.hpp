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
#ifndef DEBUG_H
#define DEBUG_H

#include <chrono>
#include <iostream>
#include <sstream>

using namespace std;

#define LOG(message) Debug::printMessage(message)
#define TRACE_LINE(file, line) Debug::addPoint(file, line)
#define STACK_TRACE(function)                                                  \
  Debug::printMessage(string("Function: ") + string(function))
#define DEBUG_STREAM(obj)                                                      \
  Debug::out.str("");                                                          \
  Debug::out << obj;                                                           \
  LOG(Debug::out.str());

class Debug {
private:
  /** Debug enable flag */
  static bool enable;

public:
  static stringstream out;
  /**
   * Set debug enable flag
   * @param enableVal Enable flag value
   */
  static void setEnable(bool enableVal);

  /**
   * Print debug message
   * @param message Message to print
   */
  static void printMessage(string message);

  /**
   * Add debug trace point
   * @param file File name
   * @param line Line number within file
   */
  static void addPoint(string file, int line);

  /**
   * Add debug trace point
   * @param file    File name
   * @param line    Line number within file
   * @param message Message to append
   */
  static void addPoint(string file, int line, string message);
};

#endif /* DEBUG_H */
