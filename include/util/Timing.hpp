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
#ifndef TIMING_H
#define TIMING_H

#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono;

class Timing {
public:
  /**
   * Start timer
   */
  void start() { startTime = high_resolution_clock::now(); }

  /**
   * End timer
   */
  void end() {
    endTime = high_resolution_clock::now();
    elapsedTime = duration_cast<second>(endTime - startTime).count();
  }

  /**
   * Print message with delta time
   */
  void print() { cout << "Elapsed time: " << elapsedTime << "s" << endl; }

  /**
   * Get the delta
   * @return Delta time value in second
   */
  double getDelta() { return this->elapsedTime; }

private:
  /** High resolution clock */
  typedef high_resolution_clock clock;
  /** Second definition */
  typedef duration<double, std::ratio<1>> second;
  /** Start time */
  time_point<clock> startTime;
  /** End time */
  time_point<clock> endTime;
  /** Delta time */
  double elapsedTime;
};

#endif /* TIMING_H */
