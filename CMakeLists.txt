# MIT License
#
# Copyright (c) 2017 Pedro Cuadra
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Define project
project(feature_tracking)

# Define required cmake version
cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -rdynamic")

# Include macros
include("${CMAKE_SOURCE_DIR}/tools/cmake/git.cmake")
include("${CMAKE_SOURCE_DIR}/tools/cmake/versions.cmake")

# Locate the packages
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Create temporal directory
set(GIT_ROOT_PATH "${CMAKE_BINARY_DIR}/git")
set(TMP_BUILD_PATH "${CMAKE_BINARY_DIR}/tmp")
file(MAKE_DIRECTORY "${TMP_BUILD_PATH}")
file(MAKE_DIRECTORY "${GIT_ROOT_PATH}")

set(TEST_SOURCE_DIR "${CMAKE_SOURCE_DIR}/test/src")

# Clone google test
set(GOOGLE_TEST_PATH "${GIT_ROOT_PATH}/googletest")
git_clone(googletest
  "${GOOGLE_TEST_URL}"
  "${GOOGLE_TEST_VERSION}"
  "${GOOGLE_TEST_PATH}"
)

find_package(OpenCV REQUIRED)

# Add google test to building system
add_subdirectory(${GOOGLE_TEST_PATH})

# Include gtest headers
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

# Add all cpp in test/src
file(GLOB_RECURSE TEST_SOURCES ${TEST_SOURCE_DIR}/*.cpp)

# Add all detectors sources
file(GLOB_RECURSE DETECTOR_SOURCES ${CMAKE_SOURCE_DIR}/src/detectors/*.cpp)
file(GLOB_RECURSE TRACKERS_SOURCES ${CMAKE_SOURCE_DIR}/src/trackers/*.cpp)
file(GLOB_RECURSE UTIL_SOURCES ${CMAKE_SOURCE_DIR}/src/util/*.cpp)

#message("${OpenCV_LIBS}")

# Create Testing executable
add_executable(tests_exec
  ${DETECTOR_SOURCES}
  ${TRACKERS_SOURCES}
  ${UTIL_SOURCES}
  ${TEST_SOURCES}
)

# Link tests to gtest and openCV
target_link_libraries(tests_exec ${OpenCV_LIBS} gtest gtest_main)

# Create run tests target
add_custom_target(run_tests
  COMMAND "${CMAKE_BINARY_DIR}/tests_exec"
  DEPENDS tests_exec
)

####
include_directories(${CMAKE_SOURCE_DIR}/include)

message(STATUS "Util Sources ${UTIL_SOURCES}")

add_executable(feature_detector_demo
  ${CMAKE_SOURCE_DIR}/src/featureDetectorDemo.cpp
  ${DETECTOR_SOURCES}
  ${TRACKERS_SOURCES}
  ${UTIL_SOURCES}
)
target_link_libraries(feature_detector_demo ${OpenCV_LIBS})

add_executable(tracking_demo
  ${DETECTOR_SOURCES}
  ${TRACKERS_SOURCES}
  ${UTIL_SOURCES}
  ${CMAKE_SOURCE_DIR}/src/trackingDemo.cpp
)
target_link_libraries(tracking_demo ${OpenCV_LIBS})
