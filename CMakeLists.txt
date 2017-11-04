# Define project
project(feature_tracking)

# Define required cmake version
cmake_minimum_required(VERSION 2.8)

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

#message("${OpenCV_LIBS}")

# Create Testing executable
add_executable(tests_exec ${TEST_SOURCES})

# Link tests to gtest and openCV
target_link_libraries(tests_exec opencv_core gtest gtest_main)

# Create run tests target
add_custom_target(run_tests COMMAND "${CMAKE_BINARY_DIR}/tests_exec" DEPENDS tests_exec)
