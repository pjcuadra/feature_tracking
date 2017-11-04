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

# Clone google test
set(GOOGLE_TEST_PATH "${GIT_ROOT_PATH}/googletest")
git_clone(googletest
  "${GOOGLE_TEST_URL}"
  "${GOOGLE_TEST_VERSION}"
  "${GOOGLE_TEST_PATH}"
)

# Add google test to building system
add_subdirectory(${GOOGLE_TEST_PATH})

find_package(OpenCV REQUIRED)
