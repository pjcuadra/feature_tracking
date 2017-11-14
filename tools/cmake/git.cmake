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

include(ExternalProject)

function(git_clone repo_name repo_url repo_tag repo_path)

  message("-- Cloning ${repo_url}:${repo_tag}")

  # Create repository holder directory
  file(MAKE_DIRECTORY "${TMP_BUILD_PATH}/${repo_name}")

  # Creat CMakeLists.txt file for git repo cloning
  configure_file("${CMAKE_SOURCE_DIR}/tools/cmake/gitCMakeLists.txt"
    "${TMP_BUILD_PATH}/${repo_name}/CMakeLists.txt"
    @ONLY)

  # Run cmake and make commands
  execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" . OUTPUT_QUIET
      WORKING_DIRECTORY "${TMP_BUILD_PATH}/${repo_name}" )
  execute_process(COMMAND "${CMAKE_COMMAND}" --build . OUTPUT_QUIET
      WORKING_DIRECTORY "${TMP_BUILD_PATH}/${repo_name}" )

endfunction(git_clone)
