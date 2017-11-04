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
