#!/bin/bash
output_build_dirs() {
  echo "Built to:"
  echo "src/Connect5_run"
  echo "test/Connect5_test"
}

# Based on the tutorial here
# https://raymii.org/s/tutorials/Cpp_project_setup_with_cmake_and_unit_tests.html
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -G "Unix Makefiles"  && make all && output_build_dirs 

