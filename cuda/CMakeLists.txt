cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(hello LANGUAGES CXX CUDA)
enable_language(CUDA)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  SET(CMAKE_CUDA_ARCHITECTURES 75)
endif()

add_executable(cuda_connect main.cu eval.cuh board.cuh board.cu engine_board.cuh engine_board.cu)

# add_executable(cuda_tests tests.cu eval.cuh)
