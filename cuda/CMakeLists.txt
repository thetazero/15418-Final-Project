cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(hello LANGUAGES CXX CUDA)
enable_language(CUDA)

add_executable(cuda_connect main.cu eval.cuh)

add_executable(cuda_tests tests.cu eval.cuh)
