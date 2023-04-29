cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(hello LANGUAGES CXX CUDA)
enable_language(CUDA)

add_executable(cuda_connect main.cu board.cpp board.h engine_board.cpp engine_board.h)

add_executable(cuda_tests tests.cu eval.cu)
