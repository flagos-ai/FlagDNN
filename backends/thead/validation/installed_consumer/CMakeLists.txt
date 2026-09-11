# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.23)

project(FlagDNNTHeadInstalledConsumer LANGUAGES C CXX)

include(CTest)

find_package(FlagDNN 0.2 CONFIG REQUIRED)

foreach(_required IN ITEMS
    FLAGDNN_SOURCE_ROOT
    FLAGDNN_INSTALLED_THEAD_PPU_SDK_ROOT
    FLAGDNN_INSTALLED_THEAD_SDK_ROOT
    FLAGDNN_INSTALLED_THEAD_PLUGIN
    FLAGDNN_INSTALLED_THEAD_JIT
    FLAGDNN_INSTALLED_THEAD_TRITON_ROOT
    FLAGDNN_INSTALLED_PYTHON_LIBRARY_DIR
    FLAGDNN_INSTALLED_THEAD_CACHE_ROOT)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "${_required} is required")
  endif()
endforeach()
unset(_required)

foreach(_required_file IN ITEMS
    "${FLAGDNN_SOURCE_ROOT}/tests/core/installed_consumer/main.c"
    "${FLAGDNN_SOURCE_ROOT}/tests/core/installed_consumer/main.cpp"
    "${FLAGDNN_INSTALLED_THEAD_PLUGIN}"
    "${FLAGDNN_INSTALLED_THEAD_JIT}")
  if(NOT EXISTS "${_required_file}")
    message(FATAL_ERROR "installed-consumer input is missing: ${_required_file}")
  endif()
endforeach()
unset(_required_file)

find_path(FLAGDNN_INSTALLED_THEAD_ACDNN_INCLUDE_DIR
  NAMES acdnn.h
  PATHS "${FLAGDNN_INSTALLED_THEAD_PPU_SDK_ROOT}/include"
  NO_DEFAULT_PATH REQUIRED)
find_library(FLAGDNN_INSTALLED_THEAD_ACDNN_LIBRARY
  NAMES acdnn
  PATHS "${FLAGDNN_INSTALLED_THEAD_PPU_SDK_ROOT}/lib"
  NO_DEFAULT_PATH REQUIRED)
find_path(FLAGDNN_INSTALLED_THEAD_CUDA_INCLUDE_DIR
  NAMES cuda.h
  PATHS "${FLAGDNN_INSTALLED_THEAD_PPU_SDK_ROOT}/CUDA_SDK/include"
  NO_DEFAULT_PATH REQUIRED)
find_library(FLAGDNN_INSTALLED_THEAD_CUDA_DRIVER
  NAMES libcuda.so.1 cuda
  PATHS "${FLAGDNN_INSTALLED_THEAD_PPU_SDK_ROOT}/CUDA_SDK/lib64"
  NO_DEFAULT_PATH REQUIRED)

add_executable(flagdnn_installed_thead_c
  "${FLAGDNN_SOURCE_ROOT}/tests/core/installed_consumer/main.c")
set_target_properties(flagdnn_installed_thead_c PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES)
target_link_libraries(flagdnn_installed_thead_c PRIVATE FlagDNN::flagdnn)
add_test(NAME installed.c COMMAND flagdnn_installed_thead_c)

add_executable(flagdnn_installed_thead_cpp
  "${FLAGDNN_SOURCE_ROOT}/tests/core/installed_consumer/main.cpp")
target_compile_features(flagdnn_installed_thead_cpp PRIVATE cxx_std_20)
target_link_libraries(flagdnn_installed_thead_cpp PRIVATE FlagDNN::flagdnn)
add_test(NAME installed.cpp COMMAND flagdnn_installed_thead_cpp)

add_executable(flagdnn_installed_thead_add thead_add.cpp)
target_compile_features(flagdnn_installed_thead_add PRIVATE cxx_std_20)
target_compile_options(flagdnn_installed_thead_add PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>)
target_include_directories(flagdnn_installed_thead_add SYSTEM PRIVATE
  "${FLAGDNN_INSTALLED_THEAD_ACDNN_INCLUDE_DIR}"
  "${FLAGDNN_INSTALLED_THEAD_CUDA_INCLUDE_DIR}")
target_link_libraries(flagdnn_installed_thead_add PRIVATE
  FlagDNN::flagdnn
  "${FLAGDNN_INSTALLED_THEAD_ACDNN_LIBRARY}"
  "${FLAGDNN_INSTALLED_THEAD_CUDA_DRIVER}"
  "${CMAKE_DL_LIBS}")
get_filename_component(_acdnn_library_directory
  "${FLAGDNN_INSTALLED_THEAD_ACDNN_LIBRARY}" DIRECTORY)
get_filename_component(_cuda_library_directory
  "${FLAGDNN_INSTALLED_THEAD_CUDA_DRIVER}" DIRECTORY)
set_target_properties(flagdnn_installed_thead_add PROPERTIES
  BUILD_RPATH
    "${_acdnn_library_directory};${_cuda_library_directory}")
unset(_acdnn_library_directory)
unset(_cuda_library_directory)

add_test(NAME installed.thead_add COMMAND flagdnn_installed_thead_add)
set_tests_properties(installed.thead_add PROPERTIES
  ENVIRONMENT
    "FLAGDNN_INSTALLED_EXPECTED_SDK=${FLAGDNN_INSTALLED_THEAD_SDK_ROOT};FLAGDNN_INSTALLED_EXPECTED_PLUGIN=${FLAGDNN_INSTALLED_THEAD_PLUGIN};FLAGDNN_INSTALLED_EXPECTED_JIT=${FLAGDNN_INSTALLED_THEAD_JIT};FLAGDNN_INSTALLED_EXPECTED_TRITON=${FLAGDNN_INSTALLED_THEAD_TRITON_ROOT};FLAGDNN_THEAD_TRITON_ROOT=${FLAGDNN_INSTALLED_THEAD_TRITON_ROOT};FLAGDNN_THEAD_PPU_SDK_ROOT=${FLAGDNN_INSTALLED_THEAD_PPU_SDK_ROOT};PPU_SDK=${FLAGDNN_INSTALLED_THEAD_PPU_SDK_ROOT};PPU_HOME=${FLAGDNN_INSTALLED_THEAD_PPU_SDK_ROOT};CUDA_PATH=${FLAGDNN_INSTALLED_THEAD_PPU_SDK_ROOT}/CUDA_SDK;TRITON_PTXAS_PATH=${FLAGDNN_INSTALLED_THEAD_PPU_SDK_ROOT}/CUDA_SDK/bin/ptxas;TRITON_IR_FORMATTER_PATH=${FLAGDNN_INSTALLED_THEAD_PPU_SDK_ROOT}/bin/llvm-irformatter;LD_LIBRARY_PATH=${FLAGDNN_INSTALLED_PYTHON_LIBRARY_DIR}:${FLAGDNN_INSTALLED_THEAD_PPU_SDK_ROOT}/targets/x86_64-linux/lib:${FLAGDNN_INSTALLED_THEAD_PPU_SDK_ROOT}/lib:${FLAGDNN_INSTALLED_THEAD_PPU_SDK_ROOT}/CUDA_SDK/lib64;PYTHONDONTWRITEBYTECODE=1;TRITON_JIT_BACKEND=CUDA;FLAGDNN_CACHE_DIRECTORY=${FLAGDNN_INSTALLED_THEAD_CACHE_ROOT}"
  TIMEOUT 1800)
