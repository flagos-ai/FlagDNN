# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.23)

if(NOT DEFINED PPU_SDK_ROOT OR PPU_SDK_ROOT STREQUAL "" OR
   NOT DEFINED TEST_ROOT OR TEST_ROOT STREQUAL "")
  message(FATAL_ERROR "PPU_SDK_ROOT and TEST_ROOT are required")
endif()
if(NOT IS_ABSOLUTE "${TEST_ROOT}" OR
   TEST_ROOT STREQUAL "/" OR
   TEST_ROOT STREQUAL "/tmp")
  message(FATAL_ERROR "TEST_ROOT must be a dedicated absolute directory")
endif()

set(_resolver
  "${CMAKE_CURRENT_LIST_DIR}/../cmake/ResolvePPUSDK.cmake")
if(NOT EXISTS "${_resolver}")
  message(FATAL_ERROR "THead PPU SDK resolver is missing: ${_resolver}")
endif()

file(REAL_PATH "${PPU_SDK_ROOT}" _real_ppu_sdk_root EXPAND_TILDE)
if(NOT IS_DIRECTORY "${_real_ppu_sdk_root}")
  message(FATAL_ERROR "real PPU SDK root does not exist: ${PPU_SDK_ROOT}")
endif()
file(STRINGS "${_real_ppu_sdk_root}/release.yaml" _real_version_lines
  REGEX "^[ \t]*version[ \t]*:")
list(LENGTH _real_version_lines _real_version_count)
if(NOT _real_version_count EQUAL 1)
  message(FATAL_ERROR "real PPU SDK has no unique version")
endif()
list(GET _real_version_lines 0 _real_version_line)
string(REGEX REPLACE
  "^[ \t]*version[ \t]*:[ \t]*['\"]?([^'\"# \t]+)['\"]?.*$"
  "\\1" _real_sdk_version "${_real_version_line}")

file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}" "${TEST_ROOT}/child")

function(_write_fake_sdk root include_acdnn version)
  file(MAKE_DIRECTORY
    "${root}/include"
    "${root}/lib"
    "${root}/CUDA_SDK/include"
    "${root}/CUDA_SDK/lib64"
    "${root}/CUDA_SDK/bin")
  file(WRITE "${root}/release.yaml" "version: ${version}\n")
  file(WRITE "${root}/include/hggc.h" "/* fake HGGC */\n")
  file(WRITE "${root}/lib/libhggc.so" "fake-hggc\n")
  file(WRITE "${root}/CUDA_SDK/include/cuda.h" "/* fake CUDA */\n")
  file(WRITE "${root}/CUDA_SDK/lib64/libcuda.so.1" "fake-driver\n")
  file(WRITE "${root}/CUDA_SDK/bin/nvcc" "#!/bin/sh\nexit 0\n")
  file(CHMOD "${root}/CUDA_SDK/bin/nvcc"
    PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE)
  if(include_acdnn)
    file(WRITE "${root}/include/acdnn.h" "/* fake acDNN */\n")
    file(WRITE "${root}/lib/libacdnn.so" "fake-acdnn\n")
  endif()
endfunction()

_write_fake_sdk("${TEST_ROOT}/valid" TRUE "9.8.7-test")
_write_fake_sdk("${TEST_ROOT}/missing-include" TRUE "9.8.7-test")
file(REMOVE_RECURSE "${TEST_ROOT}/missing-include/include")

_write_fake_sdk("${TEST_ROOT}/split-sdk" TRUE "9.8.7-test")
file(MAKE_DIRECTORY "${TEST_ROOT}/outside-sdk/lib")
file(WRITE "${TEST_ROOT}/outside-sdk/lib/libhggc.so" "outside-hggc\n")
file(REMOVE "${TEST_ROOT}/split-sdk/lib/libhggc.so")
file(CREATE_LINK
  "${TEST_ROOT}/outside-sdk/lib/libhggc.so"
  "${TEST_ROOT}/split-sdk/lib/libhggc.so" SYMBOLIC)

_write_fake_sdk("${TEST_ROOT}/driver-escape" TRUE "9.8.7-test")
file(MAKE_DIRECTORY "${TEST_ROOT}/outside-cuda")
file(WRITE "${TEST_ROOT}/outside-cuda/libcuda.so.1" "outside-driver\n")
file(REMOVE "${TEST_ROOT}/driver-escape/CUDA_SDK/lib64/libcuda.so.1")
file(CREATE_LINK
  "${TEST_ROOT}/outside-cuda/libcuda.so.1"
  "${TEST_ROOT}/driver-escape/CUDA_SDK/lib64/libcuda.so.1" SYMBOLIC)

_write_fake_sdk("${TEST_ROOT}/missing-acdnn" FALSE "9.8.7-test")
_write_fake_sdk("${TEST_ROOT}/missing-version" TRUE "9.8.7-test")
file(REMOVE "${TEST_ROOT}/missing-version/release.yaml")
_write_fake_sdk("${TEST_ROOT}/malformed-version" TRUE "bad version!")

file(WRITE "${TEST_ROOT}/child/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.23)
project(FlagDNNTHeadPPUSDKContract LANGUAGES NONE)

foreach(_required IN ITEMS RESOLVER SDK_ROOT EXPECTED_ROOT EXPECTED_VERSION
                           REQUIRE_ACDNN EXPECT_ACDNN)
  if(NOT DEFINED ${_required})
    message(FATAL_ERROR "missing child argument: ${_required}")
  endif()
endforeach()

add_library(CUDA::cuda_driver INTERFACE IMPORTED GLOBAL)
set_property(TARGET CUDA::cuda_driver PROPERTY FLAGDNN_SENTINEL "preserve")
include("${RESOLVER}")
set(FLAGDNN_THEAD_PPU_SDK_ROOT "${SDK_ROOT}" CACHE PATH "" FORCE)
if(REQUIRE_ACDNN)
  flagdnn_thead_resolve_ppu_sdk(THEAD_PPU REQUIRE_ACDNN)
else()
  flagdnn_thead_resolve_ppu_sdk(THEAD_PPU)
endif()

foreach(_required_output IN ITEMS
    ROOT VERSION INCLUDE_DIR HGGC_LIBRARY CUDA_ROOT CUDA_INCLUDE_DIR
    CUDA_DRIVER_LIBRARY CUDA_COMPILER HAS_ACDNN)
  if(NOT DEFINED THEAD_PPU_${_required_output} OR
     "${THEAD_PPU_${_required_output}}" STREQUAL "")
    message(FATAL_ERROR "resolver output is missing: ${_required_output}")
  endif()
endforeach()
if(NOT THEAD_PPU_ROOT STREQUAL EXPECTED_ROOT)
  message(FATAL_ERROR
    "SDK root is not canonical: ${THEAD_PPU_ROOT} != ${EXPECTED_ROOT}")
endif()
if(NOT THEAD_PPU_VERSION STREQUAL EXPECTED_VERSION)
  message(FATAL_ERROR
    "SDK version mismatch: ${THEAD_PPU_VERSION} != ${EXPECTED_VERSION}")
endif()
if(NOT TARGET FlagDNN::THeadCudaDriver OR
   NOT TARGET FlagDNN::THeadHggcRuntime)
  message(FATAL_ERROR "required THead imported targets are missing")
endif()
if(EXPECT_ACDNN)
  if(NOT THEAD_PPU_HAS_ACDNN OR NOT TARGET FlagDNN::THeadAcdnn)
    message(FATAL_ERROR "acDNN target is missing")
  endif()
  foreach(_output IN ITEMS ACDNN_INCLUDE_DIR ACDNN_LIBRARY)
    if(NOT DEFINED THEAD_PPU_${_output} OR
       "${THEAD_PPU_${_output}}" STREQUAL "")
      message(FATAL_ERROR "acDNN output is missing: ${_output}")
    endif()
  endforeach()
else()
  if(THEAD_PPU_HAS_ACDNN OR TARGET FlagDNN::THeadAcdnn)
    message(FATAL_ERROR "missing optional acDNN produced a target")
  endif()
endif()

get_target_property(_driver_location
  FlagDNN::THeadCudaDriver IMPORTED_LOCATION)
get_target_property(_driver_include
  FlagDNN::THeadCudaDriver INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(_hggc_location
  FlagDNN::THeadHggcRuntime IMPORTED_LOCATION)
if(NOT _driver_location STREQUAL THEAD_PPU_CUDA_DRIVER_LIBRARY OR
   NOT _driver_include STREQUAL THEAD_PPU_CUDA_INCLUDE_DIR OR
   NOT _hggc_location STREQUAL THEAD_PPU_HGGC_LIBRARY)
  message(FATAL_ERROR "THead imported target properties are incoherent")
endif()
get_target_property(_sentinel CUDA::cuda_driver FLAGDNN_SENTINEL)
if(NOT _sentinel STREQUAL "preserve")
  message(FATAL_ERROR "resolver mutated CUDA::cuda_driver")
endif()
]=])

function(_run_case name root require_acdnn expect_acdnn expect_success
         expected_root expected_version expected_error)
  set(_build "${TEST_ROOT}/build-${name}")
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
      -S "${TEST_ROOT}/child"
      -B "${_build}"
      "-DRESOLVER=${_resolver}"
      "-DSDK_ROOT=${root}"
      "-DEXPECTED_ROOT=${expected_root}"
      "-DEXPECTED_VERSION=${expected_version}"
      "-DREQUIRE_ACDNN=${require_acdnn}"
      "-DEXPECT_ACDNN=${expect_acdnn}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  set(_output "${_stdout}\n${_stderr}")
  if(expect_success)
    if(NOT _result EQUAL 0)
      message(FATAL_ERROR
        "positive PPU SDK contract ${name} failed:\n${_output}")
    endif()
  else()
    if(_result EQUAL 0)
      message(FATAL_ERROR "negative PPU SDK contract ${name} passed")
    endif()
    if(NOT _output MATCHES "${expected_error}")
      message(FATAL_ERROR
        "negative PPU SDK contract ${name} produced an unexpected error:\n"
        "${_output}")
    endif()
  endif()
endfunction()

_run_case(real-valid "${_real_ppu_sdk_root}" TRUE TRUE TRUE
  "${_real_ppu_sdk_root}" "${_real_sdk_version}" "")
_run_case(synthetic-valid "${TEST_ROOT}/valid" TRUE TRUE TRUE
  "${TEST_ROOT}/valid" "9.8.7-test" "")
_run_case(missing-include "${TEST_ROOT}/missing-include" FALSE FALSE FALSE
  "unused" "unused" "PPU SDK include directory")
_run_case(split-sdk "${TEST_ROOT}/split-sdk" FALSE TRUE FALSE
  "unused" "unused" "outside the selected PPU SDK root")
_run_case(driver-escape "${TEST_ROOT}/driver-escape" FALSE TRUE FALSE
  "unused" "unused" "outside the selected CUDA")
_run_case(missing-acdnn-optional "${TEST_ROOT}/missing-acdnn" FALSE FALSE TRUE
  "${TEST_ROOT}/missing-acdnn" "9.8.7-test" "")
_run_case(missing-acdnn-required "${TEST_ROOT}/missing-acdnn" TRUE FALSE FALSE
  "unused" "unused" "acDNN header and shared library")
_run_case(missing-version "${TEST_ROOT}/missing-version" FALSE TRUE FALSE
  "unused" "unused" "release.yaml")
_run_case(malformed-version "${TEST_ROOT}/malformed-version" FALSE TRUE FALSE
  "unused" "unused" "malformed PPU SDK version")

message(STATUS "PASS THead single-root PPU SDK CMake contracts")
