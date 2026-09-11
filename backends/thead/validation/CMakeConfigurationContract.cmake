# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.23)

foreach(_required IN ITEMS SOURCE_ROOT TEST_ROOT PPU_SDK_ROOT TRITON_JIT_ROOT
                           TRITON_ROOT PYTHON_EXECUTABLE)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "${_required} is required")
  endif()
endforeach()

function(_run_contract name)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR
      "THead ${name} contract failed:\n${_stdout}\n${_stderr}")
  endif()
endfunction()

_run_contract(environment
  "${CMAKE_COMMAND}" -E env
    "PYTHONDONTWRITEBYTECODE=1"
    "FLAGDNN_THEAD_PPU_SDK_ROOT=${PPU_SDK_ROOT}"
    "FLAGDNN_THEAD_TRITON_ROOT=${TRITON_ROOT}"
    "FLAGDNN_THEAD_TRITON_JIT_ROOT=${TRITON_JIT_ROOT}"
    "${PYTHON_EXECUTABLE}"
    "${SOURCE_ROOT}/backends/thead/validation/environment_contract.py")

_run_contract(ppu-sdk
  "${CMAKE_COMMAND}"
    "-DPPU_SDK_ROOT=${PPU_SDK_ROOT}"
    "-DTEST_ROOT=${TEST_ROOT}/ppu-sdk"
    -P
      "${SOURCE_ROOT}/backends/thead/validation/PPUSDKConfigurationContract.cmake")

_run_contract(triton-jit
  "${CMAKE_COMMAND}"
    "-DTRITON_JIT_ROOT=${TRITON_JIT_ROOT}"
    "-DTRITON_ROOT=${TRITON_ROOT}"
    "-DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}"
    "-DTEST_ROOT=${TEST_ROOT}/triton-jit"
    -P
      "${SOURCE_ROOT}/backends/thead/validation/TritonJITConfigurationContract.cmake")

message(STATUS "PASS THead CMake configuration contracts")
