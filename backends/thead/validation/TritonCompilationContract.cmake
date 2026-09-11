# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.23)

foreach(_required IN ITEMS
    PYTHON_EXECUTABLE SOURCE_ROOT TRITON_ROOT PPU_SDK_ROOT TRITON_JIT_ROOT)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "${_required} is required")
  endif()
endforeach()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "PYTHONDONTWRITEBYTECODE=1"
    "FLAGDNN_THEAD_TRITON_ROOT=${TRITON_ROOT}"
    "FLAGDNN_THEAD_PPU_SDK_ROOT=${PPU_SDK_ROOT}"
    "FLAGDNN_THEAD_TRITON_JIT_ROOT=${TRITON_JIT_ROOT}"
    "PPU_SDK=${PPU_SDK_ROOT}"
    "PPU_HOME=${PPU_SDK_ROOT}"
    "CUDA_PATH=${PPU_SDK_ROOT}/CUDA_SDK"
    "TRITON_PTXAS_PATH=${PPU_SDK_ROOT}/CUDA_SDK/bin/ptxas"
    "TRITON_IR_FORMATTER_PATH=${PPU_SDK_ROOT}/bin/llvm-irformatter"
    "TRITON_JIT_BACKEND=CUDA"
    "${PYTHON_EXECUTABLE}"
    "${SOURCE_ROOT}/backends/thead/validation/compiler_contract.py"
    --case all --compile
  RESULT_VARIABLE _result
  OUTPUT_VARIABLE _output
  ERROR_VARIABLE _error)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR
    "THead PPU-aware Triton PPU binary compilation contract failed\n"
    "stdout:\n${_output}\nstderr:\n${_error}")
endif()

message(STATUS
  "THead qualified pointwise PPU-aware Triton PPU binary compilation contract passed")
