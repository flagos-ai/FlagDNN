# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.23)

if(NOT DEFINED SOURCE_ROOT OR NOT IS_DIRECTORY "${SOURCE_ROOT}")
  message(FATAL_ERROR "SOURCE_ROOT is missing")
endif()

set(_engine "${SOURCE_ROOT}/backends/thead/engines/libtriton_jit.cpp")
if(NOT EXISTS "${_engine}")
  message(FATAL_ERROR "THead libtriton_jit engine is missing")
endif()
file(READ "${_engine}" _source)

function(_flagdnn_extract_between output begin_marker end_marker description)
  string(FIND "${_source}" "${begin_marker}" _begin)
  if(_begin EQUAL -1)
    message(FATAL_ERROR "cannot find ${description} start")
  endif()
  string(SUBSTRING "${_source}" ${_begin} -1 _tail)
  string(FIND "${_tail}" "${end_marker}" _length)
  if(_length EQUAL -1)
    message(FATAL_ERROR "cannot find ${description} end")
  endif()
  string(SUBSTRING "${_tail}" 0 ${_length} _result)
  set(${output} "${_result}" PARENT_SCOPE)
endfunction()

_flagdnn_extract_between(
  _launch_arguments
  "class LaunchArguments final"
  "std::vector<std::string_view> signature_tokens"
  "fixed launch argument pack")
foreach(_required IN ITEMS
    "std::array<ArgumentValue, FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS>"
    "std::array<void*, FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS + 2>")
  string(FIND "${_launch_arguments}" "${_required}" _offset)
  if(_offset EQUAL -1)
    message(FATAL_ERROR
      "THead launch argument pack lost fixed storage: ${_required}")
  endif()
endforeach()
foreach(_forbidden IN ITEMS
    "std::vector" "new " "make_unique" "cuMemAlloc" "cuMemFree")
  string(FIND "${_launch_arguments}" "${_forbidden}" _offset)
  if(NOT _offset EQUAL -1)
    message(FATAL_ERROR
      "THead launch argument pack performs dynamic work: ${_forbidden}")
  endif()
endforeach()

_flagdnn_extract_between(
  _execute
  "  void execute(CUstream stream,"
  "\n private:"
  "steady-state execute path")
foreach(_required IN ITEMS
    "validate_execution_arguments"
    "cuStreamIsCapturing"
    "CU_STREAM_CAPTURE_STATUS_NONE"
    "cuStreamGetCtx"
    "ContextGuard guard"
    "LaunchArguments arguments"
    "cuLaunchKernel")
  string(FIND "${_execute}" "${_required}" _offset)
  if(_offset EQUAL -1)
    message(FATAL_ERROR
      "THead steady execute lost required operation: ${_required}")
  endif()
endforeach()

foreach(_forbidden IN ITEMS
    "JitFunction::get_instance"
    "launch_jit("
    "capture_prepared_launch"
    "cuModuleLoad"
    "cuModuleUnload"
    "cuMemAlloc"
    "cuMemFree"
    "cuStreamSynchronize"
    "cuCtxSynchronize"
    "cuEventSynchronize"
    "std::filesystem"
    "std::fstream"
    "std::ifstream"
    "std::ofstream"
    "std::vector"
    "std::map"
    "getenv("
    "setenv("
    "Py_"
    "PyObject"
    "std::mutex"
    "lock_guard"
    "system("
    "fork("
    "execve(")
  string(FIND "${_execute}" "${_forbidden}" _offset)
  if(NOT _offset EQUAL -1)
    message(FATAL_ERROR
      "THead steady execute is not Graph-capture-safe: ${_forbidden}")
  endif()
endforeach()

message(STATUS
  "PASS THead steady execute is a fixed-storage direct CUDA launch path")
