# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.23)

if(NOT DEFINED SOURCE_ROOT OR NOT IS_DIRECTORY "${SOURCE_ROOT}")
  message(FATAL_ERROR "SOURCE_ROOT is missing")
endif()

set(_engine "${SOURCE_ROOT}/backends/thead/engines/libtriton_jit.cpp")
set(_compatibility_header
  "${SOURCE_ROOT}/backends/thead/engines/jit_candidate_compatibility.hpp")
set(_artifact_header "${SOURCE_ROOT}/backends/thead/artifact.hpp")
set(_backend_cmake "${SOURCE_ROOT}/backends/thead/CMakeLists.txt")
foreach(_path IN ITEMS
    "${_engine}" "${_compatibility_header}" "${_artifact_header}"
    "${_backend_cmake}")
  if(NOT EXISTS "${_path}")
    message(FATAL_ERROR "THead autotune contract input is missing: ${_path}")
  endif()
endforeach()

file(READ "${_engine}" _engine_source)
file(READ "${_compatibility_header}" _compatibility_source)
file(READ "${_artifact_header}" _artifact_source)
file(READ "${_backend_cmake}" _cmake_source)
set(_autotune_source "${_engine_source}\n${_compatibility_source}")

foreach(_required IN ITEMS
    "#include \"backends/autotune_policy.hpp\""
    "class CandidateCompatibilityError"
    "validate_candidate_resources"
    "catch (const CandidateCompatibilityError&"
    "backend::autotune::find_cached_candidate"
    "backend::autotune::discard_cached_candidate"
    "backend::autotune::select_best_candidate"
    "cuEventCreate"
    "cuEventRecord"
    "cuEventElapsedTime"
    "cuEventSynchronize"
    "FLAGDNN_THEAD_TRITON_JIT_PROVENANCE_SHA256")
  string(FIND "${_autotune_source}" "${_required}" _offset)
  if(_offset EQUAL -1)
    message(FATAL_ERROR
      "THead prepare-time autotune is missing required token: ${_required}")
  endif()
endforeach()

foreach(_required IN ITEMS
    "std::optional<unsigned int> maxnreg"
    "ppu_compiler_options")
  string(FIND "${_artifact_source}" "${_required}" _offset)
  if(_offset EQUAL -1)
    message(FATAL_ERROR
      "THead candidate identity schema is missing: ${_required}")
  endif()
endforeach()

foreach(_required IN ITEMS
    "backends/autotune_policy.cpp"
    "FLAGDNN_THEAD_TRITON_JIT_PROVENANCE_SHA256")
  string(FIND "${_cmake_source}" "${_required}" _offset)
  if(_offset EQUAL -1)
    message(FATAL_ERROR
      "THead backend build omits autotune dependency/provenance: ${_required}")
  endif()
endforeach()

string(FIND "${_engine_source}" "select_candidate(" _select_begin)
string(FIND "${_engine_source}"
  "std::vector<const ExecutionStage*> execution_order" _select_end)
string(FIND "${_engine_source}" "void execute(" _execute_begin)
if(_select_begin EQUAL -1 OR _select_end EQUAL -1 OR
   _execute_begin EQUAL -1 OR NOT _select_begin LESS _select_end OR
   NOT _select_end LESS _execute_begin)
  message(FATAL_ERROR "cannot isolate THead prepare-time autotune region")
endif()
math(EXPR _select_length "${_select_end} - ${_select_begin}")
string(SUBSTRING "${_engine_source}" ${_select_begin} ${_select_length}
  _selection_source)
if(_selection_source MATCHES
   "catch[ \\t\\r\\n]*\\([ \\t\\r\\n]*const[ \\t\\r\\n]+std::exception")
  message(FATAL_ERROR
    "THead candidate selection broadly catches std::exception; schema/ABI/JIT/runtime errors must fail the Graph")
endif()

string(SUBSTRING "${_engine_source}" ${_execute_begin} -1 _execute_source)
foreach(_forbidden IN ITEMS
    "select_candidate"
    "select_best_candidate"
    "find_cached_candidate"
    "discard_cached_candidate"
    "cuEventCreate"
    "cuEventRecord"
    "cuEventSynchronize"
    "selection_cache")
  string(FIND "${_execute_source}" "${_forbidden}" _offset)
  if(NOT _offset EQUAL -1)
    message(FATAL_ERROR
      "THead steady execute contains forbidden autotune work: ${_forbidden}")
  endif()
endforeach()

message(STATUS
  "PASS THead strict candidate compatibility and prepare-only autotune contract")
