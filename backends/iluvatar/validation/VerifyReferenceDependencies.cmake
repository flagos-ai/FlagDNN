# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

foreach(_required IN ITEMS
    REFERENCE_EXECUTABLE COREX_ROOT COREX_CUDNN COREX_CUDART COREX_DRIVER
    SOURCE_DIR)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "${_required} is required")
  endif()
endforeach()

foreach(_path IN ITEMS
    "${REFERENCE_EXECUTABLE}" "${COREX_ROOT}" "${COREX_CUDNN}"
    "${COREX_CUDART}" "${COREX_DRIVER}" "${SOURCE_DIR}")
  if(NOT EXISTS "${_path}")
    message(FATAL_ERROR "reference dependency input does not exist: ${_path}")
  endif()
endforeach()

file(REAL_PATH "${COREX_ROOT}" _corex_root)
file(REAL_PATH "${COREX_CUDNN}" _expected_cudnn)
file(REAL_PATH "${COREX_CUDART}" _expected_cudart)
file(REAL_PATH "${COREX_DRIVER}" _expected_driver)

find_program(_readelf NAMES readelf llvm-readelf REQUIRED)
find_program(_nm NAMES nm llvm-nm REQUIRED)
find_program(_ldd NAMES ldd REQUIRED)

execute_process(
  COMMAND "${_readelf}" -d "${REFERENCE_EXECUTABLE}"
  RESULT_VARIABLE _readelf_result
  OUTPUT_VARIABLE _dynamic_section
  ERROR_VARIABLE _readelf_error)
if(NOT _readelf_result EQUAL 0)
  message(FATAL_ERROR
    "cannot inspect reference executable dependencies: ${_readelf_error}")
endif()

foreach(_soname IN ITEMS libcudnn.so.7 libcudart.so.10.2 libcuda.so.1)
  string(FIND "${_dynamic_section}" "Shared library: [${_soname}]"
    _needed_offset)
  if(_needed_offset EQUAL -1)
    message(FATAL_ERROR
      "reference executable has no direct DT_NEEDED entry for ${_soname}")
  endif()
endforeach()

string(TOLOWER "${_dynamic_section}" _dynamic_section_lower)
if(_dynamic_section_lower MATCHES
   "shared library: \\[lib(cublas|torch|c10|flagdnn|miopen|hipdnn|amdhip64)")
  message(FATAL_ERROR
    "reference executable has a forbidden direct compute dependency")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "LD_LIBRARY_PATH=${_corex_root}/lib64:${_corex_root}/lib"
    "${_ldd}" "${REFERENCE_EXECUTABLE}"
  RESULT_VARIABLE _ldd_result
  OUTPUT_VARIABLE _ldd_output
  ERROR_VARIABLE _ldd_error)
if(NOT _ldd_result EQUAL 0)
  message(FATAL_ERROR "ldd failed for reference executable: ${_ldd_error}")
endif()

function(_require_resolution soname expected)
  string(REGEX MATCHALL "[^\n]+" _ldd_lines "${_ldd_output}")
  set(_resolved "")
  foreach(_line IN LISTS _ldd_lines)
    string(FIND "${_line}" "${soname} =>" _soname_offset)
    if(NOT _soname_offset EQUAL -1)
      string(FIND "${_line}" "=>" _arrow_offset)
      math(EXPR _path_offset "${_arrow_offset} + 2")
      string(SUBSTRING "${_line}" ${_path_offset} -1 _resolved)
      string(STRIP "${_resolved}" _resolved)
      string(REGEX REPLACE "[ \t].*$" "" _resolved "${_resolved}")
      break()
    endif()
  endforeach()
  if(_resolved STREQUAL "")
    message(FATAL_ERROR "ldd did not resolve ${soname}:\n${_ldd_output}")
  endif()
  if(NOT EXISTS "${_resolved}")
    message(FATAL_ERROR "${soname} resolved to a missing path: ${_resolved}")
  endif()
  file(REAL_PATH "${_resolved}" _resolved_real)
  file(REAL_PATH "${expected}" _expected_real)
  if(NOT _resolved_real STREQUAL _expected_real)
    message(FATAL_ERROR
      "${soname} resolved outside the selected CoreX provenance: "
      "${_resolved_real}; expected ${_expected_real}")
  endif()
endfunction()

_require_resolution(libcudnn.so.7 "${_expected_cudnn}")
_require_resolution(libcudart.so.10.2 "${_expected_cudart}")
_require_resolution(libcuda.so.1 "${_expected_driver}")

execute_process(
  COMMAND "${_nm}" -D --undefined-only "${REFERENCE_EXECUTABLE}"
  RESULT_VARIABLE _nm_result
  OUTPUT_VARIABLE _undefined_symbols
  ERROR_VARIABLE _nm_error)
if(NOT _nm_result EQUAL 0)
  message(FATAL_ERROR
    "cannot inspect reference executable symbols: ${_nm_error}")
endif()
string(TOLOWER "${_undefined_symbols}" _undefined_symbols_lower)
if(_undefined_symbols_lower MATCHES "(cublas|torch|flagdnn|miopen|hipdnn)")
  message(FATAL_ERROR
    "reference executable imports a forbidden compute symbol")
endif()

file(GLOB_RECURSE _reference_sources LIST_DIRECTORIES FALSE
  "${SOURCE_DIR}/*.c"
  "${SOURCE_DIR}/*.cc"
  "${SOURCE_DIR}/*.cpp"
  "${SOURCE_DIR}/*.cxx"
  "${SOURCE_DIR}/*.h"
  "${SOURCE_DIR}/*.hpp")
if(_reference_sources STREQUAL "")
  message(FATAL_ERROR "reference source scan found no source files")
endif()
foreach(_source IN LISTS _reference_sources)
  file(READ "${_source}" _source_text)
  string(TOLOWER "${_source_text}" _source_text_lower)
  if(_source_text_lower MATCHES
     "(cudnn_frontend|cudnnbackend|cublas|torch|miopen|hipdnn)")
    message(FATAL_ERROR
      "forbidden reference dependency token found in ${_source}")
  endif()
endforeach()

if(DEFINED OUTPUT_JSON AND NOT OUTPUT_JSON STREQUAL "")
  get_filename_component(_output_directory "${OUTPUT_JSON}" DIRECTORY)
  file(MAKE_DIRECTORY "${_output_directory}")
  file(WRITE "${OUTPUT_JSON}"
    "{\n"
    "  \"reference_executable\": \"${REFERENCE_EXECUTABLE}\",\n"
    "  \"cudnn\": \"${_expected_cudnn}\",\n"
    "  \"cudart\": \"${_expected_cudart}\",\n"
    "  \"cuda_driver\": \"${_expected_driver}\",\n"
    "  \"direct_forbidden_compute_dependencies\": []\n"
    "}\n")
endif()

message(STATUS
  "PASS strict CoreX cuDNN reference dependency and source boundary")
