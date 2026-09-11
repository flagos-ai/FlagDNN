# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.23)

if(NOT DEFINED BACKEND_LIBRARY OR NOT EXISTS "${BACKEND_LIBRARY}")
  message(FATAL_ERROR "BACKEND_LIBRARY is missing")
endif()
if(NOT DEFINED SOURCE_ROOT OR NOT IS_DIRECTORY "${SOURCE_ROOT}")
  message(FATAL_ERROR "SOURCE_ROOT is missing")
endif()

find_program(READELF_EXECUTABLE NAMES readelf llvm-readelf REQUIRED)
find_program(NM_EXECUTABLE NAMES nm llvm-nm REQUIRED)
find_program(LDD_EXECUTABLE NAMES ldd REQUIRED)
execute_process(
  COMMAND "${READELF_EXECUTABLE}" -d "${BACKEND_LIBRARY}"
  RESULT_VARIABLE _readelf_result
  OUTPUT_VARIABLE _dynamic
  ERROR_VARIABLE _readelf_error)
if(NOT _readelf_result EQUAL 0)
  message(FATAL_ERROR "cannot inspect THead plugin: ${_readelf_error}")
endif()
execute_process(
  COMMAND "${NM_EXECUTABLE}" -D --undefined-only "${BACKEND_LIBRARY}"
  RESULT_VARIABLE _nm_result
  OUTPUT_VARIABLE _undefined_symbols
  ERROR_VARIABLE _nm_error)
if(NOT _nm_result EQUAL 0)
  message(FATAL_ERROR "cannot inspect THead plugin symbols: ${_nm_error}")
endif()

string(TOLOWER "${_dynamic}\n${_undefined_symbols}" _binary_contract)
foreach(_forbidden IN ITEMS
    acdnn acblas cublas cudnn blas lapack
    flagdnn_backend_nvidia flagdnn_backend_hygon
    flagdnn_backend_iluvatar flagdnn_backend_ascend)
  if(_binary_contract MATCHES "${_forbidden}")
    message(FATAL_ERROR
      "THead production plugin contains forbidden reference ${_forbidden}:\n"
      "${_binary_contract}")
  endif()
endforeach()

file(GLOB_RECURSE _production_sources
  "${SOURCE_ROOT}/backends/thead/*.cpp"
  "${SOURCE_ROOT}/backends/thead/*.hpp")
list(FILTER _production_sources EXCLUDE REGEX "/validation/")
foreach(_source_file IN LISTS _production_sources)
  file(READ "${_source_file}" _source)
  string(TOLOWER "${_source}" _source_lower)
  foreach(_forbidden IN ITEMS acdnn acblas cublas cudnn)
    if(_source_lower MATCHES
       "#[ \t]*include[^\n]*${_forbidden}|${_forbidden}[a-z0-9_]*[ \t]*\\(")
      message(FATAL_ERROR
        "THead production source directly uses ${_forbidden}: ${_source_file}")
    endif()
  endforeach()
endforeach()

if(DEFINED REFERENCE_EXECUTABLE AND
   NOT "${REFERENCE_EXECUTABLE}" STREQUAL "")
  foreach(_required IN ITEMS
      REFERENCE_EXECUTABLE PPU_SDK_ROOT ACDNN_LIBRARY CUDA_DRIVER_LIBRARY)
    if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "" OR
       NOT EXISTS "${${_required}}")
      message(FATAL_ERROR "${_required} is missing")
    endif()
  endforeach()

  file(REAL_PATH "${PPU_SDK_ROOT}" _ppu_sdk_root)
  file(REAL_PATH "${ACDNN_LIBRARY}" _expected_acdnn)
  file(REAL_PATH "${CUDA_DRIVER_LIBRARY}" _expected_cuda_driver)
  foreach(_selected IN ITEMS "${_expected_acdnn}" "${_expected_cuda_driver}")
    file(RELATIVE_PATH _selected_relative "${_ppu_sdk_root}" "${_selected}")
    if(IS_ABSOLUTE "${_selected_relative}" OR
       _selected_relative STREQUAL ".." OR
       _selected_relative MATCHES "^\\.\\./")
      message(FATAL_ERROR
        "THead reference dependency resolves outside the selected PPU SDK: "
        "${_selected}")
    endif()
  endforeach()

  execute_process(
    COMMAND "${READELF_EXECUTABLE}" -d "${REFERENCE_EXECUTABLE}"
    RESULT_VARIABLE _reference_readelf_result
    OUTPUT_VARIABLE _reference_dynamic
    ERROR_VARIABLE _reference_readelf_error)
  if(NOT _reference_readelf_result EQUAL 0)
    message(FATAL_ERROR
      "cannot inspect THead reference executable: ${_reference_readelf_error}")
  endif()
  foreach(_required_soname IN ITEMS libacdnn.so libcuda.so.1)
    string(FIND "${_reference_dynamic}"
      "Shared library: [${_required_soname}]" _required_soname_offset)
    if(_required_soname_offset EQUAL -1)
      message(FATAL_ERROR
        "THead reference executable has no direct ${_required_soname} "
        "dependency")
    endif()
  endforeach()

  string(TOLOWER "${_reference_dynamic}" _reference_dynamic_lower)
  if(_reference_dynamic_lower MATCHES
     "shared library: \\[lib(acblas|cublas|cudnn|blas|lapack|torch|c10|miopen|hipdnn)")
    message(FATAL_ERROR
      "THead reference executable has a forbidden direct compute dependency")
  endif()

  execute_process(
    COMMAND "${NM_EXECUTABLE}" -D --undefined-only
      "${REFERENCE_EXECUTABLE}"
    RESULT_VARIABLE _reference_nm_result
    OUTPUT_VARIABLE _reference_symbols
    ERROR_VARIABLE _reference_nm_error)
  if(NOT _reference_nm_result EQUAL 0)
    message(FATAL_ERROR
      "cannot inspect THead reference symbols: ${_reference_nm_error}")
  endif()
  string(TOLOWER "${_reference_symbols}" _reference_symbols_lower)
  if(NOT _reference_symbols_lower MATCHES "acdnngetversion" OR
     NOT _reference_symbols_lower MATCHES "cumemalloc")
    message(FATAL_ERROR
      "THead reference executable does not exercise acDNN and PPU driver APIs")
  endif()
  if(_reference_symbols_lower MATCHES
     "(acblas|cublas|cudnn|(^|[^a-z])blas|lapack|torch|c10::|miopen|hipdnn)")
    message(FATAL_ERROR
      "THead reference executable imports a forbidden compute symbol")
  endif()

  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
      "LD_LIBRARY_PATH=${_ppu_sdk_root}/lib:${_ppu_sdk_root}/CUDA_SDK/lib64"
      "${LDD_EXECUTABLE}" "${REFERENCE_EXECUTABLE}"
    RESULT_VARIABLE _ldd_result
    OUTPUT_VARIABLE _ldd_output
    ERROR_VARIABLE _ldd_error)
  if(NOT _ldd_result EQUAL 0)
    message(FATAL_ERROR
      "cannot resolve THead reference dependencies: ${_ldd_error}")
  endif()
  string(TOLOWER "${_ldd_output}" _ldd_output_lower)
  if(_ldd_output_lower MATCHES
     "lib(acblas|cublas|cudnn|blas|lapack|torch|c10|miopen|hipdnn)")
    message(FATAL_ERROR
      "THead reference dependency closure contains a forbidden library")
  endif()

  function(_flagdnn_thead_require_resolution soname expected)
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
    if(_resolved STREQUAL "" OR NOT EXISTS "${_resolved}")
      message(FATAL_ERROR "ldd did not resolve ${soname}:\n${_ldd_output}")
    endif()
    file(REAL_PATH "${_resolved}" _resolved_real)
    file(REAL_PATH "${expected}" _expected_real)
    if(NOT _resolved_real STREQUAL _expected_real)
      message(FATAL_ERROR
        "${soname} resolved outside the selected PPU SDK: "
        "${_resolved_real}; expected ${_expected_real}")
    endif()
  endfunction()

  _flagdnn_thead_require_resolution(libacdnn.so "${_expected_acdnn}")
  _flagdnn_thead_require_resolution(libcuda.so.1 "${_expected_cuda_driver}")

  set(_reference_sources
    "${SOURCE_ROOT}/backends/thead/validation/acdnn_reference.cpp"
    "${SOURCE_ROOT}/backends/thead/validation/acdnn_reference.hpp"
    "${SOURCE_ROOT}/backends/thead/validation/capability.cpp"
    "${SOURCE_ROOT}/backends/thead/validation/capability.hpp"
    "${SOURCE_ROOT}/backends/thead/validation/tensor_io.cpp"
    "${SOURCE_ROOT}/backends/thead/validation/tensor_io.hpp")
  file(GLOB _reference_extensions LIST_DIRECTORIES FALSE
    "${SOURCE_ROOT}/backends/thead/validation/*_reference.cpp"
    "${SOURCE_ROOT}/backends/thead/validation/*_reference.hpp"
    "${SOURCE_ROOT}/backends/thead/validation/acdnn_*_dag.cpp"
    "${SOURCE_ROOT}/backends/thead/validation/acdnn_*_dag.hpp"
    "${SOURCE_ROOT}/backends/thead/validation/acdnn_fp8_codec.cpp"
    "${SOURCE_ROOT}/backends/thead/validation/acdnn_fp8_codec.hpp"
    "${SOURCE_ROOT}/backends/thead/validation/functional/*acdnn*.cpp"
    "${SOURCE_ROOT}/backends/thead/validation/functional/*acdnn*.hpp"
    "${SOURCE_ROOT}/backends/thead/validation/functional/test_autotune.cpp"
    "${SOURCE_ROOT}/backends/thead/validation/functional/test_graph.cpp"
    "${SOURCE_ROOT}/backends/thead/validation/functional/*runner*.cpp"
    "${SOURCE_ROOT}/backends/thead/validation/functional/*runner*.hpp"
    "${SOURCE_ROOT}/backends/thead/validation/benchmark/*.cpp"
    "${SOURCE_ROOT}/backends/thead/validation/benchmark/*.hpp")
  list(APPEND _reference_sources ${_reference_extensions})
  list(REMOVE_DUPLICATES _reference_sources)
  foreach(_reference_source IN LISTS _reference_sources)
    if(NOT EXISTS "${_reference_source}")
      message(FATAL_ERROR
        "THead reference source is missing: ${_reference_source}")
    endif()
    file(READ "${_reference_source}" _reference_source_text)
    string(TOLOWER "${_reference_source_text}" _reference_source_lower)
    if(_reference_source_lower MATCHES
       "(acblas|cublas|cudnn|(^|[^a-z0-9_])blas([^a-z0-9_]|$)|lapack|torch::|at::|c10::|reference_cpu|cpu_reference)")
      message(FATAL_ERROR
        "forbidden numerical reference token found in ${_reference_source}")
    endif()
  endforeach()
endif()

message(STATUS
  "PASS THead production/acDNN-only reference dependency boundary")
