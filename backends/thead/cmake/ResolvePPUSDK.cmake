# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

function(_flagdnn_thead_real_directory input output description)
  if(input STREQUAL "" OR NOT EXISTS "${input}")
    message(FATAL_ERROR "${description} does not exist: ${input}")
  endif()
  file(REAL_PATH "${input}" _resolved EXPAND_TILDE)
  if(_resolved STREQUAL "" OR NOT IS_DIRECTORY "${_resolved}")
    message(FATAL_ERROR "${description} is not a directory: ${input}")
  endif()
  set(${output} "${_resolved}" PARENT_SCOPE)
endfunction()

function(_flagdnn_thead_real_file input output description)
  if(input STREQUAL "" OR NOT EXISTS "${input}")
    message(FATAL_ERROR "${description} does not exist: ${input}")
  endif()
  file(REAL_PATH "${input}" _resolved EXPAND_TILDE)
  if(_resolved STREQUAL "" OR NOT EXISTS "${_resolved}" OR
     IS_DIRECTORY "${_resolved}")
    message(FATAL_ERROR "${description} is not a file: ${input}")
  endif()
  set(${output} "${_resolved}" PARENT_SCOPE)
endfunction()

function(_flagdnn_thead_require_below root path description root_description)
  file(RELATIVE_PATH _relative "${root}" "${path}")
  if(IS_ABSOLUTE "${_relative}" OR
     _relative STREQUAL ".." OR
     _relative MATCHES "^\\.\\./")
    message(FATAL_ERROR
      "${description} is outside the selected ${root_description} "
      "${root}: ${path}")
  endif()
endfunction()

function(_flagdnn_thead_import_library target location include_directory)
  if(TARGET "${target}")
    get_target_property(_existing_location "${target}" IMPORTED_LOCATION)
    get_target_property(_existing_include
      "${target}" INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT _existing_location STREQUAL location OR
       NOT _existing_include STREQUAL include_directory)
      message(FATAL_ERROR
        "${target} already exists with different PPU SDK provenance")
    endif()
    return()
  endif()
  add_library("${target}" SHARED IMPORTED GLOBAL)
  set_target_properties("${target}" PROPERTIES
    IMPORTED_LOCATION "${location}"
    INTERFACE_INCLUDE_DIRECTORIES "${include_directory}")
endfunction()

function(flagdnn_thead_resolve_ppu_sdk output_prefix)
  set(options REQUIRE_ACDNN)
  set(one_value_arguments ROOT)
  cmake_parse_arguments(
    FLAGDNN_THEAD_PPU
    "${options}"
    "${one_value_arguments}"
    ""
    ${ARGN})
  if(FLAGDNN_THEAD_PPU_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "unknown flagdnn_thead_resolve_ppu_sdk arguments: "
      "${FLAGDNN_THEAD_PPU_UNPARSED_ARGUMENTS}")
  endif()

  set(_requested_root "${FLAGDNN_THEAD_PPU_ROOT}")
  if(_requested_root STREQUAL "")
    set(_requested_root "${FLAGDNN_THEAD_PPU_SDK_ROOT}")
  endif()
  if(_requested_root STREQUAL "")
    set(_requested_root "/usr/local/PPU_SDK")
  endif()
  if(NOT "${FLAGDNN_THEAD_PPU_ROOT}" STREQUAL "" AND
     NOT "${FLAGDNN_THEAD_PPU_SDK_ROOT}" STREQUAL "")
    _flagdnn_thead_real_directory(
      "${FLAGDNN_THEAD_PPU_ROOT}" _argument_root "PPU SDK root argument")
    _flagdnn_thead_real_directory(
      "${FLAGDNN_THEAD_PPU_SDK_ROOT}" _cache_root
      "FLAGDNN_THEAD_PPU_SDK_ROOT")
    if(NOT _argument_root STREQUAL _cache_root)
      message(FATAL_ERROR
        "PPU SDK root argument conflicts with FLAGDNN_THEAD_PPU_SDK_ROOT")
    endif()
  endif()

  _flagdnn_thead_real_directory(
    "${_requested_root}" _root "PPU SDK root")
  _flagdnn_thead_real_file(
    "${_root}/release.yaml" _release_file "PPU SDK release.yaml")
  _flagdnn_thead_require_below(
    "${_root}" "${_release_file}" "PPU SDK release file" "PPU SDK root")
  file(STRINGS "${_release_file}" _version_lines
    REGEX "^[ \t]*version[ \t]*:")
  list(LENGTH _version_lines _version_count)
  if(NOT _version_count EQUAL 1)
    message(FATAL_ERROR
      "PPU SDK release.yaml must contain exactly one version field")
  endif()
  list(GET _version_lines 0 _version_line)
  if(NOT _version_line MATCHES
     "^[ \t]*version[ \t]*:[ \t]*['\"]?([^'\"# \t]+)['\"]?[ \t]*(#.*)?$")
    message(FATAL_ERROR
      "malformed PPU SDK version in ${_release_file}: ${_version_line}")
  endif()
  set(_version "${CMAKE_MATCH_1}")
  if(NOT _version MATCHES "^[0-9A-Za-z][0-9A-Za-z.+_-]*$")
    message(FATAL_ERROR "malformed PPU SDK version: ${_version}")
  endif()

  _flagdnn_thead_real_directory(
    "${_root}/include" _include_dir "PPU SDK include directory")
  _flagdnn_thead_require_below(
    "${_root}" "${_include_dir}" "PPU SDK include directory"
    "PPU SDK root")
  _flagdnn_thead_real_file(
    "${_include_dir}/hggc.h" _hggc_header "HGGC header")
  _flagdnn_thead_require_below(
    "${_root}" "${_hggc_header}" "HGGC header" "PPU SDK root")
  _flagdnn_thead_real_file(
    "${_root}/lib/libhggc.so" _hggc_library "HGGC shared library")
  _flagdnn_thead_require_below(
    "${_root}" "${_hggc_library}" "HGGC shared library" "PPU SDK root")

  _flagdnn_thead_real_directory(
    "${_root}/CUDA_SDK" _cuda_root "CUDA compatibility root")
  _flagdnn_thead_require_below(
    "${_root}" "${_cuda_root}" "CUDA compatibility root" "PPU SDK root")
  _flagdnn_thead_real_directory(
    "${_cuda_root}/include" _cuda_include_dir
    "CUDA compatibility include directory")
  _flagdnn_thead_real_file(
    "${_cuda_include_dir}/cuda.h" _cuda_header
    "CUDA compatibility header")
  _flagdnn_thead_require_below(
    "${_cuda_root}" "${_cuda_include_dir}"
    "CUDA compatibility include directory" "CUDA compatibility root")
  _flagdnn_thead_require_below(
    "${_cuda_root}" "${_cuda_header}" "CUDA compatibility header"
    "CUDA compatibility root")
  _flagdnn_thead_real_file(
    "${_cuda_root}/lib64/libcuda.so.1" _cuda_driver
    "CUDA compatibility Driver library")
  _flagdnn_thead_require_below(
    "${_cuda_root}" "${_cuda_driver}" "CUDA compatibility Driver library"
    "CUDA compatibility root")
  _flagdnn_thead_real_file(
    "${_cuda_root}/bin/nvcc" _cuda_compiler
    "CUDA compatibility compiler")
  _flagdnn_thead_require_below(
    "${_cuda_root}" "${_cuda_compiler}" "CUDA compatibility compiler"
    "CUDA compatibility root")

  set(_acdnn_header_input "${_root}/include/acdnn.h")
  set(_acdnn_library_input "${_root}/lib/libacdnn.so")
  set(_has_acdnn FALSE)
  set(_acdnn_header "")
  set(_acdnn_library "")
  if(EXISTS "${_acdnn_header_input}" AND EXISTS "${_acdnn_library_input}")
    _flagdnn_thead_real_file(
      "${_acdnn_header_input}" _acdnn_header "acDNN header")
    _flagdnn_thead_real_file(
      "${_acdnn_library_input}" _acdnn_library "acDNN shared library")
    _flagdnn_thead_require_below(
      "${_root}" "${_acdnn_header}" "acDNN header" "PPU SDK root")
    _flagdnn_thead_require_below(
      "${_root}" "${_acdnn_library}" "acDNN shared library"
      "PPU SDK root")
    set(_has_acdnn TRUE)
  elseif(EXISTS "${_acdnn_header_input}" OR EXISTS "${_acdnn_library_input}")
    message(FATAL_ERROR
      "selected PPU SDK contains an incomplete acDNN header/library pair")
  elseif(FLAGDNN_THEAD_PPU_REQUIRE_ACDNN)
    message(FATAL_ERROR
      "acDNN header and shared library are required for THead validation")
  endif()

  _flagdnn_thead_import_library(
    FlagDNN::THeadCudaDriver "${_cuda_driver}" "${_cuda_include_dir}")
  _flagdnn_thead_import_library(
    FlagDNN::THeadHggcRuntime "${_hggc_library}" "${_include_dir}")
  if(_has_acdnn)
    get_filename_component(_acdnn_include_dir "${_acdnn_header}" DIRECTORY)
    _flagdnn_thead_import_library(
      FlagDNN::THeadAcdnn "${_acdnn_library}" "${_acdnn_include_dir}")
  else()
    set(_acdnn_include_dir "")
  endif()

  set(${output_prefix}_ROOT "${_root}" PARENT_SCOPE)
  set(${output_prefix}_VERSION "${_version}" PARENT_SCOPE)
  set(${output_prefix}_RELEASE_FILE "${_release_file}" PARENT_SCOPE)
  set(${output_prefix}_INCLUDE_DIR "${_include_dir}" PARENT_SCOPE)
  set(${output_prefix}_HGGC_LIBRARY "${_hggc_library}" PARENT_SCOPE)
  set(${output_prefix}_CUDA_ROOT "${_cuda_root}" PARENT_SCOPE)
  set(${output_prefix}_CUDA_INCLUDE_DIR "${_cuda_include_dir}" PARENT_SCOPE)
  set(${output_prefix}_CUDA_DRIVER_LIBRARY "${_cuda_driver}" PARENT_SCOPE)
  set(${output_prefix}_CUDA_COMPILER "${_cuda_compiler}" PARENT_SCOPE)
  set(${output_prefix}_HAS_ACDNN "${_has_acdnn}" PARENT_SCOPE)
  set(${output_prefix}_ACDNN_INCLUDE_DIR
    "${_acdnn_include_dir}" PARENT_SCOPE)
  set(${output_prefix}_ACDNN_LIBRARY "${_acdnn_library}" PARENT_SCOPE)
endfunction()
