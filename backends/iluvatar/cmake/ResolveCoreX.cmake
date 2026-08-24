# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

function(_flagdnn_iluvatar_real_existing input output description)
  if(input STREQUAL "" OR NOT EXISTS "${input}")
    message(FATAL_ERROR "${description} does not exist: ${input}")
  endif()
  file(REAL_PATH "${input}" _resolved EXPAND_TILDE)
  if(_resolved STREQUAL "" OR NOT EXISTS "${_resolved}")
    message(FATAL_ERROR "${description} has an unresolved symlink: ${input}")
  endif()
  set(${output} "${_resolved}" PARENT_SCOPE)
endfunction()

function(_flagdnn_iluvatar_require_below root path description)
  file(RELATIVE_PATH _relative "${root}" "${path}")
  if(IS_ABSOLUTE "${_relative}" OR
     _relative STREQUAL ".." OR
     _relative MATCHES "^\\.\\./")
    message(FATAL_ERROR
      "${description} is outside the selected CoreX root ${root}: ${path}")
  endif()
endfunction()

function(flagdnn_iluvatar_resolve_corex output_prefix)
  set(_explicit_include "${FLAGDNN_ILUVATAR_CUDA_INCLUDE_DIR}")
  set(_explicit_driver "${FLAGDNN_ILUVATAR_CUDA_DRIVER_LIBRARY}")
  if((_explicit_include STREQUAL "" AND NOT _explicit_driver STREQUAL "") OR
     (NOT _explicit_include STREQUAL "" AND _explicit_driver STREQUAL ""))
    message(FATAL_ERROR
      "FLAGDNN_ILUVATAR_CUDA_INCLUDE_DIR and "
      "FLAGDNN_ILUVATAR_CUDA_DRIVER_LIBRARY must be set together")
  endif()

  set(_root "")
  set(_include "")
  set(_driver "")

  if(NOT _explicit_include STREQUAL "")
    _flagdnn_iluvatar_real_existing(
      "${_explicit_include}" _include "CoreX CUDA include directory")
    _flagdnn_iluvatar_real_existing(
      "${_explicit_driver}" _driver "CoreX CUDA Driver library")
    get_filename_component(_include_parent "${_include}" DIRECTORY)
    get_filename_component(_driver_directory "${_driver}" DIRECTORY)
    get_filename_component(_driver_parent "${_driver_directory}" DIRECTORY)
    if(NOT _include_parent STREQUAL _driver_parent)
      message(FATAL_ERROR
        "explicit CoreX CUDA include and Driver library do not share one root")
    endif()
    set(_root "${_include_parent}")
  else()
    set(_requested_root "${FLAGDNN_ILUVATAR_COREX_ROOT}")
    if(_requested_root STREQUAL "" AND EXISTS "/usr/local/corex")
      set(_requested_root "/usr/local/corex")
    endif()

    if(NOT _requested_root STREQUAL "")
      _flagdnn_iluvatar_real_existing(
        "${_requested_root}" _root "CoreX SDK root")
      if(NOT EXISTS "${_root}/include/cuda.h")
        message(FATAL_ERROR
          "cuda.h was not found under selected CoreX root ${_root}")
      endif()
      _flagdnn_iluvatar_real_existing(
        "${_root}/include" _include "CoreX CUDA include directory")
      find_library(_flagdnn_iluvatar_corex_driver
        NAMES cuda libcuda.so
        PATHS "${_root}/lib64" "${_root}/lib"
        NO_DEFAULT_PATH)
      if(NOT _flagdnn_iluvatar_corex_driver)
        message(FATAL_ERROR
          "libcuda was not found under selected CoreX root ${_root}")
      endif()
      _flagdnn_iluvatar_real_existing(
        "${_flagdnn_iluvatar_corex_driver}" _driver
        "CoreX CUDA Driver library")
      unset(_flagdnn_iluvatar_corex_driver CACHE)
    else()
      find_path(_flagdnn_iluvatar_corex_include NAMES cuda.h)
      find_library(_flagdnn_iluvatar_corex_driver NAMES cuda libcuda.so)
      if(NOT _flagdnn_iluvatar_corex_include OR
         NOT _flagdnn_iluvatar_corex_driver)
        message(FATAL_ERROR
          "Cannot locate CoreX; set FLAGDNN_ILUVATAR_COREX_ROOT or the "
          "explicit Iluvatar CUDA include/library pair")
      endif()
      _flagdnn_iluvatar_real_existing(
        "${_flagdnn_iluvatar_corex_include}" _include
        "CoreX CUDA include directory")
      _flagdnn_iluvatar_real_existing(
        "${_flagdnn_iluvatar_corex_driver}" _driver
        "CoreX CUDA Driver library")
      get_filename_component(_include_parent "${_include}" DIRECTORY)
      get_filename_component(_driver_directory "${_driver}" DIRECTORY)
      get_filename_component(_driver_parent "${_driver_directory}" DIRECTORY)
      if(NOT _include_parent STREQUAL _driver_parent)
        message(FATAL_ERROR
          "discovered cuda.h and libcuda do not share one CoreX root")
      endif()
      set(_root "${_include_parent}")
      unset(_flagdnn_iluvatar_corex_include CACHE)
      unset(_flagdnn_iluvatar_corex_driver CACHE)
    endif()
  endif()

  _flagdnn_iluvatar_real_existing("${_root}" _root "CoreX SDK root")
  if(NOT EXISTS "${_include}/cuda.h")
    message(FATAL_ERROR "cuda.h was not found in ${_include}")
  endif()
  _flagdnn_iluvatar_require_below(
    "${_root}" "${_include}/cuda.h" "CoreX CUDA header")
  _flagdnn_iluvatar_require_below(
    "${_root}" "${_driver}" "CoreX CUDA Driver library")

  if(NOT "${FLAGDNN_ILUVATAR_COREX_ROOT}" STREQUAL "")
    _flagdnn_iluvatar_real_existing(
      "${FLAGDNN_ILUVATAR_COREX_ROOT}" _declared_root "CoreX SDK root")
    if(NOT _declared_root STREQUAL _root)
      message(FATAL_ERROR
        "explicit CoreX CUDA paths are outside the selected "
        "FLAGDNN_ILUVATAR_COREX_ROOT")
    endif()
  endif()

  file(READ "${_include}/cuda.h" _cuda_header)
  foreach(_required_symbol IN ITEMS
      CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
      CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
      cuDeviceGetUuid
      cuDriverGetVersion)
    string(FIND "${_cuda_header}" "${_required_symbol}" _symbol_offset)
    if(_symbol_offset EQUAL -1)
      message(FATAL_ERROR
        "selected CoreX Driver API cannot qualify corex_71: missing "
        "${_required_symbol}")
    endif()
  endforeach()

  set(_sdk_version "unknown")
  if(EXISTS "${_root}/release-corex.txt")
    file(READ "${_root}/release-corex.txt" _release)
    if(_release MATCHES "CoreX SDK[ \\t]+([0-9]+\\.[0-9]+\\.[0-9]+)")
      set(_sdk_version "${CMAKE_MATCH_1}")
    endif()
  endif()
  if(_sdk_version STREQUAL "unknown")
    get_filename_component(_root_name "${_root}" NAME)
    if(_root_name MATCHES "corex-([0-9]+\\.[0-9]+\\.[0-9]+)")
      set(_sdk_version "${CMAKE_MATCH_1}")
    endif()
  endif()

  set(${output_prefix}_ROOT "${_root}" PARENT_SCOPE)
  set(${output_prefix}_CUDA_INCLUDE_DIR "${_include}" PARENT_SCOPE)
  set(${output_prefix}_CUDA_DRIVER_LIBRARY "${_driver}" PARENT_SCOPE)
  set(${output_prefix}_SDK_VERSION "${_sdk_version}" PARENT_SCOPE)
endfunction()

function(flagdnn_iluvatar_resolve_corex_cudnn output_prefix corex_root)
  _flagdnn_iluvatar_real_existing(
    "${corex_root}" _corex_root "selected CoreX SDK root")

  set(_explicit_include "${FLAGDNN_ILUVATAR_CUDNN_INCLUDE_DIR}")
  set(_explicit_library "${FLAGDNN_ILUVATAR_CUDNN_LIBRARY}")
  if((_explicit_include STREQUAL "" AND NOT _explicit_library STREQUAL "") OR
     (NOT _explicit_include STREQUAL "" AND _explicit_library STREQUAL ""))
    message(FATAL_ERROR
      "FLAGDNN_ILUVATAR_CUDNN_INCLUDE_DIR and "
      "FLAGDNN_ILUVATAR_CUDNN_LIBRARY must be set together")
  endif()

  set(_include "")
  set(_library "")
  if(NOT _explicit_include STREQUAL "")
    _flagdnn_iluvatar_real_existing(
      "${_explicit_include}" _include "CoreX cuDNN include directory")
    _flagdnn_iluvatar_real_existing(
      "${_explicit_library}" _library "CoreX cuDNN shared library")
  else()
    set(_candidate_roots "${_corex_root}")
    if(EXISTS "/usr/local/corex")
      file(REAL_PATH "/usr/local/corex" _corex_hint EXPAND_TILDE)
      list(APPEND _candidate_roots "${_corex_hint}")
    endif()
    list(REMOVE_DUPLICATES _candidate_roots)
    foreach(_candidate IN LISTS _candidate_roots)
      if(_include STREQUAL "" AND EXISTS "${_candidate}/include/cudnn.h")
        set(_include "${_candidate}/include")
      endif()
      if(_library STREQUAL "")
        find_library(_flagdnn_iluvatar_cudnn_candidate
          NAMES cudnn libcudnn.so.7
          PATHS "${_candidate}/lib64" "${_candidate}/lib"
          NO_DEFAULT_PATH)
        if(_flagdnn_iluvatar_cudnn_candidate)
          set(_library "${_flagdnn_iluvatar_cudnn_candidate}")
        endif()
        unset(_flagdnn_iluvatar_cudnn_candidate CACHE)
      endif()
      if(NOT _include STREQUAL "" AND NOT _library STREQUAL "")
        break()
      endif()
    endforeach()

    if(_include STREQUAL "")
      find_path(_flagdnn_iluvatar_cudnn_include NAMES cudnn.h)
      if(_flagdnn_iluvatar_cudnn_include)
        set(_include "${_flagdnn_iluvatar_cudnn_include}")
      endif()
      unset(_flagdnn_iluvatar_cudnn_include CACHE)
    endif()
    if(_library STREQUAL "")
      find_library(_flagdnn_iluvatar_cudnn_library
        NAMES cudnn libcudnn.so.7)
      if(_flagdnn_iluvatar_cudnn_library)
        set(_library "${_flagdnn_iluvatar_cudnn_library}")
      endif()
      unset(_flagdnn_iluvatar_cudnn_library CACHE)
    endif()
  endif()

  if(_include STREQUAL "" OR NOT EXISTS "${_include}/cudnn.h")
    message(FATAL_ERROR
      "cudnn.h was not found for the selected CoreX validation environment")
  endif()
  if(_library STREQUAL "")
    message(FATAL_ERROR
      "libcudnn.so.7 was not found for the selected CoreX validation environment")
  endif()
  _flagdnn_iluvatar_real_existing(
    "${_include}" _include "CoreX cuDNN include directory")
  _flagdnn_iluvatar_real_existing(
    "${_library}" _library "CoreX cuDNN shared library")
  _flagdnn_iluvatar_require_below(
    "${_corex_root}" "${_include}/cudnn.h" "CoreX cuDNN header")
  _flagdnn_iluvatar_require_below(
    "${_corex_root}" "${_library}" "CoreX cuDNN shared library")

  file(READ "${_include}/cudnn.h" _cudnn_header)
  foreach(_component IN ITEMS MAJOR MINOR PATCHLEVEL)
    if(NOT _cudnn_header MATCHES
       "#[ \t]*define[ \t]+CUDNN_${_component}[ \t]+([0-9]+)")
      message(FATAL_ERROR
        "selected CoreX cudnn.h does not define CUDNN_${_component}")
    endif()
    set(_version_${_component} "${CMAKE_MATCH_1}")
  endforeach()
  math(EXPR _header_version
    "${_version_MAJOR} * 1000 + ${_version_MINOR} * 100 + ${_version_PATCHLEVEL}")
  if(NOT _header_version EQUAL 7605)
    message(FATAL_ERROR
      "CoreX cuDNN header version mismatch: expected 7605, got ${_header_version}")
  endif()

  find_program(_flagdnn_iluvatar_readelf NAMES readelf llvm-readelf REQUIRED)
  execute_process(
    COMMAND "${_flagdnn_iluvatar_readelf}" -d "${_library}"
    RESULT_VARIABLE _readelf_result
    OUTPUT_VARIABLE _dynamic_section
    ERROR_VARIABLE _readelf_error)
  if(NOT _readelf_result EQUAL 0)
    message(FATAL_ERROR
      "cannot inspect CoreX cuDNN SONAME: ${_readelf_error}")
  endif()
  string(FIND "${_dynamic_section}"
    "Library soname: [libcudnn.so.7]" _soname_offset)
  if(_soname_offset EQUAL -1)
    message(FATAL_ERROR
      "selected CoreX cuDNN library must have SONAME libcudnn.so.7")
  endif()

  find_program(_flagdnn_iluvatar_nm NAMES nm llvm-nm REQUIRED)
  execute_process(
    COMMAND "${_flagdnn_iluvatar_nm}" -D --defined-only "${_library}"
    RESULT_VARIABLE _nm_result
    OUTPUT_VARIABLE _dynamic_symbols
    ERROR_VARIABLE _nm_error)
  if(NOT _nm_result EQUAL 0)
    message(FATAL_ERROR
      "cannot inspect CoreX cuDNN symbols: ${_nm_error}")
  endif()
  string(APPEND _dynamic_symbols "\n")
  foreach(_symbol IN ITEMS
      cudnnGetVersion
      cudnnGetCudartVersion
      cudnnGetErrorString
      cudnnCreate
      cudnnDestroy
      cudnnSetStream
      cudnnCreateTensorDescriptor
      cudnnSetTensorNdDescriptor
      cudnnDestroyTensorDescriptor
      cudnnTransformTensor
      cudnnCreateOpTensorDescriptor
      cudnnSetOpTensorDescriptor
      cudnnOpTensor
      cudnnDestroyOpTensorDescriptor
      cudnnCreateReduceTensorDescriptor
      cudnnSetReduceTensorDescriptor
      cudnnGetReductionWorkspaceSize
      cudnnReduceTensor
      cudnnDestroyReduceTensorDescriptor
      cudnnCreateActivationDescriptor
      cudnnSetActivationDescriptor
      cudnnActivationForward
      cudnnActivationBackward
      cudnnDestroyActivationDescriptor
      cudnnCreateConvolutionDescriptor
      cudnnSetConvolutionNdDescriptor
      cudnnConvolutionForward
      cudnnConvolutionBackwardData
      cudnnConvolutionBackwardFilter
      cudnnDestroyConvolutionDescriptor
      cudnnBatchNormalizationForwardTraining
      cudnnBatchNormalizationForwardInference
      cudnnBatchNormalizationBackward)
    string(FIND "${_dynamic_symbols}" " ${_symbol}\n" _symbol_offset)
    if(_symbol_offset EQUAL -1)
      message(FATAL_ERROR
        "selected CoreX cuDNN library is missing classic symbol ${_symbol}")
    endif()
  endforeach()

  find_library(_flagdnn_iluvatar_cudart
    NAMES cudart libcudart.so.10.2
    PATHS "${_corex_root}/lib64" "${_corex_root}/lib"
    NO_DEFAULT_PATH REQUIRED)
  _flagdnn_iluvatar_real_existing(
    "${_flagdnn_iluvatar_cudart}" _cudart
    "CoreX CUDA Runtime library")
  _flagdnn_iluvatar_require_below(
    "${_corex_root}" "${_cudart}" "CoreX CUDA Runtime library")
  unset(_flagdnn_iluvatar_cudart CACHE)

  set(${output_prefix}_INCLUDE_DIR "${_include}" PARENT_SCOPE)
  set(${output_prefix}_LIBRARY "${_library}" PARENT_SCOPE)
  set(${output_prefix}_CUDA_RUNTIME_LIBRARY "${_cudart}" PARENT_SCOPE)
  set(${output_prefix}_HEADER_VERSION "${_header_version}" PARENT_SCOPE)
  set(${output_prefix}_SONAME "libcudnn.so.7" PARENT_SCOPE)
endfunction()
