# Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0

# Ascend validation is deliberately isolated from the production plugin. Only
# targets in this directory may include ACLNN headers or link operator API
# libraries.
include("${PROJECT_SOURCE_DIR}/cmake/Operators.cmake")

if(NOT TARGET flagdnn_backend_ascend)
  message(FATAL_ERROR "Ascend validation requires FLAGDNN_BACKENDS=ascend")
endif()
if(NOT TARGET Ascend::ascendcl OR NOT TARGET Ascend::runtime)
  message(FATAL_ERROR
    "Ascend validation requires the production AscendCL imported targets")
endif()
if(NOT CANN_ROOT OR NOT IS_DIRECTORY "${CANN_ROOT}")
  message(FATAL_ERROR "Ascend validation requires a valid CANN_ROOT")
endif()
if(NOT FLAGDNN_CODEGEN_PYTHON)
  message(FATAL_ERROR "Ascend validation requires FLAGDNN_CODEGEN_PYTHON")
endif()
execute_process(
  COMMAND "${FLAGDNN_CODEGEN_PYTHON}" -I -c
    "import pathlib,sysconfig; print(pathlib.Path(sysconfig.get_paths()['purelib']).resolve())"
  RESULT_VARIABLE _flagdnn_validation_python_root_status
  OUTPUT_VARIABLE _flagdnn_validation_python_module_root
  ERROR_VARIABLE _flagdnn_validation_python_root_error
  OUTPUT_STRIP_TRAILING_WHITESPACE
  TIMEOUT 10)
if(NOT _flagdnn_validation_python_root_status EQUAL 0 OR
   NOT IS_DIRECTORY "${_flagdnn_validation_python_module_root}")
  message(FATAL_ERROR
    "Ascend validation cannot resolve the codegen Python module root: "
    "${_flagdnn_validation_python_root_error}")
endif()
file(REAL_PATH "${_flagdnn_validation_python_module_root}"
  _flagdnn_validation_python_module_root)

file(REAL_PATH "${CANN_ROOT}" _flagdnn_validation_cann_root)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(_flagdnn_validation_arch_directory "aarch64-linux")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  set(_flagdnn_validation_arch_directory "x86_64-linux")
else()
  message(FATAL_ERROR
    "Ascend validation does not support ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(_flagdnn_validation_arch_root
  "${_flagdnn_validation_cann_root}/${_flagdnn_validation_arch_directory}")
find_path(_flagdnn_validation_aclnn_include
  NAMES
    aclnn/acl_meta.h
    aclnnop/aclnn_batch_norm.h
    aclnnop/aclnn_batch_norm_elemt.h
    aclnnop/aclnn_add.h
    aclnnop/aclnn_sub.h
    aclnnop/aclnn_mul.h
    aclnnop/aclnn_div.h
    aclnnop/aclnn_eq_tensor.h
    aclnnop/aclnn_fmod_tensor.h
    aclnnop/aclnn_ge_tensor.h
    aclnnop/aclnn_gt_tensor.h
    aclnnop/aclnn_ne_tensor.h
    aclnnop/aclnn_pow_tensor_tensor.h
    aclnnop/aclnn_s_where.h
    aclnnop/aclnn_flatten.h
    aclnnop/aclnn_permute.h
    aclnnop/aclnn_slice_v2.h
    aclnnop/aclnn_minimum.h
    aclnnop/aclnn_maximum.h
    aclnnop/aclnn_abs.h
    aclnnop/aclnn_ceil.h
    aclnnop/aclnn_clamp.h
    aclnnop/aclnn_cos.h
    aclnnop/aclnn_elu.h
    aclnnop/aclnn_erf.h
    aclnnop/aclnn_exp.h
    aclnnop/aclnn_floor.h
    aclnnop/aclnn_gelu.h
    aclnnop/aclnn_gelu_v2.h
    aclnnop/aclnn_log.h
    aclnnop/aclnn_logical_and.h
    aclnnop/aclnn_logical_not.h
    aclnnop/aclnn_logical_or.h
    aclnnop/aclnn_matmul.h
    aclnnop/aclnn_convolution.h
    aclnnop/aclnn_neg.h
    aclnnop/aclnn_reciprocal.h
    aclnnop/aclnn_reduce_sum.h
    aclnnop/aclnn_mean.h
    aclnnop/aclnn_layer_norm.h
    aclnnop/aclnn_prod.h
    aclnnop/aclnn_rsqrt.h
    aclnnop/aclnn_sigmoid.h
    aclnnop/aclnn_sigmoid_backward.h
    aclnnop/aclnn_sin.h
    aclnnop/aclnn_softplus.h
    aclnnop/aclnn_sqrt.h
    aclnnop/aclnn_swish.h
    aclnnop/aclnn_tan.h
    aclnnop/aclnn_tanh.h
  PATHS
    "${_flagdnn_validation_arch_root}/include"
    "${_flagdnn_validation_arch_root}/pkg_inc"
  NO_DEFAULT_PATH
  NO_CACHE)
find_library(_flagdnn_validation_nnopbase
  NAMES nnopbase
  PATHS "${_flagdnn_validation_arch_root}/lib64"
  NO_DEFAULT_PATH
  NO_CACHE)
find_library(_flagdnn_validation_opapi_math
  NAMES opapi_math
  PATHS "${_flagdnn_validation_arch_root}/lib64"
  NO_DEFAULT_PATH
  NO_CACHE)
find_library(_flagdnn_validation_opapi_nn
  NAMES opapi_nn
  PATHS "${_flagdnn_validation_arch_root}/lib64"
  NO_DEFAULT_PATH
  NO_CACHE)
if(NOT _flagdnn_validation_aclnn_include OR
   NOT _flagdnn_validation_nnopbase OR
   NOT _flagdnn_validation_opapi_math OR
   NOT _flagdnn_validation_opapi_nn)
  message(FATAL_ERROR
    "Ascend validation requires ACLNN headers, libnnopbase.so, and "
    "the precise libopapi_math.so/libopapi_nn.so libraries from the "
    "selected CANN 9 installation")
endif()
if(NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnn/acl_meta.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_add.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_sub.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_mul.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_div.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_eq_tensor.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_ge_tensor.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_gt_tensor.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_ne_tensor.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_fmod_tensor.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_pow_tensor_tensor.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_s_where.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_minimum.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_maximum.h")
  message(FATAL_ERROR
    "Ascend validation requires binary pointwise ACLNN headers")
endif()
if(NOT EXISTS
    "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_batch_norm_elemt.h")
  message(FATAL_ERROR
    "Ascend BatchNorm inference validation requires the exact "
    "aclnn_batch_norm_elemt.h header")
endif()
if(NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_matmul.h")
  message(FATAL_ERROR
    "Ascend MatMul validation requires the exact aclnn_matmul.h header")
endif()
if(NOT EXISTS
    "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_convolution.h")
  message(FATAL_ERROR
    "Ascend convolution validation requires the exact "
    "aclnn_convolution.h header")
endif()
if(NOT EXISTS
    "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_batch_norm.h")
  message(FATAL_ERROR
    "Ascend BatchNorm validation requires the exact "
    "aclnn_batch_norm.h header")
endif()
if(NOT EXISTS
    "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_layer_norm.h")
  message(FATAL_ERROR
    "Ascend LayerNorm validation requires the exact "
    "aclnn_layer_norm.h header")
endif()
if(NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_abs.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_ceil.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_clamp.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_cos.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_erf.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_exp.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_floor.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_log.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_logical_and.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_logical_not.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_logical_or.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_neg.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_reciprocal.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_rsqrt.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_sin.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_sqrt.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_tan.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_tanh.h")
  message(FATAL_ERROR
    "Ascend validation requires unary ACLNN math primitive headers")
endif()
if(NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_reduce_sum.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_mean.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_prod.h")
  message(FATAL_ERROR
    "Ascend validation requires ACLNN reduction math primitive headers")
endif()
if(NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_flatten.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_permute.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_slice_v2.h")
  message(FATAL_ERROR
    "Ascend validation requires ACLNN layout primitive headers")
endif()
if(NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_elu.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_gelu.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_gelu_v2.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_sigmoid.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_sigmoid_backward.h" OR
   NOT EXISTS
     "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_softplus.h" OR
   NOT EXISTS "${_flagdnn_validation_aclnn_include}/aclnnop/aclnn_swish.h")
  message(FATAL_ERROR
    "Ascend validation requires unary ACLNN neural-network headers")
endif()
foreach(_flagdnn_validation_library IN ITEMS
    _flagdnn_validation_nnopbase _flagdnn_validation_opapi_math
    _flagdnn_validation_opapi_nn)
  file(REAL_PATH "${${_flagdnn_validation_library}}"
    "${_flagdnn_validation_library}")
endforeach()
get_filename_component(_flagdnn_validation_nnopbase_name
  "${_flagdnn_validation_nnopbase}" NAME)
get_filename_component(_flagdnn_validation_opapi_math_name
  "${_flagdnn_validation_opapi_math}" NAME)
get_filename_component(_flagdnn_validation_opapi_nn_name
  "${_flagdnn_validation_opapi_nn}" NAME)
if(NOT _flagdnn_validation_nnopbase_name MATCHES "^libnnopbase\\.so" OR
   NOT _flagdnn_validation_opapi_math_name MATCHES "^libopapi_math\\.so" OR
   NOT _flagdnn_validation_opapi_nn_name MATCHES "^libopapi_nn\\.so")
  message(FATAL_ERROR
    "Ascend validation resolved an imprecise ACLNN dependency set")
endif()

add_library(FlagDNNAscendValidation::nnopbase SHARED IMPORTED GLOBAL)
set_target_properties(FlagDNNAscendValidation::nnopbase PROPERTIES
  IMPORTED_LOCATION "${_flagdnn_validation_nnopbase}"
  INTERFACE_INCLUDE_DIRECTORIES "${_flagdnn_validation_aclnn_include}")
add_library(FlagDNNAscendValidation::opapi_math SHARED IMPORTED GLOBAL)
set_target_properties(FlagDNNAscendValidation::opapi_math PROPERTIES
  IMPORTED_LOCATION "${_flagdnn_validation_opapi_math}"
  INTERFACE_INCLUDE_DIRECTORIES "${_flagdnn_validation_aclnn_include}")
add_library(FlagDNNAscendValidation::opapi_nn SHARED IMPORTED GLOBAL)
set_target_properties(FlagDNNAscendValidation::opapi_nn PROPERTIES
  IMPORTED_LOCATION "${_flagdnn_validation_opapi_nn}"
  INTERFACE_INCLUDE_DIRECTORIES "${_flagdnn_validation_aclnn_include}")

function(_flagdnn_validation_imported_location target output)
  set(_flagdnn_validation_location "")
  foreach(_flagdnn_validation_property IN ITEMS
      IMPORTED_LOCATION IMPORTED_LOCATION_NOCONFIG IMPORTED_LOCATION_RELEASE
      IMPORTED_LOCATION_RELWITHDEBINFO IMPORTED_LOCATION_DEBUG)
    if(NOT _flagdnn_validation_location)
      get_target_property(_flagdnn_validation_candidate "${target}"
        "${_flagdnn_validation_property}")
      if(_flagdnn_validation_candidate AND
         EXISTS "${_flagdnn_validation_candidate}")
        file(REAL_PATH "${_flagdnn_validation_candidate}"
          _flagdnn_validation_location)
      endif()
    endif()
  endforeach()
  if(NOT _flagdnn_validation_location)
    message(FATAL_ERROR "Cannot determine imported location for ${target}")
  endif()
  set("${output}" "${_flagdnn_validation_location}" PARENT_SCOPE)
endfunction()

_flagdnn_validation_imported_location(
  Ascend::ascendcl _flagdnn_validation_ascendcl)
_flagdnn_validation_imported_location(
  Ascend::runtime _flagdnn_validation_runtime)
# Production promotes its selected imported target to global scope. Read that
# exact target so validation cannot accidentally hash a different installation.
if(TARGET TritonJIT::triton_jit)
  _flagdnn_validation_imported_location(
    TritonJIT::triton_jit _flagdnn_validation_triton_jit)
elseif(TARGET triton_jit)
  _flagdnn_validation_imported_location(
    triton_jit _flagdnn_validation_triton_jit)
else()
  message(FATAL_ERROR
    "Ascend validation cannot inspect the production NPU libtriton_jit target")
endif()

file(SHA256 "${_flagdnn_validation_ascendcl}"
  _flagdnn_validation_ascendcl_sha256)
file(SHA256 "${_flagdnn_validation_runtime}"
  _flagdnn_validation_runtime_sha256)
file(SHA256 "${_flagdnn_validation_triton_jit}"
  _flagdnn_validation_triton_jit_sha256)
file(SHA256 "${_flagdnn_validation_nnopbase}"
  _flagdnn_validation_nnopbase_sha256)
file(SHA256 "${_flagdnn_validation_opapi_math}"
  _flagdnn_validation_opapi_math_sha256)
file(SHA256 "${_flagdnn_validation_opapi_nn}"
  _flagdnn_validation_opapi_nn_sha256)

if(NOT CMAKE_READELF)
  find_program(CMAKE_READELF readelf REQUIRED)
endif()
function(_flagdnn_validation_object_id library sha256 output)
  execute_process(COMMAND "${CMAKE_READELF}" -n "${library}"
    RESULT_VARIABLE _flagdnn_build_id_status
    OUTPUT_VARIABLE _flagdnn_build_id_notes
    ERROR_VARIABLE _flagdnn_build_id_error)
  if(NOT _flagdnn_build_id_status EQUAL 0)
    message(FATAL_ERROR
      "Cannot inspect build ID for ${library}: ${_flagdnn_build_id_error}")
  endif()
  string(REGEX MATCH "Build ID: ([0-9A-Fa-f]+)"
    _flagdnn_build_id_match "${_flagdnn_build_id_notes}")
  if(CMAKE_MATCH_1)
    string(TOLOWER "${CMAKE_MATCH_1}" _flagdnn_build_id_value)
    set(_flagdnn_build_id_value "build-id:${_flagdnn_build_id_value}")
  else()
    set(_flagdnn_build_id_value "sha256:${sha256}")
  endif()
  set("${output}" "${_flagdnn_build_id_value}" PARENT_SCOPE)
endfunction()
_flagdnn_validation_object_id("${_flagdnn_validation_ascendcl}"
  "${_flagdnn_validation_ascendcl_sha256}"
  _flagdnn_validation_ascendcl_build_id)
_flagdnn_validation_object_id("${_flagdnn_validation_runtime}"
  "${_flagdnn_validation_runtime_sha256}"
  _flagdnn_validation_runtime_build_id)

set(_flagdnn_validation_install_info
  "${_flagdnn_validation_arch_root}/ascend_toolkit_install.info")
if(NOT EXISTS "${_flagdnn_validation_install_info}")
  message(FATAL_ERROR "Ascend validation cannot find CANN install metadata")
endif()
file(STRINGS "${_flagdnn_validation_install_info}"
  _flagdnn_validation_version_line REGEX "^version=")
list(LENGTH _flagdnn_validation_version_line
  _flagdnn_validation_version_count)
if(NOT _flagdnn_validation_version_count EQUAL 1)
  message(FATAL_ERROR "CANN install metadata has no unique version entry")
endif()
string(REGEX REPLACE "^version=" "" _flagdnn_validation_cann_version
  "${_flagdnn_validation_version_line}")

add_library(flagdnn_validation_ascend_platform INTERFACE)
target_compile_features(flagdnn_validation_ascend_platform
  INTERFACE cxx_std_20)
target_include_directories(flagdnn_validation_ascend_platform INTERFACE
  "${_flagdnn_validation_aclnn_include}"
  "${PROJECT_SOURCE_DIR}/backends/ascend")
target_link_libraries(flagdnn_validation_ascend_platform INTERFACE
  Ascend::ascendcl Ascend::runtime)

add_library(flagdnn_validation_ascend_tensor_io STATIC tensor_io.cpp)
target_compile_features(flagdnn_validation_ascend_tensor_io PUBLIC cxx_std_20)
target_link_libraries(flagdnn_validation_ascend_tensor_io PUBLIC
  FlagDNN::flagdnn flagdnn_validation_ascend_platform)
flagdnn_enable_warnings(flagdnn_validation_ascend_tensor_io)

add_library(flagdnn_validation_ascend_aclnn INTERFACE)
target_link_libraries(flagdnn_validation_ascend_aclnn INTERFACE
  FlagDNNAscendValidation::nnopbase
  FlagDNNAscendValidation::opapi_math)

add_library(flagdnn_validation_ascend_aclnn_nn INTERFACE)
target_link_libraries(flagdnn_validation_ascend_aclnn_nn INTERFACE
  flagdnn_validation_ascend_aclnn
  FlagDNNAscendValidation::opapi_nn)

get_filename_component(_flagdnn_validation_library_directory
  "${_flagdnn_validation_opapi_math}" DIRECTORY)

if(FLAGDNN_BUILD_TESTS)
  include("${CMAKE_CURRENT_SOURCE_DIR}/functional/Targets.cmake")
endif()

if(FLAGDNN_BUILD_BENCHMARKS)
  include("${CMAKE_CURRENT_SOURCE_DIR}/benchmark/Targets.cmake")
endif()
