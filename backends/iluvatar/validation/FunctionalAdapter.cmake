# Iluvatar-only functional validation assembly. Nothing in this file is
# installed or linked into the production backend plugin.

include("${PROJECT_SOURCE_DIR}/cmake/Operators.cmake")

set(_flagdnn_iluvatar_add_operators add)
set(_flagdnn_iluvatar_composite_operators add_square conv_bias_relu)
set(_flagdnn_iluvatar_convolution_operators
  conv_dgrad conv_fprop conv_wgrad)
set(_flagdnn_iluvatar_layout_operators reshape slice transpose)
set(_flagdnn_iluvatar_normalization_operators
  batchnorm batchnorm_inference layernorm rmsnorm)
set(_flagdnn_iluvatar_tensor_operators matmul reduction)
set(_flagdnn_iluvatar_attention_operators
  sdpa sdpa_backward sdpa_fp8 sdpa_fp8_backward)
set(_flagdnn_iluvatar_pointwise_operators
  abs binary_select ceil cmp_eq cmp_ge cmp_gt cmp_le cmp_lt cmp_neq cos div
  elu erf exp floor gelu gelu_approx_tanh identity leaky_relu log
  logical_and logical_not logical_or max min mod mul neg pow reciprocal relu
  rsqrt scale sigmoid sigmoid_backward sin softplus sqrt sub swish tan tanh)
set(_flagdnn_iluvatar_partition
  ${_flagdnn_iluvatar_add_operators}
  ${_flagdnn_iluvatar_composite_operators}
  ${_flagdnn_iluvatar_convolution_operators}
  ${_flagdnn_iluvatar_layout_operators}
  ${_flagdnn_iluvatar_normalization_operators}
  ${_flagdnn_iluvatar_tensor_operators}
  ${_flagdnn_iluvatar_attention_operators}
  ${_flagdnn_iluvatar_pointwise_operators})
set(_flagdnn_iluvatar_unique_partition ${_flagdnn_iluvatar_partition})
list(REMOVE_DUPLICATES _flagdnn_iluvatar_unique_partition)
list(LENGTH _flagdnn_iluvatar_partition _flagdnn_iluvatar_partition_count)
list(LENGTH _flagdnn_iluvatar_unique_partition
  _flagdnn_iluvatar_unique_partition_count)
if(NOT _flagdnn_iluvatar_partition_count EQUAL
       _flagdnn_iluvatar_unique_partition_count)
  message(FATAL_ERROR "Iluvatar functional partition contains duplicates")
endif()
set(_flagdnn_iluvatar_expected ${FLAGDNN_FUNCTIONAL_OPERATORS})
list(SORT _flagdnn_iluvatar_unique_partition)
list(SORT _flagdnn_iluvatar_expected)
if(NOT "${_flagdnn_iluvatar_unique_partition}" STREQUAL
       "${_flagdnn_iluvatar_expected}")
  message(FATAL_ERROR
    "Iluvatar functional partition differs: got=[${_flagdnn_iluvatar_unique_partition}] expected=[${_flagdnn_iluvatar_expected}]")
endif()

add_library(flagdnn_test_iluvatar_adapter STATIC
  $<TARGET_OBJECTS:flagdnn_test_common_objects>
  convolution_reference.cpp
  pointwise_reference.cpp
  tensor_reference.cpp
  functional/add_runner.cpp
  functional/attention_runner.cpp
  functional/composite_runner.cpp
  functional/convolution_runner.cpp
  functional/corex_cudnn_add.cpp
  functional/corex_cudnn_attention.cpp
  functional/corex_cudnn_composite.cpp
  functional/corex_cudnn_convolution.cpp
  functional/corex_cudnn_layout.cpp
  functional/corex_cudnn_matmul.cpp
  normalization_reference.cpp
  functional/corex_cudnn_pointwise.cpp
  functional/corex_cudnn_reduction.cpp
  functional/layout_runner.cpp
  functional/matmul_runner.cpp
  functional/normalization_runner.cpp
  functional/pointwise_runner.cpp
  functional/reduction_runner.cpp
  functional/runner_support.cpp)
target_compile_features(flagdnn_test_iluvatar_adapter PUBLIC cxx_std_20)
target_compile_definitions(flagdnn_test_iluvatar_adapter PRIVATE
  FLAGDNN_ILUVATAR_CUDNN_CAPABILITY_CATALOG="${CMAKE_CURRENT_SOURCE_DIR}/corex_cudnn_capabilities.json")
target_include_directories(flagdnn_test_iluvatar_adapter PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/tests")
target_link_libraries(flagdnn_test_iluvatar_adapter PUBLIC
  FlagDNN::flagdnn
  flagdnn_validation_iluvatar_reference)
flagdnn_enable_warnings(flagdnn_test_iluvatar_adapter)

get_filename_component(_flagdnn_iluvatar_cudnn_library_directory
  "${FLAGDNN_ILUVATAR_RESOLVED_CUDNN_LIBRARY}" DIRECTORY)
flagdnn_register_functional_suite(
  PLATFORM iluvatar
  ADAPTER_TARGET flagdnn_test_iluvatar_adapter
  BACKEND_TARGET flagdnn_backend_iluvatar
  COMMAND_ARGS
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${FLAGDNN_CODEGEN_COMPILER}"
  ENVIRONMENT
    "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_iluvatar>;FLAGDNN_ILUVATAR_COREX_ROOT=${FLAGDNN_ILUVATAR_RESOLVED_COREX_ROOT};FLAGDNN_ILUVATAR_TRITON_JIT_DIR=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_CONFIG_DIR};FLAGDNN_ILUVATAR_TRITON_JIT_LIBRARY=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_LIBRARY};FLAGDNN_ILUVATAR_TRITON_JIT_INCLUDE_DIR=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_INCLUDE_DIR};FLAGDNN_ILUVATAR_TRITON_JIT_SCRIPT_DIR=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_SCRIPT_DIR}"
  LABELS "corex;cudnn"
  TIMEOUT 900
  BUILD_RPATH "${_flagdnn_iluvatar_cudnn_library_directory}")
foreach(_operator IN LISTS FLAGDNN_FUNCTIONAL_OPERATORS)
  set_tests_properties("functional.iluvatar.${_operator}" PROPERTIES
    SKIP_RETURN_CODE 77)
endforeach()
