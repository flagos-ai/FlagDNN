# Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0

find_package(Threads REQUIRED)
add_library(flagdnn_test_ascend_layernorm_adapter STATIC
  $<TARGET_OBJECTS:flagdnn_test_common_objects>
  layernorm_validation.cpp
  functional/aclnn_layernorm.cpp
  functional/aclnn_layernorm_plan.cpp
  functional/layernorm_runner.cpp)
target_compile_features(
  flagdnn_test_ascend_layernorm_adapter PUBLIC cxx_std_20)
target_include_directories(
  flagdnn_test_ascend_layernorm_adapter PUBLIC
  "${PROJECT_SOURCE_DIR}/tests"
  "${PROJECT_SOURCE_DIR}/benchmark"
  "${PROJECT_SOURCE_DIR}/backends/ascend")
target_compile_definitions(
  flagdnn_test_ascend_layernorm_adapter PRIVATE
  FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
  FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}")
target_link_libraries(
  flagdnn_test_ascend_layernorm_adapter PUBLIC
  FlagDNN::flagdnn
  flagdnn_validation_ascend_platform
  flagdnn_validation_ascend_tensor_io
  flagdnn_validation_ascend_aclnn_nn)
flagdnn_enable_warnings(flagdnn_test_ascend_layernorm_adapter)

flagdnn_register_functional_suite(
  PLATFORM ascend
  ADAPTER_TARGET flagdnn_test_ascend_layernorm_adapter
  BACKEND_TARGET flagdnn_backend_ascend
  OPERATORS layernorm
  COMMAND_ARGS
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${FLAGDNN_CODEGEN_COMPILER}"
  ENVIRONMENT
    "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
  LABELS "aclnn;exact_reference;device;normalization"
  TIMEOUT 1200
  BUILD_RPATH "${_flagdnn_validation_library_directory}")

add_library(flagdnn_test_ascend_rmsnorm_adapter STATIC
  $<TARGET_OBJECTS:flagdnn_test_common_objects>
  rmsnorm_validation.cpp
  functional/aclnn_rmsnorm.cpp
  functional/aclnn_rmsnorm_plan.cpp
  functional/rmsnorm_runner.cpp)
target_compile_features(
  flagdnn_test_ascend_rmsnorm_adapter PUBLIC cxx_std_20)
target_include_directories(
  flagdnn_test_ascend_rmsnorm_adapter PUBLIC
  "${PROJECT_SOURCE_DIR}/tests"
  "${PROJECT_SOURCE_DIR}/benchmark"
  "${PROJECT_SOURCE_DIR}/backends/ascend")
target_compile_definitions(
  flagdnn_test_ascend_rmsnorm_adapter PRIVATE
  FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
  FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}")
target_link_libraries(
  flagdnn_test_ascend_rmsnorm_adapter PUBLIC
  FlagDNN::flagdnn
  flagdnn_validation_ascend_platform
  flagdnn_validation_ascend_tensor_io
  flagdnn_validation_ascend_aclnn_nn)
flagdnn_enable_warnings(flagdnn_test_ascend_rmsnorm_adapter)

flagdnn_register_functional_suite(
  PLATFORM ascend
  ADAPTER_TARGET flagdnn_test_ascend_rmsnorm_adapter
  BACKEND_TARGET flagdnn_backend_ascend
  OPERATORS rmsnorm
  COMMAND_ARGS
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${FLAGDNN_CODEGEN_COMPILER}"
  ENVIRONMENT
    "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
  LABELS "aclnn;exact_reference;device;normalization"
  TIMEOUT 1200
  BUILD_RPATH "${_flagdnn_validation_library_directory}")

add_library(flagdnn_test_ascend_batchnorm_inference_adapter STATIC
  "${PROJECT_SOURCE_DIR}/tests/common/normalization.cpp"
  batchnorm_inference_validation.cpp
  functional/aclnn_batchnorm_inference.cpp
  functional/normalization_runner.cpp)
target_compile_features(
  flagdnn_test_ascend_batchnorm_inference_adapter PUBLIC cxx_std_20)
target_include_directories(
  flagdnn_test_ascend_batchnorm_inference_adapter PUBLIC
  "${PROJECT_SOURCE_DIR}/tests"
  "${PROJECT_SOURCE_DIR}/benchmark"
  "${PROJECT_SOURCE_DIR}/backends/ascend")
target_compile_definitions(
  flagdnn_test_ascend_batchnorm_inference_adapter PRIVATE
  FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
  FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}")
target_link_libraries(
  flagdnn_test_ascend_batchnorm_inference_adapter PUBLIC
  FlagDNN::flagdnn
  flagdnn_validation_ascend_platform
  flagdnn_validation_ascend_tensor_io
  flagdnn_validation_ascend_aclnn_nn)
flagdnn_enable_warnings(
  flagdnn_test_ascend_batchnorm_inference_adapter)

flagdnn_register_functional_suite(
  PLATFORM ascend
  ADAPTER_TARGET flagdnn_test_ascend_batchnorm_inference_adapter
  BACKEND_TARGET flagdnn_backend_ascend
  OPERATORS batchnorm_inference
  COMMAND_ARGS
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${FLAGDNN_CODEGEN_COMPILER}"
  ENVIRONMENT
    "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
  LABELS "aclnn;exact_reference;device;normalization"
  TIMEOUT 1200
  BUILD_RPATH "${_flagdnn_validation_library_directory}")

add_library(flagdnn_test_ascend_batchnorm_adapter STATIC
  $<TARGET_OBJECTS:flagdnn_test_common_objects>
  batchnorm_validation.cpp
  functional/aclnn_batchnorm.cpp
  functional/batchnorm_runner.cpp)
target_compile_features(
  flagdnn_test_ascend_batchnorm_adapter PUBLIC cxx_std_20)
target_include_directories(
  flagdnn_test_ascend_batchnorm_adapter PUBLIC
  "${PROJECT_SOURCE_DIR}/tests"
  "${PROJECT_SOURCE_DIR}/benchmark"
  "${PROJECT_SOURCE_DIR}/backends/ascend")
target_compile_definitions(
  flagdnn_test_ascend_batchnorm_adapter PRIVATE
  FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
  FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}")
target_link_libraries(
  flagdnn_test_ascend_batchnorm_adapter PUBLIC
  FlagDNN::flagdnn
  flagdnn_validation_ascend_platform
  flagdnn_validation_ascend_tensor_io
  flagdnn_validation_ascend_aclnn_nn)
flagdnn_enable_warnings(flagdnn_test_ascend_batchnorm_adapter)

flagdnn_register_functional_suite(
  PLATFORM ascend
  ADAPTER_TARGET flagdnn_test_ascend_batchnorm_adapter
  BACKEND_TARGET flagdnn_backend_ascend
  OPERATORS batchnorm
  COMMAND_ARGS
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${FLAGDNN_CODEGEN_COMPILER}"
  ENVIRONMENT
    "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
  LABELS "aclnn;exact_reference;device;normalization"
  TIMEOUT 1200
  BUILD_RPATH "${_flagdnn_validation_library_directory}")

add_library(flagdnn_test_ascend_reduction_adapter STATIC
  $<TARGET_OBJECTS:flagdnn_test_common_objects>
  functional/aclnn_reduction.cpp
  functional/aclnn_reduction_plan.cpp
  functional/reduction_runner.cpp
  reduction_validation.cpp)
target_compile_features(flagdnn_test_ascend_reduction_adapter PUBLIC
  cxx_std_20)
target_include_directories(flagdnn_test_ascend_reduction_adapter PUBLIC
  "${PROJECT_SOURCE_DIR}/tests"
  "${PROJECT_SOURCE_DIR}/backends/ascend")
target_compile_definitions(flagdnn_test_ascend_reduction_adapter PRIVATE
  FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
  FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}")
target_link_libraries(flagdnn_test_ascend_reduction_adapter PUBLIC
  FlagDNN::flagdnn
  flagdnn_validation_ascend_platform
  flagdnn_validation_ascend_tensor_io
  flagdnn_validation_ascend_aclnn)
flagdnn_enable_warnings(flagdnn_test_ascend_reduction_adapter)

add_library(flagdnn_test_ascend_convolution_fprop_adapter STATIC
  $<TARGET_OBJECTS:flagdnn_test_common_objects>
  functional/aclnn_convolution.cpp
  functional/aclnn_convolution_plan.cpp
  functional/convolution_fprop_runner.cpp)
target_compile_features(
  flagdnn_test_ascend_convolution_fprop_adapter PUBLIC cxx_std_20)
target_include_directories(
  flagdnn_test_ascend_convolution_fprop_adapter PUBLIC
  "${PROJECT_SOURCE_DIR}/tests"
  "${PROJECT_SOURCE_DIR}/backends/ascend")
target_compile_definitions(
  flagdnn_test_ascend_convolution_fprop_adapter PRIVATE
  FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
  FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}")
target_link_libraries(
  flagdnn_test_ascend_convolution_fprop_adapter PUBLIC
  FlagDNN::flagdnn
  flagdnn_validation_ascend_platform
  flagdnn_validation_ascend_tensor_io
  flagdnn_validation_ascend_aclnn_nn)
flagdnn_enable_warnings(
  flagdnn_test_ascend_convolution_fprop_adapter)

flagdnn_register_functional_suite(
  PLATFORM ascend
  ADAPTER_TARGET flagdnn_test_ascend_convolution_fprop_adapter
  BACKEND_TARGET flagdnn_backend_ascend
  OPERATORS conv_fprop
  COMMAND_ARGS
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${FLAGDNN_CODEGEN_COMPILER}"
  ENVIRONMENT
    "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
  LABELS "aclnn;exact_reference;device;convolution_fprop"
  TIMEOUT 2400
  BUILD_RPATH "${_flagdnn_validation_library_directory}")

add_library(flagdnn_test_ascend_matmul_adapter STATIC
  $<TARGET_OBJECTS:flagdnn_test_common_objects>
  functional/aclnn_matmul.cpp
  functional/aclnn_matmul_plan.cpp
  functional/matmul_runner.cpp)
target_compile_features(flagdnn_test_ascend_matmul_adapter PUBLIC
  cxx_std_20)
target_include_directories(flagdnn_test_ascend_matmul_adapter PUBLIC
  "${PROJECT_SOURCE_DIR}/tests"
  "${PROJECT_SOURCE_DIR}/backends/ascend")
target_compile_definitions(flagdnn_test_ascend_matmul_adapter PRIVATE
  FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
  FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}")
target_link_libraries(flagdnn_test_ascend_matmul_adapter PUBLIC
  FlagDNN::flagdnn
  flagdnn_validation_ascend_platform
  flagdnn_validation_ascend_tensor_io
  flagdnn_validation_ascend_aclnn_nn)
flagdnn_enable_warnings(flagdnn_test_ascend_matmul_adapter)

flagdnn_register_functional_suite(
  PLATFORM ascend
  ADAPTER_TARGET flagdnn_test_ascend_matmul_adapter
  BACKEND_TARGET flagdnn_backend_ascend
  OPERATORS matmul
  COMMAND_ARGS
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${FLAGDNN_CODEGEN_COMPILER}"
  ENVIRONMENT
    "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
  LABELS "aclnn;exact_reference;device;matmul"
  TIMEOUT 1200
  BUILD_RPATH "${_flagdnn_validation_library_directory}")

add_executable(flagdnn_test_ascend_development_runtime
  functional/test_runtime.cpp)
target_compile_features(flagdnn_test_ascend_development_runtime PRIVATE
  cxx_std_20)
target_include_directories(flagdnn_test_ascend_development_runtime PRIVATE
  "${PROJECT_SOURCE_DIR}/backends/ascend")
target_compile_definitions(flagdnn_test_ascend_development_runtime PRIVATE
  FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
  FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}")
target_link_libraries(flagdnn_test_ascend_development_runtime PRIVATE
  FlagDNN::flagdnn
  flagdnn_validation_ascend_platform
  Threads::Threads)
flagdnn_enable_warnings(flagdnn_test_ascend_development_runtime)

add_test(NAME core.ascend.development_runtime.resource_cache_snapshot_host
  COMMAND "${CMAKE_COMMAND}"
    "-DEXECUTABLE=$<TARGET_FILE:flagdnn_test_ascend_development_runtime>"
    "-DCOMPILER_EXECUTABLE=${FLAGDNN_CODEGEN_PYTHON}"
    "-DCOMPILER_ENTRY=${FLAGDNN_CODEGEN_COMPILER}"
    "-DBACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    "-DMODE=resource_cache_snapshot_host"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/RunDevelopmentRuntime.cmake")
set_tests_properties(
  core.ascend.development_runtime.resource_cache_snapshot_host PROPERTIES
  LABELS "core;ascend;no_device;resource;development"
  TIMEOUT 60)

foreach(mode IN ITEMS
    ownership concurrency repeat workspace
    environment_drift no_active_sync no_active_sync_default)
  add_test(NAME integration.ascend.development_runtime.${mode}
    COMMAND flagdnn_test_ascend_development_runtime
      "${mode}"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}")
  set_tests_properties(
    integration.ascend.development_runtime.${mode} PROPERTIES
    LABELS "integration;ascend;runtime;device;development"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    RUN_SERIAL TRUE TIMEOUT 600)
endforeach()

add_test(NAME integration.ascend.development_runtime.cache
  COMMAND "${CMAKE_COMMAND}"
    "-DEXECUTABLE=$<TARGET_FILE:flagdnn_test_ascend_development_runtime>"
    "-DCOMPILER_EXECUTABLE=${FLAGDNN_CODEGEN_PYTHON}"
    "-DCOMPILER_ENTRY=${FLAGDNN_CODEGEN_COMPILER}"
    "-DBACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/RunDevelopmentRuntime.cmake")
set_tests_properties(integration.ascend.development_runtime.cache PROPERTIES
  LABELS "integration;ascend;runtime;device;development;cache"
  RUN_SERIAL TRUE TIMEOUT 600)

foreach(mode IN ITEMS
    resource_budget resource_graph_lifecycle resource_dual_handle)
  add_test(NAME integration.ascend.development_runtime.${mode}
    COMMAND "${CMAKE_COMMAND}"
      "-DEXECUTABLE=$<TARGET_FILE:flagdnn_test_ascend_development_runtime>"
      "-DCOMPILER_EXECUTABLE=${FLAGDNN_CODEGEN_PYTHON}"
      "-DCOMPILER_ENTRY=${FLAGDNN_CODEGEN_COMPILER}"
      "-DBACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
      "-DMODE=${mode}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/RunDevelopmentRuntime.cmake")
  set_tests_properties(
    integration.ascend.development_runtime.${mode} PROPERTIES
    LABELS "integration;ascend;runtime;device;development;cache;resource"
    RUN_SERIAL TRUE TIMEOUT 1200)
endforeach()

add_library(flagdnn_test_ascend_adapter STATIC
  $<TARGET_OBJECTS:flagdnn_test_common_objects>
  aclnn_unary_runtime.cpp
  functional/aclnn_add.cpp
  functional/aclnn_binary_pointwise.cpp
  functional/aclnn_ternary_pointwise.cpp
  functional/aclnn_unary_pointwise.cpp
  functional/add_runner.cpp
  functional/pointwise_runner.cpp)
target_compile_features(flagdnn_test_ascend_adapter PUBLIC cxx_std_20)
target_include_directories(flagdnn_test_ascend_adapter PUBLIC
  "${PROJECT_SOURCE_DIR}/tests"
  "${PROJECT_SOURCE_DIR}/backends/ascend")
target_compile_definitions(flagdnn_test_ascend_adapter PRIVATE
  FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
  FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}")
target_link_libraries(flagdnn_test_ascend_adapter PUBLIC
  FlagDNN::flagdnn
  flagdnn_validation_ascend_platform
  flagdnn_validation_ascend_tensor_io
  flagdnn_validation_ascend_aclnn)
flagdnn_enable_warnings(flagdnn_test_ascend_adapter)

add_library(flagdnn_test_ascend_nn_adapter STATIC
  $<TARGET_OBJECTS:flagdnn_test_common_objects>
  aclnn_unary_runtime.cpp
  functional/aclnn_add.cpp
  functional/aclnn_binary_pointwise.cpp
  functional/aclnn_ternary_pointwise.cpp
  functional/aclnn_unary_pointwise.cpp
  functional/add_runner.cpp
  functional/pointwise_runner.cpp)
target_compile_features(flagdnn_test_ascend_nn_adapter PUBLIC cxx_std_20)
target_include_directories(flagdnn_test_ascend_nn_adapter PUBLIC
  "${PROJECT_SOURCE_DIR}/tests"
  "${PROJECT_SOURCE_DIR}/backends/ascend")
target_compile_definitions(flagdnn_test_ascend_nn_adapter PRIVATE
  FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN=1
  FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
  FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}")
target_link_libraries(flagdnn_test_ascend_nn_adapter PUBLIC
  FlagDNN::flagdnn
  flagdnn_validation_ascend_platform
  flagdnn_validation_ascend_tensor_io
  flagdnn_validation_ascend_aclnn_nn)
flagdnn_enable_warnings(flagdnn_test_ascend_nn_adapter)

flagdnn_register_functional_suite(
  PLATFORM ascend
  ADAPTER_TARGET flagdnn_test_ascend_adapter
  BACKEND_TARGET flagdnn_backend_ascend
  OPERATORS
    add sub mul scale div mod pow min max
    binary_select
    cmp_eq cmp_neq cmp_gt cmp_ge cmp_lt cmp_le
    logical_not logical_and logical_or
    identity neg abs relu leaky_relu ceil floor reciprocal sqrt rsqrt
    exp log tanh sin cos tan erf
  COMMAND_ARGS
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${FLAGDNN_CODEGEN_COMPILER}"
  ENVIRONMENT
    "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
  LABELS "aclnn;exact_reference;device"
  TIMEOUT 1200
  BUILD_RPATH "${_flagdnn_validation_library_directory}")

flagdnn_register_functional_suite(
  PLATFORM ascend
  ADAPTER_TARGET flagdnn_test_ascend_reduction_adapter
  BACKEND_TARGET flagdnn_backend_ascend
  OPERATORS reduction
  COMMAND_ARGS
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${FLAGDNN_CODEGEN_COMPILER}"
  ENVIRONMENT
    "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
  LABELS "aclnn;exact_reference;device;reduction"
  TIMEOUT 1200
  BUILD_RPATH "${_flagdnn_validation_library_directory}")

# Layout references use CANN's precise math primitives and own a dedicated
# runner so pointwise symbol resolution and case-count contracts stay
# isolated.
add_library(flagdnn_test_ascend_layout_adapter STATIC
  $<TARGET_OBJECTS:flagdnn_test_common_objects>
  functional/aclnn_layout.cpp
  functional/layout_runner.cpp)
target_compile_features(
  flagdnn_test_ascend_layout_adapter PUBLIC cxx_std_20)
target_include_directories(flagdnn_test_ascend_layout_adapter PUBLIC
  "${PROJECT_SOURCE_DIR}/tests"
  "${PROJECT_SOURCE_DIR}/backends/ascend")
target_compile_definitions(flagdnn_test_ascend_layout_adapter PRIVATE
  FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
  FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}")
target_link_libraries(flagdnn_test_ascend_layout_adapter PUBLIC
  FlagDNN::flagdnn
  flagdnn_validation_ascend_platform
  flagdnn_validation_ascend_tensor_io
  flagdnn_validation_ascend_aclnn)
flagdnn_enable_warnings(flagdnn_test_ascend_layout_adapter)

flagdnn_register_functional_suite(
  PLATFORM ascend
  ADAPTER_TARGET flagdnn_test_ascend_layout_adapter
  BACKEND_TARGET flagdnn_backend_ascend
  OPERATORS reshape transpose slice
  COMMAND_ARGS
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${FLAGDNN_CODEGEN_COMPILER}"
  ENVIRONMENT
    "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
  LABELS "aclnn;exact_reference;device;layout"
  TIMEOUT 1200
  BUILD_RPATH "${_flagdnn_validation_library_directory}")

flagdnn_register_functional_suite(
  PLATFORM ascend
  ADAPTER_TARGET flagdnn_test_ascend_nn_adapter
  BACKEND_TARGET flagdnn_backend_ascend
  OPERATORS sigmoid sigmoid_backward elu gelu softplus swish gelu_approx_tanh
  COMMAND_ARGS
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${FLAGDNN_CODEGEN_COMPILER}"
  ENVIRONMENT
    "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
  LABELS "aclnn;exact_reference;device"
  TIMEOUT 1200
  BUILD_RPATH "${_flagdnn_validation_library_directory}")

add_library(flagdnn_test_ascend_conv_bias_relu_adapter STATIC
  functional/aclnn_conv_bias_relu.cpp
  functional/aclnn_convolution.cpp
  functional/aclnn_convolution_plan.cpp
  functional/conv_bias_relu_runner.cpp)
target_compile_features(
  flagdnn_test_ascend_conv_bias_relu_adapter PUBLIC cxx_std_20)
target_include_directories(
  flagdnn_test_ascend_conv_bias_relu_adapter PUBLIC
  "${PROJECT_SOURCE_DIR}/tests"
  "${PROJECT_SOURCE_DIR}/backends/ascend")
target_compile_definitions(
  flagdnn_test_ascend_conv_bias_relu_adapter PRIVATE
  FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
  FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}")
target_link_libraries(
  flagdnn_test_ascend_conv_bias_relu_adapter PUBLIC
  flagdnn_test_ascend_nn_adapter)
flagdnn_enable_warnings(flagdnn_test_ascend_conv_bias_relu_adapter)

flagdnn_register_functional_suite(
  PLATFORM ascend
  ADAPTER_TARGET flagdnn_test_ascend_conv_bias_relu_adapter
  BACKEND_TARGET flagdnn_backend_ascend
  OPERATORS conv_bias_relu
  COMMAND_ARGS
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${FLAGDNN_CODEGEN_COMPILER}"
  ENVIRONMENT
    "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
  LABELS "aclnn;exact_reference;device;conv_bias_relu"
  TIMEOUT 2400
  BUILD_RPATH "${_flagdnn_validation_library_directory}")

# AddSquare uses the same Ascend binary kernels, but its public test entry
# resolves a composite-specific runner and an independent ACLNN Mul -> Add
# reference.  Keep it in a separate archive so the generic pointwise runner
# remains the only run_pointwise_functional_test implementation.
add_library(flagdnn_test_ascend_add_square_adapter STATIC
  functional/aclnn_add_square.cpp
  functional/add_square_runner.cpp)
target_compile_features(
  flagdnn_test_ascend_add_square_adapter PUBLIC cxx_std_20)
target_include_directories(flagdnn_test_ascend_add_square_adapter PUBLIC
  "${PROJECT_SOURCE_DIR}/tests"
  "${PROJECT_SOURCE_DIR}/backends/ascend")
target_compile_definitions(flagdnn_test_ascend_add_square_adapter PRIVATE
  FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
  FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}")
target_link_libraries(flagdnn_test_ascend_add_square_adapter PUBLIC
  flagdnn_test_ascend_adapter)
flagdnn_enable_warnings(flagdnn_test_ascend_add_square_adapter)

flagdnn_register_functional_suite(
  PLATFORM ascend
  ADAPTER_TARGET flagdnn_test_ascend_add_square_adapter
  BACKEND_TARGET flagdnn_backend_ascend
  OPERATORS add_square
  COMMAND_ARGS
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${FLAGDNN_CODEGEN_COMPILER}"
  ENVIRONMENT
    "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
  LABELS "aclnn;exact_reference;device;graph"
  TIMEOUT 1200
  BUILD_RPATH "${_flagdnn_validation_library_directory}")

add_test(NAME integration.ascend.dependency_boundary
  COMMAND "${CMAKE_COMMAND}"
    "-DCORE_LIBRARY=$<TARGET_FILE:flagdnn>"
    "-DBACKEND_LIBRARY=$<TARGET_FILE:flagdnn_backend_ascend>"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/VerifyNativeDependencies.cmake")
set_tests_properties(integration.ascend.dependency_boundary PROPERTIES
  LABELS "integration;ascend;dependency")

# All reference executables use one of two shared dependency profiles. Check
# each profile once instead of registering the same readelf assertion for
# every operator executable.
foreach(_flagdnn_validation_profile IN ITEMS math nn)
  if(_flagdnn_validation_profile STREQUAL "nn")
    set(_flagdnn_validation_reference_target flagdnn_test_ascend_gelu)
    set(_flagdnn_validation_requires_opapi_nn TRUE)
  else()
    set(_flagdnn_validation_reference_target flagdnn_test_ascend_add)
    set(_flagdnn_validation_requires_opapi_nn FALSE)
  endif()
  add_test(
    NAME integration.ascend.reference_dependency_boundary.${_flagdnn_validation_profile}
    COMMAND "${CMAKE_COMMAND}"
      "-DREFERENCE_EXECUTABLE=$<TARGET_FILE:${_flagdnn_validation_reference_target}>"
      "-DREFERENCE_REQUIRES_OPAPI_NN=${_flagdnn_validation_requires_opapi_nn}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/VerifyReferenceDependencies.cmake")
  set_tests_properties(
    integration.ascend.reference_dependency_boundary.${_flagdnn_validation_profile}
    PROPERTIES LABELS "integration;ascend;aclnn;dependency")
endforeach()
