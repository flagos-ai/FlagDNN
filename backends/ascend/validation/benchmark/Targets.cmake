# Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0

add_library(flagdnn_benchmark_ascend_layernorm_adapter STATIC
  layernorm_validation.cpp
  functional/aclnn_layernorm.cpp
  functional/aclnn_layernorm_plan.cpp
  benchmark/aclnn_layernorm_provider.cpp
  benchmark/layernorm_runner.cpp
  "${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp")
target_compile_features(
  flagdnn_benchmark_ascend_layernorm_adapter PUBLIC cxx_std_20)
target_include_directories(
  flagdnn_benchmark_ascend_layernorm_adapter PUBLIC
  "${PROJECT_SOURCE_DIR}/benchmark"
  "${PROJECT_SOURCE_DIR}/tests"
  "${PROJECT_SOURCE_DIR}/backends/ascend"
  "${PROJECT_SOURCE_DIR}/src")
target_compile_definitions(
  flagdnn_benchmark_ascend_layernorm_adapter PRIVATE
  FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
  FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}"
  FLAGDNN_ASCEND_VALIDATION_CANN_VERSION="${_flagdnn_validation_cann_version}"
  FLAGDNN_ASCEND_VALIDATION_ASCENDCL_BUILD_ID="${_flagdnn_validation_ascendcl_build_id}"
  FLAGDNN_ASCEND_VALIDATION_RUNTIME_BUILD_ID="${_flagdnn_validation_runtime_build_id}"
  FLAGDNN_ASCEND_VALIDATION_TRITON_JIT_SHA256="${_flagdnn_validation_triton_jit_sha256}"
  FLAGDNN_ASCEND_VALIDATION_NNOPBASE_SHA256="${_flagdnn_validation_nnopbase_sha256}"
  FLAGDNN_ASCEND_VALIDATION_OPAPI_MATH_SHA256="${_flagdnn_validation_opapi_math_sha256}"
  FLAGDNN_ASCEND_VALIDATION_OPAPI_NN_SHA256="${_flagdnn_validation_opapi_nn_sha256}")
target_link_libraries(
  flagdnn_benchmark_ascend_layernorm_adapter PUBLIC
  flagdnn_benchmark_cases
  FlagDNN::flagdnn
  flagdnn_validation_ascend_platform
  flagdnn_validation_ascend_tensor_io
  flagdnn_validation_ascend_aclnn_nn
  ${CMAKE_DL_LIBS})
flagdnn_enable_warnings(flagdnn_benchmark_ascend_layernorm_adapter)

flagdnn_register_benchmark_suite(
  PLATFORM ascend
  ADAPTER_TARGET flagdnn_benchmark_ascend_layernorm_adapter
  BACKEND_TARGET flagdnn_backend_ascend
  OPERATORS layernorm
  COMMAND_ARGS
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${FLAGDNN_CODEGEN_COMPILER}"
  ENVIRONMENT
    "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
  LABELS "aclnn;exact_reference;raw_event;device;normalization"
  TIMEOUT 3600
  BUILD_RPATH "${_flagdnn_validation_library_directory}")

add_library(flagdnn_benchmark_ascend_rmsnorm_adapter STATIC
  rmsnorm_validation.cpp
  functional/aclnn_rmsnorm.cpp
  functional/aclnn_rmsnorm_plan.cpp
  benchmark/aclnn_rmsnorm_provider.cpp
  benchmark/rmsnorm_runner.cpp
  "${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp")
target_compile_features(
  flagdnn_benchmark_ascend_rmsnorm_adapter PUBLIC cxx_std_20)
target_include_directories(
  flagdnn_benchmark_ascend_rmsnorm_adapter PUBLIC
  "${PROJECT_SOURCE_DIR}/benchmark"
  "${PROJECT_SOURCE_DIR}/tests"
  "${PROJECT_SOURCE_DIR}/backends/ascend"
  "${PROJECT_SOURCE_DIR}/src")
target_compile_definitions(
  flagdnn_benchmark_ascend_rmsnorm_adapter PRIVATE
  FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
  FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}"
  FLAGDNN_ASCEND_VALIDATION_CANN_VERSION="${_flagdnn_validation_cann_version}"
  FLAGDNN_ASCEND_VALIDATION_ASCENDCL_BUILD_ID="${_flagdnn_validation_ascendcl_build_id}"
  FLAGDNN_ASCEND_VALIDATION_RUNTIME_BUILD_ID="${_flagdnn_validation_runtime_build_id}"
  FLAGDNN_ASCEND_VALIDATION_TRITON_JIT_SHA256="${_flagdnn_validation_triton_jit_sha256}"
  FLAGDNN_ASCEND_VALIDATION_NNOPBASE_SHA256="${_flagdnn_validation_nnopbase_sha256}"
  FLAGDNN_ASCEND_VALIDATION_OPAPI_MATH_SHA256="${_flagdnn_validation_opapi_math_sha256}"
  FLAGDNN_ASCEND_VALIDATION_OPAPI_NN_SHA256="${_flagdnn_validation_opapi_nn_sha256}")
target_link_libraries(
  flagdnn_benchmark_ascend_rmsnorm_adapter PUBLIC
  flagdnn_benchmark_cases
  FlagDNN::flagdnn
  flagdnn_validation_ascend_platform
  flagdnn_validation_ascend_tensor_io
  flagdnn_validation_ascend_aclnn_nn
  ${CMAKE_DL_LIBS})
flagdnn_enable_warnings(flagdnn_benchmark_ascend_rmsnorm_adapter)

flagdnn_register_benchmark_suite(
  PLATFORM ascend
  ADAPTER_TARGET flagdnn_benchmark_ascend_rmsnorm_adapter
  BACKEND_TARGET flagdnn_backend_ascend
  OPERATORS rmsnorm
  COMMAND_ARGS
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${FLAGDNN_CODEGEN_COMPILER}"
  ENVIRONMENT
    "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
  LABELS "aclnn;exact_reference;raw_event;device;normalization"
  TIMEOUT 3600
  BUILD_RPATH "${_flagdnn_validation_library_directory}")

add_library(flagdnn_benchmark_ascend_batchnorm_inference_adapter STATIC
  batchnorm_inference_validation.cpp
  functional/aclnn_batchnorm_inference.cpp
  benchmark/aclnn_batchnorm_inference_provider.cpp
  benchmark/batchnorm_inference_runner.cpp
  "${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp")
target_compile_features(
  flagdnn_benchmark_ascend_batchnorm_inference_adapter PUBLIC cxx_std_20)
target_include_directories(
  flagdnn_benchmark_ascend_batchnorm_inference_adapter PUBLIC
  "${PROJECT_SOURCE_DIR}/benchmark"
  "${PROJECT_SOURCE_DIR}/tests"
  "${PROJECT_SOURCE_DIR}/backends/ascend"
  "${PROJECT_SOURCE_DIR}/src")
target_compile_definitions(
  flagdnn_benchmark_ascend_batchnorm_inference_adapter PRIVATE
  FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
  FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}"
  FLAGDNN_ASCEND_VALIDATION_CANN_VERSION="${_flagdnn_validation_cann_version}"
  FLAGDNN_ASCEND_VALIDATION_ASCENDCL_BUILD_ID="${_flagdnn_validation_ascendcl_build_id}"
  FLAGDNN_ASCEND_VALIDATION_RUNTIME_BUILD_ID="${_flagdnn_validation_runtime_build_id}"
  FLAGDNN_ASCEND_VALIDATION_TRITON_JIT_SHA256="${_flagdnn_validation_triton_jit_sha256}"
  FLAGDNN_ASCEND_VALIDATION_NNOPBASE_SHA256="${_flagdnn_validation_nnopbase_sha256}"
  FLAGDNN_ASCEND_VALIDATION_OPAPI_MATH_SHA256="${_flagdnn_validation_opapi_math_sha256}"
  FLAGDNN_ASCEND_VALIDATION_OPAPI_NN_SHA256="${_flagdnn_validation_opapi_nn_sha256}")
target_link_libraries(
  flagdnn_benchmark_ascend_batchnorm_inference_adapter PUBLIC
  flagdnn_benchmark_cases
  FlagDNN::flagdnn
  flagdnn_validation_ascend_platform
  flagdnn_validation_ascend_tensor_io
  flagdnn_validation_ascend_aclnn_nn
  ${CMAKE_DL_LIBS})
flagdnn_enable_warnings(
  flagdnn_benchmark_ascend_batchnorm_inference_adapter)

flagdnn_register_benchmark_suite(
  PLATFORM ascend
  ADAPTER_TARGET flagdnn_benchmark_ascend_batchnorm_inference_adapter
  BACKEND_TARGET flagdnn_backend_ascend
  OPERATORS batchnorm_inference
  COMMAND_ARGS
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${FLAGDNN_CODEGEN_COMPILER}"
  ENVIRONMENT
    "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
  LABELS "aclnn;exact_reference;raw_event;device;normalization"
  TIMEOUT 3600
  BUILD_RPATH "${_flagdnn_validation_library_directory}")

add_library(flagdnn_benchmark_ascend_batchnorm_adapter STATIC
  batchnorm_validation.cpp
  functional/aclnn_batchnorm.cpp
  benchmark/aclnn_batchnorm_provider.cpp
  benchmark/batchnorm_runner.cpp
  "${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp")
target_compile_features(
  flagdnn_benchmark_ascend_batchnorm_adapter PUBLIC cxx_std_20)
target_include_directories(
  flagdnn_benchmark_ascend_batchnorm_adapter PUBLIC
  "${PROJECT_SOURCE_DIR}/benchmark"
  "${PROJECT_SOURCE_DIR}/tests"
  "${PROJECT_SOURCE_DIR}/backends/ascend"
  "${PROJECT_SOURCE_DIR}/src")
target_compile_definitions(
  flagdnn_benchmark_ascend_batchnorm_adapter PRIVATE
  FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
  FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}"
  FLAGDNN_ASCEND_VALIDATION_CANN_VERSION="${_flagdnn_validation_cann_version}"
  FLAGDNN_ASCEND_VALIDATION_ASCENDCL_BUILD_ID="${_flagdnn_validation_ascendcl_build_id}"
  FLAGDNN_ASCEND_VALIDATION_RUNTIME_BUILD_ID="${_flagdnn_validation_runtime_build_id}"
  FLAGDNN_ASCEND_VALIDATION_TRITON_JIT_SHA256="${_flagdnn_validation_triton_jit_sha256}"
  FLAGDNN_ASCEND_VALIDATION_NNOPBASE_SHA256="${_flagdnn_validation_nnopbase_sha256}"
  FLAGDNN_ASCEND_VALIDATION_OPAPI_MATH_SHA256="${_flagdnn_validation_opapi_math_sha256}"
  FLAGDNN_ASCEND_VALIDATION_OPAPI_NN_SHA256="${_flagdnn_validation_opapi_nn_sha256}")
target_link_libraries(
  flagdnn_benchmark_ascend_batchnorm_adapter PUBLIC
  flagdnn_benchmark_cases
  FlagDNN::flagdnn
  flagdnn_validation_ascend_platform
  flagdnn_validation_ascend_tensor_io
  flagdnn_validation_ascend_aclnn_nn
  ${CMAKE_DL_LIBS})
flagdnn_enable_warnings(flagdnn_benchmark_ascend_batchnorm_adapter)

flagdnn_register_benchmark_suite(
  PLATFORM ascend
  ADAPTER_TARGET flagdnn_benchmark_ascend_batchnorm_adapter
  BACKEND_TARGET flagdnn_backend_ascend
  OPERATORS batchnorm
  COMMAND_ARGS
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${FLAGDNN_CODEGEN_COMPILER}"
  ENVIRONMENT
    "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
  LABELS "aclnn;exact_reference;raw_event;device;normalization"
  TIMEOUT 3600
  BUILD_RPATH "${_flagdnn_validation_library_directory}")

add_library(flagdnn_benchmark_ascend_adapter STATIC
  aclnn_unary_runtime.cpp
  "${PROJECT_SOURCE_DIR}/tests/common/reduction.cpp"
  functional/aclnn_reduction.cpp
  functional/aclnn_reduction_plan.cpp
  reduction_validation.cpp
  benchmark/aclnn_provider.cpp
  benchmark/aclnn_unary_provider.cpp
  benchmark/aclnn_reduction_plan.cpp
  benchmark/aclnn_reduction_provider.cpp
  benchmark/runner.cpp
  "${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp")
target_compile_features(flagdnn_benchmark_ascend_adapter PUBLIC cxx_std_20)
target_include_directories(flagdnn_benchmark_ascend_adapter PUBLIC
  "${PROJECT_SOURCE_DIR}/benchmark"
  "${PROJECT_SOURCE_DIR}/tests"
  "${PROJECT_SOURCE_DIR}/backends/ascend"
  "${PROJECT_SOURCE_DIR}/src")
target_compile_definitions(flagdnn_benchmark_ascend_adapter PRIVATE
  FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
  FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}"
  FLAGDNN_ASCEND_VALIDATION_CANN_VERSION="${_flagdnn_validation_cann_version}"
  FLAGDNN_ASCEND_VALIDATION_ASCENDCL_BUILD_ID="${_flagdnn_validation_ascendcl_build_id}"
  FLAGDNN_ASCEND_VALIDATION_RUNTIME_BUILD_ID="${_flagdnn_validation_runtime_build_id}"
  FLAGDNN_ASCEND_VALIDATION_TRITON_JIT_SHA256="${_flagdnn_validation_triton_jit_sha256}"
  FLAGDNN_ASCEND_VALIDATION_NNOPBASE_SHA256="${_flagdnn_validation_nnopbase_sha256}"
  FLAGDNN_ASCEND_VALIDATION_OPAPI_MATH_SHA256="${_flagdnn_validation_opapi_math_sha256}")
target_link_libraries(flagdnn_benchmark_ascend_adapter PUBLIC
  flagdnn_benchmark_cases
  FlagDNN::flagdnn
  flagdnn_validation_ascend_platform
  flagdnn_validation_ascend_tensor_io
  flagdnn_validation_ascend_aclnn
  ${CMAKE_DL_LIBS})
flagdnn_enable_warnings(flagdnn_benchmark_ascend_adapter)

add_library(flagdnn_benchmark_ascend_nn_adapter STATIC
  aclnn_unary_runtime.cpp
  "${PROJECT_SOURCE_DIR}/tests/common/convolution.cpp"
  "${PROJECT_SOURCE_DIR}/tests/common/matmul.cpp"
  "${PROJECT_SOURCE_DIR}/tests/common/reduction.cpp"
  functional/aclnn_convolution.cpp
  functional/aclnn_convolution_plan.cpp
  functional/aclnn_matmul.cpp
  functional/aclnn_matmul_plan.cpp
  functional/aclnn_reduction.cpp
  functional/aclnn_reduction_plan.cpp
  reduction_validation.cpp
  benchmark/aclnn_convolution_provider.cpp
  benchmark/aclnn_provider.cpp
  benchmark/aclnn_matmul_provider.cpp
  benchmark/aclnn_unary_provider.cpp
  benchmark/aclnn_reduction_plan.cpp
  benchmark/aclnn_reduction_provider.cpp
  benchmark/runner.cpp
  "${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp")
target_compile_features(
  flagdnn_benchmark_ascend_nn_adapter PUBLIC cxx_std_20)
target_include_directories(flagdnn_benchmark_ascend_nn_adapter PUBLIC
  "${PROJECT_SOURCE_DIR}/benchmark"
  "${PROJECT_SOURCE_DIR}/tests"
  "${PROJECT_SOURCE_DIR}/backends/ascend"
  "${PROJECT_SOURCE_DIR}/src")
target_compile_definitions(flagdnn_benchmark_ascend_nn_adapter PRIVATE
  FLAGDNN_ASCEND_VALIDATION_ENABLE_CONVOLUTION_FPROP=1
  FLAGDNN_ASCEND_VALIDATION_ENABLE_MATMUL=1
  FLAGDNN_ASCEND_VALIDATION_ENABLE_OPAPI_NN=1
  FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
  FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}"
  FLAGDNN_ASCEND_VALIDATION_CANN_VERSION="${_flagdnn_validation_cann_version}"
  FLAGDNN_ASCEND_VALIDATION_ASCENDCL_BUILD_ID="${_flagdnn_validation_ascendcl_build_id}"
  FLAGDNN_ASCEND_VALIDATION_RUNTIME_BUILD_ID="${_flagdnn_validation_runtime_build_id}"
  FLAGDNN_ASCEND_VALIDATION_TRITON_JIT_SHA256="${_flagdnn_validation_triton_jit_sha256}"
  FLAGDNN_ASCEND_VALIDATION_NNOPBASE_SHA256="${_flagdnn_validation_nnopbase_sha256}"
  FLAGDNN_ASCEND_VALIDATION_OPAPI_MATH_SHA256="${_flagdnn_validation_opapi_math_sha256}"
  FLAGDNN_ASCEND_VALIDATION_OPAPI_NN_SHA256="${_flagdnn_validation_opapi_nn_sha256}")
target_link_libraries(flagdnn_benchmark_ascend_nn_adapter PUBLIC
  flagdnn_benchmark_cases
  FlagDNN::flagdnn
  flagdnn_validation_ascend_platform
  flagdnn_validation_ascend_tensor_io
  flagdnn_validation_ascend_aclnn_nn
  ${CMAKE_DL_LIBS})
flagdnn_enable_warnings(flagdnn_benchmark_ascend_nn_adapter)

add_library(flagdnn_benchmark_ascend_reduction_adapter STATIC
  aclnn_unary_runtime.cpp
  "${PROJECT_SOURCE_DIR}/tests/common/reduction.cpp"
  functional/aclnn_reduction.cpp
  functional/aclnn_reduction_plan.cpp
  reduction_validation.cpp
  benchmark/aclnn_provider.cpp
  benchmark/aclnn_unary_provider.cpp
  benchmark/aclnn_reduction_plan.cpp
  benchmark/aclnn_reduction_provider.cpp
  benchmark/runner.cpp
  "${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp")
target_compile_features(flagdnn_benchmark_ascend_reduction_adapter PUBLIC
  cxx_std_20)
target_include_directories(flagdnn_benchmark_ascend_reduction_adapter PUBLIC
  "${PROJECT_SOURCE_DIR}/benchmark"
  "${PROJECT_SOURCE_DIR}/tests"
  "${PROJECT_SOURCE_DIR}/backends/ascend"
  "${PROJECT_SOURCE_DIR}/src")
target_compile_definitions(flagdnn_benchmark_ascend_reduction_adapter PRIVATE
  FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
  FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}"
  FLAGDNN_ASCEND_VALIDATION_CANN_VERSION="${_flagdnn_validation_cann_version}"
  FLAGDNN_ASCEND_VALIDATION_ASCENDCL_BUILD_ID="${_flagdnn_validation_ascendcl_build_id}"
  FLAGDNN_ASCEND_VALIDATION_RUNTIME_BUILD_ID="${_flagdnn_validation_runtime_build_id}"
  FLAGDNN_ASCEND_VALIDATION_TRITON_JIT_SHA256="${_flagdnn_validation_triton_jit_sha256}"
  FLAGDNN_ASCEND_VALIDATION_NNOPBASE_SHA256="${_flagdnn_validation_nnopbase_sha256}"
  FLAGDNN_ASCEND_VALIDATION_OPAPI_MATH_SHA256="${_flagdnn_validation_opapi_math_sha256}")
target_link_libraries(flagdnn_benchmark_ascend_reduction_adapter PUBLIC
  flagdnn_benchmark_cases
  FlagDNN::flagdnn
  flagdnn_validation_ascend_platform
  flagdnn_validation_ascend_tensor_io
  flagdnn_validation_ascend_aclnn
  ${CMAKE_DL_LIBS})
flagdnn_enable_warnings(flagdnn_benchmark_ascend_reduction_adapter)

flagdnn_register_benchmark_suite(
  PLATFORM ascend
  ADAPTER_TARGET flagdnn_benchmark_ascend_adapter
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
  LABELS "aclnn;exact_reference;raw_event;device"
  TIMEOUT 3600
  BUILD_RPATH "${_flagdnn_validation_library_directory}")

flagdnn_register_benchmark_suite(
  PLATFORM ascend
  ADAPTER_TARGET flagdnn_benchmark_ascend_reduction_adapter
  BACKEND_TARGET flagdnn_backend_ascend
  OPERATORS reduction
  COMMAND_ARGS
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${FLAGDNN_CODEGEN_COMPILER}"
  ENVIRONMENT
    "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
  LABELS "aclnn;exact_reference;raw_event;device;reduction"
  TIMEOUT 3600
  BUILD_RPATH "${_flagdnn_validation_library_directory}")

# Layout benchmarks compare directly with the transactional ACLNN layout
# executable without importing the pointwise runner entry point.
add_library(flagdnn_benchmark_ascend_layout_adapter STATIC
  "${PROJECT_SOURCE_DIR}/tests/common/layout.cpp"
  functional/aclnn_layout.cpp
  benchmark/aclnn_layout_provider.cpp
  benchmark/layout_runner.cpp
  "${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp")
target_compile_features(
  flagdnn_benchmark_ascend_layout_adapter PUBLIC cxx_std_20)
target_include_directories(
  flagdnn_benchmark_ascend_layout_adapter PUBLIC
  "${PROJECT_SOURCE_DIR}/benchmark"
  "${PROJECT_SOURCE_DIR}/tests"
  "${PROJECT_SOURCE_DIR}/backends/ascend"
  "${PROJECT_SOURCE_DIR}/src")
target_compile_definitions(
  flagdnn_benchmark_ascend_layout_adapter PRIVATE
  FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
  FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}"
  FLAGDNN_ASCEND_VALIDATION_CANN_VERSION="${_flagdnn_validation_cann_version}"
  FLAGDNN_ASCEND_VALIDATION_ASCENDCL_BUILD_ID="${_flagdnn_validation_ascendcl_build_id}"
  FLAGDNN_ASCEND_VALIDATION_RUNTIME_BUILD_ID="${_flagdnn_validation_runtime_build_id}"
  FLAGDNN_ASCEND_VALIDATION_TRITON_JIT_SHA256="${_flagdnn_validation_triton_jit_sha256}"
  FLAGDNN_ASCEND_VALIDATION_NNOPBASE_SHA256="${_flagdnn_validation_nnopbase_sha256}"
  FLAGDNN_ASCEND_VALIDATION_OPAPI_MATH_SHA256="${_flagdnn_validation_opapi_math_sha256}")
target_link_libraries(flagdnn_benchmark_ascend_layout_adapter PUBLIC
  flagdnn_benchmark_cases
  FlagDNN::flagdnn
  flagdnn_validation_ascend_platform
  flagdnn_validation_ascend_tensor_io
  flagdnn_validation_ascend_aclnn
  ${CMAKE_DL_LIBS})
flagdnn_enable_warnings(flagdnn_benchmark_ascend_layout_adapter)

flagdnn_register_benchmark_suite(
  PLATFORM ascend
  ADAPTER_TARGET flagdnn_benchmark_ascend_layout_adapter
  BACKEND_TARGET flagdnn_backend_ascend
  OPERATORS reshape transpose slice
  COMMAND_ARGS
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${FLAGDNN_CODEGEN_COMPILER}"
  ENVIRONMENT
    "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
  LABELS "aclnn;exact_reference;raw_event;device;layout"
  TIMEOUT 3600
  BUILD_RPATH "${_flagdnn_validation_library_directory}")

flagdnn_register_benchmark_suite(
  PLATFORM ascend
  ADAPTER_TARGET flagdnn_benchmark_ascend_nn_adapter
  BACKEND_TARGET flagdnn_backend_ascend
  OPERATORS sigmoid sigmoid_backward elu gelu softplus swish gelu_approx_tanh
  COMMAND_ARGS
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${FLAGDNN_CODEGEN_COMPILER}"
  ENVIRONMENT
    "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
  LABELS "aclnn;exact_reference;raw_event;device"
  TIMEOUT 3600
  BUILD_RPATH "${_flagdnn_validation_library_directory}")

flagdnn_register_benchmark_suite(
  PLATFORM ascend
  ADAPTER_TARGET flagdnn_benchmark_ascend_nn_adapter
  BACKEND_TARGET flagdnn_backend_ascend
  OPERATORS matmul
  COMMAND_ARGS
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${FLAGDNN_CODEGEN_COMPILER}"
  ENVIRONMENT
    "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
  LABELS "aclnn;exact_reference;raw_event;device;matmul"
  TIMEOUT 3600
  BUILD_RPATH "${_flagdnn_validation_library_directory}")

flagdnn_register_benchmark_suite(
  PLATFORM ascend
  ADAPTER_TARGET flagdnn_benchmark_ascend_nn_adapter
  BACKEND_TARGET flagdnn_backend_ascend
  OPERATORS conv_fprop
  COMMAND_ARGS
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${FLAGDNN_CODEGEN_COMPILER}"
  ENVIRONMENT
    "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
  LABELS "aclnn;exact_reference;raw_event;device;convolution_fprop"
  TIMEOUT 7200
  BUILD_RPATH "${_flagdnn_validation_library_directory}")

# The composite benchmark owns a distinct run_benchmark_suite entry point;
# isolate it from the binary pointwise adapter to keep symbol resolution
# deterministic while reusing the same platform/reference primitives.
add_library(flagdnn_benchmark_ascend_add_square_adapter STATIC
  benchmark/aclnn_add_square_provider.cpp
  benchmark/add_square_runner.cpp)
target_compile_features(
  flagdnn_benchmark_ascend_add_square_adapter PUBLIC cxx_std_20)
target_include_directories(
  flagdnn_benchmark_ascend_add_square_adapter PUBLIC
  "${PROJECT_SOURCE_DIR}/benchmark"
  "${PROJECT_SOURCE_DIR}/backends/ascend"
  "${PROJECT_SOURCE_DIR}/src")
target_compile_definitions(
  flagdnn_benchmark_ascend_add_square_adapter PRIVATE
  FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
  FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}"
  FLAGDNN_ASCEND_VALIDATION_CANN_VERSION="${_flagdnn_validation_cann_version}"
  FLAGDNN_ASCEND_VALIDATION_ASCENDCL_BUILD_ID="${_flagdnn_validation_ascendcl_build_id}"
  FLAGDNN_ASCEND_VALIDATION_RUNTIME_BUILD_ID="${_flagdnn_validation_runtime_build_id}"
  FLAGDNN_ASCEND_VALIDATION_TRITON_JIT_SHA256="${_flagdnn_validation_triton_jit_sha256}"
  FLAGDNN_ASCEND_VALIDATION_NNOPBASE_SHA256="${_flagdnn_validation_nnopbase_sha256}"
  FLAGDNN_ASCEND_VALIDATION_OPAPI_MATH_SHA256="${_flagdnn_validation_opapi_math_sha256}")
target_link_libraries(flagdnn_benchmark_ascend_add_square_adapter PUBLIC
  flagdnn_benchmark_ascend_adapter)
flagdnn_enable_warnings(flagdnn_benchmark_ascend_add_square_adapter)

flagdnn_register_benchmark_suite(
  PLATFORM ascend
  ADAPTER_TARGET flagdnn_benchmark_ascend_add_square_adapter
  BACKEND_TARGET flagdnn_backend_ascend
  OPERATORS add_square
  COMMAND_ARGS
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${FLAGDNN_CODEGEN_COMPILER}"
  ENVIRONMENT
    "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
  LABELS "aclnn;exact_reference;raw_event;device;graph"
  TIMEOUT 3600
  BUILD_RPATH "${_flagdnn_validation_library_directory}")
