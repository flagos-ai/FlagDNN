# NVIDIA-only validation assembly. Production backend targets never link this
# directory, and none of these targets are installed.

include("${PROJECT_SOURCE_DIR}/cmake/Operators.cmake")

if(NOT TARGET flagdnn_backend_nvidia)
  message(FATAL_ERROR
    "NVIDIA validation requires FLAGDNN_BACKENDS=nvidia")
endif()
if(NOT FLAGDNN_CODEGEN_PYTHON)
  message(FATAL_ERROR
    "NVIDIA validation requires FLAGDNN_CODEGEN_PYTHON")
endif()

list(APPEND CMAKE_MODULE_PATH
  "${PROJECT_SOURCE_DIR}/backends/nvidia/cmake")
find_package(CUDAToolkit REQUIRED)
find_package(CUDNN REQUIRED)

get_filename_component(_flagdnn_nvidia_validation_cudnn_directory
  "${CUDNN_LIBRARY}" DIRECTORY)

# CUDA context, stream, buffer, and event support is shared by functional and
# performance validation. Keeping it in one interface target prevents either
# top-level workload directory from becoming a platform SDK owner.
add_library(flagdnn_validation_nvidia_platform INTERFACE)
target_compile_features(flagdnn_validation_nvidia_platform INTERFACE cxx_std_20)
target_include_directories(flagdnn_validation_nvidia_platform INTERFACE
  ${CUDAToolkit_INCLUDE_DIRS}
  "${PROJECT_SOURCE_DIR}/backends/nvidia")
target_link_libraries(flagdnn_validation_nvidia_platform INTERFACE
  CUDA::cuda_driver
  CUDA::cudart)

add_library(flagdnn_validation_nvidia_tensor_io STATIC tensor_io.cpp)
target_compile_features(flagdnn_validation_nvidia_tensor_io PUBLIC cxx_std_20)
target_link_libraries(flagdnn_validation_nvidia_tensor_io PUBLIC
  FlagDNN::flagdnn
  flagdnn_validation_nvidia_platform)
flagdnn_enable_warnings(flagdnn_validation_nvidia_tensor_io)

if(FLAGDNN_BUILD_TESTS)
  add_library(flagdnn_test_nvidia_adapter STATIC
    $<TARGET_OBJECTS:flagdnn_test_common_objects>
    functional/add_runner.cpp
    functional/attention_runner.cpp
    functional/composite_runner.cpp
    functional/convolution_runner.cpp
    functional/cudnn_add.cpp
    functional/cudnn_attention.cpp
    functional/cudnn_composite.cpp
    functional/cudnn_convolution.cpp
    functional/cudnn_graph.cpp
    functional/cudnn_layout.cpp
    functional/cudnn_matmul.cpp
    functional/cudnn_normalization.cpp
    functional/cudnn_pointwise.cpp
    functional/cudnn_reduction.cpp
    functional/layout_runner.cpp
    functional/matmul_runner.cpp
    functional/normalization_runner.cpp
    functional/pointwise_runner.cpp
    functional/reduction_runner.cpp)
  target_compile_features(flagdnn_test_nvidia_adapter PUBLIC cxx_std_20)
  target_include_directories(flagdnn_test_nvidia_adapter PUBLIC
    "${PROJECT_SOURCE_DIR}/tests")
  target_link_libraries(flagdnn_test_nvidia_adapter PUBLIC
    FlagDNN::flagdnn
    flagdnn_validation_nvidia_platform
    flagdnn_validation_nvidia_tensor_io
    CUDNN::frontend
    CUDA::nvrtc)
  flagdnn_enable_warnings(flagdnn_test_nvidia_adapter)

  flagdnn_register_functional_suite(
    PLATFORM nvidia
    ADAPTER_TARGET flagdnn_test_nvidia_adapter
    BACKEND_TARGET flagdnn_backend_nvidia
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_nvidia>"
    LABELS cudnn
    TIMEOUT 600
    BUILD_RPATH "${_flagdnn_nvidia_validation_cudnn_directory}")

  # cuDNN cannot express every tested layout/logical-not shape. The primary
  # tests still execute FlagDNN on the GPU with a host oracle; these tests make
  # the unavailable vendor capability explicit.
  add_test(NAME capability.nvidia.cudnn.matmul
    COMMAND flagdnn_test_nvidia_matmul
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}")
  set_tests_properties(capability.nvidia.cudnn.matmul PROPERTIES
    LABELS "capability;nvidia;cudnn;matmul"
    RUN_SERIAL TRUE
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_nvidia>;FLAGDNN_MATMUL_MODE=capability;FLAGDNN_MATMUL_CASE=matmul_fp16_2x1x17x30"
    TIMEOUT 60)

  add_test(NAME functional.nvidia.matmul.host
    COMMAND flagdnn_test_nvidia_matmul
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}")
  set_tests_properties(functional.nvidia.matmul.host PROPERTIES
    LABELS "functional;correctness;nvidia;matmul;host_oracle"
    RUN_SERIAL TRUE
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_nvidia>;FLAGDNN_MATMUL_MODE=host"
    TIMEOUT 600)

  foreach(operator IN ITEMS reshape transpose slice)
    if(operator STREQUAL "slice")
      set(capability_case "slice_fp32_case0")
    else()
      set(capability_case "${operator}_fp32")
    endif()
    add_test(NAME "capability.nvidia.cudnn.${operator}"
      COMMAND "flagdnn_test_nvidia_${operator}"
        "${FLAGDNN_CODEGEN_PYTHON}"
        "${FLAGDNN_CODEGEN_COMPILER}")
    set_tests_properties("capability.nvidia.cudnn.${operator}" PROPERTIES
      LABELS "capability;nvidia;cudnn;layout;${operator}"
      RUN_SERIAL TRUE
      ENVIRONMENT
        "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_nvidia>;FLAGDNN_LAYOUT_EXPECT_REFERENCE_UNAVAILABLE=1;FLAGDNN_LAYOUT_CASE=${capability_case}"
      TIMEOUT 60)
  endforeach()

  add_test(NAME capability.nvidia.cudnn.logical_not
    COMMAND flagdnn_test_nvidia_logical_not
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}")
  set_tests_properties(capability.nvidia.cudnn.logical_not PROPERTIES
    LABELS "capability;nvidia;cudnn;pointwise;logical_not"
    RUN_SERIAL TRUE
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_nvidia>;FLAGDNN_POINTWISE_EXPECT_REFERENCE_UNAVAILABLE=1;FLAGDNN_POINTWISE_CASE=logical_not_bool_1x1x16"
    TIMEOUT 60)

  add_test(NAME integration.nvidia.dependency_boundary
    COMMAND "${CMAKE_COMMAND}"
      "-DCORE_LIBRARY=$<TARGET_FILE:flagdnn>"
      "-DBACKEND_LIBRARY=$<TARGET_FILE:flagdnn_backend_nvidia>"
      "-DEXECUTION_ENGINE=${FLAGDNN_EXECUTION_ENGINE}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/VerifyNativeDependencies.cmake")
  set_tests_properties(integration.nvidia.dependency_boundary PROPERTIES
    LABELS "integration;nvidia;dependency")

  add_test(NAME integration.nvidia.reference_dependency_boundary
    COMMAND "${CMAKE_COMMAND}"
      "-DREFERENCE_EXECUTABLE=$<TARGET_FILE:flagdnn_test_nvidia_add_square>"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/VerifyReferenceDependencies.cmake")
  set_tests_properties(
    integration.nvidia.reference_dependency_boundary PROPERTIES
    LABELS "integration;nvidia;cudnn;dependency")

  if(FLAGDNN_EXECUTION_ENGINE STREQUAL "libtriton_jit")
    add_executable(flagdnn_test_nvidia_jit functional/test_jit.cpp)
    target_compile_features(flagdnn_test_nvidia_jit PRIVATE cxx_std_20)
    target_link_libraries(flagdnn_test_nvidia_jit PRIVATE FlagDNN::flagdnn)
    add_dependencies(flagdnn_test_nvidia_jit flagdnn_backend_nvidia)
    flagdnn_enable_warnings(flagdnn_test_nvidia_jit)
    add_test(NAME integration.nvidia.jit
      COMMAND flagdnn_test_nvidia_jit
        "${FLAGDNN_CODEGEN_PYTHON}"
        "${FLAGDNN_CODEGEN_COMPILER}")
    set_tests_properties(integration.nvidia.jit PROPERTIES
      LABELS "integration;nvidia;libtriton_jit;autotune"
      ENVIRONMENT
        "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_nvidia>"
      RUN_SERIAL TRUE
      TIMEOUT 1200)
  endif()

  add_executable(flagdnn_test_nvidia_runtime functional/test_runtime.cpp)
  target_compile_features(flagdnn_test_nvidia_runtime PRIVATE cxx_std_20)
  target_link_libraries(flagdnn_test_nvidia_runtime PRIVATE
    FlagDNN::flagdnn
    flagdnn_validation_nvidia_platform)
  add_dependencies(flagdnn_test_nvidia_runtime flagdnn_backend_nvidia)
  flagdnn_enable_warnings(flagdnn_test_nvidia_runtime)
  if(FLAGDNN_EXECUTION_ENGINE STREQUAL "libtriton_jit")
    target_compile_definitions(flagdnn_test_nvidia_runtime PRIVATE
      FLAGDNN_EXPECT_LIBTRITON_JIT=1)
  endif()
  add_test(NAME integration.nvidia.runtime
    COMMAND flagdnn_test_nvidia_runtime
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}")
  set_tests_properties(integration.nvidia.runtime PROPERTIES
    LABELS "integration;nvidia;runtime"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_nvidia>"
    RUN_SERIAL TRUE
    TIMEOUT 300)

  add_executable(flagdnn_test_nvidia_graph functional/test_graph.cpp)
  target_compile_features(flagdnn_test_nvidia_graph PRIVATE cxx_std_20)
  target_link_libraries(flagdnn_test_nvidia_graph PRIVATE
    FlagDNN::flagdnn
    flagdnn_validation_nvidia_platform)
  add_dependencies(flagdnn_test_nvidia_graph flagdnn_backend_nvidia)
  flagdnn_enable_warnings(flagdnn_test_nvidia_graph)
  if(FLAGDNN_EXECUTION_ENGINE STREQUAL "libtriton_jit")
    target_compile_definitions(flagdnn_test_nvidia_graph PRIVATE
      FLAGDNN_EXPECT_LIBTRITON_JIT=1)
  endif()
  add_test(NAME integration.nvidia.graph
    COMMAND flagdnn_test_nvidia_graph
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}")
  set_tests_properties(integration.nvidia.graph PROPERTIES
    LABELS "integration;nvidia;graph"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_nvidia>"
    RUN_SERIAL TRUE
    TIMEOUT 180)
endif()

if(FLAGDNN_BUILD_BENCHMARKS)
  add_library(flagdnn_benchmark_nvidia_adapter STATIC
    benchmark/batchnorm_inference.cpp
    benchmark/conv_backward.cpp
    benchmark/conv_fprop.cpp
    benchmark/cudnn_common.cpp
    benchmark/cudnn_provider.cpp
    benchmark/graph.cpp
    benchmark/matmul.cpp
    benchmark/pointwise.cpp
    benchmark/reduction.cpp
    benchmark/relu.cpp
    benchmark/reshape.cpp
    benchmark/runner.cpp
    benchmark/slice.cpp
    benchmark/transpose.cpp)
  target_compile_features(flagdnn_benchmark_nvidia_adapter PUBLIC cxx_std_20)
  target_include_directories(flagdnn_benchmark_nvidia_adapter PUBLIC
    "${PROJECT_SOURCE_DIR}/benchmark")
  target_link_libraries(flagdnn_benchmark_nvidia_adapter PUBLIC
    flagdnn_benchmark_cases
    FlagDNN::flagdnn
    flagdnn_validation_nvidia_platform
    flagdnn_validation_nvidia_tensor_io
    CUDNN::frontend
    CUDA::nvrtc)
  flagdnn_enable_warnings(flagdnn_benchmark_nvidia_adapter)

  flagdnn_register_benchmark_suite(
    PLATFORM nvidia
    ADAPTER_TARGET flagdnn_benchmark_nvidia_adapter
    BACKEND_TARGET flagdnn_backend_nvidia
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_nvidia>"
    LABELS cudnn
    TIMEOUT 1800
    BUILD_RPATH "${_flagdnn_nvidia_validation_cudnn_directory}")
endif()
