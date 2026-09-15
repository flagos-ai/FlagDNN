# NVIDIA-only validation assembly. Production backend targets never link this
# directory, and none of these targets are installed.

include("${PROJECT_SOURCE_DIR}/cmake/Operators.cmake")

# Standalone operators without an executable cuDNN comparator on the
# qualified NVIDIA stack. The unified runner consumes the same manifest.
set(_flagdnn_cudnn_unsupported_manifest
  "${CMAKE_CURRENT_SOURCE_DIR}/cudnn_unsupported_operators.txt")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${_flagdnn_cudnn_unsupported_manifest}")
file(STRINGS "${_flagdnn_cudnn_unsupported_manifest}"
  _flagdnn_cudnn_unsupported_operators)
set(_flagdnn_cudnn_functional_operators ${FLAGDNN_FUNCTIONAL_OPERATORS})
set(_flagdnn_cudnn_extended_operators ${FLAGDNN_EXTENDED_OPERATORS})
list(REMOVE_ITEM _flagdnn_cudnn_functional_operators
  ${_flagdnn_cudnn_unsupported_operators})
list(REMOVE_ITEM _flagdnn_cudnn_extended_operators
  ${_flagdnn_cudnn_unsupported_operators})


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

# Attention correctness and performance share the same inputs, references,
# and output validation. This target also supports benchmark-only builds.
add_library(flagdnn_validation_nvidia_attention STATIC
  "${PROJECT_SOURCE_DIR}/tests/common/attention.cpp"
  functional/attention_runner.cpp
  functional/cudnn_attention.cpp
  functional/cudnn_graph.cpp)
target_compile_features(flagdnn_validation_nvidia_attention PUBLIC cxx_std_20)
target_include_directories(flagdnn_validation_nvidia_attention
  PUBLIC "${PROJECT_SOURCE_DIR}/tests" "${PROJECT_SOURCE_DIR}/benchmark"
  PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(flagdnn_validation_nvidia_attention PUBLIC
  FlagDNN::flagdnn flagdnn_validation_nvidia_platform
  flagdnn_validation_nvidia_tensor_io CUDNN::frontend CUDA::nvrtc)
flagdnn_enable_warnings(flagdnn_validation_nvidia_attention)

# Reuse the neutral case objects when functional tests are enabled. A
# benchmark-only build compiles the same sources into this private archive.
if(TARGET flagdnn_test_common_objects)
  set(_flagdnn_native_case_sources $<TARGET_OBJECTS:flagdnn_test_common_objects>)
else()
  set(_flagdnn_native_case_sources
    "${PROJECT_SOURCE_DIR}/tests/common/matmul.cpp"
    "${PROJECT_SOURCE_DIR}/tests/common/convolution.cpp"
    "${PROJECT_SOURCE_DIR}/tests/common/reduction.cpp"
    "${PROJECT_SOURCE_DIR}/tests/common/layout.cpp"
    "${PROJECT_SOURCE_DIR}/tests/common/pointwise.cpp"
    "${PROJECT_SOURCE_DIR}/tests/common/add.cpp"
    "${PROJECT_SOURCE_DIR}/tests/common/composite.cpp"
    "${PROJECT_SOURCE_DIR}/tests/common/statistics.cpp"
    "${PROJECT_SOURCE_DIR}/tests/common/normalization_extended.cpp"
    "${PROJECT_SOURCE_DIR}/tests/common/index.cpp"
    "${PROJECT_SOURCE_DIR}/tests/common/position_embedding.cpp"
    "${PROJECT_SOURCE_DIR}/tests/common/random.cpp"
    "${PROJECT_SOURCE_DIR}/tests/common/resample.cpp"
    "${PROJECT_SOURCE_DIR}/tests/common/causal_convolution.cpp"
    "${PROJECT_SOURCE_DIR}/tests/common/fp8_matmul.cpp"
    "${PROJECT_SOURCE_DIR}/tests/common/moe_matmul.cpp")
endif()

add_library(flagdnn_validation_nvidia_extended STATIC
  ${_flagdnn_native_case_sources}
  benchmark/native_runner.cpp
  benchmark/dtype_boolean.cpp
  benchmark/native_copy.cpp
  benchmark/native_dtype.cpp
  benchmark/native_matrix.cpp
  functional/cudnn_matmul.cpp
  functional/cudnn_convolution.cpp
  functional/cudnn_reduction.cpp
  functional/cudnn_extended.cpp
  functional/cudnn_layout.cpp
  functional/cudnn_resample.cpp
  functional/cudnn_pointwise.cpp
  functional/pointwise_runner.cpp
  functional/layout_runner.cpp
  functional/cudnn_causal_convolution.cpp
  functional/statistics_runner.cpp
  functional/normalization_extended_runner.cpp
  functional/index_runner.cpp
  functional/resample_runner.cpp
  functional/causal_convolution_runner.cpp
  functional/fp8_matmul_runner.cpp
  functional/moe_matmul_runner.cpp)
target_compile_features(flagdnn_validation_nvidia_extended PUBLIC cxx_std_20)
target_include_directories(flagdnn_validation_nvidia_extended PUBLIC
  "${PROJECT_SOURCE_DIR}/tests" "${PROJECT_SOURCE_DIR}/benchmark")
target_link_libraries(flagdnn_validation_nvidia_extended PUBLIC
  FlagDNN::flagdnn flagdnn_validation_nvidia_platform flagdnn_validation_nvidia_tensor_io
  flagdnn_validation_nvidia_attention CUDNN::frontend CUDA::nvrtc)
find_library(FLAGDNN_VALIDATION_CUDNN_EXT_LIBRARY
  NAMES cudnn_ext libcudnn_ext.so.9
  HINTS "${_flagdnn_nvidia_validation_cudnn_directory}" REQUIRED)
target_link_libraries(flagdnn_validation_nvidia_extended PUBLIC
  "${FLAGDNN_VALIDATION_CUDNN_EXT_LIBRARY}")
flagdnn_enable_warnings(flagdnn_validation_nvidia_extended)

if(FLAGDNN_BUILD_TESTS)
  add_executable(flagdnn_test_nvidia_generation_contract
    functional/test_generation_contract.cpp)
  target_link_libraries(flagdnn_test_nvidia_generation_contract PRIVATE
    FlagDNN::flagdnn flagdnn_validation_nvidia_platform
    flagdnn_validation_nvidia_extended)
  flagdnn_enable_warnings(flagdnn_test_nvidia_generation_contract)
  add_test(NAME integration.nvidia.generation_contract
    COMMAND flagdnn_test_nvidia_generation_contract
      "${FLAGDNN_CODEGEN_PYTHON}" "${FLAGDNN_CODEGEN_COMPILER}")
  set_tests_properties(integration.nvidia.generation_contract PROPERTIES
    LABELS "integration;nvidia;runtime;correctness"
    ENVIRONMENT "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_nvidia>"
    RUN_SERIAL TRUE TIMEOUT 120)
  add_executable(flagdnn_test_nvidia_artifact_contract
    artifact_contract.cpp
    ../artifact.cpp ../error.cpp
    "${PROJECT_SOURCE_DIR}/src/runtime/json.cpp"
    "${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp")
  target_compile_features(flagdnn_test_nvidia_artifact_contract PRIVATE cxx_std_20)
  target_include_directories(flagdnn_test_nvidia_artifact_contract PRIVATE
    "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/include" "${PROJECT_BINARY_DIR}/generated/include")
  target_link_libraries(flagdnn_test_nvidia_artifact_contract PRIVATE
    flagdnn_validation_nvidia_platform)
  flagdnn_enable_warnings(flagdnn_test_nvidia_artifact_contract)
  add_test(NAME integration.nvidia.artifact_contract
    COMMAND "${FLAGDNN_CODEGEN_PYTHON}" -B
      "${CMAKE_CURRENT_SOURCE_DIR}/artifact_contract.py"
      "$<TARGET_FILE:flagdnn_test_nvidia_artifact_contract>" "${PROJECT_VERSION}")
  set_tests_properties(integration.nvidia.artifact_contract PROPERTIES
    LABELS "integration;nvidia;compiler")

  add_test(NAME integration.nvidia.compiler_contract
    COMMAND "${FLAGDNN_CODEGEN_PYTHON}" -B
      "${CMAKE_CURRENT_SOURCE_DIR}/compiler_contract.py")
  set_tests_properties(integration.nvidia.compiler_contract PROPERTIES
    LABELS "integration;nvidia;compiler")

  add_executable(flagdnn_test_nvidia_execution_contract
    functional/test_execution_contract.cpp)
  target_compile_features(flagdnn_test_nvidia_execution_contract PRIVATE cxx_std_20)
  target_link_libraries(flagdnn_test_nvidia_execution_contract PRIVATE
    FlagDNN::flagdnn flagdnn_validation_nvidia_platform
    flagdnn_validation_nvidia_tensor_io flagdnn_validation_nvidia_extended)
  add_dependencies(flagdnn_test_nvidia_execution_contract flagdnn_backend_nvidia)
  flagdnn_enable_warnings(flagdnn_test_nvidia_execution_contract)
  add_test(NAME integration.nvidia.execution_contract
    COMMAND flagdnn_test_nvidia_execution_contract
      "${FLAGDNN_CODEGEN_PYTHON}" "${FLAGDNN_CODEGEN_COMPILER}")
  set_tests_properties(integration.nvidia.execution_contract PROPERTIES
    LABELS "integration;nvidia;runtime;correctness"
    ENVIRONMENT "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_nvidia>"
    RUN_SERIAL TRUE TIMEOUT 600)

  add_test(NAME integration.nvidia.ieee_range
    COMMAND flagdnn_test_nvidia_execution_contract
      "${FLAGDNN_CODEGEN_PYTHON}" "${FLAGDNN_CODEGEN_COMPILER}" ieee-range)
  set_tests_properties(integration.nvidia.ieee_range PROPERTIES
    LABELS "integration;nvidia;runtime;correctness;cudnn;ieee"
    ENVIRONMENT "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_nvidia>;NVIDIA_TF32_OVERRIDE=0"
    RUN_SERIAL TRUE TIMEOUT 600)

  add_test(NAME integration.nvidia.tensor_map_first
    COMMAND flagdnn_test_nvidia_execution_contract
      "${FLAGDNN_CODEGEN_PYTHON}" "${FLAGDNN_CODEGEN_COMPILER}" tensor-map-first)
  set_tests_properties(integration.nvidia.tensor_map_first PROPERTIES
    LABELS "integration;nvidia;runtime;correctness"
    ENVIRONMENT "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_nvidia>;TRITON_JIT_BACKEND="
    RUN_SERIAL TRUE TIMEOUT 180)

  add_test(NAME integration.nvidia.short_map_first
    COMMAND flagdnn_test_nvidia_execution_contract
      "${FLAGDNN_CODEGEN_PYTHON}" "${FLAGDNN_CODEGEN_COMPILER}" short-map-first)
  set_tests_properties(integration.nvidia.short_map_first PROPERTIES
    LABELS "integration;nvidia;runtime;correctness"
    ENVIRONMENT "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_nvidia>;TRITON_JIT_BACKEND="
    RUN_SERIAL TRUE TIMEOUT 180)

  add_library(flagdnn_test_nvidia_adapter STATIC
    $<TARGET_OBJECTS:flagdnn_test_common_objects>
    functional/add_runner.cpp
    functional/composite_runner.cpp
    functional/convolution_runner.cpp
    functional/cudnn_add.cpp
    functional/cudnn_composite.cpp
    functional/cudnn_normalization.cpp
    functional/matmul_runner.cpp
    functional/normalization_runner.cpp
    functional/reduction_runner.cpp)
  target_compile_features(flagdnn_test_nvidia_adapter PUBLIC cxx_std_20)
  target_include_directories(flagdnn_test_nvidia_adapter PUBLIC
    "${PROJECT_SOURCE_DIR}/tests")
  target_link_libraries(flagdnn_test_nvidia_adapter PUBLIC
    flagdnn_validation_nvidia_extended
    flagdnn_validation_nvidia_attention
    FlagDNN::flagdnn
    flagdnn_validation_nvidia_platform
    flagdnn_validation_nvidia_tensor_io
    CUDNN::frontend
    CUDA::nvrtc)
  flagdnn_enable_warnings(flagdnn_test_nvidia_adapter)

  flagdnn_register_functional_suite(
    PLATFORM nvidia
    OPERATORS ${_flagdnn_cudnn_functional_operators}
    ADAPTER_TARGET flagdnn_test_nvidia_adapter
    BACKEND_TARGET flagdnn_backend_nvidia
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_nvidia>"
    LABELS cudnn
    TIMEOUT 3600
    BUILD_RPATH "${_flagdnn_nvidia_validation_cudnn_directory}")

  # cuDNN reads its TF32 override at process startup; IEEE and TF32 cases
  # therefore use separate CTest processes with identical precision per pair.
  foreach(operator IN ITEMS matmul conv_fprop conv_dgrad conv_wgrad)
    foreach(precision IN ITEMS ieee tf32)
      if(precision STREQUAL "ieee")
        set(input_precision 1)
        set(tf32_override 0)
      else()
        set(input_precision 2)
        set(tf32_override 1)
      endif()
      add_test(NAME "functional.nvidia.${operator}.${precision}"
        COMMAND "flagdnn_test_nvidia_${operator}"
          "${FLAGDNN_CODEGEN_PYTHON}" "${FLAGDNN_CODEGEN_COMPILER}")
      set_tests_properties("functional.nvidia.${operator}.${precision}" PROPERTIES
        LABELS "functional;correctness;nvidia;cudnn;${operator};${precision}"
        RUN_SERIAL TRUE TIMEOUT 3600
        ENVIRONMENT "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_nvidia>;FLAGDNN_INPUT_PRECISION=${input_precision};NVIDIA_TF32_OVERRIDE=${tf32_override}")
    endforeach()
  endforeach()

  add_test(NAME functional.nvidia.matmul.fp8
    COMMAND flagdnn_test_nvidia_matmul_fp8
      "${FLAGDNN_CODEGEN_PYTHON}" "${FLAGDNN_CODEGEN_COMPILER}")
  set_tests_properties(functional.nvidia.matmul.fp8 PROPERTIES
    LABELS "functional;correctness;nvidia;matmul;cudnn;fp8"
    RUN_SERIAL TRUE TIMEOUT 1200
    ENVIRONMENT "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_nvidia>;FLAGDNN_FP8_MATMUL_API=matmul")

  add_test(NAME integration.nvidia.dependency_boundary
    COMMAND "${CMAKE_COMMAND}"
      "-DCORE_LIBRARY=$<TARGET_FILE:flagdnn>"
      "-DBACKEND_LIBRARY=$<TARGET_FILE:flagdnn_backend_nvidia>"
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

  add_executable(flagdnn_test_nvidia_runtime functional/test_runtime.cpp)
  target_compile_features(flagdnn_test_nvidia_runtime PRIVATE cxx_std_20)
  target_link_libraries(flagdnn_test_nvidia_runtime PRIVATE
    FlagDNN::flagdnn
    flagdnn_validation_nvidia_platform
    flagdnn_validation_nvidia_extended)
  add_dependencies(flagdnn_test_nvidia_runtime flagdnn_backend_nvidia)
  flagdnn_enable_warnings(flagdnn_test_nvidia_runtime)
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
    flagdnn_validation_nvidia_platform
    flagdnn_validation_nvidia_extended)
  add_dependencies(flagdnn_test_nvidia_graph flagdnn_backend_nvidia)
  flagdnn_enable_warnings(flagdnn_test_nvidia_graph)
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
    flagdnn_validation_nvidia_extended
    FlagDNN::flagdnn
    flagdnn_validation_nvidia_platform
    flagdnn_validation_nvidia_tensor_io
    CUDNN::frontend
    CUDA::nvrtc)
  flagdnn_enable_warnings(flagdnn_benchmark_nvidia_adapter)

  # These operators use the paired dtype runners below, which provide
  # their actual cuDNN Graph or legacy API comparison and measurements.
  set(_flagdnn_nvidia_excluded_benchmarks
    "${CMAKE_CURRENT_SOURCE_DIR}/benchmark/dtype_only_operators.txt")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_flagdnn_nvidia_excluded_benchmarks}")
  file(STRINGS "${_flagdnn_nvidia_excluded_benchmarks}"
    _flagdnn_nvidia_excluded_operators)
  set(_flagdnn_nvidia_benchmark_operators ${FLAGDNN_BENCHMARK_OPERATORS})
  list(REMOVE_ITEM _flagdnn_nvidia_benchmark_operators
    ${_flagdnn_nvidia_excluded_operators})

  flagdnn_register_benchmark_suite(
    PLATFORM nvidia
    OPERATORS ${_flagdnn_nvidia_benchmark_operators}
    ADAPTER_TARGET flagdnn_benchmark_nvidia_adapter
    BACKEND_TARGET flagdnn_backend_nvidia
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_nvidia>"
    LABELS cudnn
    # Complete multi-stage convolution tuning can exceed 30 minutes on a cold
    # artifact cache. Match the unified NVIDIA runner's per-operator budget.
    TIMEOUT 3600
    BUILD_RPATH "${_flagdnn_nvidia_validation_cudnn_directory}")
  set(_flagdnn_nvidia_additional_benchmarks
    "${CMAKE_CURRENT_SOURCE_DIR}/benchmark/additional_operators.txt")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_flagdnn_nvidia_additional_benchmarks}")
  file(STRINGS "${_flagdnn_nvidia_additional_benchmarks}"
    _flagdnn_nvidia_attention_operators)
  list(REMOVE_ITEM _flagdnn_nvidia_attention_operators ${FLAGDNN_EXTENDED_OPERATORS})
  flagdnn_register_benchmark_suite(
    PLATFORM nvidia
    OPERATORS ${_flagdnn_nvidia_attention_operators}
    ADAPTER_TARGET flagdnn_validation_nvidia_attention
    BACKEND_TARGET flagdnn_backend_nvidia
    COMMAND_ARGS "${FLAGDNN_CODEGEN_PYTHON}" "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_nvidia>"
    LABELS cudnn
    TIMEOUT 3600
    BUILD_RPATH "${_flagdnn_nvidia_validation_cudnn_directory}")
  flagdnn_register_benchmark_suite(
    PLATFORM nvidia
    OPERATORS ${_flagdnn_cudnn_extended_operators}
    ADAPTER_TARGET flagdnn_validation_nvidia_extended
    BACKEND_TARGET flagdnn_backend_nvidia
    COMMAND_ARGS "${FLAGDNN_CODEGEN_PYTHON}" "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_nvidia>"
    LABELS cudnn
    TIMEOUT 3600)
  add_test(NAME benchmark.nvidia.matmul.fp8
    COMMAND flagdnn_benchmark_nvidia_matmul_fp8
      "${FLAGDNN_CODEGEN_PYTHON}" "${FLAGDNN_CODEGEN_COMPILER}")
  set_tests_properties(benchmark.nvidia.matmul.fp8 PROPERTIES
    LABELS "benchmark;cudnn;nvidia;matmul;fp8"
    RUN_SERIAL TRUE TIMEOUT 1200
    ENVIRONMENT "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_nvidia>;FLAGDNN_FP8_MATMUL_API=matmul")
  add_executable(flagdnn_benchmark_nvidia_dtype "${PROJECT_SOURCE_DIR}/benchmark/dtype_main.cpp")
  target_link_libraries(flagdnn_benchmark_nvidia_dtype PRIVATE flagdnn_validation_nvidia_extended)
  flagdnn_enable_warnings(flagdnn_benchmark_nvidia_dtype)
  foreach(category IN ITEMS boolean copy ieee tf32 fp32_output)
    if(category STREQUAL "boolean")
      set(dtype_operators logical_and logical_or logical_not)
    elseif(category STREQUAL "copy")
      set(dtype_operators identity reshape transpose slice)
    elseif(category STREQUAL "ieee" OR category STREQUAL "tf32")
      set(dtype_operators matmul conv_fprop conv_dgrad conv_wgrad)
    else()
      set(dtype_operators reduction)
    endif()
    set(dtype_runner_category "${category}")
    set(dtype_environment "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_nvidia>")
    if(category STREQUAL "ieee")
      set(dtype_runner_category precision)
      list(APPEND dtype_environment "FLAGDNN_INPUT_PRECISION=1" "NVIDIA_TF32_OVERRIDE=0")
    elseif(category STREQUAL "tf32")
      set(dtype_runner_category precision)
      list(APPEND dtype_environment "FLAGDNN_INPUT_PRECISION=2" "NVIDIA_TF32_OVERRIDE=1")
    endif()
    foreach(operator IN LISTS dtype_operators)
      add_test(NAME "benchmark.nvidia.${operator}.${category}"
        COMMAND flagdnn_benchmark_nvidia_dtype "${FLAGDNN_CODEGEN_PYTHON}" "${FLAGDNN_CODEGEN_COMPILER}" "${operator}" "${dtype_runner_category}")
      set_tests_properties("benchmark.nvidia.${operator}.${category}" PROPERTIES
        LABELS "benchmark;cudnn;nvidia;${operator};${category}" RUN_SERIAL TRUE TIMEOUT 3600
        ENVIRONMENT "${dtype_environment}")
    endforeach()
  endforeach()
endif()
