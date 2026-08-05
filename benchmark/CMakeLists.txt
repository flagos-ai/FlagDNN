# Platform-neutral native C++ performance workloads and registration contract.
# Device SDK discovery and provider implementations belong to
# backends/<platform>/validation.

include("${PROJECT_SOURCE_DIR}/cmake/Operators.cmake")

add_library(flagdnn_benchmark_cases STATIC
  common/add_cases.cpp
  common/all_cases.cpp
  common/batchnorm_inference_cases.cpp
  common/binary_pointwise_cases.cpp
  common/conv_backward_cases.cpp
  common/conv_fprop_cases.cpp
  common/conv_graph_cases.cpp
  common/matmul_cases.cpp
  common/pointwise_cases.cpp
  common/pointwise_graph_cases.cpp
  common/reduction_cases.cpp
  common/relu_cases.cpp
  common/reshape_cases.cpp
  common/slice_cases.cpp
  common/transpose_cases.cpp)
target_compile_features(flagdnn_benchmark_cases PUBLIC cxx_std_20)
target_include_directories(flagdnn_benchmark_cases PUBLIC
  "${PROJECT_SOURCE_DIR}/benchmark")
target_link_libraries(flagdnn_benchmark_cases PUBLIC FlagDNN::flagdnn)
flagdnn_enable_warnings(flagdnn_benchmark_cases)

add_executable(flagdnn_benchmark_catalog_contract
  common/catalog_contract.cpp)
target_compile_features(flagdnn_benchmark_catalog_contract PRIVATE cxx_std_20)
target_link_libraries(flagdnn_benchmark_catalog_contract PRIVATE
  flagdnn_benchmark_cases)
flagdnn_enable_warnings(flagdnn_benchmark_catalog_contract)
add_test(NAME benchmark.catalog_contract
  COMMAND flagdnn_benchmark_catalog_contract)
set_tests_properties(benchmark.catalog_contract PROPERTIES
  LABELS "benchmark;contract")

add_test(NAME benchmark.catalog_dependency_boundary
  COMMAND "${CMAKE_COMMAND}"
    "-DBINARY=$<TARGET_FILE:flagdnn_benchmark_catalog_contract>"
    -P "${PROJECT_SOURCE_DIR}/cmake/VerifyNoVendorDependencies.cmake")
set_tests_properties(benchmark.catalog_dependency_boundary PROPERTIES
  LABELS "benchmark;contract")

# A backend validation CMake file calls this after defining its complete
# benchmark adapter. The workload catalog remains unaware of CUDA, cuDNN,
# CANN, ACL, or any other platform SDK.
function(flagdnn_register_benchmark_suite)
  set(options)
  set(one_value_arguments
    PLATFORM ADAPTER_TARGET BACKEND_TARGET TIMEOUT BUILD_RPATH)
  set(multi_value_arguments COMMAND_ARGS ENVIRONMENT LABELS)
  cmake_parse_arguments(
    FLAGDNN_BENCH
    "${options}"
    "${one_value_arguments}"
    "${multi_value_arguments}"
    ${ARGN})

  if(NOT FLAGDNN_BENCH_PLATFORM MATCHES "^[a-z][a-z0-9_]*$")
    message(FATAL_ERROR "Benchmark platform name is invalid")
  endif()
  if(NOT TARGET "${FLAGDNN_BENCH_ADAPTER_TARGET}")
    message(FATAL_ERROR
      "Benchmark adapter target does not exist: ${FLAGDNN_BENCH_ADAPTER_TARGET}")
  endif()
  if(NOT TARGET "${FLAGDNN_BENCH_BACKEND_TARGET}")
    message(FATAL_ERROR
      "Benchmark backend target does not exist: ${FLAGDNN_BENCH_BACKEND_TARGET}")
  endif()
  if(NOT FLAGDNN_BENCH_TIMEOUT)
    set(FLAGDNN_BENCH_TIMEOUT 1800)
  endif()

  foreach(operator IN LISTS FLAGDNN_BENCHMARK_OPERATORS)
    set(entry "${PROJECT_SOURCE_DIR}/benchmark/test_${operator}.cpp")
    if(NOT EXISTS "${entry}")
      message(FATAL_ERROR "Benchmark operator ${operator} has no C++ entry")
    endif()

    set(target "flagdnn_benchmark_${FLAGDNN_BENCH_PLATFORM}_${operator}")
    add_executable("${target}" "${entry}")
    target_compile_features("${target}" PRIVATE cxx_std_20)
    target_link_libraries("${target}" PRIVATE
      "${FLAGDNN_BENCH_ADAPTER_TARGET}")
    add_dependencies("${target}" "${FLAGDNN_BENCH_BACKEND_TARGET}")
    flagdnn_enable_warnings("${target}")
    if(FLAGDNN_BENCH_BUILD_RPATH)
      set_target_properties("${target}" PROPERTIES
        BUILD_RPATH "${FLAGDNN_BENCH_BUILD_RPATH}")
    endif()

    set(test_name "benchmark.${FLAGDNN_BENCH_PLATFORM}.${operator}")
    add_test(NAME "${test_name}"
      COMMAND "${target}" ${FLAGDNN_BENCH_COMMAND_ARGS})
    set_tests_properties("${test_name}" PROPERTIES
      LABELS
        "benchmark;performance;${FLAGDNN_BENCH_PLATFORM};${operator};${FLAGDNN_BENCH_LABELS}"
      RUN_SERIAL TRUE
      ENVIRONMENT "${FLAGDNN_BENCH_ENVIRONMENT}"
      TIMEOUT "${FLAGDNN_BENCH_TIMEOUT}")
  endforeach()
endfunction()
