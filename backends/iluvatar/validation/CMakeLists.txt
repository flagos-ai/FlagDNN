# Iluvatar production discovery is independent of CoreX cuDNN. Reference
# targets are introduced only in the dedicated validation phase.

if(FLAGDNN_BUILD_TESTS OR FLAGDNN_BUILD_BENCHMARKS)
  add_library(flagdnn_validation_iluvatar_reference STATIC
    corex_cudnn_capability.cpp
    corex_cudnn_reference.cpp
    corex_cudnn_status.cpp
    reference_tensor.cpp
    "${PROJECT_SOURCE_DIR}/src/runtime/json.cpp")
  target_compile_features(flagdnn_validation_iluvatar_reference PUBLIC
    cxx_std_20)
  target_compile_definitions(flagdnn_validation_iluvatar_reference PRIVATE
    FLAGDNN_ILUVATAR_CUDNN_CAPABILITY_CATALOG="${CMAKE_CURRENT_SOURCE_DIR}/corex_cudnn_capabilities.json")
  target_include_directories(flagdnn_validation_iluvatar_reference
    SYSTEM PUBLIC
      "${FLAGDNN_ILUVATAR_RESOLVED_CUDA_INCLUDE_DIR}"
      "${FLAGDNN_ILUVATAR_RESOLVED_CUDNN_INCLUDE_DIR}")
  target_include_directories(flagdnn_validation_iluvatar_reference PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/tests"
    PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}")
  target_link_libraries(flagdnn_validation_iluvatar_reference PUBLIC
    FlagDNN::flagdnn
    "${FLAGDNN_ILUVATAR_RESOLVED_CUDNN_LIBRARY}"
    "${FLAGDNN_ILUVATAR_RESOLVED_CUDA_RUNTIME_LIBRARY}"
    "${FLAGDNN_ILUVATAR_RESOLVED_CUDA_DRIVER_LIBRARY}"
    ${CMAKE_DL_LIBS})
  flagdnn_enable_warnings(flagdnn_validation_iluvatar_reference)

endif()

if(FLAGDNN_BUILD_TESTS)
  add_executable(flagdnn_test_iluvatar_validation_contract
    validation_contract.cpp
    $<TARGET_OBJECTS:flagdnn_test_common_objects>)
  target_compile_features(flagdnn_test_iluvatar_validation_contract PRIVATE
    cxx_std_20)
  target_compile_definitions(flagdnn_test_iluvatar_validation_contract PRIVATE
    FLAGDNN_ILUVATAR_CUDNN_CAPABILITY_CATALOG="${CMAKE_CURRENT_SOURCE_DIR}/corex_cudnn_capabilities.json")
  target_include_directories(flagdnn_test_iluvatar_validation_contract PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}")
  target_link_libraries(flagdnn_test_iluvatar_validation_contract PRIVATE
    flagdnn_validation_iluvatar_reference)
  flagdnn_enable_warnings(flagdnn_test_iluvatar_validation_contract)
  add_test(NAME integration.iluvatar.validation_contract
    COMMAND flagdnn_test_iluvatar_validation_contract
      "${CMAKE_CURRENT_SOURCE_DIR}/corex_cudnn_capabilities.json"
      "${PROJECT_SOURCE_DIR}/cmake/Operators.cmake")
  set_tests_properties(integration.iluvatar.validation_contract PROPERTIES
    LABELS "integration;iluvatar;cudnn;capability;catalog"
    TIMEOUT 120)

  add_test(NAME integration.iluvatar.catalog_closure
    COMMAND flagdnn_test_iluvatar_validation_contract
      "${CMAKE_CURRENT_SOURCE_DIR}/corex_cudnn_capabilities.json"
      "${PROJECT_SOURCE_DIR}/cmake/Operators.cmake"
      --require-closed)
  set_tests_properties(integration.iluvatar.catalog_closure PROPERTIES
    LABELS "integration;iluvatar;cudnn;capability;catalog;closure"
    TIMEOUT 120)

  add_library(flagdnn_test_iluvatar_cudnn_symbol_probe OBJECT
    corex_cudnn_symbol_probe.cpp)
  target_compile_features(flagdnn_test_iluvatar_cudnn_symbol_probe PRIVATE
    cxx_std_20)
  target_include_directories(flagdnn_test_iluvatar_cudnn_symbol_probe
    SYSTEM PRIVATE "${FLAGDNN_ILUVATAR_RESOLVED_CUDNN_INCLUDE_DIR}")
  flagdnn_enable_warnings(flagdnn_test_iluvatar_cudnn_symbol_probe)

  add_executable(flagdnn_test_iluvatar_corex_cudnn_environment
    functional/test_corex_cudnn_environment.cpp)
  target_compile_features(
    flagdnn_test_iluvatar_corex_cudnn_environment PRIVATE cxx_std_20)
  target_include_directories(
    flagdnn_test_iluvatar_corex_cudnn_environment SYSTEM PRIVATE
    "${FLAGDNN_ILUVATAR_RESOLVED_CUDA_INCLUDE_DIR}"
    "${FLAGDNN_ILUVATAR_RESOLVED_CUDNN_INCLUDE_DIR}")
  target_link_libraries(flagdnn_test_iluvatar_corex_cudnn_environment PRIVATE
    "${FLAGDNN_ILUVATAR_RESOLVED_CUDNN_LIBRARY}"
    "${FLAGDNN_ILUVATAR_RESOLVED_CUDA_RUNTIME_LIBRARY}"
    "${FLAGDNN_ILUVATAR_RESOLVED_CUDA_DRIVER_LIBRARY}"
    ${CMAKE_DL_LIBS})
  target_link_directories(
    flagdnn_test_iluvatar_corex_cudnn_environment PRIVATE
    "${FLAGDNN_ILUVATAR_RESOLVED_COREX_ROOT}/lib64"
    "${FLAGDNN_ILUVATAR_RESOLVED_COREX_ROOT}/lib")
  flagdnn_enable_warnings(flagdnn_test_iluvatar_corex_cudnn_environment)
  add_dependencies(flagdnn_test_iluvatar_corex_cudnn_environment
    flagdnn_test_iluvatar_cudnn_symbol_probe)

  add_executable(flagdnn_test_iluvatar_reference_runtime
    functional/test_corex_cudnn_reference_runtime.cpp)
  target_compile_features(flagdnn_test_iluvatar_reference_runtime PRIVATE
    cxx_std_20)
  target_link_libraries(flagdnn_test_iluvatar_reference_runtime PRIVATE
    flagdnn_validation_iluvatar_reference)
  flagdnn_enable_warnings(flagdnn_test_iluvatar_reference_runtime)
  add_test(NAME integration.iluvatar.reference_runtime
    COMMAND flagdnn_test_iluvatar_reference_runtime)
  set_tests_properties(integration.iluvatar.reference_runtime PROPERTIES
    LABELS "integration;iluvatar;corex;cudnn;reference;runtime"
    RUN_SERIAL TRUE
    TIMEOUT 120)


  add_test(NAME integration.iluvatar.corex_cudnn_environment
    COMMAND "${CMAKE_COMMAND}"
      "-DSOURCE_ROOT=${PROJECT_SOURCE_DIR}"
      "-DTEST_ROOT=${PROJECT_BINARY_DIR}/corex-cudnn-contract/iluvatar"
      "-DPROBE_EXECUTABLE=$<TARGET_FILE:flagdnn_test_iluvatar_corex_cudnn_environment>"
      "-DC_COMPILER=${CMAKE_C_COMPILER}"
      "-DCOREX_ROOT=${FLAGDNN_ILUVATAR_RESOLVED_COREX_ROOT}"
      "-DCOREX_CUDNN_INCLUDE=${FLAGDNN_ILUVATAR_RESOLVED_CUDNN_INCLUDE_DIR}"
      "-DCOREX_CUDNN=${FLAGDNN_ILUVATAR_RESOLVED_CUDNN_LIBRARY}"
      "-DCOREX_CUDART=${FLAGDNN_ILUVATAR_RESOLVED_CUDA_RUNTIME_LIBRARY}"
      "-DCOREX_DRIVER=${FLAGDNN_ILUVATAR_RESOLVED_CUDA_DRIVER_LIBRARY}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CorexCudnnEnvironmentContract.cmake")
  set_tests_properties(integration.iluvatar.corex_cudnn_environment PROPERTIES
    LABELS "integration;iluvatar;corex;cudnn;environment"
    RUN_SERIAL TRUE
    TIMEOUT 300)

  add_test(NAME integration.iluvatar.reference_dependency_boundary
    COMMAND "${CMAKE_COMMAND}"
      "-DREFERENCE_EXECUTABLE=$<TARGET_FILE:flagdnn_test_iluvatar_reference_runtime>"
      "-DCOREX_ROOT=${FLAGDNN_ILUVATAR_RESOLVED_COREX_ROOT}"
      "-DCOREX_CUDNN=${FLAGDNN_ILUVATAR_RESOLVED_CUDNN_LIBRARY}"
      "-DCOREX_CUDART=${FLAGDNN_ILUVATAR_RESOLVED_CUDA_RUNTIME_LIBRARY}"
      "-DCOREX_DRIVER=${FLAGDNN_ILUVATAR_RESOLVED_CUDA_DRIVER_LIBRARY}"
      "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
      "-DOUTPUT_JSON=${PROJECT_BINARY_DIR}/qualification/reference-dependency-boundary.json"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/VerifyReferenceDependencies.cmake")
  set_tests_properties(
    integration.iluvatar.reference_dependency_boundary PROPERTIES
    LABELS "integration;iluvatar;dependency;cudnn;reference"
    RUN_SERIAL TRUE
    TIMEOUT 120)

  add_test(NAME integration.iluvatar.cmake_configuration_contract
    COMMAND "${CMAKE_COMMAND}"
      "-DSOURCE_ROOT=${PROJECT_SOURCE_DIR}"
      "-DTEST_ROOT=${PROJECT_BINARY_DIR}/cmake-contract/iluvatar"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/CMakeConfigurationContract.cmake")
  set_tests_properties(
    integration.iluvatar.cmake_configuration_contract PROPERTIES
    LABELS "integration;iluvatar;cmake;contract"
    TIMEOUT 300)

  set(_flagdnn_iluvatar_context_test_sources
    functional/test_context.cpp)
  foreach(_source IN ITEMS
      "${PROJECT_SOURCE_DIR}/backends/iluvatar/context.cpp"
      "${PROJECT_SOURCE_DIR}/backends/iluvatar/error.cpp")
    if(EXISTS "${_source}")
      list(APPEND _flagdnn_iluvatar_context_test_sources "${_source}")
    endif()
  endforeach()
  add_executable(flagdnn_test_iluvatar_context
    ${_flagdnn_iluvatar_context_test_sources})
  target_compile_features(flagdnn_test_iluvatar_context PRIVATE cxx_std_20)
  target_include_directories(flagdnn_test_iluvatar_context PRIVATE
    "${PROJECT_SOURCE_DIR}"
    "${FLAGDNN_ILUVATAR_RESOLVED_CUDA_INCLUDE_DIR}")
  target_link_libraries(flagdnn_test_iluvatar_context PRIVATE
    "${FLAGDNN_ILUVATAR_RESOLVED_CUDA_DRIVER_LIBRARY}")
  flagdnn_enable_warnings(flagdnn_test_iluvatar_context)
  add_test(NAME integration.iluvatar.context
    COMMAND flagdnn_test_iluvatar_context)
  set_tests_properties(integration.iluvatar.context PROPERTIES
    LABELS "integration;iluvatar;corex;context"
    RUN_SERIAL TRUE
    TIMEOUT 60)

  set(_flagdnn_iluvatar_artifact_test_sources
    artifact_contract.cpp
    "${PROJECT_SOURCE_DIR}/backends/iluvatar/context.cpp"
    "${PROJECT_SOURCE_DIR}/backends/iluvatar/error.cpp"
    "${PROJECT_SOURCE_DIR}/src/runtime/json.cpp"
    "${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp")
  if(EXISTS "${PROJECT_SOURCE_DIR}/backends/iluvatar/artifact.cpp")
    list(APPEND _flagdnn_iluvatar_artifact_test_sources
      "${PROJECT_SOURCE_DIR}/backends/iluvatar/artifact.cpp")
  endif()
  add_executable(flagdnn_test_iluvatar_artifact_contract
    ${_flagdnn_iluvatar_artifact_test_sources})
  target_compile_features(flagdnn_test_iluvatar_artifact_contract PRIVATE
    cxx_std_20)
  target_include_directories(flagdnn_test_iluvatar_artifact_contract PRIVATE
    "${PROJECT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_BINARY_DIR}/generated/include"
    "${FLAGDNN_ILUVATAR_RESOLVED_CUDA_INCLUDE_DIR}")
  target_link_libraries(flagdnn_test_iluvatar_artifact_contract PRIVATE
    "${FLAGDNN_ILUVATAR_RESOLVED_CUDA_DRIVER_LIBRARY}")
  flagdnn_enable_warnings(flagdnn_test_iluvatar_artifact_contract)
  add_test(NAME integration.iluvatar.artifact_contract
    COMMAND flagdnn_test_iluvatar_artifact_contract)
  set_tests_properties(integration.iluvatar.artifact_contract PROPERTIES
    LABELS "integration;iluvatar;artifact;contract"
    TIMEOUT 60)

  add_test(NAME integration.iluvatar.compiler_contract
    COMMAND "${FLAGDNN_CODEGEN_PYTHON}"
      "${CMAKE_CURRENT_SOURCE_DIR}/compiler_contract.py"
      "${PROJECT_SOURCE_DIR}")
  set_tests_properties(integration.iluvatar.compiler_contract PROPERTIES
    LABELS "integration;iluvatar;compiler;contract"
    ENVIRONMENT
      "FLAGDNN_ILUVATAR_COREX_ROOT=${FLAGDNN_ILUVATAR_RESOLVED_COREX_ROOT};FLAGDNN_ILUVATAR_TRITON_JIT_DIR=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_CONFIG_DIR};FLAGDNN_ILUVATAR_TRITON_JIT_LIBRARY=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_LIBRARY};FLAGDNN_ILUVATAR_TRITON_JIT_INCLUDE_DIR=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_INCLUDE_DIR};FLAGDNN_ILUVATAR_TRITON_JIT_SCRIPT_DIR=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_SCRIPT_DIR}"
    RUN_SERIAL TRUE
    TIMEOUT 300)

  add_test(NAME integration.iluvatar.run_tests_adapter_contract
    COMMAND "${FLAGDNN_CODEGEN_PYTHON}"
      "${CMAKE_CURRENT_SOURCE_DIR}/run_tests_adapter_contract.py"
      "${PROJECT_SOURCE_DIR}")
  set_tests_properties(
    integration.iluvatar.run_tests_adapter_contract PROPERTIES
    LABELS "integration;iluvatar;runner;adapter;contract"
    TIMEOUT 120)

  add_executable(flagdnn_test_iluvatar_jit
    functional/test_jit.cpp)
  target_compile_features(flagdnn_test_iluvatar_jit PRIVATE cxx_std_20)
  target_compile_definitions(flagdnn_test_iluvatar_jit PRIVATE BACKEND_IX=1)
  target_include_directories(flagdnn_test_iluvatar_jit SYSTEM PRIVATE
    "${FLAGDNN_ILUVATAR_RESOLVED_CUDA_INCLUDE_DIR}"
    "${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_INCLUDE_DIR}")
  target_include_directories(flagdnn_test_iluvatar_jit PRIVATE
    "${PROJECT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_BINARY_DIR}/generated/include")
  target_link_libraries(flagdnn_test_iluvatar_jit PRIVATE
    FlagDNN::flagdnn
    "${FLAGDNN_ILUVATAR_RESOLVED_CUDA_DRIVER_LIBRARY}"
    "${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_TARGET}"
    ${CMAKE_DL_LIBS})
  target_link_directories(flagdnn_test_iluvatar_jit PRIVATE
    "${FLAGDNN_ILUVATAR_RESOLVED_COREX_ROOT}/lib64"
    "${FLAGDNN_ILUVATAR_RESOLVED_COREX_ROOT}/lib")
  flagdnn_enable_warnings(flagdnn_test_iluvatar_jit)
  add_dependencies(flagdnn_test_iluvatar_jit flagdnn_backend_iluvatar)
  add_test(NAME integration.iluvatar.jit
    COMMAND flagdnn_test_iluvatar_jit
      "$<TARGET_FILE:flagdnn_backend_iluvatar>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py")
  set_tests_properties(integration.iluvatar.jit PROPERTIES
    LABELS "integration;iluvatar;jit;functional"
    ENVIRONMENT
      "FLAGDNN_ILUVATAR_COREX_ROOT=${FLAGDNN_ILUVATAR_RESOLVED_COREX_ROOT};FLAGDNN_ILUVATAR_TRITON_JIT_DIR=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_CONFIG_DIR};FLAGDNN_ILUVATAR_TRITON_JIT_LIBRARY=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_LIBRARY};FLAGDNN_ILUVATAR_TRITON_JIT_INCLUDE_DIR=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_INCLUDE_DIR};FLAGDNN_ILUVATAR_TRITON_JIT_SCRIPT_DIR=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_SCRIPT_DIR}"
    RUN_SERIAL TRUE
    TIMEOUT 600)

  add_executable(flagdnn_test_iluvatar_autotune
    functional/test_autotune.cpp
    "${PROJECT_SOURCE_DIR}/backends/autotune_policy.cpp"
    "${PROJECT_SOURCE_DIR}/src/runtime/json.cpp")
  target_compile_features(flagdnn_test_iluvatar_autotune PRIVATE cxx_std_20)
  target_compile_definitions(flagdnn_test_iluvatar_autotune PRIVATE
    BACKEND_IX=1)
  target_include_directories(flagdnn_test_iluvatar_autotune SYSTEM PRIVATE
    "${FLAGDNN_ILUVATAR_RESOLVED_CUDA_INCLUDE_DIR}"
    "${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_INCLUDE_DIR}")
  target_include_directories(flagdnn_test_iluvatar_autotune PRIVATE
    "${PROJECT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_BINARY_DIR}/generated/include")
  target_link_libraries(flagdnn_test_iluvatar_autotune PRIVATE
    FlagDNN::flagdnn
    "${FLAGDNN_ILUVATAR_RESOLVED_CUDA_DRIVER_LIBRARY}"
    "${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_TARGET}"
    ${CMAKE_DL_LIBS})
  target_link_directories(flagdnn_test_iluvatar_autotune PRIVATE
    "${FLAGDNN_ILUVATAR_RESOLVED_COREX_ROOT}/lib64"
    "${FLAGDNN_ILUVATAR_RESOLVED_COREX_ROOT}/lib")
  flagdnn_enable_warnings(flagdnn_test_iluvatar_autotune)
  add_dependencies(flagdnn_test_iluvatar_autotune flagdnn_backend_iluvatar)
  add_test(NAME integration.iluvatar.autotune
    COMMAND flagdnn_test_iluvatar_autotune
      "$<TARGET_FILE:flagdnn_backend_iluvatar>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py")
  set_tests_properties(integration.iluvatar.autotune PROPERTIES
    LABELS "integration;iluvatar;autotune;corex;event;cache"
    ENVIRONMENT
      "FLAGDNN_ILUVATAR_COREX_ROOT=${FLAGDNN_ILUVATAR_RESOLVED_COREX_ROOT};FLAGDNN_ILUVATAR_TRITON_JIT_DIR=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_CONFIG_DIR};FLAGDNN_ILUVATAR_TRITON_JIT_LIBRARY=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_LIBRARY};FLAGDNN_ILUVATAR_TRITON_JIT_INCLUDE_DIR=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_INCLUDE_DIR};FLAGDNN_ILUVATAR_TRITON_JIT_SCRIPT_DIR=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_SCRIPT_DIR}"
    RUN_SERIAL TRUE
    TIMEOUT 1200)

  add_executable(flagdnn_test_iluvatar_graph
    functional/test_graph.cpp)
  target_compile_features(flagdnn_test_iluvatar_graph PRIVATE cxx_std_20)
  target_compile_definitions(flagdnn_test_iluvatar_graph PRIVATE BACKEND_IX=1)
  target_include_directories(flagdnn_test_iluvatar_graph SYSTEM PRIVATE
    "${FLAGDNN_ILUVATAR_RESOLVED_CUDA_INCLUDE_DIR}"
    "${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_INCLUDE_DIR}")
  target_include_directories(flagdnn_test_iluvatar_graph PRIVATE
    "${PROJECT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_BINARY_DIR}/generated/include")
  target_link_libraries(flagdnn_test_iluvatar_graph PRIVATE
    FlagDNN::flagdnn
    "${FLAGDNN_ILUVATAR_RESOLVED_CUDA_DRIVER_LIBRARY}"
    "${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_TARGET}"
    ${CMAKE_DL_LIBS})
  target_link_directories(flagdnn_test_iluvatar_graph PRIVATE
    "${FLAGDNN_ILUVATAR_RESOLVED_COREX_ROOT}/lib64"
    "${FLAGDNN_ILUVATAR_RESOLVED_COREX_ROOT}/lib")
  flagdnn_enable_warnings(flagdnn_test_iluvatar_graph)
  add_dependencies(flagdnn_test_iluvatar_graph flagdnn_backend_iluvatar)
  add_test(NAME integration.iluvatar.graph
    COMMAND flagdnn_test_iluvatar_graph
      "$<TARGET_FILE:flagdnn_backend_iluvatar>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py")
  set_tests_properties(integration.iluvatar.graph PROPERTIES
    LABELS "integration;iluvatar;graph;capture;replay;corex"
    ENVIRONMENT
      "FLAGDNN_ILUVATAR_COREX_ROOT=${FLAGDNN_ILUVATAR_RESOLVED_COREX_ROOT};FLAGDNN_ILUVATAR_TRITON_JIT_DIR=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_CONFIG_DIR};FLAGDNN_ILUVATAR_TRITON_JIT_LIBRARY=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_LIBRARY};FLAGDNN_ILUVATAR_TRITON_JIT_INCLUDE_DIR=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_INCLUDE_DIR};FLAGDNN_ILUVATAR_TRITON_JIT_SCRIPT_DIR=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_SCRIPT_DIR}"
    RUN_SERIAL TRUE
    TIMEOUT 1200)

  add_executable(flagdnn_test_iluvatar_runtime
    functional/test_runtime.cpp)
  target_compile_features(flagdnn_test_iluvatar_runtime PRIVATE cxx_std_20)
  target_include_directories(flagdnn_test_iluvatar_runtime SYSTEM PRIVATE
    "${FLAGDNN_ILUVATAR_RESOLVED_CUDA_INCLUDE_DIR}")
  target_include_directories(flagdnn_test_iluvatar_runtime PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_BINARY_DIR}/generated/include")
  target_link_libraries(flagdnn_test_iluvatar_runtime PRIVATE
    FlagDNN::flagdnn
    "${FLAGDNN_ILUVATAR_RESOLVED_CUDA_DRIVER_LIBRARY}")
  flagdnn_enable_warnings(flagdnn_test_iluvatar_runtime)
  add_dependencies(flagdnn_test_iluvatar_runtime flagdnn_backend_iluvatar)
  add_test(NAME integration.iluvatar.runtime
    COMMAND flagdnn_test_iluvatar_runtime
      "$<TARGET_FILE:flagdnn_backend_iluvatar>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py")
  set_tests_properties(integration.iluvatar.runtime PROPERTIES
    LABELS "integration;iluvatar;runtime;cache;stream"
    ENVIRONMENT
      "FLAGDNN_ILUVATAR_COREX_ROOT=${FLAGDNN_ILUVATAR_RESOLVED_COREX_ROOT};FLAGDNN_ILUVATAR_TRITON_JIT_DIR=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_CONFIG_DIR};FLAGDNN_ILUVATAR_TRITON_JIT_LIBRARY=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_LIBRARY};FLAGDNN_ILUVATAR_TRITON_JIT_INCLUDE_DIR=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_INCLUDE_DIR};FLAGDNN_ILUVATAR_TRITON_JIT_SCRIPT_DIR=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_SCRIPT_DIR}"
    RUN_SERIAL TRUE
    TIMEOUT 600)

  add_test(NAME integration.iluvatar.installed_consumer
    COMMAND "${CMAKE_COMMAND}"
      "-DSOURCE_ROOT=${PROJECT_SOURCE_DIR}"
      "-DBUILD_ROOT=${PROJECT_BINARY_DIR}"
      "-DTEST_ROOT=${PROJECT_BINARY_DIR}/installed-consumer/iluvatar"
      "-DCODEGEN_PYTHON=${FLAGDNN_CODEGEN_PYTHON}"
      "-DEXTERNAL_PYTHONPATH=${FLAGDNN_ILUVATAR_RESOLVED_PYTHONPATH}"
      "-DCOREX_ROOT=${FLAGDNN_ILUVATAR_RESOLVED_COREX_ROOT}"
      "-DCOREX_DRIVER=${FLAGDNN_ILUVATAR_RESOLVED_CUDA_DRIVER_LIBRARY}"
      "-DJIT_CONFIG_DIR=/usr/local/lib/cmake/TritonJIT"
      "-DJIT_LIBRARY=/usr/local/lib/libtriton_jit.so"
      "-DJIT_INCLUDE_DIR=/usr/local/include"
      "-DJIT_SCRIPT_DIR=/usr/local/share/triton_jit/scripts"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/InstalledConsumerContract.cmake")
  set_tests_properties(integration.iluvatar.installed_consumer PROPERTIES
    LABELS "integration;iluvatar;install;consumer"
    RUN_SERIAL TRUE
    TIMEOUT 1200)

  add_test(NAME integration.iluvatar.dependency_boundary
    COMMAND "${CMAKE_COMMAND}"
      "-DBUILD_PLUGIN=$<TARGET_FILE:flagdnn_backend_iluvatar>"
      "-DCOREX_ROOT=${FLAGDNN_ILUVATAR_RESOLVED_COREX_ROOT}"
      "-DCOREX_DRIVER=${FLAGDNN_ILUVATAR_RESOLVED_CUDA_DRIVER_LIBRARY}"
      "-DJIT_LIBRARY=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_LIBRARY}"
      "-DJIT_CONFIG_DIR=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_CONFIG_DIR}"
      "-DOUTPUT_JSON=${PROJECT_BINARY_DIR}/qualification/dependency-boundary.json"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/VerifyProductionDependencies.cmake")
  set_tests_properties(integration.iluvatar.dependency_boundary PROPERTIES
    LABELS "integration;iluvatar;dependency;abi"
    RUN_SERIAL TRUE
    TIMEOUT 120)

  include("${CMAKE_CURRENT_SOURCE_DIR}/FunctionalAdapter.cmake")

endif()

if(FLAGDNN_BUILD_BENCHMARKS)
  include("${PROJECT_SOURCE_DIR}/cmake/Operators.cmake")

  set(_flagdnn_iluvatar_benchmark_pointwise
    abs add binary_select ceil cmp_eq cmp_ge cmp_gt cmp_le cmp_lt cmp_neq cos
    div elu erf exp floor gelu gelu_approx_tanh identity leaky_relu log
    logical_and logical_not logical_or max min mod mul neg pow reciprocal relu
    rsqrt scale sigmoid sigmoid_backward sin softplus sqrt sub swish tan tanh)
  set(_flagdnn_iluvatar_benchmark_reduction reduction)
  set(_flagdnn_iluvatar_benchmark_layout reshape slice transpose)
  set(_flagdnn_iluvatar_benchmark_matmul matmul)
  set(_flagdnn_iluvatar_benchmark_convolution
    conv_dgrad conv_fprop conv_wgrad)
  set(_flagdnn_iluvatar_benchmark_normalization
    batchnorm batchnorm_inference layernorm rmsnorm)
  set(_flagdnn_iluvatar_benchmark_graph add_square conv_bias_relu)
  set(_flagdnn_iluvatar_benchmark_partition
    ${_flagdnn_iluvatar_benchmark_pointwise}
    ${_flagdnn_iluvatar_benchmark_reduction}
    ${_flagdnn_iluvatar_benchmark_layout}
    ${_flagdnn_iluvatar_benchmark_matmul}
    ${_flagdnn_iluvatar_benchmark_convolution}
    ${_flagdnn_iluvatar_benchmark_normalization}
    ${_flagdnn_iluvatar_benchmark_graph})
  set(_flagdnn_iluvatar_benchmark_unique
    ${_flagdnn_iluvatar_benchmark_partition})
  list(REMOVE_DUPLICATES _flagdnn_iluvatar_benchmark_unique)
  list(LENGTH _flagdnn_iluvatar_benchmark_partition
    _flagdnn_iluvatar_benchmark_partition_count)
  list(LENGTH _flagdnn_iluvatar_benchmark_unique
    _flagdnn_iluvatar_benchmark_unique_count)
  if(NOT _flagdnn_iluvatar_benchmark_partition_count EQUAL
         _flagdnn_iluvatar_benchmark_unique_count)
    message(FATAL_ERROR "Iluvatar benchmark partition contains duplicates")
  endif()
  set(_flagdnn_iluvatar_benchmark_expected
    ${FLAGDNN_BENCHMARK_OPERATORS})
  list(REMOVE_ITEM _flagdnn_iluvatar_benchmark_expected
    ${FLAGDNN_ACTIVATION_BACKWARD_EXTENSIONS})
  list(SORT _flagdnn_iluvatar_benchmark_unique)
  list(SORT _flagdnn_iluvatar_benchmark_expected)
  if(NOT "${_flagdnn_iluvatar_benchmark_unique}" STREQUAL
         "${_flagdnn_iluvatar_benchmark_expected}")
    message(FATAL_ERROR
      "Iluvatar benchmark partition differs: got=[${_flagdnn_iluvatar_benchmark_unique}] expected=[${_flagdnn_iluvatar_benchmark_expected}]")
  endif()

  add_library(flagdnn_benchmark_iluvatar_adapter STATIC
    benchmark/convolution.cpp
    benchmark/corex_cudnn_capability_policy.cpp
    benchmark/corex_cudnn_provider.cpp
    benchmark/cuda_graph.cpp
    benchmark/graph.cpp
    benchmark/layout.cpp
    benchmark/matmul.cpp
    benchmark/normalization.cpp
    benchmark/pointwise.cpp
    benchmark/reduction.cpp
    benchmark/runner.cpp
    convolution_reference.cpp
    normalization_reference.cpp
    pointwise_reference.cpp
    tensor_reference.cpp
    functional/runner_support.cpp)
  target_compile_features(flagdnn_benchmark_iluvatar_adapter PUBLIC
    cxx_std_20)
  target_compile_definitions(flagdnn_benchmark_iluvatar_adapter PRIVATE
    FLAGDNN_ILUVATAR_CUDNN_CAPABILITY_CATALOG="${CMAKE_CURRENT_SOURCE_DIR}/corex_cudnn_capabilities.json")
  target_include_directories(flagdnn_benchmark_iluvatar_adapter
    SYSTEM PUBLIC
      "${FLAGDNN_ILUVATAR_RESOLVED_CUDA_INCLUDE_DIR}"
      "${FLAGDNN_ILUVATAR_RESOLVED_CUDNN_INCLUDE_DIR}")
  target_include_directories(flagdnn_benchmark_iluvatar_adapter PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/benchmark"
    "${PROJECT_SOURCE_DIR}/tests")
  target_link_libraries(flagdnn_benchmark_iluvatar_adapter PUBLIC
    FlagDNN::flagdnn
    flagdnn_benchmark_cases
    flagdnn_validation_iluvatar_reference
    "${FLAGDNN_ILUVATAR_RESOLVED_CUDNN_LIBRARY}"
    "${FLAGDNN_ILUVATAR_RESOLVED_CUDA_RUNTIME_LIBRARY}"
    "${FLAGDNN_ILUVATAR_RESOLVED_CUDA_DRIVER_LIBRARY}"
    ${CMAKE_DL_LIBS})
  target_link_directories(flagdnn_benchmark_iluvatar_adapter PUBLIC
    "${FLAGDNN_ILUVATAR_RESOLVED_COREX_ROOT}/lib64"
    "${FLAGDNN_ILUVATAR_RESOLVED_COREX_ROOT}/lib")
  flagdnn_enable_warnings(flagdnn_benchmark_iluvatar_adapter)

  add_executable(flagdnn_test_iluvatar_benchmark_runner_contract
    benchmark/capability_contract.cpp
    benchmark/runner_contract.cpp)
  target_compile_features(
    flagdnn_test_iluvatar_benchmark_runner_contract PRIVATE cxx_std_20)
  target_include_directories(
    flagdnn_test_iluvatar_benchmark_runner_contract SYSTEM PRIVATE
    "${FLAGDNN_ILUVATAR_RESOLVED_CUDA_INCLUDE_DIR}")
  target_include_directories(
    flagdnn_test_iluvatar_benchmark_runner_contract PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/benchmark")
  target_link_libraries(
    flagdnn_test_iluvatar_benchmark_runner_contract PRIVATE
    flagdnn_benchmark_iluvatar_adapter
    FlagDNN::flagdnn
    "${FLAGDNN_ILUVATAR_RESOLVED_CUDA_RUNTIME_LIBRARY}")
  flagdnn_enable_warnings(flagdnn_test_iluvatar_benchmark_runner_contract)
  add_test(NAME integration.iluvatar.benchmark_runner_contract
    COMMAND flagdnn_test_iluvatar_benchmark_runner_contract)
  set_tests_properties(
    integration.iluvatar.benchmark_runner_contract PROPERTIES
    LABELS "integration;iluvatar;benchmark;graph;contract"
    TIMEOUT 120)

  get_filename_component(_flagdnn_iluvatar_benchmark_cudnn_directory
    "${FLAGDNN_ILUVATAR_RESOLVED_CUDNN_LIBRARY}" DIRECTORY)
  flagdnn_register_benchmark_suite(
    PLATFORM iluvatar
    OPERATORS ${_flagdnn_iluvatar_benchmark_partition}
    ADAPTER_TARGET flagdnn_benchmark_iluvatar_adapter
    BACKEND_TARGET flagdnn_backend_iluvatar
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_iluvatar>;FLAGDNN_ILUVATAR_COREX_ROOT=${FLAGDNN_ILUVATAR_RESOLVED_COREX_ROOT};FLAGDNN_ILUVATAR_TRITON_JIT_DIR=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_CONFIG_DIR};FLAGDNN_ILUVATAR_TRITON_JIT_LIBRARY=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_LIBRARY};FLAGDNN_ILUVATAR_TRITON_JIT_INCLUDE_DIR=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_INCLUDE_DIR};FLAGDNN_ILUVATAR_TRITON_JIT_SCRIPT_DIR=${FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_SCRIPT_DIR}"
    LABELS "corex;cudnn;cuda_graph;paired"
    TIMEOUT 3600
    BUILD_RPATH "${_flagdnn_iluvatar_benchmark_cudnn_directory}")
  foreach(_operator IN LISTS _flagdnn_iluvatar_benchmark_partition)
    set_tests_properties("benchmark.iluvatar.${_operator}" PROPERTIES
      SKIP_RETURN_CODE 77)
  endforeach()
endif()
