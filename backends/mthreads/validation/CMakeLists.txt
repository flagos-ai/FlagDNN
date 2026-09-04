include("${PROJECT_SOURCE_DIR}/cmake/Operators.cmake")

if(NOT TARGET flagdnn_backend_mthreads)
  message(FATAL_ERROR
    "mthreads validation requires flagdnn_backend_mthreads")
endif()

add_test(
  NAME integration.mthreads.cmake_configuration_contract
  COMMAND
    "${CMAKE_COMMAND}"
    "-DFLAGDNN_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
    "-DFLAGDNN_BINARY_ROOT=${CMAKE_CURRENT_BINARY_DIR}/configuration-contract"
    "-DFLAGDNN_CODEGEN_PYTHON=${FLAGDNN_CODEGEN_PYTHON}"
    "-DFLAGDNN_MUSA_ROOT=${FLAGDNN_MTHREADS_MUSA_ROOT}"
    "-DFLAGDNN_TRITON_JIT_DIR=${FLAGDNN_MTHREADS_TRITON_JIT_DIR}"
    -P
    "${CMAKE_CURRENT_SOURCE_DIR}/integration/VerifyDependencies.cmake")
set_tests_properties(
  integration.mthreads.cmake_configuration_contract
  PROPERTIES
    LABELS "integration;mthreads"
    TIMEOUT 600)

add_executable(
  flagdnn_test_mthreads_runtime
  integration/runtime_contract.cpp)
target_compile_features(
  flagdnn_test_mthreads_runtime PRIVATE cxx_std_20)
target_include_directories(
  flagdnn_test_mthreads_runtime
  PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(
  flagdnn_test_mthreads_runtime
  PRIVATE
    FlagDNNMthreads::musa_driver
    ${CMAKE_DL_LIBS})
flagdnn_enable_warnings(flagdnn_test_mthreads_runtime)
add_dependencies(
  flagdnn_test_mthreads_runtime
  flagdnn_backend_mthreads)

add_executable(
  flagdnn_test_mthreads_request_capture
  integration/request_capture.cpp)
target_compile_features(
  flagdnn_test_mthreads_request_capture PRIVATE cxx_std_20)
target_link_libraries(
  flagdnn_test_mthreads_request_capture
  PRIVATE FlagDNN::flagdnn)
flagdnn_enable_warnings(flagdnn_test_mthreads_request_capture)
add_dependencies(
  flagdnn_test_mthreads_request_capture
  flagdnn_backend_mthreads)

add_executable(
  flagdnn_test_mthreads_artifact
  integration/artifact_contract.cpp
  ${PROJECT_SOURCE_DIR}/backends/mthreads/artifact.cpp
  ${PROJECT_SOURCE_DIR}/backends/mthreads/error.cpp
  ${PROJECT_SOURCE_DIR}/src/runtime/json.cpp
  ${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp)
target_compile_features(
  flagdnn_test_mthreads_artifact PRIVATE cxx_std_20)
target_include_directories(
  flagdnn_test_mthreads_artifact
  PRIVATE
    "${PROJECT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_BINARY_DIR}/generated/include")
target_link_libraries(
  flagdnn_test_mthreads_artifact
  PRIVATE
    FlagDNNMthreads::musa_driver
    FlagDNNMthreads::musa_runtime)
flagdnn_enable_warnings(flagdnn_test_mthreads_artifact)

add_executable(
  flagdnn_test_mthreads_jit_add
  functional/jit_add_contract.cpp)
target_compile_features(
  flagdnn_test_mthreads_jit_add PRIVATE cxx_std_20)
target_include_directories(
  flagdnn_test_mthreads_jit_add
  PRIVATE
    "${FLAGDNN_MTHREADS_VALIDATION_MUDNN_INCLUDE_DIR}")
target_link_libraries(
  flagdnn_test_mthreads_jit_add
  PRIVATE
    FlagDNN::flagdnn
    FlagDNNMthreads::musa_runtime
    FlagDNNMthreads::triton_jit_private
    "${FLAGDNN_MTHREADS_VALIDATION_MUDNN_LIBRARY}")
flagdnn_enable_warnings(flagdnn_test_mthreads_jit_add)
add_dependencies(
  flagdnn_test_mthreads_jit_add
  flagdnn_backend_mthreads)

file(READ
  "${FLAGDNN_MTHREADS_VALIDATION_ENVIRONMENT_REPORT}"
  _flagdnn_mthreads_validation_environment_json)
string(JSON _flagdnn_mthreads_torch_init GET
  "${_flagdnn_mthreads_validation_environment_json}"
  python modules torch realpath)
get_filename_component(
  _flagdnn_mthreads_torch_package
  "${_flagdnn_mthreads_torch_init}"
  DIRECTORY)
set(_flagdnn_mthreads_validation_torch_libraries
    "${_flagdnn_mthreads_torch_package}/lib/libtorch.so"
    "${_flagdnn_mthreads_torch_package}/lib/libtorch_cpu.so"
    "${_flagdnn_mthreads_torch_package}/lib/libc10.so")
foreach(_flagdnn_mthreads_torch_library IN LISTS
    _flagdnn_mthreads_validation_torch_libraries)
  if(NOT EXISTS "${_flagdnn_mthreads_torch_library}")
    message(FATAL_ERROR
      "validated Torch runtime library disappeared: "
      "${_flagdnn_mthreads_torch_library}")
  endif()
endforeach()
target_link_libraries(
  flagdnn_test_mthreads_jit_add
  PRIVATE ${_flagdnn_mthreads_validation_torch_libraries})
string(JSON _flagdnn_mthreads_python_prefix GET
  "${_flagdnn_mthreads_validation_environment_json}"
  python prefix)
string(JSON _flagdnn_mthreads_musa_root GET
  "${_flagdnn_mthreads_validation_environment_json}"
  resources musa_root realpath)
set(_flagdnn_mthreads_test_library_path
    "${_flagdnn_mthreads_python_prefix}/lib"
    "${_flagdnn_mthreads_torch_package}/lib"
    "${_flagdnn_mthreads_musa_root}/lib"
    "/usr/lib/x86_64-linux-gnu"
    "/lib/x86_64-linux-gnu")
list(APPEND _flagdnn_mthreads_test_library_path
  ${FLAGDNN_MTHREADS_VALIDATION_RUNTIME_LIBRARY_DIRECTORIES})
list(REMOVE_DUPLICATES _flagdnn_mthreads_test_library_path)
list(JOIN
  _flagdnn_mthreads_test_library_path
  ":"
  _flagdnn_mthreads_test_library_path_value)

get_filename_component(
  _flagdnn_mthreads_mudnn_library_directory
  "${FLAGDNN_MTHREADS_VALIDATION_MUDNN_LIBRARY}"
  DIRECTORY)

add_library(
  flagdnn_validation_mthreads_support STATIC
  mudnn_add.cpp
  mudnn_attention.cpp
  mudnn_composite.cpp
  mudnn_convolution.cpp
  mudnn_layout.cpp
  mudnn_matmul.cpp
  mudnn_normalization.cpp
  mudnn_pointwise.cpp
  mudnn_reduction.cpp
  musa_driver.cpp
  tensor_io.cpp)
target_compile_features(
  flagdnn_validation_mthreads_support PUBLIC cxx_std_20)
target_include_directories(
  flagdnn_validation_mthreads_support
  PUBLIC "${PROJECT_SOURCE_DIR}")
target_include_directories(
  flagdnn_validation_mthreads_support
  SYSTEM PUBLIC
    "${FLAGDNN_MTHREADS_VALIDATION_MUDNN_INCLUDE_DIR}")
target_link_libraries(
  flagdnn_validation_mthreads_support
  PUBLIC
    FlagDNN::flagdnn
    FlagDNNMthreads::musa_runtime
    "${FLAGDNN_MTHREADS_VALIDATION_MUDNN_LIBRARY}")
flagdnn_enable_warnings(flagdnn_validation_mthreads_support)

if(FLAGDNN_BUILD_TESTS)
  add_library(
    flagdnn_test_mthreads_adapter STATIC
    $<TARGET_OBJECTS:flagdnn_test_common_objects>
    functional/add_runner.cpp
    functional/attention_runner.cpp
    functional/composite_runner.cpp
    functional/convolution_runner.cpp
    functional/layout_runner.cpp
    functional/matmul_runner.cpp
    functional/mudnn_add.cpp
    functional/mudnn_attention.cpp
    functional/mudnn_composite.cpp
    functional/mudnn_convolution.cpp
    functional/mudnn_layout.cpp
    functional/mudnn_matmul.cpp
    functional/mudnn_normalization.cpp
    functional/mudnn_pointwise.cpp
    functional/mudnn_reduction.cpp
    functional/normalization_runner.cpp
    functional/pointwise_runner.cpp
    functional/reduction_runner.cpp
    mudnn_reference.cpp)
  target_compile_features(
    flagdnn_test_mthreads_adapter PUBLIC cxx_std_20)
  target_include_directories(
    flagdnn_test_mthreads_adapter
    PUBLIC
      "${PROJECT_SOURCE_DIR}"
      "${PROJECT_SOURCE_DIR}/tests")
  target_include_directories(
    flagdnn_test_mthreads_adapter
    SYSTEM PUBLIC
      "${FLAGDNN_MTHREADS_VALIDATION_MUDNN_INCLUDE_DIR}")
  target_link_libraries(
    flagdnn_test_mthreads_adapter
    PUBLIC
      FlagDNN::flagdnn
      flagdnn_validation_mthreads_support)
  flagdnn_enable_warnings(flagdnn_test_mthreads_adapter)

  flagdnn_register_functional_suite(
    PLATFORM mthreads
    OPERATORS ${FLAGDNN_FUNCTIONAL_OPERATORS}
    ADAPTER_TARGET flagdnn_test_mthreads_adapter
    BACKEND_TARGET flagdnn_backend_mthreads
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_mthreads>"
      "FLAGDNN_MTHREADS_ENVIRONMENT_REPORT=${FLAGDNN_MTHREADS_VALIDATION_ENVIRONMENT_REPORT}"
      "FLAGDNN_MTHREADS_CONVOLUTION_CACHE_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/convolution-functional-cache"
      "LD_LIBRARY_PATH=${_flagdnn_mthreads_test_library_path_value}"
      "TRITON_JIT_BACKEND=MTGPU"
      "TORCH_DEVICE_BACKEND_AUTOLOAD=0"
      "PYTHONDONTWRITEBYTECODE=1"
    LABELS mudnn
    TIMEOUT 1800
    BUILD_RPATH "${_flagdnn_mthreads_mudnn_library_directory}")
  set_tests_properties(
    functional.mthreads.abs
    functional.mthreads.add
    functional.mthreads.add_square
    functional.mthreads.sdpa
    functional.mthreads.sdpa_backward
    functional.mthreads.sdpa_fp8
    functional.mthreads.sdpa_fp8_backward
    functional.mthreads.batchnorm
    functional.mthreads.batchnorm_inference
    functional.mthreads.binary_select
    functional.mthreads.ceil
    functional.mthreads.cmp_eq
    functional.mthreads.cmp_ge
    functional.mthreads.cmp_gt
    functional.mthreads.cmp_le
    functional.mthreads.cmp_lt
    functional.mthreads.cmp_neq
    functional.mthreads.cos
    functional.mthreads.conv_bias_relu
    functional.mthreads.conv_dgrad
    functional.mthreads.conv_fprop
    functional.mthreads.conv_wgrad
    functional.mthreads.div
    functional.mthreads.elu
    functional.mthreads.erf
    functional.mthreads.exp
    functional.mthreads.floor
    functional.mthreads.gelu
    functional.mthreads.gelu_approx_tanh
    functional.mthreads.identity
    functional.mthreads.leaky_relu
    functional.mthreads.layernorm
    functional.mthreads.log
    functional.mthreads.logical_and
    functional.mthreads.logical_not
    functional.mthreads.logical_or
    functional.mthreads.max
    functional.mthreads.matmul
    functional.mthreads.min
    functional.mthreads.mod
    functional.mthreads.mul
    functional.mthreads.neg
    functional.mthreads.pow
    functional.mthreads.reciprocal
    functional.mthreads.reduction
    functional.mthreads.relu
    functional.mthreads.reshape
    functional.mthreads.rmsnorm
    functional.mthreads.rsqrt
    functional.mthreads.scale
    functional.mthreads.sigmoid
    functional.mthreads.sigmoid_backward
    functional.mthreads.sin
    functional.mthreads.slice
    functional.mthreads.softplus
    functional.mthreads.sqrt
    functional.mthreads.sub
    functional.mthreads.swish
    functional.mthreads.tan
    functional.mthreads.tanh
    functional.mthreads.transpose
    PROPERTIES RESOURCE_LOCK mthreads_gpu)
  set_tests_properties(
    functional.mthreads.conv_bias_relu
    functional.mthreads.conv_dgrad
    functional.mthreads.conv_fprop
    functional.mthreads.conv_wgrad
    functional.mthreads.batchnorm
    functional.mthreads.batchnorm_inference
    functional.mthreads.layernorm
    functional.mthreads.rmsnorm
    functional.mthreads.sdpa
    functional.mthreads.sdpa_backward
    functional.mthreads.sdpa_fp8
    functional.mthreads.sdpa_fp8_backward
    PROPERTIES TIMEOUT 7200)
endif()

if(FLAGDNN_BUILD_BENCHMARKS)
  add_library(
    flagdnn_benchmark_mthreads_adapter STATIC
    benchmark/add_runner.cpp)
  target_compile_features(
    flagdnn_benchmark_mthreads_adapter PUBLIC cxx_std_20)
  target_include_directories(
    flagdnn_benchmark_mthreads_adapter
    PUBLIC
      "${PROJECT_SOURCE_DIR}"
      "${PROJECT_SOURCE_DIR}/benchmark")
  target_link_libraries(
    flagdnn_benchmark_mthreads_adapter
    PUBLIC
      flagdnn_benchmark_cases
      flagdnn_validation_mthreads_support)
  flagdnn_enable_warnings(flagdnn_benchmark_mthreads_adapter)

  flagdnn_register_benchmark_suite(
    PLATFORM mthreads
    OPERATORS ${FLAGDNN_BENCHMARK_OPERATORS}
    ADAPTER_TARGET flagdnn_benchmark_mthreads_adapter
    BACKEND_TARGET flagdnn_backend_mthreads
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_mthreads>"
      "FLAGDNN_MTHREADS_ENVIRONMENT_REPORT=${FLAGDNN_MTHREADS_VALIDATION_ENVIRONMENT_REPORT}"
      "FLAGDNN_BENCHMARK_CACHE_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/benchmark-cache"
      "FLAGDNN_BENCHMARK_AUTOTUNE=1"
      "LD_LIBRARY_PATH=${_flagdnn_mthreads_test_library_path_value}"
      "TRITON_JIT_BACKEND=MTGPU"
      "TORCH_DEVICE_BACKEND_AUTOLOAD=0"
      "PYTHONDONTWRITEBYTECODE=1"
    LABELS mudnn
    TIMEOUT 7200
    BUILD_RPATH "${_flagdnn_mthreads_mudnn_library_directory}")
  set_tests_properties(
    benchmark.mthreads.abs
    benchmark.mthreads.add
    benchmark.mthreads.add_square
    benchmark.mthreads.batchnorm
    benchmark.mthreads.batchnorm_inference
    benchmark.mthreads.binary_select
    benchmark.mthreads.ceil
    benchmark.mthreads.cmp_eq
    benchmark.mthreads.cmp_ge
    benchmark.mthreads.cmp_gt
    benchmark.mthreads.cmp_le
    benchmark.mthreads.cmp_lt
    benchmark.mthreads.cmp_neq
    benchmark.mthreads.cos
    benchmark.mthreads.conv_bias_relu
    benchmark.mthreads.conv_dgrad
    benchmark.mthreads.conv_fprop
    benchmark.mthreads.conv_wgrad
    benchmark.mthreads.div
    benchmark.mthreads.elu
    benchmark.mthreads.erf
    benchmark.mthreads.exp
    benchmark.mthreads.floor
    benchmark.mthreads.gelu
    benchmark.mthreads.gelu_approx_tanh
    benchmark.mthreads.identity
    benchmark.mthreads.leaky_relu
    benchmark.mthreads.layernorm
    benchmark.mthreads.log
    benchmark.mthreads.logical_and
    benchmark.mthreads.logical_not
    benchmark.mthreads.logical_or
    benchmark.mthreads.max
    benchmark.mthreads.matmul
    benchmark.mthreads.min
    benchmark.mthreads.mod
    benchmark.mthreads.mul
    benchmark.mthreads.neg
    benchmark.mthreads.pow
    benchmark.mthreads.reciprocal
    benchmark.mthreads.reduction
    benchmark.mthreads.relu
    benchmark.mthreads.reshape
    benchmark.mthreads.rmsnorm
    benchmark.mthreads.rsqrt
    benchmark.mthreads.scale
    benchmark.mthreads.sigmoid
    benchmark.mthreads.sigmoid_backward
    benchmark.mthreads.sin
    benchmark.mthreads.slice
    benchmark.mthreads.softplus
    benchmark.mthreads.sqrt
    benchmark.mthreads.sub
    benchmark.mthreads.swish
    benchmark.mthreads.tan
    benchmark.mthreads.tanh
    benchmark.mthreads.transpose
    PROPERTIES RESOURCE_LOCK mthreads_gpu)
endif()

add_test(
  NAME integration.mthreads.dependency_boundary
  COMMAND
    "${CMAKE_COMMAND}"
    "-DBACKEND_LIBRARY=$<TARGET_FILE:flagdnn_backend_mthreads>"
    "-DBACKEND_NAME=mthreads"
    "-DREQUIRED_DEPENDENCIES=libmusa.so;libmusart;libtriton_jit;libpython"
    "-DFORBIDDEN_DEPENDENCIES=libmudnn;libcudnn;libcublas;libhipdnn;libMIOpen;libmiopen;librocblas;libhipblas;libacl;libnnopbase;libopapi;libtorch"
    -P "${PROJECT_SOURCE_DIR}/cmake/VerifyBackendPlugin.cmake")
set_tests_properties(
  integration.mthreads.dependency_boundary
  PROPERTIES
    LABELS "integration;mthreads;dependency;abi"
    TIMEOUT 120)

if(FLAGDNN_BUILD_TESTS)
  add_test(
    NAME integration.mthreads.reference_dependency_boundary
    COMMAND
      "${CMAKE_COMMAND}"
      "-DREFERENCE_EXECUTABLE=$<TARGET_FILE:flagdnn_test_mthreads_add>"
      -P
        "${CMAKE_CURRENT_SOURCE_DIR}/integration/VerifyReferenceDependencies.cmake")
  set_tests_properties(
    integration.mthreads.reference_dependency_boundary
    PROPERTIES
      LABELS "integration;mthreads;mudnn;dependency"
      TIMEOUT 120)

  add_test(
    NAME integration.mthreads.installed_consumer
    COMMAND
      "${CMAKE_COMMAND}"
      "-DSOURCE_ROOT=${PROJECT_SOURCE_DIR}"
      "-DBUILD_ROOT=${PROJECT_BINARY_DIR}"
      "-DBUILD_PLUGIN=$<TARGET_FILE:flagdnn_backend_mthreads>"
      "-DPLUGIN_SONAME=$<TARGET_SONAME_FILE_NAME:flagdnn_backend_mthreads>"
      "-DJIT_SONAME=${FLAGDNN_MTHREADS_VALIDATION_JIT_SONAME}"
      "-DJIT_SHA256=${FLAGDNN_MTHREADS_VALIDATION_JIT_PRIVATE_SHA256}"
      "-DJIT_PROVENANCE_SHA256=${FLAGDNN_MTHREADS_VALIDATION_JIT_PROVENANCE_SHA256}"
      "-DSTANDALONE_SHA256=${FLAGDNN_MTHREADS_VALIDATION_JIT_STANDALONE_SHA256}"
      "-DGEN_SSIG_SHA256=${FLAGDNN_MTHREADS_VALIDATION_JIT_GEN_SSIG_SHA256}"
      "-DENVIRONMENT_IDENTITY=${FLAGDNN_MTHREADS_VALIDATION_ENVIRONMENT_IDENTITY}"
      "-DTEST_ROOT=${PROJECT_BINARY_DIR}/installed-consumer-contract/mthreads"
      "-DCODEGEN_PYTHON=${FLAGDNN_CODEGEN_PYTHON}"
      "-DINSTALL_CONFIG=$<CONFIG>"
      "-DMUSA_ROOT=${FLAGDNN_MTHREADS_VALIDATION_MUSA_ROOT}"
      "-DMUDNN_INCLUDE_DIR=${FLAGDNN_MTHREADS_VALIDATION_MUDNN_INCLUDE_DIR}"
      "-DMUDNN_LIBRARY=${FLAGDNN_MTHREADS_VALIDATION_MUDNN_LIBRARY}"
      "-DVALIDATION_LIBRARY_PATH=${_flagdnn_mthreads_test_library_path_value}"
      -P
        "${CMAKE_CURRENT_SOURCE_DIR}/integration/InstalledConsumerContract.cmake")
  set_tests_properties(
    integration.mthreads.installed_consumer
    PROPERTIES
      LABELS
        "integration;mthreads;install;consumer;mudnn;libtriton_jit;autotune;graph"
      RUN_SERIAL TRUE
      TIMEOUT 1800
      ENVIRONMENT
        "LD_LIBRARY_PATH=${_flagdnn_mthreads_test_library_path_value};TRITON_JIT_BACKEND=MTGPU;TORCH_DEVICE_BACKEND_AUTOLOAD=0;PYTHONDONTWRITEBYTECODE=1")
endif()

if(FLAGDNN_BUILD_BENCHMARKS)
  add_test(
    NAME integration.mthreads.benchmark_reference_dependency_boundary
    COMMAND
      "${CMAKE_COMMAND}"
      "-DREFERENCE_EXECUTABLE=$<TARGET_FILE:flagdnn_benchmark_mthreads_add>"
      -P
        "${CMAKE_CURRENT_SOURCE_DIR}/integration/VerifyReferenceDependencies.cmake")
  set_tests_properties(
    integration.mthreads.benchmark_reference_dependency_boundary
    PROPERTIES
      LABELS "integration;mthreads;mudnn;benchmark;dependency"
      TIMEOUT 120)
endif()

add_test(
  NAME integration.mthreads.run_tests_adapter_contract
  COMMAND
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${CMAKE_CURRENT_SOURCE_DIR}/integration/run_tests_adapter_contract.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/run_tests_adapter.py")
set_tests_properties(
  integration.mthreads.run_tests_adapter_contract
  PROPERTIES
    LABELS "integration;mthreads;runner;contract"
    TIMEOUT 60
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")

add_test(
  NAME integration.mthreads.runtime
  COMMAND
    flagdnn_test_mthreads_runtime
    "$<TARGET_FILE:flagdnn_backend_mthreads>")
set_tests_properties(
  integration.mthreads.runtime
  PROPERTIES
    LABELS "integration;mthreads"
    TIMEOUT 300
    ENVIRONMENT
      "LD_LIBRARY_PATH=${_flagdnn_mthreads_test_library_path_value};TRITON_JIT_BACKEND=MTGPU;TORCH_DEVICE_BACKEND_AUTOLOAD=0")

add_test(
  NAME integration.mthreads.compiler_contract
  COMMAND
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${CMAKE_CURRENT_SOURCE_DIR}/integration/compiler_contract.py"
    --provider "${PROJECT_SOURCE_DIR}/backends/mthreads/compiler.py"
    --compiler "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
    --target "musa-mtgpu-cc31-w32"
    --environment-report
      "${FLAGDNN_MTHREADS_VALIDATION_ENVIRONMENT_REPORT}"
    --capture-executable
      "$<TARGET_FILE:flagdnn_test_mthreads_request_capture>"
    --capture-compiler
      "${CMAKE_CURRENT_SOURCE_DIR}/integration/capture_compiler.py"
    --python "${FLAGDNN_CODEGEN_PYTHON}")
set_tests_properties(
  integration.mthreads.compiler_contract
  PROPERTIES
    LABELS "integration;mthreads"
    TIMEOUT 900
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_mthreads>;FLAGDNN_EXECUTION_ENGINE=libtriton_jit;FLAGDNN_MTHREADS_ENVIRONMENT_REPORT=${FLAGDNN_MTHREADS_VALIDATION_ENVIRONMENT_REPORT};LD_LIBRARY_PATH=${_flagdnn_mthreads_test_library_path_value};TRITON_JIT_BACKEND=MTGPU;TORCH_DEVICE_BACKEND_AUTOLOAD=0;PYTHONDONTWRITEBYTECODE=1")

add_test(
  NAME integration.mthreads.artifact_contract
  COMMAND
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${CMAKE_CURRENT_SOURCE_DIR}/integration/artifact_mutation_contract.py"
    --provider "${PROJECT_SOURCE_DIR}/backends/mthreads/compiler.py"
    --compiler "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
    --parser "$<TARGET_FILE:flagdnn_test_mthreads_artifact>"
    --capture-executable
      "$<TARGET_FILE:flagdnn_test_mthreads_request_capture>"
    --capture-compiler
      "${CMAKE_CURRENT_SOURCE_DIR}/integration/capture_compiler.py"
    --python "${FLAGDNN_CODEGEN_PYTHON}"
    --target "musa-mtgpu-cc31-w32"
    --flagdnn-version "${PROJECT_VERSION}")
set_tests_properties(
  integration.mthreads.artifact_contract
  PROPERTIES
    LABELS "integration;mthreads"
    TIMEOUT 900
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_mthreads>;FLAGDNN_BACKEND_ROOT=${PROJECT_SOURCE_DIR}/backends;FLAGDNN_MTHREADS_ENVIRONMENT_REPORT=${FLAGDNN_MTHREADS_VALIDATION_ENVIRONMENT_REPORT};LD_LIBRARY_PATH=${_flagdnn_mthreads_test_library_path_value};TRITON_JIT_BACKEND=MTGPU;TORCH_DEVICE_BACKEND_AUTOLOAD=0;PYTHONDONTWRITEBYTECODE=1")

add_test(
  NAME integration.mthreads.jit_add
  COMMAND
    flagdnn_test_mthreads_jit_add
    "$<TARGET_FILE:flagdnn_backend_mthreads>"
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py")
set_tests_properties(
  integration.mthreads.jit_add
  PROPERTIES
    LABELS "functional;integration;mthreads"
    TIMEOUT 1800
    RESOURCE_LOCK mthreads_gpu
    ENVIRONMENT
      "FLAGDNN_MTHREADS_ENVIRONMENT_REPORT=${FLAGDNN_MTHREADS_VALIDATION_ENVIRONMENT_REPORT};LD_LIBRARY_PATH=${_flagdnn_mthreads_test_library_path_value};TRITON_JIT_BACKEND=MTGPU;TORCH_DEVICE_BACKEND_AUTOLOAD=0;PYTHONDONTWRITEBYTECODE=1")

add_test(
  NAME unit.mthreads.environment_identity
  COMMAND
    "${FLAGDNN_CODEGEN_PYTHON}"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_environment_identity.py")
set_tests_properties(
  unit.mthreads.environment_identity
  PROPERTIES
    LABELS "unit;mthreads;environment"
    TIMEOUT 120
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
