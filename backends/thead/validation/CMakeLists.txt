# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET flagdnn_backend_thead)
  message(FATAL_ERROR "THead backend target is missing")
endif()

foreach(_property IN ITEMS
    FLAGDNN_THEAD_PPU_SDK_ROOT
    FLAGDNN_THEAD_PPU_SDK_VERSION
    FLAGDNN_THEAD_CUDA_DRIVER_LIBRARY
    FLAGDNN_THEAD_TRITON_JIT_ROOT
    FLAGDNN_THEAD_TRITON_JIT_LIBRARY
    FLAGDNN_THEAD_TRITON_JIT_SONAME
    FLAGDNN_THEAD_TRITON_JIT_SHA256
    FLAGDNN_THEAD_TRITON_JIT_PROVENANCE_SHA256
    FLAGDNN_THEAD_TRITON_JIT_STANDALONE_SHA256
    FLAGDNN_THEAD_TRITON_JIT_GEN_SSIG_SHA256
    FLAGDNN_THEAD_TRITON_ROOT
    FLAGDNN_THEAD_TRITON_METADATA
    FLAGDNN_THEAD_TRITON_JIT_INCLUDE_DIR
    FLAGDNN_THEAD_TORCH_INCLUDE
    FLAGDNN_THEAD_TORCH_API_INCLUDE
    FLAGDNN_THEAD_TORCH_CXX11_ABI)
  get_target_property(_value flagdnn_backend_thead "${_property}")
  if(NOT _value OR _value MATCHES "-NOTFOUND$")
    message(FATAL_ERROR
      "THead backend target has no validation property ${_property}")
  endif()
  set(_${_property} "${_value}")
endforeach()

set(_flagdnn_thead_codegen_environment
  "PYTHONDONTWRITEBYTECODE=1;FLAGDNN_THEAD_PPU_SDK_ROOT=${_FLAGDNN_THEAD_PPU_SDK_ROOT};FLAGDNN_THEAD_TRITON_JIT_ROOT=${_FLAGDNN_THEAD_TRITON_JIT_ROOT};FLAGDNN_THEAD_TRITON_ROOT=${_FLAGDNN_THEAD_TRITON_ROOT};PPU_SDK=${_FLAGDNN_THEAD_PPU_SDK_ROOT};PPU_HOME=${_FLAGDNN_THEAD_PPU_SDK_ROOT};CUDA_PATH=${_FLAGDNN_THEAD_PPU_SDK_ROOT}/CUDA_SDK;TRITON_PTXAS_PATH=${_FLAGDNN_THEAD_PPU_SDK_ROOT}/CUDA_SDK/bin/ptxas;TRITON_IR_FORMATTER_PATH=${_FLAGDNN_THEAD_PPU_SDK_ROOT}/bin/llvm-irformatter;TRITON_JIT_BACKEND=CUDA")

if(FLAGDNN_BUILD_TESTS OR FLAGDNN_BUILD_BENCHMARKS)
  if(NOT TARGET FlagDNN::THeadAcdnn)
    message(FATAL_ERROR
      "THead validation requires the selected PPU SDK acDNN target")
  endif()
  get_target_property(_flagdnn_thead_acdnn_library
    FlagDNN::THeadAcdnn IMPORTED_LOCATION)
  if(NOT _flagdnn_thead_acdnn_library OR
     _flagdnn_thead_acdnn_library MATCHES "-NOTFOUND$")
    message(FATAL_ERROR "THead acDNN target has no imported library")
  endif()

  add_library(flagdnn_validation_thead_reference STATIC
    acdnn_convolution_reference.cpp
    acdnn_composite_reference.cpp
    acdnn_layout_reference.cpp
    acdnn_matmul_reference.cpp
    acdnn_normalization_reference.cpp
    acdnn_reduction_reference.cpp
    acdnn_reference.cpp
    backend_pointwise_reference.cpp
    capability.cpp
    numeric_types.cpp
    pointwise_reference.cpp
    tensor_io.cpp
    "${PROJECT_SOURCE_DIR}/src/runtime/json.cpp")
  target_compile_features(flagdnn_validation_thead_reference PUBLIC
    cxx_std_20)
  target_compile_definitions(flagdnn_validation_thead_reference PRIVATE
    FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG="${CMAKE_CURRENT_SOURCE_DIR}/capability.json")
  target_include_directories(flagdnn_validation_thead_reference PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/tests"
    PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}")
  target_link_libraries(flagdnn_validation_thead_reference PUBLIC
    FlagDNN::flagdnn
    FlagDNN::THeadAcdnn
    FlagDNN::THeadCudaDriver)
  flagdnn_enable_warnings(flagdnn_validation_thead_reference)
endif()

add_test(NAME integration.thead.cmake_configuration_contract
  COMMAND "${CMAKE_COMMAND}"
    "-DSOURCE_ROOT=${PROJECT_SOURCE_DIR}"
    "-DTEST_ROOT=${PROJECT_BINARY_DIR}/cmake-contract/thead"
    "-DPPU_SDK_ROOT=${_FLAGDNN_THEAD_PPU_SDK_ROOT}"
    "-DTRITON_JIT_ROOT=${_FLAGDNN_THEAD_TRITON_JIT_ROOT}"
    "-DTRITON_ROOT=${_FLAGDNN_THEAD_TRITON_ROOT}"
    "-DPYTHON_EXECUTABLE=${FLAGDNN_CODEGEN_PYTHON}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/CMakeConfigurationContract.cmake")
set_tests_properties(integration.thead.cmake_configuration_contract PROPERTIES
  LABELS "integration;thead;cmake;contract"
  RUN_SERIAL TRUE
  TIMEOUT 120)

add_test(NAME integration.thead.dependency_boundary
  COMMAND "${CMAKE_COMMAND}"
    "-DBACKEND_LIBRARY=$<TARGET_FILE:flagdnn_backend_thead>"
    "-DBACKEND_NAME=thead"
    "-DBACKEND_ABI_VERSION=2"
    "-DREQUIRED_DEPENDENCIES=libcuda.so.1;libtriton_jit.so"
    "-DFORBIDDEN_DEPENDENCIES=libacdnn;libacblas;libcublas;libcudnn;libblas;liblapack;libflagdnn_backend_nvidia;libflagdnn_backend_hygon;libflagdnn_backend_iluvatar;libflagdnn_backend_ascend"
    -P "${PROJECT_SOURCE_DIR}/cmake/VerifyBackendPlugin.cmake")
set_tests_properties(integration.thead.dependency_boundary PROPERTIES
  LABELS "integration;thead;dependency;contract"
  RUN_SERIAL TRUE)

set(_flagdnn_thead_reference_dependency_arguments
  "-DBACKEND_LIBRARY=$<TARGET_FILE:flagdnn_backend_thead>"
  "-DSOURCE_ROOT=${PROJECT_SOURCE_DIR}")
if(FLAGDNN_BUILD_TESTS)
  set(_flagdnn_thead_extended_unary_operations
    leaky_relu log cos rsqrt sin tan softplus swish gelu_approx_tanh div pow
    sigmoid_backward reciprocal add_square cmp_eq cmp_neq cmp_gt cmp_ge
    cmp_lt cmp_le)
  list(APPEND _flagdnn_thead_reference_dependency_arguments
    "-DREFERENCE_EXECUTABLE=$<TARGET_FILE:flagdnn_test_thead_validation_contract>"
    "-DPPU_SDK_ROOT=${_FLAGDNN_THEAD_PPU_SDK_ROOT}"
    "-DACDNN_LIBRARY=${_flagdnn_thead_acdnn_library}"
    "-DCUDA_DRIVER_LIBRARY=${_FLAGDNN_THEAD_CUDA_DRIVER_LIBRARY}")
endif()
add_test(NAME integration.thead.reference_dependency_boundary
  COMMAND "${CMAKE_COMMAND}"
    ${_flagdnn_thead_reference_dependency_arguments}
    -P "${CMAKE_CURRENT_SOURCE_DIR}/VerifyReferenceDependencies.cmake")
unset(_flagdnn_thead_reference_dependency_arguments)
set_tests_properties(
  integration.thead.reference_dependency_boundary PROPERTIES
  LABELS "integration;thead;acdnn;dependency;contract"
  RUN_SERIAL TRUE)

add_test(NAME integration.thead.run_tests_adapter_contract
  COMMAND "${FLAGDNN_CODEGEN_PYTHON}"
    "${CMAKE_CURRENT_SOURCE_DIR}/run_tests_adapter_contract.py")
set_tests_properties(
  integration.thead.run_tests_adapter_contract PROPERTIES
  LABELS "integration;thead;runner;adapter;contract"
  ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
  RUN_SERIAL TRUE
  TIMEOUT 120)

add_test(NAME integration.thead.catalog_closure_contract
  COMMAND "${CMAKE_COMMAND}"
    "-DSOURCE_ROOT=${PROJECT_SOURCE_DIR}"
    "-DBUILD_ROOT=${PROJECT_BINARY_DIR}"
    "-DCTEST_COMMAND=${CMAKE_CTEST_COMMAND}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/CatalogClosureContract.cmake")
set_tests_properties(
  integration.thead.catalog_closure_contract PROPERTIES
  LABELS "integration;thead;catalog;closure;contract"
  RUN_SERIAL TRUE
  TIMEOUT 60)

if(FLAGDNN_BUILD_TESTS)
  add_test(NAME integration.thead.installed_consumer
    COMMAND "${CMAKE_COMMAND}"
      "-DSOURCE_ROOT=${PROJECT_SOURCE_DIR}"
      "-DBUILD_ROOT=${PROJECT_BINARY_DIR}"
      "-DTEST_ROOT=${PROJECT_BINARY_DIR}/installed-consumer-contract/thead"
      "-DCODEGEN_PYTHON=${FLAGDNN_CODEGEN_PYTHON}"
      "-DPPU_SDK_ROOT=${_FLAGDNN_THEAD_PPU_SDK_ROOT}"
      "-DBUILD_PLUGIN=$<TARGET_FILE:flagdnn_backend_thead>"
      "-DPLUGIN_SONAME=$<TARGET_SONAME_FILE_NAME:flagdnn_backend_thead>"
      "-DJIT_ROOT=${_FLAGDNN_THEAD_TRITON_JIT_ROOT}"
      "-DJIT_SONAME=${_FLAGDNN_THEAD_TRITON_JIT_SONAME}"
      "-DJIT_SHA256=${_FLAGDNN_THEAD_TRITON_JIT_SHA256}"
      "-DJIT_PROVENANCE_SHA256=${_FLAGDNN_THEAD_TRITON_JIT_PROVENANCE_SHA256}"
      "-DSTANDALONE_SHA256=${_FLAGDNN_THEAD_TRITON_JIT_STANDALONE_SHA256}"
      "-DGEN_SSIG_SHA256=${_FLAGDNN_THEAD_TRITON_JIT_GEN_SSIG_SHA256}"
      "-DTRITON_ROOT=${_FLAGDNN_THEAD_TRITON_ROOT}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/InstalledConsumerContract.cmake")
  set_tests_properties(integration.thead.installed_consumer PROPERTIES
    LABELS "integration;thead;install;consumer;acdnn;libtriton_jit;autotune"
    RUN_SERIAL TRUE
    TIMEOUT 1800)

  add_executable(flagdnn_test_thead_validation_contract
    validation_contract.cpp
    $<TARGET_OBJECTS:flagdnn_test_common_objects>)
  target_compile_features(flagdnn_test_thead_validation_contract PRIVATE
    cxx_std_20)
  target_compile_definitions(flagdnn_test_thead_validation_contract PRIVATE
    FLAGDNN_THEAD_PPU_SDK_VERSION="${_FLAGDNN_THEAD_PPU_SDK_VERSION}")
  target_include_directories(flagdnn_test_thead_validation_contract PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}")
  target_link_libraries(flagdnn_test_thead_validation_contract PRIVATE
    FlagDNN::flagdnn
    flagdnn_validation_thead_reference)
  set_target_properties(flagdnn_test_thead_validation_contract PROPERTIES
    BUILD_RPATH
      "${_FLAGDNN_THEAD_PPU_SDK_ROOT}/lib;${_FLAGDNN_THEAD_PPU_SDK_ROOT}/CUDA_SDK/lib64")
  flagdnn_enable_warnings(flagdnn_test_thead_validation_contract)

  add_test(NAME integration.thead.validation_contract
    COMMAND flagdnn_test_thead_validation_contract --api-contract)
  set_tests_properties(integration.thead.validation_contract PROPERTIES
    LABELS "integration;thead;acdnn;validation;contract"
    RUN_SERIAL TRUE
    TIMEOUT 120)

  add_test(NAME integration.thead.capability_contract
    COMMAND flagdnn_test_thead_validation_contract
      --catalog-contract
      "${CMAKE_CURRENT_SOURCE_DIR}/capability.json"
      "${PROJECT_SOURCE_DIR}/cmake/Operators.cmake"
      "${_FLAGDNN_THEAD_PPU_SDK_VERSION}")
  set_tests_properties(integration.thead.capability_contract PROPERTIES
    LABELS "integration;thead;acdnn;capability;catalog"
    RUN_SERIAL TRUE
    TIMEOUT 120)

  add_executable(flagdnn_test_thead_artifact
    functional/test_artifact.cpp
    "${PROJECT_SOURCE_DIR}/backends/thead/artifact.cpp"
    "${PROJECT_SOURCE_DIR}/backends/thead/error.cpp"
    "${PROJECT_SOURCE_DIR}/src/runtime/json.cpp"
    "${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp")
  target_compile_features(flagdnn_test_thead_artifact PRIVATE cxx_std_20)
  target_include_directories(flagdnn_test_thead_artifact PRIVATE
    "${PROJECT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_BINARY_DIR}/generated/include")
  target_link_libraries(flagdnn_test_thead_artifact PRIVATE
    FlagDNN::THeadCudaDriver)
  flagdnn_enable_warnings(flagdnn_test_thead_artifact)
  add_test(NAME integration.thead.artifact_contract
    COMMAND flagdnn_test_thead_artifact)
  set_tests_properties(integration.thead.artifact_contract PROPERTIES
    LABELS "integration;thead;artifact;contract"
    RUN_SERIAL TRUE
    TIMEOUT 60)

  add_test(NAME integration.thead.compiler_contract
    COMMAND "${FLAGDNN_CODEGEN_PYTHON}"
      "${CMAKE_CURRENT_SOURCE_DIR}/compiler_contract.py")
  set_tests_properties(integration.thead.compiler_contract PROPERTIES
    LABELS "integration;thead;compiler;identity;contract"
    ENVIRONMENT
      "${_flagdnn_thead_codegen_environment}"
    RUN_SERIAL TRUE
    TIMEOUT 180)

  add_executable(flagdnn_test_thead_jit
    functional/test_jit.cpp)
  target_compile_features(flagdnn_test_thead_jit PRIVATE cxx_std_20)
  target_include_directories(flagdnn_test_thead_jit PRIVATE
    "${PROJECT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_BINARY_DIR}/generated/include")
  target_include_directories(flagdnn_test_thead_jit SYSTEM PRIVATE
    "${_FLAGDNN_THEAD_TRITON_JIT_INCLUDE_DIR}"
    "${_FLAGDNN_THEAD_TORCH_INCLUDE}"
    "${_FLAGDNN_THEAD_TORCH_API_INCLUDE}")
  target_compile_definitions(flagdnn_test_thead_jit PRIVATE
    _GLIBCXX_USE_CXX11_ABI=${_FLAGDNN_THEAD_TORCH_CXX11_ABI})
  target_link_libraries(flagdnn_test_thead_jit PRIVATE
    FlagDNN::flagdnn
    FlagDNN::THeadCudaDriver
    FlagDNN::THeadTritonJIT
    "${CMAKE_DL_LIBS}")
  set_property(TARGET flagdnn_test_thead_jit APPEND PROPERTY BUILD_RPATH
    "$<TARGET_FILE_DIR:flagdnn_backend_thead>/flagdnn/thead")
  if(UNIX AND NOT APPLE)
    target_link_options(flagdnn_test_thead_jit PRIVATE
      "LINKER:--disable-new-dtags")
  endif()
  add_dependencies(flagdnn_test_thead_jit flagdnn_backend_thead)
  flagdnn_enable_warnings(flagdnn_test_thead_jit)
  add_test(NAME integration.thead.jit
    COMMAND flagdnn_test_thead_jit
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py")
  add_test(NAME integration.thead.jit_mul
    COMMAND flagdnn_test_thead_jit
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation mul)
  add_test(NAME integration.thead.jit_sub
    COMMAND flagdnn_test_thead_jit
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation sub)
  add_test(NAME integration.thead.jit_min
    COMMAND flagdnn_test_thead_jit
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation min)
  add_test(NAME integration.thead.jit_max
    COMMAND flagdnn_test_thead_jit
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation max)
  add_test(NAME integration.thead.jit_scale
    COMMAND flagdnn_test_thead_jit
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation scale)
  add_test(NAME integration.thead.jit_relu
    COMMAND flagdnn_test_thead_jit
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation relu)
  add_test(NAME integration.thead.jit_sigmoid
    COMMAND flagdnn_test_thead_jit
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation sigmoid)
  add_test(NAME integration.thead.jit_tanh
    COMMAND flagdnn_test_thead_jit
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation tanh)
  add_test(NAME integration.thead.jit_elu
    COMMAND flagdnn_test_thead_jit
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation elu)
  add_test(NAME integration.thead.jit_identity
    COMMAND flagdnn_test_thead_jit
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation identity)
  add_test(NAME integration.thead.jit_gelu
    COMMAND flagdnn_test_thead_jit
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation gelu)
  add_test(NAME integration.thead.jit_sqrt
    COMMAND flagdnn_test_thead_jit
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation sqrt)
  add_test(NAME integration.thead.jit_neg
    COMMAND flagdnn_test_thead_jit
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation neg)
  add_test(NAME integration.thead.jit_abs
    COMMAND flagdnn_test_thead_jit
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation abs)
  add_test(NAME integration.thead.jit_ceil
    COMMAND flagdnn_test_thead_jit
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation ceil)
  add_test(NAME integration.thead.jit_floor
    COMMAND flagdnn_test_thead_jit
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation floor)
  add_test(NAME integration.thead.jit_exp
    COMMAND flagdnn_test_thead_jit
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation exp)
  set(_flagdnn_thead_extended_jit_tests)
  foreach(_flagdnn_thead_operation IN LISTS
      _flagdnn_thead_extended_unary_operations)
    add_test(NAME "integration.thead.jit_${_flagdnn_thead_operation}"
      COMMAND flagdnn_test_thead_jit
        "$<TARGET_FILE:flagdnn_backend_thead>"
        "${FLAGDNN_CODEGEN_PYTHON}"
        "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
        --operation "${_flagdnn_thead_operation}")
    list(APPEND _flagdnn_thead_extended_jit_tests
      "integration.thead.jit_${_flagdnn_thead_operation}")
  endforeach()
  set_tests_properties(
    integration.thead.jit integration.thead.jit_mul
    integration.thead.jit_sub integration.thead.jit_min
    integration.thead.jit_max integration.thead.jit_scale
    integration.thead.jit_relu integration.thead.jit_sigmoid
    integration.thead.jit_tanh integration.thead.jit_elu
    integration.thead.jit_identity integration.thead.jit_gelu
    integration.thead.jit_sqrt integration.thead.jit_neg
    integration.thead.jit_abs integration.thead.jit_ceil
    integration.thead.jit_floor integration.thead.jit_exp
    ${_flagdnn_thead_extended_jit_tests}
    PROPERTIES
    LABELS "integration;thead;jit;graph"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_thead>;${_flagdnn_thead_codegen_environment}"
    RUN_SERIAL TRUE
    TIMEOUT 300)

  add_executable(flagdnn_test_thead_graph
    functional/test_graph.cpp)
  target_compile_features(flagdnn_test_thead_graph PRIVATE cxx_std_20)
  target_include_directories(flagdnn_test_thead_graph PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_BINARY_DIR}/generated/include")
  target_include_directories(flagdnn_test_thead_graph SYSTEM PRIVATE
    "${_FLAGDNN_THEAD_TRITON_JIT_INCLUDE_DIR}"
    "${_FLAGDNN_THEAD_TORCH_INCLUDE}"
    "${_FLAGDNN_THEAD_TORCH_API_INCLUDE}")
  target_compile_definitions(flagdnn_test_thead_graph PRIVATE
    _GLIBCXX_USE_CXX11_ABI=${_FLAGDNN_THEAD_TORCH_CXX11_ABI}
    FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG="${CMAKE_CURRENT_SOURCE_DIR}/capability.json"
    FLAGDNN_THEAD_PPU_SDK_VERSION="${_FLAGDNN_THEAD_PPU_SDK_VERSION}")
  target_link_libraries(flagdnn_test_thead_graph PRIVATE
    FlagDNN::flagdnn
    flagdnn_validation_thead_reference
    FlagDNN::THeadCudaDriver
    FlagDNN::THeadTritonJIT
    "${CMAKE_DL_LIBS}")
  add_dependencies(flagdnn_test_thead_graph flagdnn_backend_thead)
  set_target_properties(flagdnn_test_thead_graph PROPERTIES
    BUILD_RPATH
      "$<TARGET_FILE_DIR:flagdnn_backend_thead>/flagdnn/thead;${_FLAGDNN_THEAD_PPU_SDK_ROOT}/lib;${_FLAGDNN_THEAD_PPU_SDK_ROOT}/CUDA_SDK/lib64")
  if(UNIX AND NOT APPLE)
    target_link_options(flagdnn_test_thead_graph PRIVATE
      "LINKER:--disable-new-dtags"
      "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/allocation_probe.map")
  endif()
  flagdnn_enable_warnings(flagdnn_test_thead_graph)

  add_test(NAME integration.thead.graph
    COMMAND flagdnn_test_thead_graph
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py")
  add_test(NAME integration.thead.graph_mul
    COMMAND flagdnn_test_thead_graph
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation mul)
  add_test(NAME integration.thead.graph_sub
    COMMAND flagdnn_test_thead_graph
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation sub)
  add_test(NAME integration.thead.graph_min
    COMMAND flagdnn_test_thead_graph
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation min)
  add_test(NAME integration.thead.graph_max
    COMMAND flagdnn_test_thead_graph
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation max)
  add_test(NAME integration.thead.graph_scale
    COMMAND flagdnn_test_thead_graph
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation scale)
  add_test(NAME integration.thead.graph_relu
    COMMAND flagdnn_test_thead_graph
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation relu)
  add_test(NAME integration.thead.graph_sigmoid
    COMMAND flagdnn_test_thead_graph
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation sigmoid)
  add_test(NAME integration.thead.graph_tanh
    COMMAND flagdnn_test_thead_graph
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation tanh)
  add_test(NAME integration.thead.graph_elu
    COMMAND flagdnn_test_thead_graph
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation elu)
  add_test(NAME integration.thead.graph_identity
    COMMAND flagdnn_test_thead_graph
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation identity)
  add_test(NAME integration.thead.graph_gelu
    COMMAND flagdnn_test_thead_graph
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation gelu)
  add_test(NAME integration.thead.graph_sqrt
    COMMAND flagdnn_test_thead_graph
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation sqrt)
  add_test(NAME integration.thead.graph_neg
    COMMAND flagdnn_test_thead_graph
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation neg)
  add_test(NAME integration.thead.graph_abs
    COMMAND flagdnn_test_thead_graph
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation abs)
  add_test(NAME integration.thead.graph_ceil
    COMMAND flagdnn_test_thead_graph
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation ceil)
  add_test(NAME integration.thead.graph_floor
    COMMAND flagdnn_test_thead_graph
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation floor)
  add_test(NAME integration.thead.graph_exp
    COMMAND flagdnn_test_thead_graph
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation exp)
  set(_flagdnn_thead_extended_graph_tests)
  foreach(_flagdnn_thead_operation IN LISTS
      _flagdnn_thead_extended_unary_operations)
    add_test(NAME "integration.thead.graph_${_flagdnn_thead_operation}"
      COMMAND flagdnn_test_thead_graph
        "$<TARGET_FILE:flagdnn_backend_thead>"
        "${FLAGDNN_CODEGEN_PYTHON}"
        "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
        --operation "${_flagdnn_thead_operation}")
    list(APPEND _flagdnn_thead_extended_graph_tests
      "integration.thead.graph_${_flagdnn_thead_operation}")
  endforeach()
  set_tests_properties(
    integration.thead.graph integration.thead.graph_mul
    integration.thead.graph_sub integration.thead.graph_min
    integration.thead.graph_max integration.thead.graph_scale
    integration.thead.graph_relu integration.thead.graph_sigmoid
    integration.thead.graph_tanh integration.thead.graph_elu
    integration.thead.graph_identity integration.thead.graph_gelu
    integration.thead.graph_sqrt integration.thead.graph_neg
    integration.thead.graph_abs integration.thead.graph_ceil
    integration.thead.graph_floor integration.thead.graph_exp
    ${_flagdnn_thead_extended_graph_tests}
    PROPERTIES
    LABELS "integration;thead;graph;capture;acdnn"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_thead>;${_flagdnn_thead_codegen_environment}"
    RUN_SERIAL TRUE
    TIMEOUT 600)

  add_test(NAME integration.thead.graph_capture_safety_contract
    COMMAND "${CMAKE_COMMAND}"
      "-DSOURCE_ROOT=${PROJECT_SOURCE_DIR}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/GraphCaptureSafetyContract.cmake")
  set_tests_properties(
    integration.thead.graph_capture_safety_contract PROPERTIES
    LABELS "integration;thead;graph;capture;contract"
    RUN_SERIAL TRUE
    TIMEOUT 30)

  add_executable(flagdnn_test_thead_autotune
    functional/test_autotune.cpp
    "${PROJECT_SOURCE_DIR}/backends/autotune_policy.cpp")
  target_compile_features(flagdnn_test_thead_autotune PRIVATE cxx_std_20)
  target_include_directories(flagdnn_test_thead_autotune PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/tests"
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_BINARY_DIR}/generated/include")
  target_include_directories(flagdnn_test_thead_autotune SYSTEM PRIVATE
    "${_FLAGDNN_THEAD_TRITON_JIT_INCLUDE_DIR}"
    "${_FLAGDNN_THEAD_TORCH_INCLUDE}"
    "${_FLAGDNN_THEAD_TORCH_API_INCLUDE}")
  target_compile_definitions(flagdnn_test_thead_autotune PRIVATE
    _GLIBCXX_USE_CXX11_ABI=${_FLAGDNN_THEAD_TORCH_CXX11_ABI}
    FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG="${CMAKE_CURRENT_SOURCE_DIR}/capability.json"
    FLAGDNN_THEAD_PPU_SDK_VERSION="${_FLAGDNN_THEAD_PPU_SDK_VERSION}")
  target_link_libraries(flagdnn_test_thead_autotune PRIVATE
    FlagDNN::flagdnn
    flagdnn_validation_thead_reference
    FlagDNN::THeadCudaDriver
    FlagDNN::THeadTritonJIT
    "${CMAKE_DL_LIBS}")
  add_dependencies(flagdnn_test_thead_autotune flagdnn_backend_thead)
  set_target_properties(flagdnn_test_thead_autotune PROPERTIES
    BUILD_RPATH
      "$<TARGET_FILE_DIR:flagdnn_backend_thead>/flagdnn/thead;${_FLAGDNN_THEAD_PPU_SDK_ROOT}/lib;${_FLAGDNN_THEAD_PPU_SDK_ROOT}/CUDA_SDK/lib64")
  if(UNIX AND NOT APPLE)
    target_link_options(flagdnn_test_thead_autotune PRIVATE
      "LINKER:--disable-new-dtags")
  endif()
  flagdnn_enable_warnings(flagdnn_test_thead_autotune)

  add_test(NAME integration.thead.autotune_policy_contract
    COMMAND flagdnn_test_thead_autotune --policy-contract)
  set_tests_properties(integration.thead.autotune_policy_contract PROPERTIES
    LABELS "integration;thead;autotune;policy;contract"
    RUN_SERIAL TRUE
    TIMEOUT 60)

  add_test(NAME integration.thead.autotune
    COMMAND flagdnn_test_thead_autotune
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py")
  add_test(NAME integration.thead.autotune_mul
    COMMAND flagdnn_test_thead_autotune
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation mul)
  add_test(NAME integration.thead.autotune_sub
    COMMAND flagdnn_test_thead_autotune
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation sub)
  add_test(NAME integration.thead.autotune_min
    COMMAND flagdnn_test_thead_autotune
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation min)
  add_test(NAME integration.thead.autotune_max
    COMMAND flagdnn_test_thead_autotune
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation max)
  add_test(NAME integration.thead.autotune_scale
    COMMAND flagdnn_test_thead_autotune
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation scale)
  add_test(NAME integration.thead.autotune_relu
    COMMAND flagdnn_test_thead_autotune
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation relu)
  add_test(NAME integration.thead.autotune_sigmoid
    COMMAND flagdnn_test_thead_autotune
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation sigmoid)
  add_test(NAME integration.thead.autotune_tanh
    COMMAND flagdnn_test_thead_autotune
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation tanh)
  add_test(NAME integration.thead.autotune_elu
    COMMAND flagdnn_test_thead_autotune
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation elu)
  add_test(NAME integration.thead.autotune_identity
    COMMAND flagdnn_test_thead_autotune
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation identity)
  add_test(NAME integration.thead.autotune_gelu
    COMMAND flagdnn_test_thead_autotune
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation gelu)
  add_test(NAME integration.thead.autotune_sqrt
    COMMAND flagdnn_test_thead_autotune
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation sqrt)
  add_test(NAME integration.thead.autotune_neg
    COMMAND flagdnn_test_thead_autotune
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation neg)
  add_test(NAME integration.thead.autotune_abs
    COMMAND flagdnn_test_thead_autotune
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation abs)
  add_test(NAME integration.thead.autotune_ceil
    COMMAND flagdnn_test_thead_autotune
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation ceil)
  add_test(NAME integration.thead.autotune_floor
    COMMAND flagdnn_test_thead_autotune
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation floor)
  add_test(NAME integration.thead.autotune_exp
    COMMAND flagdnn_test_thead_autotune
      "$<TARGET_FILE:flagdnn_backend_thead>"
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
      --operation exp)
  set(_flagdnn_thead_extended_autotune_tests)
  foreach(_flagdnn_thead_operation IN LISTS
      _flagdnn_thead_extended_unary_operations)
    add_test(NAME "integration.thead.autotune_${_flagdnn_thead_operation}"
      COMMAND flagdnn_test_thead_autotune
        "$<TARGET_FILE:flagdnn_backend_thead>"
        "${FLAGDNN_CODEGEN_PYTHON}"
        "${PROJECT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
        --operation "${_flagdnn_thead_operation}")
    list(APPEND _flagdnn_thead_extended_autotune_tests
      "integration.thead.autotune_${_flagdnn_thead_operation}")
  endforeach()
  set_tests_properties(integration.thead.autotune
    integration.thead.autotune_mul integration.thead.autotune_sub
    integration.thead.autotune_min integration.thead.autotune_max
    integration.thead.autotune_scale integration.thead.autotune_relu
    integration.thead.autotune_sigmoid integration.thead.autotune_tanh
    integration.thead.autotune_elu integration.thead.autotune_identity
    integration.thead.autotune_gelu integration.thead.autotune_sqrt
    integration.thead.autotune_neg integration.thead.autotune_abs
    integration.thead.autotune_ceil integration.thead.autotune_floor
    integration.thead.autotune_exp
    ${_flagdnn_thead_extended_autotune_tests}
    PROPERTIES
    LABELS "integration;thead;autotune;jit;acdnn"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_thead>;${_flagdnn_thead_codegen_environment}"
    RUN_SERIAL TRUE
    TIMEOUT 900)

  add_test(NAME integration.thead.jit_candidate_compatibility_contract
    COMMAND "${CMAKE_COMMAND}"
      "-DSOURCE_ROOT=${PROJECT_SOURCE_DIR}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/JitCandidateCompatibilityContract.cmake")
  set_tests_properties(
    integration.thead.jit_candidate_compatibility_contract PROPERTIES
    LABELS "integration;thead;autotune;jit;contract"
    RUN_SERIAL TRUE
    TIMEOUT 30)

  add_test(NAME integration.thead.triton_compilation
    COMMAND "${CMAKE_COMMAND}"
      "-DPYTHON_EXECUTABLE=${FLAGDNN_CODEGEN_PYTHON}"
      "-DSOURCE_ROOT=${PROJECT_SOURCE_DIR}"
      "-DTRITON_ROOT=${_FLAGDNN_THEAD_TRITON_ROOT}"
      "-DPPU_SDK_ROOT=${_FLAGDNN_THEAD_PPU_SDK_ROOT}"
      "-DTRITON_JIT_ROOT=${_FLAGDNN_THEAD_TRITON_JIT_ROOT}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/TritonCompilationContract.cmake")
  set_tests_properties(integration.thead.triton_compilation PROPERTIES
    LABELS "integration;thead;jit;artifact;contract"
    RUN_SERIAL TRUE
    # The exhaustive contract compiles every qualified family, including the
    # typed/strided, normalization, MatMul and convolution variants.
    TIMEOUT 900)

  add_executable(flagdnn_test_thead_runtime
    functional/test_runtime.cpp)
  target_compile_features(flagdnn_test_thead_runtime PRIVATE cxx_std_20)
  target_include_directories(flagdnn_test_thead_runtime PRIVATE
    "${PROJECT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_BINARY_DIR}/generated/include")
  target_link_libraries(flagdnn_test_thead_runtime PRIVATE
    FlagDNN::flagdnn
    FlagDNN::THeadCudaDriver
    "${CMAKE_DL_LIBS}")
  add_dependencies(flagdnn_test_thead_runtime flagdnn_backend_thead)
  flagdnn_enable_warnings(flagdnn_test_thead_runtime)

  add_test(NAME integration.thead.runtime
    COMMAND flagdnn_test_thead_runtime
      "$<TARGET_FILE:flagdnn_backend_thead>")
  set_tests_properties(integration.thead.runtime PROPERTIES
    LABELS "integration;thead;runtime;context;stream"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_thead>"
    RUN_SERIAL TRUE
    TIMEOUT 60)

  unset(_flagdnn_thead_extended_jit_tests)
  unset(_flagdnn_thead_extended_graph_tests)
  unset(_flagdnn_thead_extended_autotune_tests)
  unset(_flagdnn_thead_extended_unary_operations)
  unset(_flagdnn_thead_operation)

  add_library(flagdnn_test_thead_adapter STATIC
    $<TARGET_OBJECTS:flagdnn_test_common_objects>
    functional/acdnn_add.cpp
    functional/add_runner.cpp
    functional/composite_runner.cpp
    functional/convolution_runner.cpp
    functional/layout_runner.cpp
    functional/matmul_runner.cpp
    functional/normalization_runner.cpp
    functional/pointwise_runner.cpp
    functional/reduction_runner.cpp
    functional/unsupported_runner.cpp)
  target_compile_features(flagdnn_test_thead_adapter PUBLIC cxx_std_20)
  target_compile_definitions(flagdnn_test_thead_adapter PRIVATE
    FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG="${CMAKE_CURRENT_SOURCE_DIR}/capability.json"
    FLAGDNN_THEAD_PPU_SDK_VERSION="${_FLAGDNN_THEAD_PPU_SDK_VERSION}")
  target_include_directories(flagdnn_test_thead_adapter PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/tests")
  target_link_libraries(flagdnn_test_thead_adapter PUBLIC
    FlagDNN::flagdnn
    flagdnn_validation_thead_reference)
  flagdnn_enable_warnings(flagdnn_test_thead_adapter)

  include("${PROJECT_SOURCE_DIR}/cmake/Operators.cmake")
  flagdnn_register_functional_suite(
    PLATFORM thead
    ADAPTER_TARGET flagdnn_test_thead_adapter
    BACKEND_TARGET flagdnn_backend_thead
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_thead>;FLAGDNN_THEAD_QUALIFY_PROBES=1;${_flagdnn_thead_codegen_environment}"
    LABELS "acdnn;ppu"
    TIMEOUT 900
    BUILD_RPATH
      "${_FLAGDNN_THEAD_PPU_SDK_ROOT}/lib;${_FLAGDNN_THEAD_PPU_SDK_ROOT}/CUDA_SDK/lib64")
  foreach(_flagdnn_thead_operator IN LISTS FLAGDNN_FUNCTIONAL_OPERATORS)
    set_tests_properties(
      "functional.thead.${_flagdnn_thead_operator}" PROPERTIES
      SKIP_RETURN_CODE 77)
  endforeach()
  unset(_flagdnn_thead_operator)
endif()

if(FLAGDNN_BUILD_BENCHMARKS)
  add_library(flagdnn_benchmark_thead_adapter STATIC
    benchmark/acdnn_provider.cpp
    benchmark/pointwise.cpp
    benchmark/runner.cpp)
  target_compile_features(flagdnn_benchmark_thead_adapter PUBLIC cxx_std_20)
  target_compile_definitions(flagdnn_benchmark_thead_adapter PRIVATE
    FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG="${CMAKE_CURRENT_SOURCE_DIR}/capability.json"
    FLAGDNN_THEAD_BENCHMARK_CATALOG="${CMAKE_CURRENT_SOURCE_DIR}/benchmark/comparable_cases.json"
    FLAGDNN_THEAD_PPU_SDK_VERSION="${_FLAGDNN_THEAD_PPU_SDK_VERSION}")
  target_include_directories(flagdnn_benchmark_thead_adapter PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/benchmark"
    PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}")
  target_link_libraries(flagdnn_benchmark_thead_adapter PUBLIC
    FlagDNN::flagdnn
    flagdnn_benchmark_cases
    flagdnn_validation_thead_reference)
  flagdnn_enable_warnings(flagdnn_benchmark_thead_adapter)

  add_library(flagdnn_benchmark_thead_unsupported_adapter STATIC
    benchmark/unsupported_runner.cpp)
  target_compile_features(
    flagdnn_benchmark_thead_unsupported_adapter PUBLIC cxx_std_20)
  target_compile_definitions(
    flagdnn_benchmark_thead_unsupported_adapter PRIVATE
    FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG="${CMAKE_CURRENT_SOURCE_DIR}/capability.json"
    FLAGDNN_THEAD_PPU_SDK_VERSION="${_FLAGDNN_THEAD_PPU_SDK_VERSION}")
  target_include_directories(
    flagdnn_benchmark_thead_unsupported_adapter PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/benchmark")
  target_link_libraries(
    flagdnn_benchmark_thead_unsupported_adapter PUBLIC
    flagdnn_benchmark_cases
    flagdnn_validation_thead_reference)
  flagdnn_enable_warnings(flagdnn_benchmark_thead_unsupported_adapter)

  include("${PROJECT_SOURCE_DIR}/cmake/Operators.cmake")
  flagdnn_register_benchmark_suite(
    PLATFORM thead
    ADAPTER_TARGET flagdnn_benchmark_thead_adapter
    BACKEND_TARGET flagdnn_backend_thead
    OPERATORS add sub mul min max scale relu leaky_relu sigmoid tanh elu identity gelu sqrt neg abs ceil floor exp log cos rsqrt sin tan softplus swish gelu_approx_tanh div pow mod sigmoid_backward reciprocal add_square cmp_eq cmp_neq cmp_gt cmp_ge cmp_lt cmp_le reshape transpose slice reduction batchnorm batchnorm_inference layernorm rmsnorm matmul conv_fprop conv_dgrad conv_wgrad conv_bias_relu
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_thead>;FLAGDNN_THEAD_QUALIFY_PROBES=1;${_flagdnn_thead_codegen_environment}"
    LABELS "acdnn;paired;ppu"
    TIMEOUT 1800
    BUILD_RPATH
      "${_FLAGDNN_THEAD_PPU_SDK_ROOT}/lib;${_FLAGDNN_THEAD_PPU_SDK_ROOT}/CUDA_SDK/lib64")
  set_tests_properties(
    benchmark.thead.add benchmark.thead.sub benchmark.thead.mul
    benchmark.thead.min benchmark.thead.max benchmark.thead.scale
    benchmark.thead.relu benchmark.thead.leaky_relu
    benchmark.thead.sigmoid benchmark.thead.tanh
    benchmark.thead.elu benchmark.thead.identity benchmark.thead.gelu
    benchmark.thead.sqrt benchmark.thead.neg benchmark.thead.abs
    benchmark.thead.ceil benchmark.thead.floor benchmark.thead.exp
    benchmark.thead.log benchmark.thead.cos benchmark.thead.rsqrt
    benchmark.thead.sin benchmark.thead.tan benchmark.thead.softplus
    benchmark.thead.swish benchmark.thead.gelu_approx_tanh
    benchmark.thead.div benchmark.thead.pow benchmark.thead.mod
    benchmark.thead.sigmoid_backward benchmark.thead.reciprocal
    benchmark.thead.add_square benchmark.thead.cmp_eq
    benchmark.thead.cmp_neq benchmark.thead.cmp_gt benchmark.thead.cmp_ge
    benchmark.thead.cmp_lt benchmark.thead.cmp_le benchmark.thead.reshape
    benchmark.thead.transpose benchmark.thead.slice benchmark.thead.reduction
    benchmark.thead.batchnorm benchmark.thead.batchnorm_inference
    benchmark.thead.layernorm benchmark.thead.rmsnorm
    benchmark.thead.matmul benchmark.thead.conv_fprop
    benchmark.thead.conv_dgrad benchmark.thead.conv_wgrad
    benchmark.thead.conv_bias_relu
    PROPERTIES
    SKIP_RETURN_CODE 77)

  set(_flagdnn_thead_unsupported_benchmark_operators
    ${FLAGDNN_BENCHMARK_OPERATORS})
  list(REMOVE_ITEM _flagdnn_thead_unsupported_benchmark_operators
    add sub mul min max scale relu leaky_relu sigmoid tanh elu identity gelu sqrt neg abs ceil floor exp log cos rsqrt sin tan softplus swish gelu_approx_tanh div pow mod sigmoid_backward reciprocal add_square cmp_eq cmp_neq cmp_gt cmp_ge cmp_lt cmp_le reshape transpose slice reduction batchnorm batchnorm_inference layernorm rmsnorm matmul conv_fprop conv_dgrad conv_wgrad conv_bias_relu)
  flagdnn_register_benchmark_suite(
    PLATFORM thead
    ADAPTER_TARGET flagdnn_benchmark_thead_unsupported_adapter
    BACKEND_TARGET flagdnn_backend_thead
    OPERATORS ${_flagdnn_thead_unsupported_benchmark_operators}
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_thead>"
    LABELS "acdnn;explicit-unsupported;ppu"
    TIMEOUT 300
    BUILD_RPATH
      "${_FLAGDNN_THEAD_PPU_SDK_ROOT}/lib;${_FLAGDNN_THEAD_PPU_SDK_ROOT}/CUDA_SDK/lib64")
  foreach(_flagdnn_thead_operator IN LISTS
      _flagdnn_thead_unsupported_benchmark_operators)
    set_tests_properties(
      "benchmark.thead.${_flagdnn_thead_operator}" PROPERTIES
      SKIP_RETURN_CODE 77)
  endforeach()
  unset(_flagdnn_thead_operator)
  unset(_flagdnn_thead_unsupported_benchmark_operators)
endif()
