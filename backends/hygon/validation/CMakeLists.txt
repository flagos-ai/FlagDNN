# Hygon validation deliberately has one vendor reference SDK: hipDNN. The
# functional-only CPU semantic oracle is repository code and does not add a
# vendor dependency. Production backend targets do not link validation targets.

include("${PROJECT_SOURCE_DIR}/cmake/Operators.cmake")

if(NOT TARGET flagdnn_backend_hygon)
  message(FATAL_ERROR "Hygon validation requires FLAGDNN_BACKENDS=hygon")
endif()
if(NOT FLAGDNN_CODEGEN_PYTHON OR NOT FLAGDNN_CODEGEN_COMPILER)
  message(FATAL_ERROR "Hygon validation requires the codegen Python/provider")
endif()

find_package(hip CONFIG REQUIRED)
find_path(FLAGDNN_HIPDNN_INCLUDE_DIR
  NAMES hipdnn.h
  HINTS
    "$ENV{HIPDNN_ROOT}"
    "$ENV{DTK_ROOT}"
    /opt/dtk/hipdnn
    /opt/dtk
  PATH_SUFFIXES include hipdnn/include
  REQUIRED)
find_library(FLAGDNN_HIPDNN_LIBRARY
  NAMES hipdnn
  HINTS
    "$ENV{HIPDNN_ROOT}"
    "$ENV{DTK_ROOT}"
    /opt/dtk/hipdnn
    /opt/dtk
  PATH_SUFFIXES lib lib64
  REQUIRED)
get_filename_component(_flagdnn_hipdnn_library_directory
  "${FLAGDNN_HIPDNN_LIBRARY}" DIRECTORY)

set(_flagdnn_hygon_operators
  add sub mul div pow max min mod add_square cmp_eq
  abs ceil cos elu erf exp floor gelu gelu_approx_tanh identity
  leaky_relu log logical_not neg reciprocal relu rsqrt sigmoid sin
  softplus sqrt swish tan tanh
  binary_select cmp_ge cmp_gt cmp_le cmp_lt cmp_neq logical_and
  logical_or scale sigmoid_backward
  ${FLAGDNN_ACTIVATION_BACKWARD_EXTENSIONS})

# These functional tests deliberately use the repository CPU semantic oracle.
# Keep this Hygon routing policy separate from the set of operations that the
# CPU reference library may implement now or in the future.
set(_flagdnn_hygon_cpu_reference_operators div pow mod cmp_eq)

set(_flagdnn_hygon_tensor_operators reshape transpose slice reduction matmul)

set(_flagdnn_hygon_convolution_operators
  conv_fprop conv_dgrad conv_wgrad conv_bias_relu)

set(_flagdnn_hygon_attention_operators
  sdpa sdpa_backward sdpa_fp8 sdpa_fp8_backward)

set(_flagdnn_hygon_normalization_operators
  layernorm rmsnorm batchnorm batchnorm_inference)

# Keep the Hygon family partitions exhaustive as the public operator catalog
# evolves.  Sorting also makes duplicate or misspelled registrations fail the
# equality check instead of silently disappearing from tools/run_tests.py.
set(_flagdnn_hygon_functional_catalog
  ${FLAGDNN_EXTENDED_OPERATORS}
  ${_flagdnn_hygon_operators}
  ${_flagdnn_hygon_tensor_operators}
  ${_flagdnn_hygon_convolution_operators}
  ${_flagdnn_hygon_attention_operators}
  ${_flagdnn_hygon_normalization_operators})
set(_flagdnn_hygon_expected_functional_catalog
  ${FLAGDNN_FUNCTIONAL_OPERATORS})
list(SORT _flagdnn_hygon_functional_catalog)
list(SORT _flagdnn_hygon_expected_functional_catalog)
if(NOT "${_flagdnn_hygon_functional_catalog}" STREQUAL
   "${_flagdnn_hygon_expected_functional_catalog}")
  message(FATAL_ERROR
    "Hygon functional registrations do not match the public catalog")
endif()

set(_flagdnn_hygon_benchmark_catalog
  ${_flagdnn_hygon_operators}
  ${_flagdnn_hygon_tensor_operators}
  ${_flagdnn_hygon_convolution_operators}
  ${_flagdnn_hygon_normalization_operators})
set(_flagdnn_hygon_expected_benchmark_catalog
  ${FLAGDNN_BENCHMARK_OPERATORS})
list(SORT _flagdnn_hygon_benchmark_catalog)
list(SORT _flagdnn_hygon_expected_benchmark_catalog)
if(NOT "${_flagdnn_hygon_benchmark_catalog}" STREQUAL
   "${_flagdnn_hygon_expected_benchmark_catalog}")
  message(FATAL_ERROR
    "Hygon benchmark registrations do not match the public catalog")
endif()

add_library(flagdnn_validation_hygon_platform INTERFACE)
target_compile_features(flagdnn_validation_hygon_platform INTERFACE cxx_std_20)
target_include_directories(flagdnn_validation_hygon_platform INTERFACE
  "${FLAGDNN_HIPDNN_INCLUDE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(flagdnn_validation_hygon_platform INTERFACE hip::host)

add_library(flagdnn_validation_hygon_reference STATIC
  activation_layout.cpp
  attention_reference.cpp
  convolution_reference.cpp
  hipdnn_reference.cpp
  normalization_reference.cpp
  pointwise_reference.cpp
  tensor_reference.cpp
  tensor_io.cpp)
target_compile_features(flagdnn_validation_hygon_reference PUBLIC cxx_std_20)
target_link_libraries(flagdnn_validation_hygon_reference PUBLIC
  FlagDNN::flagdnn
  flagdnn_validation_hygon_platform
  "${FLAGDNN_HIPDNN_LIBRARY}")
flagdnn_enable_warnings(flagdnn_validation_hygon_reference)

# Shared shape generators are also available in benchmark-only builds.
if(TARGET flagdnn_test_common_objects)
  set(_flagdnn_hygon_case_sources $<TARGET_OBJECTS:flagdnn_test_common_objects>)
else()
  set(_flagdnn_hygon_case_sources)
  foreach(family IN ITEMS add attention composite convolution index layout matmul
      normalization pointwise reduction statistics moe_matmul fp8_matmul
      causal_convolution resample random position_embedding normalization_extended)
    list(APPEND _flagdnn_hygon_case_sources "${PROJECT_SOURCE_DIR}/tests/common/${family}.cpp")
  endforeach()
endif()
add_library(flagdnn_validation_hygon_cases STATIC
  ${_flagdnn_hygon_case_sources}
  functional/extended_runner.cpp
  functional/attention_runner.cpp)
target_compile_features(flagdnn_validation_hygon_cases PUBLIC cxx_std_20)
target_include_directories(flagdnn_validation_hygon_cases PUBLIC
  "${PROJECT_SOURCE_DIR}/tests" "${PROJECT_SOURCE_DIR}/benchmark"
  "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(flagdnn_validation_hygon_cases PUBLIC
  FlagDNN::flagdnn flagdnn_validation_hygon_reference flagdnn_validation_cpu_reference)
flagdnn_enable_warnings(flagdnn_validation_hygon_cases)

if(FLAGDNN_BUILD_TESTS)
  add_executable(flagdnn_test_hygon_jit_compatibility
    functional/test_jit_compatibility.cpp
    "../error.cpp")
  target_compile_features(flagdnn_test_hygon_jit_compatibility PRIVATE cxx_std_20)
  target_include_directories(flagdnn_test_hygon_jit_compatibility PRIVATE
    "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include")
  target_link_libraries(flagdnn_test_hygon_jit_compatibility PRIVATE hip::host)
  flagdnn_enable_warnings(flagdnn_test_hygon_jit_compatibility)
  add_test(NAME integration.hygon.jit_candidate_compatibility_contract
    COMMAND flagdnn_test_hygon_jit_compatibility)
  set_tests_properties(
    integration.hygon.jit_candidate_compatibility_contract PROPERTIES
    LABELS "integration;hygon;libtriton_jit;autotune;contract")

  add_test(NAME integration.hygon.jit_global_state_contract
    COMMAND "${CMAKE_COMMAND}"
      "-DSOURCE_FILE=${PROJECT_SOURCE_DIR}/backends/hygon/engines/libtriton_jit.cpp"
      -P
        "${CMAKE_CURRENT_SOURCE_DIR}/JitGlobalStateContract.cmake")
  set_tests_properties(
    integration.hygon.jit_global_state_contract PROPERTIES
    LABELS "integration;hygon;libtriton_jit;contract")

  add_test(NAME integration.hygon.convolution_validation_static_contract
    COMMAND "${CMAKE_COMMAND}"
      "-DREFERENCE_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/convolution_reference.cpp"
      "-DBENCHMARK_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/benchmark/convolution.cpp"
      "-DRUNNER_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/benchmark/runner.cpp"
      -P
        "${CMAKE_CURRENT_SOURCE_DIR}/ConvolutionValidationContract.cmake")
  set_tests_properties(
    integration.hygon.convolution_validation_static_contract PROPERTIES
    LABELS "integration;hygon;validation;convolution;contract")

  add_test(NAME integration.hygon.cmake_configuration_contract
    COMMAND "${CMAKE_COMMAND}"
      "-DSOURCE_ROOT=${PROJECT_SOURCE_DIR}"
      "-DTEST_ROOT=${PROJECT_BINARY_DIR}/cmake-contract/hygon-triton-jit"
      -P
        "${CMAKE_CURRENT_SOURCE_DIR}/TritonJITConfigurationContract.cmake")
  set_tests_properties(
    integration.hygon.cmake_configuration_contract PROPERTIES
    LABELS "integration;hygon;cmake;contract")

  # Compile NVIDIA's native catalog selectors with a recording executor only.
  # This optional contract never links CUDA/cuDNN or changes production targets.
  add_library(flagdnn_hygon_catalog_probe OBJECT benchmark/native_dtype.cpp)
  target_compile_definitions(flagdnn_hygon_catalog_probe PRIVATE
    run_native_dtype_benchmark=run_hygon_dtype_catalog FLAGDNN_HYGON_CATALOG_PROBE=1)
  target_link_libraries(flagdnn_hygon_catalog_probe PRIVATE flagdnn_validation_hygon_cases)
  flagdnn_enable_warnings(flagdnn_hygon_catalog_probe)
  add_executable(flagdnn_test_hygon_dtype_catalog
    catalog/dtype_contract.cpp
    $<TARGET_OBJECTS:flagdnn_hygon_catalog_probe>
    "${PROJECT_SOURCE_DIR}/backends/nvidia/validation/benchmark/native_matrix.cpp"
    "${PROJECT_SOURCE_DIR}/backends/nvidia/validation/benchmark/native_copy.cpp"
    "${PROJECT_SOURCE_DIR}/backends/nvidia/validation/benchmark/native_dtype.cpp"
    "${PROJECT_SOURCE_DIR}/backends/nvidia/validation/benchmark/dtype_boolean.cpp"
    "${PROJECT_SOURCE_DIR}/backends/nvidia/validation/functional/layout_runner.cpp")
  target_include_directories(flagdnn_test_hygon_dtype_catalog BEFORE PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/catalog" "${PROJECT_SOURCE_DIR}/backends/nvidia")
  target_link_libraries(flagdnn_test_hygon_dtype_catalog PRIVATE flagdnn_validation_hygon_cases)
  flagdnn_enable_warnings(flagdnn_test_hygon_dtype_catalog)
  add_test(NAME integration.hygon.dtype_catalog COMMAND flagdnn_test_hygon_dtype_catalog)
  set_tests_properties(integration.hygon.dtype_catalog PROPERTIES LABELS "integration;hygon;catalog")
  add_test(NAME integration.hygon.catalog_registration
    COMMAND "${FLAGDNN_CODEGEN_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/catalog_contract.py"
      "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
  set_tests_properties(integration.hygon.catalog_registration PROPERTIES
    LABELS "integration;hygon;catalog")


  add_executable(flagdnn_test_hygon_validation_contract
    benchmark/convolution.cpp
    validation_contract.cpp)
  target_compile_features(flagdnn_test_hygon_validation_contract PRIVATE
    cxx_std_20)
  target_include_directories(flagdnn_test_hygon_validation_contract PRIVATE
    "${PROJECT_SOURCE_DIR}/benchmark"
    "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(flagdnn_test_hygon_validation_contract PRIVATE
    flagdnn_validation_hygon_reference)
  flagdnn_enable_warnings(flagdnn_test_hygon_validation_contract)
  add_test(NAME integration.hygon.validation_contract
    COMMAND flagdnn_test_hygon_validation_contract)
  set_tests_properties(integration.hygon.validation_contract PROPERTIES
    LABELS "integration;hygon;validation;contract")

  add_library(flagdnn_test_hygon_adapter STATIC
    "${PROJECT_SOURCE_DIR}/tests/common/pointwise_runner.cpp"
    functional/add_runner.cpp
    functional/composite_runner.cpp
    functional/convolution_runner.cpp
    functional/hipdnn_add.cpp
    functional/hipdnn_attention.cpp
    functional/hipdnn_composite.cpp
    functional/hipdnn_convolution.cpp
    functional/hipdnn_layout.cpp
    functional/hipdnn_matmul.cpp
    functional/hipdnn_normalization.cpp
    functional/hipdnn_pointwise.cpp
    functional/hipdnn_reduction.cpp
    functional/layout_runner.cpp
    functional/matmul_runner.cpp
    functional/normalization_runner.cpp
    functional/pointwise_runner.cpp
    functional/reduction_runner.cpp)
  target_compile_features(flagdnn_test_hygon_adapter PUBLIC cxx_std_20)
  target_include_directories(flagdnn_test_hygon_adapter PUBLIC
    "${PROJECT_SOURCE_DIR}/tests"
    "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(flagdnn_test_hygon_adapter PUBLIC
    FlagDNN::flagdnn
    flagdnn_validation_cpu_reference
    flagdnn_validation_hygon_cases
    flagdnn_validation_hygon_platform
    flagdnn_validation_hygon_reference)
  flagdnn_enable_warnings(flagdnn_test_hygon_adapter)

  flagdnn_register_functional_suite(
    PLATFORM hygon
    ADAPTER_TARGET flagdnn_test_hygon_adapter
    BACKEND_TARGET flagdnn_backend_hygon
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_hygon>"
    LABELS "hipdnn;cpu-reference"
    TIMEOUT 7200
    BUILD_RPATH "${_flagdnn_hipdnn_library_directory}")

  foreach(operator IN LISTS FLAGDNN_FUNCTIONAL_OPERATORS)
    set_tests_properties("functional.hygon.${operator}" PROPERTIES
      SKIP_RETURN_CODE 77)
  endforeach()

  foreach(operator IN ITEMS matmul conv_fprop conv_dgrad conv_wgrad)
    foreach(precision IN ITEMS ieee tf32)
      if(precision STREQUAL "ieee")
        set(input_precision 1)
      else()
        set(input_precision 2)
      endif()
      add_test(NAME "functional.hygon.${operator}.${precision}"
        COMMAND "flagdnn_test_hygon_${operator}"
          "${FLAGDNN_CODEGEN_PYTHON}" "${FLAGDNN_CODEGEN_COMPILER}")
      set_tests_properties("functional.hygon.${operator}.${precision}" PROPERTIES
        LABELS "functional;correctness;hygon;hipdnn;${operator};${precision}"
        RUN_SERIAL TRUE TIMEOUT 14400 SKIP_RETURN_CODE 77
        ENVIRONMENT "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_hygon>;FLAGDNN_INPUT_PRECISION=${input_precision}")
    endforeach()
    set_property(TEST "functional.hygon.${operator}" APPEND PROPERTY
      ENVIRONMENT "FLAGDNN_INPUT_PRECISION=0")
  endforeach()
  add_test(NAME functional.hygon.matmul.fp8 COMMAND flagdnn_test_hygon_matmul_fp8
    "${FLAGDNN_CODEGEN_PYTHON}" "${FLAGDNN_CODEGEN_COMPILER}")
  set_tests_properties(functional.hygon.matmul.fp8 PROPERTIES
    LABELS "functional;correctness;hygon;hipdnn;matmul;fp8"
    RUN_SERIAL TRUE TIMEOUT 14400 SKIP_RETURN_CODE 77
    ENVIRONMENT "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_hygon>;FLAGDNN_FP8_MATMUL_API=matmul")
  set_property(TEST functional.hygon.matmul_fp8 APPEND PROPERTY
    ENVIRONMENT "FLAGDNN_FP8_MATMUL_API=matmul_fp8")

  add_executable(flagdnn_test_hygon_private_convolution functional/test_private_convolution.cpp)
  target_link_libraries(flagdnn_test_hygon_private_convolution PRIVATE flagdnn_test_hygon_adapter)
  flagdnn_enable_warnings(flagdnn_test_hygon_private_convolution)
  add_test(NAME integration.hygon.private_convolution COMMAND flagdnn_test_hygon_private_convolution
    "${FLAGDNN_CODEGEN_PYTHON}" "${FLAGDNN_CODEGEN_COMPILER}")
  set_tests_properties(integration.hygon.private_convolution PROPERTIES
    LABELS "integration;hygon;convolution" RUN_SERIAL TRUE TIMEOUT 14400 SKIP_RETURN_CODE 77
    ENVIRONMENT "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_hygon>;FLAGDNN_INPUT_PRECISION=0")

  # Functional host references exercise the complete public dtype/shape
  # catalog, including cases that previously returned immediately as SKIP.
  # Each cold-cache shape compiles a graph; keep a bounded budget for the
  # largest corpora without removing any cases.
  set_tests_properties(functional.hygon.reduction functional.hygon.matmul_fp8
    functional.hygon.matmul PROPERTIES TIMEOUT 14400)

  foreach(operator IN LISTS _flagdnn_hygon_cpu_reference_operators)
    set_tests_properties("functional.hygon.${operator}" PROPERTIES
      LABELS "functional;correctness;hygon;${operator};cpu-reference")
  endforeach()

  add_executable(flagdnn_test_hygon_normalization_stability
    functional/test_normalization_stability.cpp)
  target_compile_features(flagdnn_test_hygon_normalization_stability PRIVATE
    cxx_std_20)
  target_link_libraries(flagdnn_test_hygon_normalization_stability PRIVATE
    flagdnn_test_hygon_adapter)
  add_dependencies(flagdnn_test_hygon_normalization_stability
    flagdnn_backend_hygon)
  flagdnn_enable_warnings(flagdnn_test_hygon_normalization_stability)
  set_target_properties(flagdnn_test_hygon_normalization_stability PROPERTIES
    BUILD_RPATH "${_flagdnn_hipdnn_library_directory}")
  add_test(NAME integration.hygon.normalization_stability
    COMMAND flagdnn_test_hygon_normalization_stability
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}")
  set_tests_properties(integration.hygon.normalization_stability PROPERTIES
    LABELS "integration;hygon;normalization;stability;correctness"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_hygon>"
    RUN_SERIAL TRUE
    TIMEOUT 1200)

  add_test(NAME integration.hygon.dependency_boundary
    COMMAND "${CMAKE_COMMAND}"
      "-DBACKEND_LIBRARY=$<TARGET_FILE:flagdnn_backend_hygon>"
      "-DBACKEND_NAME=hygon"
      "-DREQUIRED_DEPENDENCIES=libgalaxyhip;libtriton_jit;libpython"
      "-DFORBIDDEN_DEPENDENCIES=libhipdnn;libMIOpen;libmiopen;librocblas;libhipblas;libcuda;libcudnn;libcublas"
      -P "${PROJECT_SOURCE_DIR}/cmake/VerifyBackendPlugin.cmake")
  set_tests_properties(integration.hygon.dependency_boundary PROPERTIES
    LABELS "integration;hygon;dependency")

  add_test(NAME integration.hygon.compiler_contract
    COMMAND "${FLAGDNN_CODEGEN_PYTHON}"
      "${CMAKE_CURRENT_SOURCE_DIR}/compiler_contract.py"
      "${PROJECT_SOURCE_DIR}")
  set_tests_properties(integration.hygon.compiler_contract PROPERTIES
    LABELS "integration;hygon;compiler"
    TIMEOUT 60)

  add_test(NAME integration.hygon.reference_dependency_boundary
    COMMAND "${CMAKE_COMMAND}"
      "-DREFERENCE_EXECUTABLE=$<TARGET_FILE:flagdnn_test_hygon_add>"
      -P
        "${CMAKE_CURRENT_SOURCE_DIR}/VerifyReferenceDependencies.cmake")
  set_tests_properties(
    integration.hygon.reference_dependency_boundary PROPERTIES
    LABELS "integration;hygon;hipdnn;dependency")

  if(FLAGDNN_EXECUTION_ENGINE STREQUAL "libtriton_jit")
    set(Python_EXECUTABLE "${FLAGDNN_CODEGEN_PYTHON}")
    find_package(Python REQUIRED COMPONENTS Development)
    add_executable(flagdnn_test_hygon_jit functional/test_jit.cpp)
    target_compile_features(flagdnn_test_hygon_jit PRIVATE cxx_std_20)
    target_link_libraries(flagdnn_test_hygon_jit PRIVATE
      FlagDNN::flagdnn
      Python::Python)
    add_dependencies(flagdnn_test_hygon_jit flagdnn_backend_hygon)
    flagdnn_enable_warnings(flagdnn_test_hygon_jit)
    add_test(NAME integration.hygon.jit
      COMMAND flagdnn_test_hygon_jit
        "${FLAGDNN_CODEGEN_PYTHON}"
        "${FLAGDNN_CODEGEN_COMPILER}")
    set_tests_properties(integration.hygon.jit PROPERTIES
      LABELS "integration;hygon;libtriton_jit;autotune"
      ENVIRONMENT
        "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_hygon>"
      RUN_SERIAL TRUE
      TIMEOUT 1200)

    add_executable(flagdnn_test_hygon_pointwise_smoke
      functional/test_pointwise_smoke.cpp)
    target_compile_features(flagdnn_test_hygon_pointwise_smoke PRIVATE
      cxx_std_20)
    target_link_libraries(flagdnn_test_hygon_pointwise_smoke PRIVATE
      FlagDNN::flagdnn
      flagdnn_validation_hygon_platform
      flagdnn_validation_hygon_reference)
    add_dependencies(flagdnn_test_hygon_pointwise_smoke
      flagdnn_backend_hygon)
    flagdnn_enable_warnings(flagdnn_test_hygon_pointwise_smoke)
    add_test(NAME integration.hygon.pointwise_smoke
      COMMAND flagdnn_test_hygon_pointwise_smoke
        "${FLAGDNN_CODEGEN_PYTHON}"
        "${FLAGDNN_CODEGEN_COMPILER}")
    set_tests_properties(integration.hygon.pointwise_smoke PROPERTIES
      LABELS "integration;hygon;libtriton_jit;autotune;pointwise;smoke"
      ENVIRONMENT
        "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_hygon>"
      RUN_SERIAL TRUE
      TIMEOUT 1200)

    get_target_property(_flagdnn_hygon_installed_jit_soname
      flagdnn_backend_hygon FLAGDNN_HYGON_TRITON_JIT_SONAME)
    get_target_property(_flagdnn_hygon_installed_jit_sha256
      flagdnn_backend_hygon FLAGDNN_HYGON_TRITON_JIT_SHA256)
    get_target_property(_flagdnn_hygon_installed_jit_provenance_sha256
      flagdnn_backend_hygon
      FLAGDNN_HYGON_TRITON_JIT_PROVENANCE_SHA256)
    get_target_property(_flagdnn_hygon_installed_jit_build_identity
      flagdnn_backend_hygon FLAGDNN_HYGON_TRITON_JIT_BUILD_IDENTITY)
    get_target_property(_flagdnn_hygon_installed_standalone_sha256
      flagdnn_backend_hygon
      FLAGDNN_HYGON_TRITON_JIT_STANDALONE_SHA256)
    get_target_property(_flagdnn_hygon_installed_gen_ssig_sha256
      flagdnn_backend_hygon FLAGDNN_HYGON_TRITON_JIT_GEN_SSIG_SHA256)
    get_target_property(_flagdnn_hygon_installed_environment_helper_sha256
      flagdnn_backend_hygon
      FLAGDNN_HYGON_PYTHON_ENVIRONMENT_HELPER_SHA256)
    foreach(_flagdnn_hygon_installed_metadata IN ITEMS
        _flagdnn_hygon_installed_jit_soname
        _flagdnn_hygon_installed_jit_sha256
        _flagdnn_hygon_installed_jit_provenance_sha256
        _flagdnn_hygon_installed_jit_build_identity
        _flagdnn_hygon_installed_standalone_sha256
        _flagdnn_hygon_installed_gen_ssig_sha256
        _flagdnn_hygon_installed_environment_helper_sha256)
      if("${${_flagdnn_hygon_installed_metadata}}" STREQUAL "" OR
         "${${_flagdnn_hygon_installed_metadata}}" MATCHES "-NOTFOUND$")
        message(FATAL_ERROR
          "Hygon installed-consumer metadata is missing: "
          "${_flagdnn_hygon_installed_metadata}")
      endif()
    endforeach()

    add_test(NAME integration.hygon.installed_consumer
      COMMAND "${CMAKE_COMMAND}"
        "-DSOURCE_ROOT=${PROJECT_SOURCE_DIR}"
        "-DBUILD_ROOT=${PROJECT_BINARY_DIR}"
        "-DBUILD_PLUGIN=$<TARGET_FILE:flagdnn_backend_hygon>"
        "-DPLUGIN_SONAME=$<TARGET_SONAME_FILE_NAME:flagdnn_backend_hygon>"
        "-DJIT_SONAME=${_flagdnn_hygon_installed_jit_soname}"
        "-DJIT_SHA256=${_flagdnn_hygon_installed_jit_sha256}"
        "-DJIT_PROVENANCE_SHA256=${_flagdnn_hygon_installed_jit_provenance_sha256}"
        "-DJIT_BUILD_IDENTITY=${_flagdnn_hygon_installed_jit_build_identity}"
        "-DSTANDALONE_SHA256=${_flagdnn_hygon_installed_standalone_sha256}"
        "-DGEN_SSIG_SHA256=${_flagdnn_hygon_installed_gen_ssig_sha256}"
        "-DENVIRONMENT_HELPER_SHA256=${_flagdnn_hygon_installed_environment_helper_sha256}"
        "-DTEST_ROOT=${PROJECT_BINARY_DIR}/installed-consumer-contract/hygon"
        "-DCODEGEN_PYTHON=${FLAGDNN_CODEGEN_PYTHON}"
        "-DINSTALL_CONFIG=${CMAKE_BUILD_TYPE}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/InstalledConsumerContract.cmake")
    set_tests_properties(integration.hygon.installed_consumer PROPERTIES
      LABELS "integration;hygon;install;consumer;hipdnn;libtriton_jit;autotune"
      RUN_SERIAL TRUE
      TIMEOUT 1800)
  endif()

  add_executable(flagdnn_test_hygon_runtime functional/test_runtime.cpp)
  target_compile_features(flagdnn_test_hygon_runtime PRIVATE cxx_std_20)
  target_link_libraries(flagdnn_test_hygon_runtime PRIVATE
    FlagDNN::flagdnn
    flagdnn_validation_hygon_platform
    flagdnn_validation_hygon_reference)
  add_dependencies(flagdnn_test_hygon_runtime flagdnn_backend_hygon)
  flagdnn_enable_warnings(flagdnn_test_hygon_runtime)
  if(FLAGDNN_EXECUTION_ENGINE STREQUAL "libtriton_jit")
    target_compile_definitions(flagdnn_test_hygon_runtime PRIVATE
      FLAGDNN_EXPECT_LIBTRITON_JIT=1)
  endif()
  add_test(NAME integration.hygon.runtime
    COMMAND flagdnn_test_hygon_runtime
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}")
  set_tests_properties(integration.hygon.runtime PROPERTIES
    LABELS "integration;hygon;runtime"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_hygon>"
    RUN_SERIAL TRUE
    # The runtime contract intentionally exercises compiler identity invalidation,
    # offline-cache recovery, and real Hygon kernels. Leave enough headroom for
    # cold Triton compilation while retaining a finite test-level timeout.
    TIMEOUT 900
    SKIP_RETURN_CODE 77)

  add_executable(flagdnn_test_hygon_graph functional/test_graph.cpp)
  target_compile_features(flagdnn_test_hygon_graph PRIVATE cxx_std_20)
  target_link_libraries(flagdnn_test_hygon_graph PRIVATE
    FlagDNN::flagdnn
    flagdnn_validation_hygon_platform
    flagdnn_validation_hygon_reference)
  add_dependencies(flagdnn_test_hygon_graph flagdnn_backend_hygon)
  flagdnn_enable_warnings(flagdnn_test_hygon_graph)
  if(FLAGDNN_EXECUTION_ENGINE STREQUAL "libtriton_jit")
    target_compile_definitions(flagdnn_test_hygon_graph PRIVATE
      FLAGDNN_EXPECT_LIBTRITON_JIT=1)
  endif()
  add_test(NAME integration.hygon.graph
    COMMAND flagdnn_test_hygon_graph
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}")
  set_tests_properties(integration.hygon.graph PROPERTIES
    LABELS "integration;hygon;graph"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_hygon>"
    RUN_SERIAL TRUE
    TIMEOUT 180
    SKIP_RETURN_CODE 77)
endif()

if(FLAGDNN_BUILD_BENCHMARKS)
  add_library(flagdnn_benchmark_hygon_adapter STATIC
    benchmark/runner.cpp
    benchmark/native_runner.cpp
    benchmark/native_dtype.cpp
    benchmark/hipdnn_provider.cpp
    benchmark/pointwise.cpp
    benchmark/tensor.cpp
    benchmark/convolution.cpp
    benchmark/normalization.cpp)
  target_compile_features(flagdnn_benchmark_hygon_adapter PUBLIC cxx_std_20)
  target_include_directories(flagdnn_benchmark_hygon_adapter PUBLIC
    "${PROJECT_SOURCE_DIR}/benchmark"
    "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(flagdnn_benchmark_hygon_adapter PUBLIC
    FlagDNN::flagdnn
    flagdnn_benchmark_cases
    flagdnn_validation_hygon_cases
    flagdnn_validation_hygon_platform
    flagdnn_validation_hygon_reference)
  flagdnn_enable_warnings(flagdnn_benchmark_hygon_adapter)

  file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/benchmark/additional_operators.txt"
    _flagdnn_hygon_additional_benchmarks)
  file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/benchmark/dtype_only_operators.txt"
    _flagdnn_hygon_dtype_only)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/benchmark/additional_operators.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/benchmark/dtype_only_operators.txt")
  list(REMOVE_ITEM _flagdnn_hygon_benchmark_catalog ${_flagdnn_hygon_dtype_only})
  list(APPEND _flagdnn_hygon_benchmark_catalog ${_flagdnn_hygon_additional_benchmarks})

  flagdnn_register_benchmark_suite(
    PLATFORM hygon
    OPERATORS ${_flagdnn_hygon_benchmark_catalog}
    ADAPTER_TARGET flagdnn_benchmark_hygon_adapter
    BACKEND_TARGET flagdnn_backend_hygon
    COMMAND_ARGS
      "${FLAGDNN_CODEGEN_PYTHON}"
      "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_hygon>"
    LABELS hipdnn
    TIMEOUT 1800
    BUILD_RPATH "${_flagdnn_hipdnn_library_directory}")

  add_executable(flagdnn_benchmark_hygon_dtype "${PROJECT_SOURCE_DIR}/benchmark/dtype_main.cpp")
  target_link_libraries(flagdnn_benchmark_hygon_dtype PRIVATE flagdnn_benchmark_hygon_adapter)
  flagdnn_enable_warnings(flagdnn_benchmark_hygon_dtype)
  foreach(category IN ITEMS boolean copy ieee tf32 fp32_output)
    set(dtype_category "${category}")
    set(dtype_environment "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_hygon>")
    if(category STREQUAL "boolean")
      set(dtype_operators logical_and logical_or logical_not)
    elseif(category STREQUAL "copy")
      set(dtype_operators identity reshape transpose slice)
    elseif(category STREQUAL "fp32_output")
      set(dtype_operators reduction)
    else()
      set(dtype_operators matmul conv_fprop conv_dgrad conv_wgrad)
      set(dtype_category precision)
      if(category STREQUAL "ieee")
        list(APPEND dtype_environment "FLAGDNN_INPUT_PRECISION=1")
      else()
        list(APPEND dtype_environment "FLAGDNN_INPUT_PRECISION=2")
      endif()
    endif()
    foreach(operator IN LISTS dtype_operators)
      add_test(NAME "benchmark.hygon.${operator}.${category}"
        COMMAND flagdnn_benchmark_hygon_dtype "${FLAGDNN_CODEGEN_PYTHON}"
          "${FLAGDNN_CODEGEN_COMPILER}" "${operator}" "${dtype_category}")
      set_tests_properties("benchmark.hygon.${operator}.${category}" PROPERTIES
        LABELS "benchmark;hygon;hipdnn;${operator};${category}"
        RUN_SERIAL TRUE TIMEOUT 14400 SKIP_RETURN_CODE 77 ENVIRONMENT "${dtype_environment}")
    endforeach()
  endforeach()
  add_test(NAME benchmark.hygon.matmul.fp8 COMMAND flagdnn_benchmark_hygon_matmul_fp8
    "${FLAGDNN_CODEGEN_PYTHON}" "${FLAGDNN_CODEGEN_COMPILER}")
  set_tests_properties(benchmark.hygon.matmul.fp8 PROPERTIES
    LABELS "benchmark;hygon;hipdnn;matmul;fp8" RUN_SERIAL TRUE TIMEOUT 14400 SKIP_RETURN_CODE 77
    ENVIRONMENT "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_hygon>;FLAGDNN_FP8_MATMUL_API=matmul")
  set_property(TEST benchmark.hygon.matmul_fp8 APPEND PROPERTY
    ENVIRONMENT "FLAGDNN_FP8_MATMUL_API=matmul_fp8")

  foreach(operator IN LISTS _flagdnn_hygon_benchmark_catalog)
    set_tests_properties("benchmark.hygon.${operator}" PROPERTIES
      SKIP_RETURN_CODE 77)
  endforeach()
endif()

unset(_flagdnn_hygon_operators)
unset(_flagdnn_hygon_cpu_reference_operators)
unset(_flagdnn_hygon_tensor_operators)
unset(_flagdnn_hygon_convolution_operators)
unset(_flagdnn_hygon_attention_operators)
unset(_flagdnn_hygon_normalization_operators)
unset(_flagdnn_hygon_functional_catalog)
unset(_flagdnn_hygon_expected_functional_catalog)
unset(_flagdnn_hygon_benchmark_catalog)
unset(_flagdnn_hygon_expected_benchmark_catalog)
unset(_flagdnn_hipdnn_library_directory)
