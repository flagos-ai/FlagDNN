if(FLAGDNN_BUILD_TESTS)
  set(_ascend_native_case_sources $<TARGET_OBJECTS:flagdnn_test_common_objects>)
else()
  file(GLOB _ascend_native_case_sources CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/tests/common/*.cpp")
  list(FILTER _ascend_native_case_sources EXCLUDE REGEX "/capability_gate\\.cpp$")
endif()
find_library(_ascend_validation_opapi_cv NAMES opapi_cv
  PATHS "${_flagdnn_validation_library_directory}" NO_DEFAULT_PATH REQUIRED)
file(SHA256 "${_ascend_validation_opapi_cv}" _ascend_validation_opapi_cv_sha256)
set(_ascend_extended_operators
  sdpa sdpa_backward conv_dgrad conv_wgrad
  moe_grouped_matmul moe_grouped_matmul_bwd causal_conv1d rope rope_backward resample concatenate gen_index genstats bn_finalize instancenorm adalayernorm
  instancenorm_backward adalayernorm_backward layernorm_backward rmsnorm_backward batchnorm_backward)
# Ascend owns the adapters; workloads stay in the shared test catalog.
if(NOT TARGET flagdnn_validation_cpu_reference)
  add_subdirectory("${PROJECT_SOURCE_DIR}/reference/cpu"
    "${CMAKE_CURRENT_BINARY_DIR}/cpu-reference")
endif()
add_library(flagdnn_validation_ascend_extended STATIC
  ${_ascend_native_case_sources}
  functional/attention_runner.cpp
  functional/dtype_runner.cpp
  functional/aclnn_convolution_backward.cpp
  functional/convolution_backward_runner.cpp
  functional/aclnn_moe_matmul.cpp
  functional/moe_matmul_runner.cpp
  functional/aclnn_causal_convolution.cpp
  functional/causal_convolution_runner.cpp
  functional/aclnn_rope.cpp
  functional/rope_runner.cpp
  functional/aclnn_resample.cpp
  functional/resample_runner.cpp
  functional/aclnn_index.cpp
  functional/index_runner.cpp
  functional/aclnn_extended_normalization.cpp
  functional/normalization_extended_runner.cpp
  functional/aclnn_statistics.cpp
  functional/statistics_runner.cpp
  benchmark/conv_bias_relu_runner.cpp
  benchmark/native_dtype.cpp
  benchmark/native_matrix.cpp
  functional/aclnn_matmul.cpp
  functional/aclnn_matmul_plan.cpp
  functional/aclnn_convolution.cpp
  functional/aclnn_convolution_plan.cpp
  benchmark/native_runner.cpp
  functional/aclnn_activation_backward.cpp
  functional/activation_backward_runner.cpp
  benchmark/activation_backward_runner.cpp
  "${PROJECT_SOURCE_DIR}/src/runtime/json.cpp")
target_compile_features(flagdnn_validation_ascend_extended PUBLIC cxx_std_20)
target_include_directories(flagdnn_validation_ascend_extended PUBLIC
  "${PROJECT_SOURCE_DIR}/tests" "${PROJECT_SOURCE_DIR}/benchmark"
  "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/backends/ascend")
target_link_libraries(flagdnn_validation_ascend_extended PUBLIC FlagDNN::flagdnn
  flagdnn_validation_cpu_reference
  flagdnn_validation_ascend_platform flagdnn_validation_ascend_tensor_io
  flagdnn_validation_ascend_aclnn_nn "${_ascend_validation_opapi_cv}")
target_compile_definitions(flagdnn_validation_ascend_extended PRIVATE
  FLAGDNN_ASCEND_VALIDATION_OPAPI_CV_SHA256="${_ascend_validation_opapi_cv_sha256}"
  FLAGDNN_ASCEND_VALIDATION_CANN_ROOT="${_flagdnn_validation_cann_root}"
  FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT="${_flagdnn_validation_python_module_root}"
  FLAGDNN_ASCEND_VALIDATION_ASCENDCL_BUILD_ID="${_flagdnn_validation_ascendcl_build_id}"
  FLAGDNN_ASCEND_VALIDATION_RUNTIME_BUILD_ID="${_flagdnn_validation_runtime_build_id}"
  FLAGDNN_ASCEND_VALIDATION_TRITON_JIT_SHA256="${_flagdnn_validation_triton_jit_sha256}"
  FLAGDNN_ASCEND_VALIDATION_NNOPBASE_SHA256="${_flagdnn_validation_nnopbase_sha256}"
  FLAGDNN_ASCEND_VALIDATION_OPAPI_MATH_SHA256="${_flagdnn_validation_opapi_math_sha256}"
  FLAGDNN_ASCEND_VALIDATION_OPAPI_NN_SHA256="${_flagdnn_validation_opapi_nn_sha256}")
flagdnn_enable_warnings(flagdnn_validation_ascend_extended)
if(FLAGDNN_BUILD_TESTS)
  flagdnn_register_functional_suite(PLATFORM ascend
    ADAPTER_TARGET flagdnn_validation_ascend_extended BACKEND_TARGET flagdnn_backend_ascend
    OPERATORS ${FLAGDNN_ACTIVATION_BACKWARD_EXTENSIONS} ${_ascend_extended_operators}
    COMMAND_ARGS "${FLAGDNN_CODEGEN_PYTHON}" "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    LABELS "aclnn;device" TIMEOUT 3600 BUILD_RPATH "${_flagdnn_validation_library_directory}")
endif()
if(FLAGDNN_BUILD_BENCHMARKS)
  target_link_libraries(flagdnn_validation_ascend_extended PUBLIC flagdnn_benchmark_cases)
  flagdnn_register_benchmark_suite(PLATFORM ascend
    ADAPTER_TARGET flagdnn_validation_ascend_extended BACKEND_TARGET flagdnn_backend_ascend
    OPERATORS conv_bias_relu ${FLAGDNN_ACTIVATION_BACKWARD_EXTENSIONS} ${_ascend_extended_operators}
    COMMAND_ARGS "${FLAGDNN_CODEGEN_PYTHON}" "${FLAGDNN_CODEGEN_COMPILER}"
    ENVIRONMENT "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>"
    LABELS "aclnn;device" TIMEOUT 3600 BUILD_RPATH "${_flagdnn_validation_library_directory}")
endif()


foreach(operator IN LISTS _ascend_extended_operators FLAGDNN_ACTIVATION_BACKWARD_EXTENSIONS)
  foreach(suite IN ITEMS functional benchmark)
    if(TEST "${suite}.ascend.${operator}")
      set_tests_properties("${suite}.ascend.${operator}" PROPERTIES SKIP_RETURN_CODE 77)
    endif()
  endforeach()
endforeach()

# The shared convolution-gradient performance catalog includes large model
# shapes. IEEE FP32 execution plus first-use compilation can exceed one hour.
foreach(operator IN ITEMS conv_dgrad conv_wgrad)
  if(TEST "benchmark.ascend.${operator}")
    set_tests_properties("benchmark.ascend.${operator}" PROPERTIES TIMEOUT 7200)
  endif()
endforeach()

# Keep every public operator visible to the unified runner, including genuine
# native-library capability gaps. Each gate names the unsupported contract.
add_executable(flagdnn_ascend_capability_gate capability_gate.cpp)
target_compile_features(flagdnn_ascend_capability_gate PRIVATE cxx_std_20)
flagdnn_enable_warnings(flagdnn_ascend_capability_gate)
foreach(operator IN ITEMS rng matmul_fp8 sdpa_fp8 sdpa_fp8_backward)
  foreach(suite IN ITEMS functional benchmark)
    if((suite STREQUAL "functional" AND FLAGDNN_BUILD_TESTS) OR
       (suite STREQUAL "benchmark" AND FLAGDNN_BUILD_BENCHMARKS))
      add_test(NAME "${suite}.ascend.${operator}"
        COMMAND flagdnn_ascend_capability_gate "${operator}")
      set_tests_properties("${suite}.ascend.${operator}" PROPERTIES
        SKIP_RETURN_CODE 77 LABELS "${suite};ascend;${operator};capability")
    endif()
  endforeach()
endforeach()

add_executable(flagdnn_ascend_dtype "${PROJECT_SOURCE_DIR}/benchmark/dtype_main.cpp")
target_link_libraries(flagdnn_ascend_dtype PRIVATE flagdnn_validation_ascend_extended)
flagdnn_enable_warnings(flagdnn_ascend_dtype)
if(FLAGDNN_BUILD_TESTS)
  foreach(operator IN ITEMS add sub mul div mod pow min max neg abs relu identity
      cmp_eq cmp_neq cmp_lt cmp_le cmp_gt cmp_ge binary_select reshape transpose slice reduction)
    add_test(NAME "functional.ascend.${operator}.dtype" COMMAND flagdnn_ascend_dtype
      "${FLAGDNN_CODEGEN_PYTHON}" "${FLAGDNN_CODEGEN_COMPILER}" "${operator}" functional_dtype)
    set_tests_properties("functional.ascend.${operator}.dtype" PROPERTIES
      LABELS "functional;ascend;${operator};dtype;aclnn" RUN_SERIAL TRUE TIMEOUT 3600 SKIP_RETURN_CODE 77
      ENVIRONMENT "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>")
  endforeach()
endif()
if(FLAGDNN_BUILD_BENCHMARKS)
  foreach(category IN ITEMS boolean copy ieee tf32 fp32_output)
    set(runner_category "${category}")
    set(environment "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_ascend>")
    if(category STREQUAL "boolean")
      set(operators logical_and logical_or logical_not)
    elseif(category STREQUAL "copy")
      set(operators identity reshape transpose slice)
    elseif(category STREQUAL "ieee" OR category STREQUAL "tf32")
      set(operators matmul conv_fprop conv_dgrad conv_wgrad)
      set(runner_category precision)
      if(category STREQUAL "ieee")
        list(APPEND environment "FLAGDNN_INPUT_PRECISION=1")
      else()
        list(APPEND environment "FLAGDNN_INPUT_PRECISION=2")
      endif()
    else()
      set(operators reduction)
    endif()
    foreach(operator IN LISTS operators)
      add_test(NAME "benchmark.ascend.${operator}.${category}" COMMAND flagdnn_ascend_dtype
        "${FLAGDNN_CODEGEN_PYTHON}" "${FLAGDNN_CODEGEN_COMPILER}" "${operator}" "${runner_category}")
      set_tests_properties("benchmark.ascend.${operator}.${category}" PROPERTIES
        LABELS "benchmark;ascend;${operator};${category};aclnn" RUN_SERIAL TRUE TIMEOUT 3600 SKIP_RETURN_CODE 77
        ENVIRONMENT "${environment}")
    endforeach()
  endforeach()
endif()

if(FLAGDNN_BUILD_TESTS)
  add_test(NAME core.ascend.codegen_contract
    COMMAND "${FLAGDNN_CODEGEN_PYTHON}"
      "${CMAKE_CURRENT_SOURCE_DIR}/compiler_contract.py")
  set_tests_properties(core.ascend.codegen_contract PROPERTIES
    LABELS "core;ascend;contract;no_device" TIMEOUT 60)
endif()
if(FLAGDNN_BUILD_TESTS)
  add_executable(flagdnn_ascend_native_plan_contract native_plan_contract.cpp)
  target_link_libraries(flagdnn_ascend_native_plan_contract PRIVATE flagdnn_validation_ascend_extended)
  flagdnn_enable_warnings(flagdnn_ascend_native_plan_contract)
  add_test(NAME core.ascend.native_plan_contract COMMAND flagdnn_ascend_native_plan_contract)
  set_tests_properties(core.ascend.native_plan_contract PROPERTIES
    LABELS "core;ascend;contract;no_device" TIMEOUT 60)
  add_executable(flagdnn_ascend_artifact_contract
    artifact_contract.cpp
    ../extended_artifact.cpp
    ../error.cpp
    "${PROJECT_SOURCE_DIR}/src/runtime/json.cpp"
    "${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp")
  target_compile_features(flagdnn_ascend_artifact_contract PRIVATE cxx_std_20)
  target_include_directories(flagdnn_ascend_artifact_contract PRIVATE
    "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/include" "${PROJECT_BINARY_DIR}/backends/ascend"
    "$<TARGET_PROPERTY:Ascend::ascendcl,INTERFACE_INCLUDE_DIRECTORIES>")
  flagdnn_enable_warnings(flagdnn_ascend_artifact_contract)
  add_test(NAME core.ascend.artifact_contract
    COMMAND "${FLAGDNN_CODEGEN_PYTHON}"
      "${CMAKE_CURRENT_SOURCE_DIR}/artifact_contract.py"
      $<TARGET_FILE:flagdnn_ascend_artifact_contract>)
  set_tests_properties(core.ascend.artifact_contract PROPERTIES
    LABELS "core;ascend;contract;no_device" TIMEOUT 60)
endif()
