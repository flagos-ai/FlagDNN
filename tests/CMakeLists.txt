include("${PROJECT_SOURCE_DIR}/cmake/Operators.cmake")


# Platform-neutral case data and FlagDNN Graph construction. Platform adapters
# consume these objects and provide device I/O plus an independent reference.
add_library(flagdnn_test_common_objects OBJECT
  common/add.cpp
  common/attention.cpp
  common/composite.cpp
  common/convolution.cpp
  common/layout.cpp
  common/matmul.cpp
  common/normalization.cpp
  common/pointwise.cpp
  common/reduction.cpp)
target_compile_features(flagdnn_test_common_objects PUBLIC cxx_std_20)
target_include_directories(flagdnn_test_common_objects PUBLIC
  "${PROJECT_SOURCE_DIR}/tests")
target_link_libraries(flagdnn_test_common_objects PUBLIC FlagDNN::flagdnn)
flagdnn_enable_warnings(flagdnn_test_common_objects)

# A backend validation CMake file calls this once after defining its adapter archive. The
# common layer stays unaware of CUDA, cuDNN, CANN, ACL, or any other device SDK.
function(flagdnn_register_functional_suite)
  set(options)
  set(one_value_arguments
    PLATFORM ADAPTER_TARGET BACKEND_TARGET TIMEOUT BUILD_RPATH)
  set(multi_value_arguments COMMAND_ARGS ENVIRONMENT LABELS)
  cmake_parse_arguments(
    FLAGDNN_SUITE
    "${options}"
    "${one_value_arguments}"
    "${multi_value_arguments}"
    ${ARGN})

  if(NOT FLAGDNN_SUITE_PLATFORM MATCHES "^[a-z][a-z0-9_]*$")
    message(FATAL_ERROR "Functional-test platform name is invalid")
  endif()
  if(NOT TARGET "${FLAGDNN_SUITE_ADAPTER_TARGET}")
    message(FATAL_ERROR
      "Functional-test adapter target does not exist: ${FLAGDNN_SUITE_ADAPTER_TARGET}")
  endif()
  if(NOT TARGET "${FLAGDNN_SUITE_BACKEND_TARGET}")
    message(FATAL_ERROR
      "Functional-test backend target does not exist: ${FLAGDNN_SUITE_BACKEND_TARGET}")
  endif()
  if(NOT FLAGDNN_SUITE_TIMEOUT)
    set(FLAGDNN_SUITE_TIMEOUT 600)
  endif()

  foreach(operator IN LISTS FLAGDNN_FUNCTIONAL_OPERATORS)
    set(target "flagdnn_test_${FLAGDNN_SUITE_PLATFORM}_${operator}")
    add_executable("${target}"
      "${PROJECT_SOURCE_DIR}/tests/test_${operator}.cpp")
    target_compile_features("${target}" PRIVATE cxx_std_20)
    target_link_libraries("${target}" PRIVATE
      "${FLAGDNN_SUITE_ADAPTER_TARGET}")
    add_dependencies("${target}" "${FLAGDNN_SUITE_BACKEND_TARGET}")
    flagdnn_enable_warnings("${target}")
    if(FLAGDNN_SUITE_BUILD_RPATH)
      set_target_properties("${target}" PROPERTIES
        BUILD_RPATH "${FLAGDNN_SUITE_BUILD_RPATH}")
    endif()

    set(test_name "functional.${FLAGDNN_SUITE_PLATFORM}.${operator}")
    add_test(NAME "${test_name}"
      COMMAND "${target}" ${FLAGDNN_SUITE_COMMAND_ARGS})
    set_tests_properties("${test_name}" PROPERTIES
      LABELS
        "functional;correctness;${FLAGDNN_SUITE_PLATFORM};${operator};${FLAGDNN_SUITE_LABELS}"
      RUN_SERIAL TRUE
      ENVIRONMENT "${FLAGDNN_SUITE_ENVIRONMENT}"
      TIMEOUT "${FLAGDNN_SUITE_TIMEOUT}")
  endforeach()
endfunction()

# Core tests exercise public headers, the C ABI, JSON, and the backend plugin
# contract without importing any real platform SDK.
add_executable(flagdnn_json_contract
  core/json_contract.cpp
  "${PROJECT_SOURCE_DIR}/src/runtime/json.cpp")
target_compile_features(flagdnn_json_contract PRIVATE cxx_std_20)
target_include_directories(flagdnn_json_contract PRIVATE
  "${PROJECT_SOURCE_DIR}/src"
  "${PROJECT_SOURCE_DIR}")
flagdnn_enable_warnings(flagdnn_json_contract)
add_test(NAME core.json_contract COMMAND flagdnn_json_contract)

flagdnn_add_backend_plugin(contract
  SOURCES
    core/contract_backend.cpp
    "${PROJECT_SOURCE_DIR}/src/runtime/json.cpp"
    "${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp")
add_test(NAME core.backend_dependency_boundary
  COMMAND "${CMAKE_COMMAND}"
    "-DBACKEND_LIBRARY=$<TARGET_FILE:flagdnn_backend_contract>"
    "-DBACKEND_NAME=contract"
    "-DFORBIDDEN_DEPENDENCIES=libcuda;libcudnn;libcublas;libascendcl;libacl;libpython;libtorch"
    -P "${PROJECT_SOURCE_DIR}/cmake/VerifyBackendPlugin.cmake")

add_executable(flagdnn_contract_compiler
  core/contract_compiler.cpp
  "${PROJECT_SOURCE_DIR}/src/runtime/json.cpp"
  "${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp")
target_compile_features(flagdnn_contract_compiler PRIVATE cxx_std_20)
target_include_directories(flagdnn_contract_compiler PRIVATE
  "${PROJECT_SOURCE_DIR}/src")
flagdnn_enable_warnings(flagdnn_contract_compiler)

add_executable(flagdnn_backend_contract_smoke core/backend_contract_smoke.cpp)
target_compile_features(flagdnn_backend_contract_smoke PRIVATE cxx_std_20)
target_link_libraries(flagdnn_backend_contract_smoke PRIVATE FlagDNN::flagdnn)
flagdnn_enable_warnings(flagdnn_backend_contract_smoke)
add_dependencies(flagdnn_backend_contract_smoke
  flagdnn_backend_contract flagdnn_contract_compiler)
add_test(NAME core.backend_contract
  COMMAND flagdnn_backend_contract_smoke
    "$<TARGET_FILE:flagdnn_contract_compiler>"
    contract)
set_tests_properties(core.backend_contract PROPERTIES
  ENVIRONMENT
    "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:flagdnn_backend_contract>;FLAGDNN_EXECUTION_ENGINE=external_artifact"
  TIMEOUT 60)

add_executable(flagdnn_c_header_smoke core/c_header_smoke.c)
target_link_libraries(flagdnn_c_header_smoke PRIVATE FlagDNN::flagdnn)
set_target_properties(flagdnn_c_header_smoke PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES)
add_test(NAME core.c_header COMMAND flagdnn_c_header_smoke)

add_executable(flagdnn_api_contract core/native_api_contract.c)
target_link_libraries(flagdnn_api_contract PRIVATE FlagDNN::flagdnn)
set_target_properties(flagdnn_api_contract PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES)
add_test(NAME core.c_api COMMAND flagdnn_api_contract)

add_executable(flagdnn_convolution_backward_api_contract
  core/convolution_backward_c_api_contract.c)
target_link_libraries(flagdnn_convolution_backward_api_contract
  PRIVATE FlagDNN::flagdnn)
set_target_properties(flagdnn_convolution_backward_api_contract PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES)
add_test(NAME core.convolution_backward_api
  COMMAND flagdnn_convolution_backward_api_contract)

add_executable(flagdnn_batchnorm_inference_api_contract
  core/batchnorm_inference_c_api_contract.c)
target_link_libraries(flagdnn_batchnorm_inference_api_contract
  PRIVATE FlagDNN::flagdnn)
set_target_properties(flagdnn_batchnorm_inference_api_contract PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES)
add_test(NAME core.batchnorm_inference_api
  COMMAND flagdnn_batchnorm_inference_api_contract)

add_executable(flagdnn_cpp_header_smoke core/cpp_header_smoke.cpp)
target_compile_features(flagdnn_cpp_header_smoke PRIVATE cxx_std_20)
target_link_libraries(flagdnn_cpp_header_smoke PRIVATE FlagDNN::flagdnn)
add_test(NAME core.cpp_header COMMAND flagdnn_cpp_header_smoke)

add_executable(flagdnn_frontend_contract core/frontend_header_smoke.cpp)
target_compile_features(flagdnn_frontend_contract PRIVATE cxx_std_20)
target_link_libraries(flagdnn_frontend_contract PRIVATE FlagDNN::flagdnn)
flagdnn_enable_warnings(flagdnn_frontend_contract)
add_test(NAME core.frontend_api COMMAND flagdnn_frontend_contract)

add_test(NAME core.test_architecture
  COMMAND "${CMAKE_COMMAND}"
    "-DSOURCE_ROOT=${PROJECT_SOURCE_DIR}"
    -P "${PROJECT_SOURCE_DIR}/cmake/VerifyPerOperatorTestLayout.cmake")
