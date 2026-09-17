# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

foreach(_required IN ITEMS
    SOURCE_ROOT TEST_ROOT PROBE_EXECUTABLE C_COMPILER COREX_ROOT
    COREX_CUDNN_INCLUDE COREX_CUDNN COREX_CUDART COREX_DRIVER)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "${_required} is required")
  endif()
endforeach()

file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}/child")

set(_required_classic_symbols
  cudnnGetVersion
  cudnnGetCudartVersion
  cudnnGetErrorString
  cudnnCreate
  cudnnDestroy
  cudnnSetStream
  cudnnCreateTensorDescriptor
  cudnnSetTensorNdDescriptor
  cudnnDestroyTensorDescriptor
  cudnnTransformTensor
  cudnnCreateOpTensorDescriptor
  cudnnSetOpTensorDescriptor
  cudnnOpTensor
  cudnnCreateReduceTensorDescriptor
  cudnnSetReduceTensorDescriptor
  cudnnGetReductionWorkspaceSize
  cudnnReduceTensor
  cudnnDestroyReduceTensorDescriptor
  cudnnCreateActivationDescriptor
  cudnnSetActivationDescriptor
  cudnnActivationForward
  cudnnActivationBackward
  cudnnDestroyActivationDescriptor
  cudnnCreateConvolutionDescriptor
  cudnnSetConvolutionNdDescriptor
  cudnnConvolutionForward
  cudnnConvolutionBackwardData
  cudnnConvolutionBackwardFilter
  cudnnDestroyConvolutionDescriptor
  cudnnBatchNormalizationForwardTraining
  cudnnBatchNormalizationForwardInference
  cudnnBatchNormalizationBackward)

function(_write_fake_cudnn root soname runtime_version omit_symbol header_version)
  file(MAKE_DIRECTORY "${root}/include" "${root}/lib64")
  file(COPY "${COREX_CUDNN_INCLUDE}/cudnn.h" DESTINATION "${root}/include")
  if(NOT header_version EQUAL 7605)
    file(READ "${root}/include/cudnn.h" _header)
    math(EXPR _header_major "${header_version} / 1000")
    math(EXPR _header_minor "(${header_version} % 1000) / 100")
    math(EXPR _header_patch "${header_version} % 100")
    string(REGEX REPLACE
      "#[ \t]*define[ \t]+CUDNN_MAJOR[ \t]+[0-9]+"
      "#define CUDNN_MAJOR ${_header_major}" _header "${_header}")
    string(REGEX REPLACE
      "#[ \t]*define[ \t]+CUDNN_MINOR[ \t]+[0-9]+"
      "#define CUDNN_MINOR ${_header_minor}" _header "${_header}")
    string(REGEX REPLACE
      "#[ \t]*define[ \t]+CUDNN_PATCHLEVEL[ \t]+[0-9]+"
      "#define CUDNN_PATCHLEVEL ${_header_patch}" _header "${_header}")
    file(WRITE "${root}/include/cudnn.h" "${_header}")
  endif()

  set(_source
    "#include <stddef.h>\n"
    "size_t cudnnGetVersion(void) { return ${runtime_version}; }\n"
    "size_t cudnnGetCudartVersion(void) { return 0; }\n"
    "const char *cudnnGetErrorString(int status) { "
    "(void)status; return \"fake cuDNN status\"; }\n"
    "int cudnnCreate(void **handle) { *handle = (void*)1; return 0; }\n"
    "int cudnnDestroy(void *handle) { (void)handle; return 0; }\n"
    "int cudnnSetStream(void *handle, void *stream) { "
    "(void)handle; (void)stream; return 0; }\n")
  foreach(_symbol IN LISTS _required_classic_symbols)
    if(_symbol STREQUAL "cudnnGetVersion" OR
       _symbol STREQUAL "cudnnGetCudartVersion" OR
       _symbol STREQUAL "cudnnGetErrorString" OR
       _symbol STREQUAL "cudnnCreate" OR
       _symbol STREQUAL "cudnnDestroy" OR
       _symbol STREQUAL "cudnnSetStream" OR
       _symbol STREQUAL "${omit_symbol}")
      continue()
    endif()
    string(APPEND _source "int ${_symbol}(void) { return 0; }\n")
  endforeach()
  file(WRITE "${root}/fake_cudnn.c" "${_source}")
  execute_process(
    COMMAND "${C_COMPILER}" -shared -fPIC "-Wl,-soname,${soname}"
      -o "${root}/lib64/libcudnn.so.7" "${root}/fake_cudnn.c"
    RESULT_VARIABLE _compile_result
    OUTPUT_VARIABLE _compile_output
    ERROR_VARIABLE _compile_error)
  if(NOT _compile_result EQUAL 0)
    message(FATAL_ERROR
      "cannot create fake cuDNN fixture: ${_compile_output}${_compile_error}")
  endif()

  file(WRITE "${root}/fake_cudart.c" "int fake_cudart(void) { return 0; }\n")
  execute_process(
    COMMAND "${C_COMPILER}" -shared -fPIC
      "-Wl,-soname,libcudart.so.10.2"
      -o "${root}/lib64/libcudart.so.10.2" "${root}/fake_cudart.c"
    RESULT_VARIABLE _cudart_result
    OUTPUT_VARIABLE _cudart_output
    ERROR_VARIABLE _cudart_error)
  if(NOT _cudart_result EQUAL 0)
    message(FATAL_ERROR
      "cannot create fake CUDA Runtime fixture: "
      "${_cudart_output}${_cudart_error}")
  endif()
endfunction()

_write_fake_cudnn(
  "${TEST_ROOT}/good" libcudnn.so.7 7605 "" 7605)
_write_fake_cudnn(
  "${TEST_ROOT}/other-root" libcudnn.so.7 7605 "" 7605)
_write_fake_cudnn(
  "${TEST_ROOT}/wrong-soname" libcudnn.so.8 7605 "" 7605)
_write_fake_cudnn(
  "${TEST_ROOT}/header-mismatch" libcudnn.so.7 7605 "" 8900)
_write_fake_cudnn(
  "${TEST_ROOT}/runtime-mismatch-root" libcudnn.so.7 7604 "" 7605)
_write_fake_cudnn(
  "${TEST_ROOT}/missing-classic" libcudnn.so.7 7605
  cudnnReduceTensor 7605)
file(MAKE_DIRECTORY "${TEST_ROOT}/host-nvidia/include"
  "${TEST_ROOT}/host-nvidia/lib64")
file(COPY "${TEST_ROOT}/good/include/cudnn.h"
  DESTINATION "${TEST_ROOT}/host-nvidia/include")
file(COPY "${TEST_ROOT}/good/lib64/libcudnn.so.7"
  DESTINATION "${TEST_ROOT}/host-nvidia/lib64")
file(MAKE_DIRECTORY "${TEST_ROOT}/runtime-mismatch")
file(COPY "${TEST_ROOT}/runtime-mismatch-root/lib64/libcudnn.so.7"
  DESTINATION "${TEST_ROOT}/runtime-mismatch")

file(WRITE "${TEST_ROOT}/child/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.23)
project(CorexCudnnResolverContract LANGUAGES C CXX)

foreach(_required IN ITEMS SOURCE_ROOT TEST_ROOT CASE REAL_COREX_ROOT
    REAL_CUDNN_INCLUDE REAL_CUDNN)
  if(NOT DEFINED ${_required})
    message(FATAL_ERROR "missing child input ${_required}")
  endif()
endforeach()
include("${SOURCE_ROOT}/backends/iluvatar/cmake/ResolveCoreX.cmake")

if(CASE STREQUAL "valid")
  set(_selected_root "${REAL_COREX_ROOT}")
  set(FLAGDNN_ILUVATAR_CUDNN_INCLUDE_DIR
    "${REAL_CUDNN_INCLUDE}" CACHE PATH "" FORCE)
  set(FLAGDNN_ILUVATAR_CUDNN_LIBRARY
    "${REAL_CUDNN}" CACHE FILEPATH "" FORCE)
elseif(CASE STREQUAL "host_nvidia")
  set(_selected_root "${REAL_COREX_ROOT}")
  set(FLAGDNN_ILUVATAR_CUDNN_INCLUDE_DIR
    "${TEST_ROOT}/host-nvidia/include" CACHE PATH "" FORCE)
  set(FLAGDNN_ILUVATAR_CUDNN_LIBRARY
    "${TEST_ROOT}/host-nvidia/lib64/libcudnn.so.7" CACHE FILEPATH "" FORCE)
elseif(CASE STREQUAL "mixed_root")
  set(_selected_root "${TEST_ROOT}/good")
  set(FLAGDNN_ILUVATAR_CUDNN_INCLUDE_DIR
    "${TEST_ROOT}/good/include" CACHE PATH "" FORCE)
  set(FLAGDNN_ILUVATAR_CUDNN_LIBRARY
    "${TEST_ROOT}/other-root/lib64/libcudnn.so.7" CACHE FILEPATH "" FORCE)
elseif(CASE STREQUAL "wrong_soname")
  set(_selected_root "${TEST_ROOT}/wrong-soname")
  set(FLAGDNN_ILUVATAR_CUDNN_INCLUDE_DIR
    "${TEST_ROOT}/wrong-soname/include" CACHE PATH "" FORCE)
  set(FLAGDNN_ILUVATAR_CUDNN_LIBRARY
    "${TEST_ROOT}/wrong-soname/lib64/libcudnn.so.7" CACHE FILEPATH "" FORCE)
elseif(CASE STREQUAL "header_mismatch")
  set(_selected_root "${TEST_ROOT}/header-mismatch")
  set(FLAGDNN_ILUVATAR_CUDNN_INCLUDE_DIR
    "${TEST_ROOT}/header-mismatch/include" CACHE PATH "" FORCE)
  set(FLAGDNN_ILUVATAR_CUDNN_LIBRARY
    "${TEST_ROOT}/header-mismatch/lib64/libcudnn.so.7" CACHE FILEPATH "" FORCE)
elseif(CASE STREQUAL "missing_classic")
  set(_selected_root "${TEST_ROOT}/missing-classic")
  set(FLAGDNN_ILUVATAR_CUDNN_INCLUDE_DIR
    "${TEST_ROOT}/missing-classic/include" CACHE PATH "" FORCE)
  set(FLAGDNN_ILUVATAR_CUDNN_LIBRARY
    "${TEST_ROOT}/missing-classic/lib64/libcudnn.so.7" CACHE FILEPATH "" FORCE)
else()
  message(FATAL_ERROR "unknown child case ${CASE}")
endif()

flagdnn_iluvatar_resolve_corex_cudnn(CUDNN "${_selected_root}")
if(CASE STREQUAL "valid")
  foreach(_output IN ITEMS CUDNN_INCLUDE_DIR CUDNN_LIBRARY
      CUDNN_CUDA_RUNTIME_LIBRARY CUDNN_HEADER_VERSION CUDNN_SONAME)
    if(NOT DEFINED ${_output} OR "${${_output}}" STREQUAL "")
      message(FATAL_ERROR "missing resolver output ${_output}")
    endif()
  endforeach()
  if(NOT CUDNN_HEADER_VERSION EQUAL 7605 OR
     NOT CUDNN_SONAME STREQUAL "libcudnn.so.7")
    message(FATAL_ERROR "unexpected resolved CoreX cuDNN identity")
  endif()
endif()
]=])

function(_run_resolver_case name expect_success expected_error)
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
      -S "${TEST_ROOT}/child"
      -B "${TEST_ROOT}/build-${name}"
      "-DSOURCE_ROOT=${SOURCE_ROOT}"
      "-DTEST_ROOT=${TEST_ROOT}"
      "-DCASE=${name}"
      "-DREAL_COREX_ROOT=${COREX_ROOT}"
      "-DREAL_CUDNN_INCLUDE=${COREX_CUDNN_INCLUDE}"
      "-DREAL_CUDNN=${COREX_CUDNN}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  set(_output "${_stdout}\n${_stderr}")
  if(expect_success)
    if(NOT _result EQUAL 0)
      message(FATAL_ERROR
        "positive CoreX cuDNN resolver case ${name} failed:\n${_output}")
    endif()
  else()
    if(_result EQUAL 0)
      message(FATAL_ERROR
        "negative CoreX cuDNN resolver case ${name} unexpectedly passed")
    endif()
    if(NOT _output MATCHES "${expected_error}")
      message(FATAL_ERROR
        "negative CoreX cuDNN resolver case ${name} returned an unexpected "
        "error:\n${_output}")
    endif()
  endif()
endfunction()

_run_resolver_case(valid TRUE "")
_run_resolver_case(host_nvidia FALSE "outside the selected CoreX root")
_run_resolver_case(mixed_root FALSE "outside the selected CoreX root")
_run_resolver_case(wrong_soname FALSE "SONAME libcudnn.so.7")
_run_resolver_case(header_mismatch FALSE "header version mismatch")
_run_resolver_case(missing_classic FALSE "missing classic symbol")

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "LD_LIBRARY_PATH=${COREX_ROOT}/lib64:${COREX_ROOT}/lib"
    "${PROBE_EXECUTABLE}" "${COREX_CUDNN}"
  RESULT_VARIABLE _probe_result
  OUTPUT_VARIABLE _probe_output
  ERROR_VARIABLE _probe_error)
if(NOT _probe_result EQUAL 0)
  message(FATAL_ERROR
    "real CoreX cuDNN environment probe failed:\n${_probe_output}${_probe_error}")
endif()
if(NOT _probe_output MATCHES
   "header=7605 runtime=7605.*target=corex_71")
  message(FATAL_ERROR
    "real CoreX cuDNN environment identity is incomplete: ${_probe_output}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "LD_LIBRARY_PATH=${TEST_ROOT}/runtime-mismatch:${COREX_ROOT}/lib64:${COREX_ROOT}/lib"
    "${PROBE_EXECUTABLE}"
    "${TEST_ROOT}/runtime-mismatch/libcudnn.so.7"
  RESULT_VARIABLE _runtime_mismatch_result
  OUTPUT_VARIABLE _runtime_mismatch_output
  ERROR_VARIABLE _runtime_mismatch_error)
set(_runtime_mismatch_log
  "${_runtime_mismatch_output}${_runtime_mismatch_error}")
if(_runtime_mismatch_result EQUAL 0 OR
   NOT _runtime_mismatch_log MATCHES "runtime version mismatch")
  message(FATAL_ERROR
    "runtime-version-mismatch fixture was not rejected:\n"
    "${_runtime_mismatch_log}")
endif()

set(_dependency_script
  "${SOURCE_ROOT}/backends/iluvatar/validation/VerifyReferenceDependencies.cmake")
execute_process(
  COMMAND "${CMAKE_COMMAND}"
    "-DREFERENCE_EXECUTABLE=${PROBE_EXECUTABLE}"
    "-DCOREX_ROOT=${COREX_ROOT}"
    "-DCOREX_CUDNN=${COREX_CUDNN}"
    "-DCOREX_CUDART=${COREX_CUDART}"
    "-DCOREX_DRIVER=${COREX_DRIVER}"
    "-DSOURCE_DIR=${SOURCE_ROOT}/backends/iluvatar/validation"
    -P "${_dependency_script}"
  RESULT_VARIABLE _dependency_result
  OUTPUT_VARIABLE _dependency_output
  ERROR_VARIABLE _dependency_error)
if(NOT _dependency_result EQUAL 0)
  message(FATAL_ERROR
    "positive reference dependency boundary failed:\n"
    "${_dependency_output}${_dependency_error}")
endif()

# Check the literal exception separately from direct calls and real imports.
function(_check_reference_boundary name executable source_name source_text
    expect_success expected_error)
  set(_source_dir "${TEST_ROOT}/reference-boundary-${name}")
  file(MAKE_DIRECTORY "${_source_dir}")
  file(WRITE "${_source_dir}/${source_name}" "${source_text}")
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
      "-DREFERENCE_EXECUTABLE=${executable}"
      "-DCOREX_ROOT=${COREX_ROOT}"
      "-DCOREX_CUDNN=${COREX_CUDNN}"
      "-DCOREX_CUDART=${COREX_CUDART}"
      "-DCOREX_DRIVER=${COREX_DRIVER}"
      "-DSOURCE_DIR=${_source_dir}"
      -P "${_dependency_script}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  set(_log "${_stdout}${_stderr}")
  if(expect_success)
    if(NOT _result EQUAL 0)
      message(FATAL_ERROR "reference boundary ${name} failed:\n${_log}")
    endif()
  elseif(_result EQUAL 0 OR NOT _log MATCHES "${expected_error}")
    message(FATAL_ERROR
      "reference boundary ${name} was not rejected as expected:\n${_log}")
  endif()
  message(STATUS "PASS reference boundary fixture ${name}")
endfunction()

set(_capability_query [=[
#include <dlfcn.h>
void *query() { return dlsym(RTLD_DEFAULT, "cudnnBackendCreateDescriptor"); }
]=])
foreach(_probe IN ITEMS dnn_api_probe dnn_capability_gate extended_reference)
  _check_reference_boundary("${_probe}-query" "${PROBE_EXECUTABLE}"
    "${_probe}.cpp" "${_capability_query}" TRUE "")
  _check_reference_boundary("${_probe}-direct-call" "${PROBE_EXECUTABLE}"
    "${_probe}.cpp"
    "${_capability_query}\nvoid call() { cudnnBackendCreateDescriptor(); }\n"
    FALSE "forbidden reference dependency token")
endforeach()
_check_reference_boundary(frontend-header "${PROBE_EXECUTABLE}"
  dnn_api_probe.cpp "#include <cudnn_frontend.h>\n"
  FALSE "forbidden reference dependency token")
_check_reference_boundary(unlisted-source "${PROBE_EXECUTABLE}"
  unrelated.cpp "${_capability_query}"
  FALSE "forbidden reference dependency token")
_check_reference_boundary(other-graph-symbol "${PROBE_EXECUTABLE}"
  dnn_api_probe.cpp [=[void *query() { return dlsym(0, "cudnnBackendExecute"); }]=]
  FALSE "forbidden reference dependency token")

# A clean source scan must not hide a forbidden symbol imported by a binary.
# The fixture is inspected only; it is never executed as a DNN implementation.
set(_graph_fixture "${TEST_ROOT}/reference-graph-import")
file(MAKE_DIRECTORY "${_graph_fixture}")
file(WRITE "${_graph_fixture}/library.c"
  "int cudnnBackendCreateDescriptor(void) { return 0; }\n")
execute_process(
  COMMAND "${C_COMPILER}" -shared -fPIC
    -Wl,-soname,libreference_graph_fixture.so
    "${_graph_fixture}/library.c"
    -o "${_graph_fixture}/libreference_graph_fixture.so"
  RESULT_VARIABLE _library_result
  OUTPUT_VARIABLE _library_output ERROR_VARIABLE _library_error)
if(NOT _library_result EQUAL 0)
  message(FATAL_ERROR
    "cannot build reference graph fixture: ${_library_output}${_library_error}")
endif()
file(WRITE "${_graph_fixture}/main.c"
  "extern int cudnnBackendCreateDescriptor(void);\n"
  "int main(void) { return cudnnBackendCreateDescriptor(); }\n")
execute_process(
  COMMAND "${C_COMPILER}" "${_graph_fixture}/main.c"
    -Wl,--no-as-needed
    "${COREX_CUDNN}" "${COREX_CUDART}" "${COREX_DRIVER}"
    "${_graph_fixture}/libreference_graph_fixture.so"
    "-Wl,-rpath,${_graph_fixture}"
    -o "${_graph_fixture}/importer"
  RESULT_VARIABLE _importer_result
  OUTPUT_VARIABLE _importer_output ERROR_VARIABLE _importer_error)
if(NOT _importer_result EQUAL 0)
  message(FATAL_ERROR
    "cannot build reference graph importer: ${_importer_output}${_importer_error}")
endif()
_check_reference_boundary(imported-graph-symbol "${_graph_fixture}/importer"
  dnn_api_probe.cpp "${_capability_query}"
  FALSE "reference executable imports a forbidden compute symbol")

string(STRIP "${_probe_output}" _probe_output)
string(REPLACE "\\" "\\\\" _probe_json "${_probe_output}")
string(REPLACE "\"" "\\\"" _probe_json "${_probe_json}")
file(WRITE "${TEST_ROOT}/corex-cudnn-environment.json"
  "{\n"
  "  \"corex_root\": \"${COREX_ROOT}\",\n"
  "  \"cudnn_header\": 7605,\n"
  "  \"cudnn_runtime\": 7605,\n"
  "  \"target\": \"corex_71\",\n"
  "  \"library\": \"${COREX_CUDNN}\",\n"
  "  \"probe\": \"${_probe_json}\"\n"
  "}\n")

message(STATUS "${_probe_output}")
message(STATUS "PASS CoreX cuDNN environment and negative contracts")
