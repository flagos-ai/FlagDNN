# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED SOURCE_ROOT OR NOT DEFINED TEST_ROOT)
  message(FATAL_ERROR "SOURCE_ROOT and TEST_ROOT are required")
endif()

set(_corex_resolver
  "${SOURCE_ROOT}/backends/iluvatar/cmake/ResolveCoreX.cmake")
set(_jit_resolver
  "${SOURCE_ROOT}/backends/iluvatar/cmake/ResolveTritonJIT.cmake")
if(NOT EXISTS "${_corex_resolver}" OR NOT EXISTS "${_jit_resolver}")
  message(FATAL_ERROR "Iluvatar CMake resolvers are missing")
endif()

file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}/child")

function(_write_fake_corex root include_cuda driver_library)
  file(MAKE_DIRECTORY "${root}/include" "${root}/lib64")
  if(include_cuda)
    file(WRITE "${root}/include/cuda.h"
      "#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75\n"
      "#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76\n"
      "typedef struct CUuuid_st { char bytes[16]; } CUuuid;\n"
      "int cuDeviceGetUuid(CUuuid*, int);\n"
      "int cuDriverGetVersion(int*);\n")
  endif()
  if(driver_library)
    file(WRITE "${root}/lib64/libcuda.so" "fake-corex-driver\n")
  endif()
endfunction()

function(_write_fake_jit root backend with_ix_definition)
  file(MAKE_DIRECTORY
    "${root}/lib/cmake/TritonJIT"
    "${root}/lib"
    "${root}/include/triton_jit/backends"
    "${root}/share/triton_jit/scripts")
  file(WRITE "${root}/lib/libtriton_jit.so" "fake-triton-jit\n")
  foreach(_header IN ITEMS
      triton_jit/backends/ix_backend.h
      triton_jit/triton_jit_function.h
      triton_jit/jit_utils.h)
    file(WRITE "${root}/include/${_header}" "// ${_header}\n")
  endforeach()
  file(WRITE "${root}/share/triton_jit/scripts/standalone_compile.py"
    "# fake standalone compiler\n")
  file(WRITE "${root}/share/triton_jit/scripts/gen_ssig.py"
    "# fake signature helper\n")
  if(with_ix_definition)
    set(_definitions "BACKEND_IX")
  else()
    set(_definitions "")
  endif()
  file(WRITE "${root}/lib/cmake/TritonJIT/TritonJITConfig.cmake"
    "set(TritonJIT_BACKEND \"${backend}\")\n"
    "if(NOT TARGET TritonJIT::triton_jit)\n"
    "  add_library(TritonJIT::triton_jit SHARED IMPORTED)\n"
    "  set_target_properties(TritonJIT::triton_jit PROPERTIES\n"
    "    IMPORTED_LOCATION \"${root}/lib/libtriton_jit.so\"\n"
    "    INTERFACE_INCLUDE_DIRECTORIES \"${root}/include\"\n"
    "    INTERFACE_COMPILE_DEFINITIONS \"${_definitions}\")\n"
    "endif()\n")
endfunction()

_write_fake_corex("${TEST_ROOT}/missing-header" FALSE TRUE)
_write_fake_corex("${TEST_ROOT}/missing-driver" TRUE FALSE)
_write_fake_jit("${TEST_ROOT}/jit-cuda" CUDA TRUE)
_write_fake_jit("${TEST_ROOT}/jit-no-ix-definition" IX FALSE)

file(WRITE "${TEST_ROOT}/child/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.23)
project(FlagDNNIluvatarCMakeContract LANGUAGES C CXX)

if(NOT DEFINED SOURCE_ROOT OR NOT DEFINED TEST_ROOT OR NOT DEFINED CASE)
  message(FATAL_ERROR "SOURCE_ROOT, TEST_ROOT and CASE are required")
endif()

if(CASE STREQUAL "external_artifact")
  set(FLAGDNN_EXECUTION_ENGINE external_artifact CACHE STRING "" FORCE)
  set(FLAGDNN_CODEGEN_PYTHON /usr/local/bin/python3 CACHE FILEPATH "" FORCE)
  add_subdirectory(
    "${SOURCE_ROOT}/backends/iluvatar"
    "${CMAKE_CURRENT_BINARY_DIR}/iluvatar")
  message(FATAL_ERROR "external_artifact unexpectedly configured")
endif()

include("${SOURCE_ROOT}/backends/iluvatar/cmake/ResolveCoreX.cmake")
include("${SOURCE_ROOT}/backends/iluvatar/cmake/ResolveTritonJIT.cmake")

if(CASE STREQUAL "valid")
  set(FLAGDNN_ILUVATAR_COREX_ROOT /usr/local/corex CACHE PATH "" FORCE)
  set(FLAGDNN_ILUVATAR_TRITON_JIT_DIR
    /usr/local/lib/cmake/TritonJIT CACHE PATH "" FORCE)
  flagdnn_iluvatar_resolve_corex(COREX)
  flagdnn_iluvatar_resolve_triton_jit(JIT)
  foreach(_required IN ITEMS
      COREX_ROOT COREX_CUDA_INCLUDE_DIR COREX_CUDA_DRIVER_LIBRARY
      COREX_SDK_VERSION JIT_TARGET JIT_CONFIG_DIR JIT_LIBRARY
      JIT_INCLUDE_DIR JIT_SCRIPT_DIR JIT_BACKEND)
    if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
      message(FATAL_ERROR "resolver output is missing: ${_required}")
    endif()
  endforeach()
  if(NOT COREX_ROOT STREQUAL "/usr/local/corex-4.4.0")
    message(FATAL_ERROR "CoreX root was not canonicalized: ${COREX_ROOT}")
  endif()
  if(NOT JIT_BACKEND STREQUAL "IX")
    message(FATAL_ERROR "valid package did not resolve as IX")
  endif()
elseif(CASE STREQUAL "missing_cuda_h")
  set(FLAGDNN_ILUVATAR_COREX_ROOT
    "${TEST_ROOT}/missing-header" CACHE PATH "" FORCE)
  flagdnn_iluvatar_resolve_corex(COREX)
elseif(CASE STREQUAL "missing_libcuda")
  set(FLAGDNN_ILUVATAR_COREX_ROOT
    "${TEST_ROOT}/missing-driver" CACHE PATH "" FORCE)
  flagdnn_iluvatar_resolve_corex(COREX)
elseif(CASE STREQUAL "wrong_jit_backend")
  set(FLAGDNN_ILUVATAR_TRITON_JIT_DIR
    "${TEST_ROOT}/jit-cuda/lib/cmake/TritonJIT" CACHE PATH "" FORCE)
  flagdnn_iluvatar_resolve_triton_jit(JIT)
elseif(CASE STREQUAL "missing_backend_ix")
  set(FLAGDNN_ILUVATAR_TRITON_JIT_DIR
    "${TEST_ROOT}/jit-no-ix-definition/lib/cmake/TritonJIT"
    CACHE PATH "" FORCE)
  flagdnn_iluvatar_resolve_triton_jit(JIT)
else()
  message(FATAL_ERROR "unknown contract case: ${CASE}")
endif()
]=])

function(_run_case name expect_success expected_error)
  set(_build "${TEST_ROOT}/build-${name}")
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
      -S "${TEST_ROOT}/child"
      -B "${_build}"
      "-DSOURCE_ROOT=${SOURCE_ROOT}"
      "-DTEST_ROOT=${TEST_ROOT}"
      "-DCASE=${name}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  set(_output "${_stdout}\n${_stderr}")
  if(expect_success)
    if(NOT _result EQUAL 0)
      message(FATAL_ERROR
        "positive CMake contract ${name} failed:\n${_output}")
    endif()
  else()
    if(_result EQUAL 0)
      message(FATAL_ERROR "negative CMake contract ${name} passed")
    endif()
    if(NOT _output MATCHES "${expected_error}")
      message(FATAL_ERROR
        "negative CMake contract ${name} produced an unexpected error:\n"
        "${_output}")
    endif()
  endif()
endfunction()

_run_case(valid TRUE "")
_run_case(external_artifact FALSE "supports only.*libtriton_jit")
_run_case(missing_cuda_h FALSE "cuda.h")
_run_case(missing_libcuda FALSE "libcuda")
_run_case(wrong_jit_backend FALSE "requires.*IX")
_run_case(missing_backend_ix FALSE "BACKEND_IX")

message(STATUS "PASS Iluvatar CoreX and IX JIT CMake contracts")
