cmake_minimum_required(VERSION 3.23)

project(
  FlagDNN
  VERSION 0.1.0
  DESCRIPTION "Native C/C++ runtime for FlagDNN"
  LANGUAGES C CXX)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
include(CTest)

configure_file(
  cmake/flagdnn_version.h.in
  "${CMAKE_CURRENT_BINARY_DIR}/generated/include/flagdnn/version.h"
  @ONLY)

option(FLAGDNN_BUILD_TESTS "Build FlagDNN native tests" ${BUILD_TESTING})
option(FLAGDNN_BUILD_BENCHMARKS
       "Build native FlagDNN operator performance benchmarks" OFF)
option(FLAGDNN_WARNINGS_AS_ERRORS "Treat native compiler warnings as errors" OFF)

set(FLAGDNN_BACKENDS "nvidia"
    CACHE STRING "Semicolon-separated backend plugins to build")
set(FLAGDNN_DEFAULT_BACKEND "nvidia"
    CACHE STRING "Backend selected by FLAGDNN_BACKEND_AUTO")
set(FLAGDNN_EXECUTION_ENGINE "libtriton_jit"
    CACHE STRING
    "Default execution engine: libtriton_jit or external_artifact")
set_property(CACHE FLAGDNN_EXECUTION_ENGINE PROPERTY STRINGS
  libtriton_jit external_artifact)
if(NOT FLAGDNN_DEFAULT_BACKEND MATCHES "^[a-z][a-z0-9_]*$")
  message(FATAL_ERROR
    "FLAGDNN_DEFAULT_BACKEND must match [a-z][a-z0-9_]*")
endif()
if(NOT FLAGDNN_EXECUTION_ENGINE STREQUAL "libtriton_jit" AND
   NOT FLAGDNN_EXECUTION_ENGINE STREQUAL "external_artifact")
  message(FATAL_ERROR
    "FLAGDNN_EXECUTION_ENGINE must be libtriton_jit or external_artifact")
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)
set(FLAGDNN_CODEGEN_PYTHON "${Python3_EXECUTABLE}"
    CACHE FILEPATH "Python executable containing FlagTree/Triton")
set(FLAGDNN_CODEGEN_COMPILER
    "${CMAKE_CURRENT_SOURCE_DIR}/compiler/flagdnn_codegen/main.py"
    CACHE FILEPATH "FlagDNN external compiler entry point")

function(flagdnn_enable_warnings target)
  target_compile_options(${target} PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>)
  if(FLAGDNN_WARNINGS_AS_ERRORS)
    target_compile_options(${target} PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Werror>)
  endif()
endfunction()

include(cmake/FlagDNNBackendPlugin.cmake)

set(FLAGDNN_RESOURCE_INSTALL_DIR
    "${CMAKE_INSTALL_DATADIR}/flagdnn")
file(RELATIVE_PATH FLAGDNN_RESOURCE_PATH_FROM_LIBRARY
  "${CMAKE_INSTALL_FULL_LIBDIR}"
  "${CMAKE_INSTALL_FULL_DATADIR}/flagdnn")

add_subdirectory(src)
add_subdirectory(backends)

install(DIRECTORY compiler/flagdnn_codegen
  DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/compiler"
  FILES_MATCHING
    PATTERN "*.py"
    PATTERN "__pycache__" EXCLUDE)
install(FILES kernels/registry.json
  DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/kernels")
install(DIRECTORY kernels/common/
  DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/kernels/common"
  FILES_MATCHING
    PATTERN "*.py"
    PATTERN "__pycache__" EXCLUDE)

configure_package_config_file(
  cmake/FlagDNNConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/FlagDNNConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/FlagDNN")
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/FlagDNNConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
install(EXPORT FlagDNNTargets
  FILE FlagDNNTargets.cmake
  NAMESPACE FlagDNN::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/FlagDNN")
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/FlagDNNConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/FlagDNNConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/FlagDNN")

if(FLAGDNN_BUILD_TESTS OR FLAGDNN_BUILD_BENCHMARKS)
  enable_testing()
endif()
if(FLAGDNN_BUILD_TESTS)
  add_subdirectory(tests)
endif()
if(FLAGDNN_BUILD_BENCHMARKS)
  add_subdirectory(benchmark)
endif()

# Platform validation is assembled only after tests and benchmark have defined
# their platform-neutral registration contracts. The same backend list controls
# production plugins and their validation adapters.
if(FLAGDNN_BUILD_TESTS OR FLAGDNN_BUILD_BENCHMARKS)
  set(_flagdnn_validation_platforms ${FLAGDNN_BACKENDS})
  list(REMOVE_DUPLICATES _flagdnn_validation_platforms)
  foreach(platform IN LISTS _flagdnn_validation_platforms)
    if(NOT platform MATCHES "^[a-z][a-z0-9_]*$")
      message(FATAL_ERROR
        "Invalid FLAGDNN_BACKENDS entry: ${platform}")
    endif()
    set(validation_directory
      "${CMAKE_CURRENT_SOURCE_DIR}/backends/${platform}/validation")
    if(NOT EXISTS "${validation_directory}/CMakeLists.txt")
      message(FATAL_ERROR
        "Backend ${platform} has no validation implementation at ${validation_directory}")
    endif()
    add_subdirectory("backends/${platform}/validation")
  endforeach()
endif()
