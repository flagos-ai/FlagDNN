cmake_minimum_required(VERSION 3.23)

project(FlagDNNMthreadsInstalledConsumer LANGUAGES C CXX)

include(CTest)

find_package(FlagDNN 0.2 CONFIG REQUIRED)

foreach(_required_variable IN ITEMS
    FLAGDNN_INSTALLED_MTHREADS_SDK_ROOT
    FLAGDNN_INSTALLED_MTHREADS_PLUGIN
    FLAGDNN_INSTALLED_MTHREADS_JIT
    FLAGDNN_INSTALLED_MTHREADS_SCRIPT_DIR
    FLAGDNN_INSTALLED_MTHREADS_CACHE
    FLAGDNN_INSTALLED_MUSA_ROOT
    FLAGDNN_INSTALLED_MUDNN_INCLUDE_DIR
    FLAGDNN_INSTALLED_MUDNN_LIBRARY
    FLAGDNN_INSTALLED_LIBRARY_PATH)
  if("${${_required_variable}}" STREQUAL "")
    message(FATAL_ERROR "${_required_variable} is required")
  endif()
endforeach()

find_path(
  FLAGDNN_INSTALLED_MUSA_INCLUDE_DIR
  NAMES musa_runtime_api.h
  PATHS "${FLAGDNN_INSTALLED_MUSA_ROOT}/include"
  NO_DEFAULT_PATH
  REQUIRED)
find_library(
  FLAGDNN_INSTALLED_MUSART_LIBRARY
  NAMES musart
  PATHS "${FLAGDNN_INSTALLED_MUSA_ROOT}/lib"
  NO_DEFAULT_PATH
  REQUIRED)

add_executable(flagdnn_installed_mthreads_c c_smoke.c)
target_link_libraries(
  flagdnn_installed_mthreads_c PRIVATE FlagDNN::flagdnn)
set_target_properties(
  flagdnn_installed_mthreads_c
  PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED YES)

add_executable(flagdnn_installed_mthreads_add mthreads_add.cpp)
target_compile_features(
  flagdnn_installed_mthreads_add PRIVATE cxx_std_20)
target_include_directories(
  flagdnn_installed_mthreads_add
  PRIVATE
    "${FLAGDNN_INSTALLED_MUSA_INCLUDE_DIR}"
    "${FLAGDNN_INSTALLED_MUDNN_INCLUDE_DIR}")
target_link_libraries(
  flagdnn_installed_mthreads_add
  PRIVATE
    FlagDNN::flagdnn
    "${FLAGDNN_INSTALLED_MUSART_LIBRARY}"
    "${FLAGDNN_INSTALLED_MUDNN_LIBRARY}"
    "${CMAKE_DL_LIBS}")
get_filename_component(
  _flagdnn_installed_mudnn_library_directory
  "${FLAGDNN_INSTALLED_MUDNN_LIBRARY}"
  DIRECTORY)
set_target_properties(
  flagdnn_installed_mthreads_add
  PROPERTIES BUILD_RPATH
    "${FLAGDNN_INSTALLED_MUSA_ROOT}/lib;${_flagdnn_installed_mudnn_library_directory}")

add_test(
  NAME installed.mthreads_c
  COMMAND flagdnn_installed_mthreads_c)
set_tests_properties(
  installed.mthreads_c
  PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${FLAGDNN_INSTALLED_LIBRARY_PATH}"
    TIMEOUT 60)

add_test(
  NAME installed.mthreads_add
  COMMAND flagdnn_installed_mthreads_add)
set_tests_properties(
  installed.mthreads_add
  PROPERTIES
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:FlagDNN::flagdnn>;FLAGDNN_INSTALLED_EXPECTED_SDK=${FLAGDNN_INSTALLED_MTHREADS_SDK_ROOT};FLAGDNN_INSTALLED_EXPECTED_PLUGIN=${FLAGDNN_INSTALLED_MTHREADS_PLUGIN};FLAGDNN_INSTALLED_EXPECTED_JIT=${FLAGDNN_INSTALLED_MTHREADS_JIT};FLAGDNN_INSTALLED_EXPECTED_JIT_SCRIPT_DIR=${FLAGDNN_INSTALLED_MTHREADS_SCRIPT_DIR};FLAGDNN_INSTALLED_CACHE=${FLAGDNN_INSTALLED_MTHREADS_CACHE};LD_LIBRARY_PATH=${FLAGDNN_INSTALLED_LIBRARY_PATH}"
    RUN_SERIAL TRUE
    TIMEOUT 1200)

