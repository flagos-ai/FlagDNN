cmake_minimum_required(VERSION 3.23)

project(FlagDNNIluvatarInstalledConsumer LANGUAGES CXX)

include(CTest)

# Use the SDK under test. A hard-coded 0.1 requirement can reject this
# repository's 0.2 SDK and silently load a system-wide older installation.
find_package(FlagDNN CONFIG REQUIRED PATHS "${FlagDNN_DIR}" NO_DEFAULT_PATH)

foreach(_required IN ITEMS
    FLAGDNN_INSTALLED_SDK
    FLAGDNN_INSTALLED_LIBRARY_DIRECTORY
    FLAGDNN_INSTALLED_CODEGEN_PYTHON
    FLAGDNN_INSTALLED_COMPILER_WRAPPER
    FLAGDNN_INSTALLED_COMPILER_LOG
    FLAGDNN_INSTALLED_CACHE_DIRECTORY
    FLAGDNN_INSTALLED_COREX_ROOT
    FLAGDNN_INSTALLED_COREX_DRIVER
    FLAGDNN_INSTALLED_TRITON_JIT_CONFIG_DIR
    FLAGDNN_INSTALLED_TRITON_JIT_LIBRARY
    FLAGDNN_INSTALLED_TRITON_JIT_INCLUDE_DIR
    FLAGDNN_INSTALLED_TRITON_JIT_SCRIPT_DIR)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "" OR
     NOT EXISTS "${${_required}}")
    message(FATAL_ERROR "${_required} must name an existing path")
  endif()
endforeach()
if(NOT DEFINED FLAGDNN_INSTALLED_EXTERNAL_PYTHONPATH OR
   FLAGDNN_INSTALLED_EXTERNAL_PYTHONPATH STREQUAL "")
  message(FATAL_ERROR
    "FLAGDNN_INSTALLED_EXTERNAL_PYTHONPATH must be nonempty")
endif()

set(_provider
  "${FLAGDNN_INSTALLED_SDK}/share/flagdnn/backends/iluvatar/compiler.py")
set(_platform_registry
  "${FLAGDNN_INSTALLED_SDK}/share/flagdnn/backends/iluvatar/kernels/registry.json")
set(_platform_binary_kernel
  "${FLAGDNN_INSTALLED_SDK}/share/flagdnn/backends/iluvatar/kernels/binary.py")
set(_platform_unary_kernel
  "${FLAGDNN_INSTALLED_SDK}/share/flagdnn/backends/iluvatar/kernels/unary.py")
set(_platform_layout_kernel
  "${FLAGDNN_INSTALLED_SDK}/share/flagdnn/backends/iluvatar/kernels/layout.py")
set(_platform_tuning
  "${FLAGDNN_INSTALLED_SDK}/share/flagdnn/backends/iluvatar/tuning/common.yaml")
foreach(_resource IN ITEMS
    "${_provider}"
    "${_platform_registry}"
    "${_platform_binary_kernel}"
    "${_platform_unary_kernel}"
    "${_platform_layout_kernel}"
    "${_platform_tuning}")
  if(NOT EXISTS "${_resource}")
    message(FATAL_ERROR "installed consumer resource is missing: ${_resource}")
  endif()
endforeach()

add_executable(flagdnn_iluvatar_installed_consumer main.cpp)
target_compile_features(flagdnn_iluvatar_installed_consumer PRIVATE cxx_std_20)
target_link_libraries(flagdnn_iluvatar_installed_consumer PRIVATE
  FlagDNN::flagdnn
  ${CMAKE_DL_LIBS})

add_test(NAME installed.iluvatar.graph
  COMMAND flagdnn_iluvatar_installed_consumer)
set_tests_properties(installed.iluvatar.graph PROPERTIES
  ENVIRONMENT
    "FLAGDNN_BACKEND_PATH=${FLAGDNN_INSTALLED_LIBRARY_DIRECTORY};FLAGDNN_COMPILER_EXECUTABLE=${FLAGDNN_INSTALLED_COMPILER_WRAPPER};FLAGDNN_CACHE_DIRECTORY=${FLAGDNN_INSTALLED_CACHE_DIRECTORY};FLAGDNN_INSTALLED_SDK=${FLAGDNN_INSTALLED_SDK};FLAGDNN_INSTALLED_PROVIDER=${_provider};FLAGDNN_INSTALLED_PLATFORM_REGISTRY=${_platform_registry};FLAGDNN_INSTALLED_PLATFORM_BINARY_KERNEL=${_platform_binary_kernel};FLAGDNN_INSTALLED_PLATFORM_UNARY_KERNEL=${_platform_unary_kernel};FLAGDNN_INSTALLED_PLATFORM_LAYOUT_KERNEL=${_platform_layout_kernel};FLAGDNN_INSTALLED_PLATFORM_TUNING=${_platform_tuning};FLAGDNN_INSTALLED_EXPECTED_JIT=${FLAGDNN_INSTALLED_TRITON_JIT_LIBRARY};FLAGDNN_INSTALLED_COREX_DRIVER=${FLAGDNN_INSTALLED_COREX_DRIVER};FLAGDNN_ILUVATAR_COREX_ROOT=${FLAGDNN_INSTALLED_COREX_ROOT};FLAGDNN_ILUVATAR_TRITON_JIT_DIR=${FLAGDNN_INSTALLED_TRITON_JIT_CONFIG_DIR};FLAGDNN_ILUVATAR_TRITON_JIT_LIBRARY=${FLAGDNN_INSTALLED_TRITON_JIT_LIBRARY};FLAGDNN_ILUVATAR_TRITON_JIT_INCLUDE_DIR=${FLAGDNN_INSTALLED_TRITON_JIT_INCLUDE_DIR};FLAGDNN_ILUVATAR_TRITON_JIT_SCRIPT_DIR=${FLAGDNN_INSTALLED_TRITON_JIT_SCRIPT_DIR};FLAGDNN_TEST_REAL_PYTHON=${FLAGDNN_INSTALLED_CODEGEN_PYTHON};FLAGDNN_TEST_COMPILER_LOG=${FLAGDNN_INSTALLED_COMPILER_LOG};FLAGDNN_TEST_EXTERNAL_PYTHONPATH=${FLAGDNN_INSTALLED_EXTERNAL_PYTHONPATH}"
  TIMEOUT 600)
