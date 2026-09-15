cmake_minimum_required(VERSION 3.23)

project(FlagDNNInstalledConsumer LANGUAGES C CXX)

include(CTest)

find_package(FlagDNN 0.2 CONFIG REQUIRED)

option(FLAGDNN_INSTALLED_NVIDIA_ADD
  "Build and run an installed-SDK NVIDIA Add integration consumer" OFF)
option(FLAGDNN_INSTALLED_HYGON_ADD
  "Build and run an installed-SDK Hygon Add integration consumer" OFF)
option(FLAGDNN_INSTALLED_ASCEND_ADD
  "Build an installed-SDK Ascend Add integration consumer" OFF)
option(FLAGDNN_INSTALLED_ASCEND_RUN
  "Register the installed-SDK Ascend Add device integration test" OFF)
set(FLAGDNN_INSTALLED_CODEGEN_PYTHON "" CACHE FILEPATH
  "Python interpreter containing Triton for an installed GPU test")
set(FLAGDNN_INSTALLED_HYGON_SDK_ROOT "" CACHE PATH
  "Isolated FlagDNN SDK root for the installed Hygon contract")
set(FLAGDNN_INSTALLED_HYGON_JIT_LIBRARY "" CACHE FILEPATH
  "Hygon-private installed libtriton_jit image")
set(FLAGDNN_INSTALLED_HYGON_JIT_SCRIPT_DIR "" CACHE PATH
  "Private script directory resolved by the installed HCU JIT image")
set(CANN_ROOT "" CACHE PATH "CANN 9 root for the installed Ascend consumer")

add_executable(flagdnn_installed_c_consumer main.c)
target_link_libraries(flagdnn_installed_c_consumer PRIVATE FlagDNN::flagdnn)
set_target_properties(flagdnn_installed_c_consumer PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES)
add_test(NAME installed.c COMMAND flagdnn_installed_c_consumer)

add_executable(flagdnn_installed_cpp_consumer main.cpp)
target_compile_features(flagdnn_installed_cpp_consumer PRIVATE cxx_std_20)
target_link_libraries(flagdnn_installed_cpp_consumer PRIVATE FlagDNN::flagdnn)
add_test(NAME installed.cpp COMMAND flagdnn_installed_cpp_consumer)

if(FLAGDNN_INSTALLED_NVIDIA_ADD)
  if(NOT FLAGDNN_INSTALLED_CODEGEN_PYTHON OR
     NOT EXISTS "${FLAGDNN_INSTALLED_CODEGEN_PYTHON}")
    message(FATAL_ERROR
      "FLAGDNN_INSTALLED_CODEGEN_PYTHON must name the test Python containing Triton")
  endif()
  find_package(CUDAToolkit REQUIRED)
  # The comparator is a consumer test dependency, never an installed FlagDNN
  # dependency. Prefer explicitly configured SDK paths, then the test venv.
  execute_process(COMMAND "${FLAGDNN_INSTALLED_CODEGEN_PYTHON}" -c
    "import sysconfig; print(sysconfig.get_path('purelib'))"
    OUTPUT_VARIABLE _flagdnn_consumer_python_packages
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
  find_path(FLAGDNN_INSTALLED_CUDNN_FRONTEND_INCLUDE_DIR cudnn_frontend.h
    HINTS "$ENV{CUDNN_FRONTEND_ROOT}" "${_flagdnn_consumer_python_packages}"
    PATH_SUFFIXES include REQUIRED)
  find_path(FLAGDNN_INSTALLED_CUDNN_INCLUDE_DIR cudnn.h
    HINTS "$ENV{CUDNN_ROOT}" "${_flagdnn_consumer_python_packages}/nvidia/cudnn"
    PATH_SUFFIXES include REQUIRED)
  find_library(FLAGDNN_INSTALLED_CUDNN_LIBRARY NAMES cudnn libcudnn.so.9
    HINTS "$ENV{CUDNN_ROOT}" "${_flagdnn_consumer_python_packages}/nvidia/cudnn"
    PATH_SUFFIXES lib lib64 REQUIRED)
  add_executable(flagdnn_installed_nvidia_add nvidia_add.cpp)
  target_compile_features(flagdnn_installed_nvidia_add PRIVATE cxx_std_20)
  target_link_libraries(flagdnn_installed_nvidia_add PRIVATE
    FlagDNN::flagdnn CUDA::cuda_driver CUDA::cudart CUDA::nvrtc
    "${FLAGDNN_INSTALLED_CUDNN_LIBRARY}")
  target_include_directories(flagdnn_installed_nvidia_add SYSTEM PRIVATE
    "${FLAGDNN_INSTALLED_CUDNN_FRONTEND_INCLUDE_DIR}"
    "${FLAGDNN_INSTALLED_CUDNN_INCLUDE_DIR}")

  add_test(NAME installed.nvidia_add COMMAND flagdnn_installed_nvidia_add)
  set_tests_properties(installed.nvidia_add PROPERTIES
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:FlagDNN::flagdnn>;FLAGDNN_COMPILER_EXECUTABLE=${FLAGDNN_INSTALLED_CODEGEN_PYTHON}"
    TIMEOUT 600)
endif()

if(FLAGDNN_INSTALLED_HYGON_ADD)
  if(NOT FLAGDNN_INSTALLED_CODEGEN_PYTHON OR
     NOT EXISTS "${FLAGDNN_INSTALLED_CODEGEN_PYTHON}")
    message(FATAL_ERROR
      "FLAGDNN_INSTALLED_CODEGEN_PYTHON must name the test Python containing Triton")
  endif()
  foreach(_flagdnn_installed_hygon_resource IN ITEMS
      FLAGDNN_INSTALLED_HYGON_SDK_ROOT
      FLAGDNN_INSTALLED_HYGON_JIT_LIBRARY
      FLAGDNN_INSTALLED_HYGON_JIT_SCRIPT_DIR)
    if("${${_flagdnn_installed_hygon_resource}}" STREQUAL "" OR
       NOT EXISTS "${${_flagdnn_installed_hygon_resource}}")
      message(FATAL_ERROR
        "${_flagdnn_installed_hygon_resource} must name an installed SDK resource")
    endif()
  endforeach()
  find_package(hip CONFIG REQUIRED)
  if(NOT TARGET hip::host)
    message(FATAL_ERROR "DTK HIP package did not export hip::host")
  endif()
  find_path(FLAGDNN_INSTALLED_HIPDNN_INCLUDE_DIR
    NAMES hipdnn.h
    HINTS "$ENV{HIPDNN_ROOT}" "$ENV{DTK_ROOT}" /opt/dtk/hipdnn /opt/dtk
    PATH_SUFFIXES include hipdnn/include
    REQUIRED)
  find_library(FLAGDNN_INSTALLED_HIPDNN_LIBRARY
    NAMES hipdnn
    HINTS "$ENV{HIPDNN_ROOT}" "$ENV{DTK_ROOT}" /opt/dtk/hipdnn /opt/dtk
    PATH_SUFFIXES lib lib64
    REQUIRED)

  add_executable(flagdnn_installed_hygon_add hygon_add.cpp)
  target_compile_features(flagdnn_installed_hygon_add PRIVATE cxx_std_20)
  target_include_directories(flagdnn_installed_hygon_add PRIVATE
    "${FLAGDNN_INSTALLED_HIPDNN_INCLUDE_DIR}")
  target_link_libraries(flagdnn_installed_hygon_add PRIVATE
    FlagDNN::flagdnn
    hip::host
    "${FLAGDNN_INSTALLED_HIPDNN_LIBRARY}"
    "${CMAKE_DL_LIBS}")
  get_filename_component(_flagdnn_installed_hipdnn_library_directory
    "${FLAGDNN_INSTALLED_HIPDNN_LIBRARY}" DIRECTORY)
  set_target_properties(flagdnn_installed_hygon_add PROPERTIES
    BUILD_RPATH "${_flagdnn_installed_hipdnn_library_directory}")

  add_test(NAME installed.hygon_add COMMAND flagdnn_installed_hygon_add)
  set_tests_properties(installed.hygon_add PROPERTIES
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:FlagDNN::flagdnn>;FLAGDNN_COMPILER_EXECUTABLE=${FLAGDNN_INSTALLED_CODEGEN_PYTHON};FLAGDNN_INSTALLED_EXPECTED_SDK=${FLAGDNN_INSTALLED_HYGON_SDK_ROOT};FLAGDNN_INSTALLED_EXPECTED_JIT=${FLAGDNN_INSTALLED_HYGON_JIT_LIBRARY};FLAGDNN_INSTALLED_EXPECTED_JIT_SCRIPT_DIR=${FLAGDNN_INSTALLED_HYGON_JIT_SCRIPT_DIR}"
    TIMEOUT 1200)
  unset(_flagdnn_installed_hipdnn_library_directory)
endif()

if(FLAGDNN_INSTALLED_ASCEND_ADD)
  if(NOT CANN_ROOT OR NOT IS_DIRECTORY "${CANN_ROOT}")
    message(FATAL_ERROR
      "CANN_ROOT must name the CANN 9 installation for the Ascend consumer")
  endif()
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(_flagdnn_consumer_ascend_arch aarch64-linux)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    set(_flagdnn_consumer_ascend_arch x86_64-linux)
  else()
    message(FATAL_ERROR
      "Unsupported host processor for Ascend: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
  find_path(FLAGDNN_INSTALLED_ASCEND_INCLUDE_DIR acl/acl.h
    PATHS
      "${CANN_ROOT}/${_flagdnn_consumer_ascend_arch}/include"
      "${CANN_ROOT}/${_flagdnn_consumer_ascend_arch}/pkg_inc"
    NO_DEFAULT_PATH REQUIRED)
  find_library(FLAGDNN_INSTALLED_ASCENDCL_LIBRARY ascendcl
    PATHS "${CANN_ROOT}/${_flagdnn_consumer_ascend_arch}/lib64"
    NO_DEFAULT_PATH REQUIRED)
  find_library(FLAGDNN_INSTALLED_ASCEND_RUNTIME_LIBRARY runtime
    PATHS "${CANN_ROOT}/${_flagdnn_consumer_ascend_arch}/lib64"
    NO_DEFAULT_PATH REQUIRED)

  add_executable(flagdnn_installed_ascend_add ascend_add.cpp)
  target_compile_features(flagdnn_installed_ascend_add PRIVATE cxx_std_20)
  target_include_directories(flagdnn_installed_ascend_add PRIVATE
    "${FLAGDNN_INSTALLED_ASCEND_INCLUDE_DIR}")
  target_link_libraries(flagdnn_installed_ascend_add PRIVATE
    FlagDNN::flagdnn
    "${FLAGDNN_INSTALLED_ASCENDCL_LIBRARY}"
    "${FLAGDNN_INSTALLED_ASCEND_RUNTIME_LIBRARY}")

  if(FLAGDNN_INSTALLED_ASCEND_RUN)
    if(NOT FLAGDNN_INSTALLED_CODEGEN_PYTHON OR
       NOT EXISTS "${FLAGDNN_INSTALLED_CODEGEN_PYTHON}")
      message(FATAL_ERROR
        "FLAGDNN_INSTALLED_CODEGEN_PYTHON must name the Ascend compiler Python")
    endif()
    add_test(NAME installed.ascend_add COMMAND flagdnn_installed_ascend_add)
    set_tests_properties(installed.ascend_add PROPERTIES
      ENVIRONMENT
        "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:FlagDNN::flagdnn>;FLAGDNN_COMPILER_EXECUTABLE=${FLAGDNN_INSTALLED_CODEGEN_PYTHON}"
      TIMEOUT 600)
  endif()
endif()
