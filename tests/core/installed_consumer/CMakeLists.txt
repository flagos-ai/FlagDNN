cmake_minimum_required(VERSION 3.23)

project(FlagDNNInstalledConsumer LANGUAGES C CXX)

include(CTest)

find_package(FlagDNN 0.1 CONFIG REQUIRED)

option(FLAGDNN_INSTALLED_NVIDIA_ADD
  "Build and run an installed-SDK NVIDIA Add integration consumer" OFF)
set(FLAGDNN_INSTALLED_CODEGEN_PYTHON "" CACHE FILEPATH
  "Python interpreter containing Triton for the installed NVIDIA test")

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
  add_executable(flagdnn_installed_nvidia_add nvidia_add.cpp)
  target_compile_features(flagdnn_installed_nvidia_add PRIVATE cxx_std_20)
  target_link_libraries(flagdnn_installed_nvidia_add PRIVATE
    FlagDNN::flagdnn CUDA::cuda_driver)

  add_test(NAME installed.nvidia_add COMMAND flagdnn_installed_nvidia_add)
  set_tests_properties(installed.nvidia_add PROPERTIES
    ENVIRONMENT
      "FLAGDNN_BACKEND_PATH=$<TARGET_FILE_DIR:FlagDNN::flagdnn>;FLAGDNN_COMPILER_EXECUTABLE=${FLAGDNN_INSTALLED_CODEGEN_PYTHON}"
    TIMEOUT 600)
endif()
