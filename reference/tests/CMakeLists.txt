list(APPEND CMAKE_MODULE_PATH
  "${PROJECT_SOURCE_DIR}/backends/nvidia/cmake")
find_package(CUDAToolkit REQUIRED)
find_package(CUDNN REQUIRED)

add_executable(flagdnn_reference_cpu_pointwise_cudnn
  cpu_pointwise_reference.cpp)
target_compile_features(flagdnn_reference_cpu_pointwise_cudnn PRIVATE
  cxx_std_20)
target_link_libraries(flagdnn_reference_cpu_pointwise_cudnn PRIVATE
  flagdnn_validation_cpu_reference
  CUDA::cudart
  CUDA::nvrtc
  CUDNN::frontend)
flagdnn_enable_warnings(flagdnn_reference_cpu_pointwise_cudnn)

get_filename_component(_flagdnn_reference_cudnn_directory
  "${CUDNN_LIBRARY}" DIRECTORY)
set_target_properties(flagdnn_reference_cpu_pointwise_cudnn PROPERTIES
  BUILD_RPATH "${_flagdnn_reference_cudnn_directory}")

set(_flagdnn_reference_pointwise_modes div pow mod cmp_eq)
foreach(mode IN LISTS _flagdnn_reference_pointwise_modes)
  set(test_name "reference.cpu.pointwise.${mode}.cudnn")
  add_test(NAME "${test_name}"
    COMMAND flagdnn_reference_cpu_pointwise_cudnn --mode "${mode}")
  set_tests_properties("${test_name}" PROPERTIES
    LABELS "reference;correctness;cpu;nvidia;cudnn;pointwise;${mode}"
    RUN_SERIAL TRUE
    TIMEOUT 120)
endforeach()
