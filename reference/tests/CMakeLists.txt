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

# Independent CPU/cuDNN correctness; no FlagDNN GPU kernel or timing runner.
add_executable(flagdnn_reference_cpu_cudnn
  cpu_cudnn.cpp
  cpu_cudnn_matrix.cpp
  cpu_cudnn_normalization.cpp
  cpu_cudnn_attention.cpp
  cpu_cudnn_storage.cpp
  cpu_cudnn_pointwise.cpp
  cpu_cudnn_extended.cpp
  cpu_cudnn_native.cpp
  cpu_cudnn_benchmark.cpp)
target_compile_features(flagdnn_reference_cpu_cudnn PRIVATE cxx_std_20)
target_link_libraries(flagdnn_reference_cpu_cudnn PRIVATE
  flagdnn_validation_cpu_reference
  flagdnn_test_nvidia_adapter
  flagdnn_validation_nvidia_extended)
flagdnn_enable_warnings(flagdnn_reference_cpu_cudnn)
set_target_properties(flagdnn_reference_cpu_cudnn PROPERTIES
  BUILD_RPATH "${_flagdnn_reference_cudnn_directory}")
foreach(family IN ITEMS matrix matrix_tf32 normalization attention storage
    pointwise statistics resample causal_convolution fp8_matmul moe_matmul formulas)
  set(test_name "reference.cpu.${family}.cudnn")
  add_test(NAME "${test_name}" COMMAND flagdnn_reference_cpu_cudnn "${family}")
  set_tests_properties("${test_name}" PROPERTIES
    LABELS "reference;correctness;cpu;nvidia;cudnn;${family}"
    ENVIRONMENT "NVIDIA_TF32_OVERRIDE=0"
    RUN_SERIAL TRUE TIMEOUT 600)
  if(family STREQUAL "matrix_tf32")
    set_tests_properties("${test_name}" PROPERTIES
      ENVIRONMENT "NVIDIA_TF32_OVERRIDE=1")
  endif()
endforeach()
add_custom_target(flagdnn_reference_cpu_cudnn_tests DEPENDS
  flagdnn_reference_cpu_cudnn flagdnn_reference_cpu_pointwise_cudnn)
