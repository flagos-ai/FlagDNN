add_library(flagdnn_validation_cpu_reference STATIC
  pointwise.cpp
  attention.cpp
  matrix.cpp
  random.cpp
  index.cpp
  moe_matmul.cpp
  fp8_matmul.cpp
  resample.cpp
  layout.cpp
  normalization.cpp
  causal_convolution.cpp
  benchmark.cpp
  statistics.cpp
  position_embedding.cpp)

target_compile_features(flagdnn_validation_cpu_reference PUBLIC cxx_std_20)
target_include_directories(flagdnn_validation_cpu_reference PUBLIC
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_BINARY_DIR}/generated/include")
flagdnn_enable_warnings(flagdnn_validation_cpu_reference)
