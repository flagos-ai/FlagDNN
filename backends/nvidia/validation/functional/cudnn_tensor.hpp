/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_NVIDIA_CUDNN_TENSOR_HPP_
#define FLAGDNN_NVIDIA_CUDNN_TENSOR_HPP_
#include <limits>
#include <stdexcept>
#include <vector>

#include "validation/functional/cudnn_graph.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
namespace flagdnn::testing::cuda {
class TensorDescriptor {
 public:
  explicit TensorDescriptor(const TestTensor& tensor) {
    std::vector<int> dims, strides;
    for (const auto dim : tensor.dimensions) dims.push_back(checked_int(dim));
    for (const auto stride : tensor.strides)
      strides.push_back(checked_int(stride));
    while (dims.size() < 4) {
      dims.insert(dims.begin(), 1);
      strides.insert(strides.begin(), 1);
    }
    cudnnDataType_t type;
    switch (tensor.data_type) {
      case FLAGDNN_DATA_FLOAT32:
        type = CUDNN_DATA_FLOAT;
        break;
      case FLAGDNN_DATA_FLOAT16:
        type = CUDNN_DATA_HALF;
        break;
      case FLAGDNN_DATA_BFLOAT16:
        type = CUDNN_DATA_BFLOAT16;
        break;
      default:
        throw std::invalid_argument("cuDNN tensor dtype unsupported");
    }
    check_cudnn(cudnnCreateTensorDescriptor(&descriptor_),
                "cudnnCreateTensorDescriptor");
    const auto status = cudnnSetTensorNdDescriptor(
        descriptor_, type, static_cast<int>(dims.size()), dims.data(),
        strides.data());
    if (status != CUDNN_STATUS_SUCCESS) {
      (void)cudnnDestroyTensorDescriptor(descriptor_);
      descriptor_ = nullptr;
      check_cudnn(status, "cudnnSetTensorNdDescriptor");
    }
  }
  ~TensorDescriptor() { (void)cudnnDestroyTensorDescriptor(descriptor_); }
  TensorDescriptor(const TensorDescriptor&) = delete;
  TensorDescriptor& operator=(const TensorDescriptor&) = delete;
  cudnnTensorDescriptor_t get() const { return descriptor_; }

 private:
  static int checked_int(std::int64_t value) {
    if (value <= 0 || value > std::numeric_limits<int>::max())
      throw std::invalid_argument("cuDNN tensor dimension/stride out of range");
    return static_cast<int>(value);
  }
  cudnnTensorDescriptor_t descriptor_ = nullptr;
};

}  // namespace flagdnn::testing::cuda
#pragma GCC diagnostic pop
#endif
