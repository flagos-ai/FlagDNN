// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "corex_cudnn_reference.hpp"
#include "reference_tensor.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace validation = flagdnn::iluvatar::validation;

int main() {
  try {
    if (cuInit(0) != CUDA_SUCCESS || cudaSetDevice(0) != cudaSuccess) {
      throw std::runtime_error("cannot initialize CoreX device 0");
    }
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
      throw std::runtime_error("cannot create caller CoreX stream");
    }
    {
      validation::CorexCudnnHandle handle;
      handle.bind_stream(reinterpret_cast<flagdnnStream_t>(stream));
      flagdnn::testing::TestTensor tensor{
          1, FLAGDNN_DATA_FLOAT32, {2, 3, 5}, {15, 5, 1}, 4};
      const validation::ReferenceTensor reference =
          validation::make_reference_tensor(tensor);
      if (reference.dimensions != std::vector<int>({1, 2, 3, 5}) ||
          reference.strides != std::vector<int>({30, 15, 5, 1}) ||
          reference.byte_offset != 4) {
        throw std::runtime_error("reference tensor conversion is incorrect");
      }
      validation::CorexCudnnTensorDescriptor tensor_descriptor(reference);
      validation::CorexCudnnTensorDescriptor moved_tensor_descriptor(
          std::move(tensor_descriptor));
      validation::CorexCudnnFilterDescriptor filter_descriptor;
      constexpr std::array filter_dimensions{4, 3, 3, 3};
      filter_descriptor.set(CUDNN_DATA_FLOAT, filter_dimensions);
      validation::CorexCudnnConvolutionDescriptor convolution_descriptor;
      validation::CorexCudnnOpTensorDescriptor op_descriptor;
      validation::CorexCudnnReductionDescriptor reduction_descriptor;
      validation::CorexCudnnActivationDescriptor activation_descriptor;
      validation::CorexDeviceWorkspace workspace(64);
      if (moved_tensor_descriptor.get() == nullptr ||
          filter_descriptor.get() == nullptr ||
          convolution_descriptor.get() == nullptr ||
          op_descriptor.get() == nullptr ||
          reduction_descriptor.get() == nullptr ||
          activation_descriptor.get() == nullptr ||
          workspace.data() == nullptr || workspace.size() != 64) {
        throw std::runtime_error("CoreX cuDNN RAII object is invalid");
      }
      if (validation::CorexCudnnFlashAttentionDescriptor::symbols_available()) {
        throw std::runtime_error(
            "baseline unexpectedly exports Flash Attention symbols");
      }
    }
    if (cudaStreamDestroy(stream) != cudaSuccess) {
      throw std::runtime_error("cannot destroy caller CoreX stream");
    }
    std::cout << "PASS strict CoreX cuDNN RAII/reference runtime" << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "FAIL strict CoreX cuDNN RAII/reference runtime: "
              << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
