// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <dlfcn.h>

#include <array>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

bool check_cuda_driver(CUresult status, std::string_view operation) {
  if (status == CUDA_SUCCESS) {
    return true;
  }
  const char *detail = nullptr;
  (void)cuGetErrorString(status, &detail);
  std::cerr << operation << " failed: "
            << (detail == nullptr ? "unknown CoreX Driver error" : detail)
            << '\n';
  return false;
}

bool check_cuda_runtime(cudaError_t status, std::string_view operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << operation << " failed: " << cudaGetErrorString(status) << '\n';
  return false;
}

bool check_cudnn(cudnnStatus_t status, std::string_view operation) {
  if (status == CUDNN_STATUS_SUCCESS) {
    return true;
  }
  std::cerr << operation << " failed: " << cudnnGetErrorString(status) << '\n';
  return false;
}

} // namespace

int main(int argc, char **argv) {
  static_assert(CUDNN_VERSION == 7605,
                "Iluvatar validation requires CoreX cuDNN header 7.6.5");
  if (argc != 2) {
    std::cerr << "usage: test_corex_cudnn_environment <libcudnn.so.7>\n";
    return EXIT_FAILURE;
  }

  if (!check_cuda_driver(cuInit(0), "cuInit")) {
    return EXIT_FAILURE;
  }
  int device_count = 0;
  if (!check_cuda_driver(cuDeviceGetCount(&device_count), "cuDeviceGetCount") ||
      device_count < 1) {
    std::cerr << "CoreX device 0 is unavailable\n";
    return EXIT_FAILURE;
  }
  CUdevice device = 0;
  if (!check_cuda_driver(cuDeviceGet(&device, 0), "cuDeviceGet")) {
    return EXIT_FAILURE;
  }
  int major = 0;
  int minor = 0;
  if (!check_cuda_driver(
          cuDeviceGetAttribute(
              &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
          "cuDeviceGetAttribute(major)") ||
      !check_cuda_driver(
          cuDeviceGetAttribute(
              &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
          "cuDeviceGetAttribute(minor)")) {
    return EXIT_FAILURE;
  }
  if (major != 7 || minor != 1) {
    std::cerr << "CoreX target mismatch: expected corex_71, got corex_" << major
              << minor << '\n';
    return EXIT_FAILURE;
  }

  if (!check_cuda_runtime(cudaSetDevice(0), "cudaSetDevice") ||
      !check_cuda_runtime(cudaFree(nullptr), "cudaFree(init)")) {
    return EXIT_FAILURE;
  }

  const std::size_t runtime_version = cudnnGetVersion();
  const std::size_t reported_cudart_version = cudnnGetCudartVersion();
  if (runtime_version != 7605) {
    std::cerr << "CoreX cuDNN runtime version mismatch: expected 7605, got "
              << runtime_version << '\n';
    return EXIT_FAILURE;
  }

  void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr) {
    std::cerr << "dlopen failed for selected CoreX cuDNN: " << dlerror()
              << '\n';
    return EXIT_FAILURE;
  }
  constexpr std::array classic_symbols{
      "cudnnGetVersion",
      "cudnnGetCudartVersion",
      "cudnnGetErrorString",
      "cudnnCreate",
      "cudnnDestroy",
      "cudnnSetStream",
      "cudnnCreateTensorDescriptor",
      "cudnnSetTensorNdDescriptor",
      "cudnnDestroyTensorDescriptor",
      "cudnnTransformTensor",
      "cudnnCreateOpTensorDescriptor",
      "cudnnSetOpTensorDescriptor",
      "cudnnOpTensor",
      "cudnnCreateReduceTensorDescriptor",
      "cudnnSetReduceTensorDescriptor",
      "cudnnGetReductionWorkspaceSize",
      "cudnnReduceTensor",
      "cudnnDestroyReduceTensorDescriptor",
      "cudnnCreateActivationDescriptor",
      "cudnnSetActivationDescriptor",
      "cudnnActivationForward",
      "cudnnActivationBackward",
      "cudnnDestroyActivationDescriptor",
      "cudnnCreateConvolutionDescriptor",
      "cudnnSetConvolutionNdDescriptor",
      "cudnnConvolutionForward",
      "cudnnConvolutionBackwardData",
      "cudnnConvolutionBackwardFilter",
      "cudnnDestroyConvolutionDescriptor",
      "cudnnBatchNormalizationForwardTraining",
      "cudnnBatchNormalizationForwardInference",
      "cudnnBatchNormalizationBackward",
  };
  for (const char *symbol : classic_symbols) {
    if (dlsym(library, symbol) == nullptr) {
      std::cerr << "selected CoreX cuDNN is missing classic symbol " << symbol
                << '\n';
      (void)dlclose(library);
      return EXIT_FAILURE;
    }
  }

  constexpr std::array flash_symbols{
      "cudnnCreateFlashAttnDescriptor", "cudnnDestroyFlashAttnDescriptor",
      "cudnnGetFlashAttnBuffers",       "cudnnFlashAttnForward",
      "cudnnFlashAttnBackward",
  };
  std::size_t flash_symbol_count = 0;
  for (const char *symbol : flash_symbols) {
    flash_symbol_count += dlsym(library, symbol) == nullptr ? 0U : 1U;
  }
  if (flash_symbol_count != 0 && flash_symbol_count != flash_symbols.size()) {
    std::cerr << "CoreX cuDNN exports only a partial Flash Attention ABI\n";
    (void)dlclose(library);
    return EXIT_FAILURE;
  }
  (void)dlclose(library);

  cudaStream_t stream = nullptr;
  if (!check_cuda_runtime(cudaStreamCreate(&stream), "cudaStreamCreate")) {
    return EXIT_FAILURE;
  }
  cudnnHandle_t handle = nullptr;
  if (!check_cudnn(cudnnCreate(&handle), "cudnnCreate") ||
      !check_cudnn(cudnnSetStream(handle, stream), "cudnnSetStream") ||
      !check_cudnn(cudnnDestroy(handle), "cudnnDestroy")) {
    (void)cudaStreamDestroy(stream);
    return EXIT_FAILURE;
  }
  if (!check_cuda_runtime(cudaStreamDestroy(stream), "cudaStreamDestroy")) {
    return EXIT_FAILURE;
  }

  std::cout << "PASS CoreX cuDNN environment: header=" << CUDNN_VERSION
            << " runtime=" << runtime_version
            << " cudnnGetCudartVersion=" << reported_cudart_version
            << " target=corex_" << major << minor << " flash_attention_symbols="
            << (flash_symbol_count == flash_symbols.size() ? "present"
                                                           : "absent")
            << '\n';
  return EXIT_SUCCESS;
}
