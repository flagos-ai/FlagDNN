// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "corex_cudnn_reference.hpp"

#include "corex_cudnn_status.hpp"
#include "reference_tensor.hpp"

#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace flagdnn::iluvatar::validation {
namespace {

template <typename Function>
Function dynamic_function(const char *name) noexcept {
  static_assert(std::is_pointer_v<Function>);
  void *symbol = dlsym(RTLD_DEFAULT, name);
  Function function = nullptr;
  static_assert(sizeof(function) == sizeof(symbol));
  std::memcpy(&function, &symbol, sizeof(function));
  return function;
}

void check_cuda_runtime(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) +
                             " failed: " + cudaGetErrorString(status));
  }
}

template <typename Descriptor, cudnnStatus_t (*Destroy)(Descriptor)>
void destroy_noexcept(Descriptor &descriptor) noexcept {
  if (descriptor != nullptr) {
    (void)Destroy(descriptor);
    descriptor = nullptr;
  }
}

template <typename Descriptor, cudnnStatus_t (*Destroy)(Descriptor)>
void move_assign(Descriptor &destination, Descriptor &source) noexcept {
  destroy_noexcept<Descriptor, Destroy>(destination);
  destination = std::exchange(source, nullptr);
}

} // namespace

CorexCudnnHandle::CorexCudnnHandle() {
  check_cudnn(cudnnCreate(&handle_), "cudnnCreate");
}

CorexCudnnHandle::~CorexCudnnHandle() noexcept {
  destroy_noexcept<cudnnHandle_t, cudnnDestroy>(handle_);
}

CorexCudnnHandle::CorexCudnnHandle(CorexCudnnHandle &&other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)) {}

CorexCudnnHandle &
CorexCudnnHandle::operator=(CorexCudnnHandle &&other) noexcept {
  if (this != &other) {
    move_assign<cudnnHandle_t, cudnnDestroy>(handle_, other.handle_);
  }
  return *this;
}

void CorexCudnnHandle::bind_stream(flagdnnStream_t stream) {
  if (stream == nullptr) {
    throw std::invalid_argument("CoreX cuDNN caller stream is null");
  }
  check_cudnn(cudnnSetStream(handle_, reinterpret_cast<cudaStream_t>(stream)),
              "cudnnSetStream");
}

CorexCudnnTensorDescriptor::CorexCudnnTensorDescriptor() {
  check_cudnn(cudnnCreateTensorDescriptor(&descriptor_),
              "cudnnCreateTensorDescriptor");
}

CorexCudnnTensorDescriptor::CorexCudnnTensorDescriptor(
    const ReferenceTensor &tensor)
    : CorexCudnnTensorDescriptor() {
  set(tensor);
}

CorexCudnnTensorDescriptor::~CorexCudnnTensorDescriptor() noexcept {
  destroy_noexcept<cudnnTensorDescriptor_t, cudnnDestroyTensorDescriptor>(
      descriptor_);
}

CorexCudnnTensorDescriptor::CorexCudnnTensorDescriptor(
    CorexCudnnTensorDescriptor &&other) noexcept
    : descriptor_(std::exchange(other.descriptor_, nullptr)) {}

CorexCudnnTensorDescriptor &CorexCudnnTensorDescriptor::operator=(
    CorexCudnnTensorDescriptor &&other) noexcept {
  if (this != &other) {
    move_assign<cudnnTensorDescriptor_t, cudnnDestroyTensorDescriptor>(
        descriptor_, other.descriptor_);
  }
  return *this;
}

void CorexCudnnTensorDescriptor::set(const ReferenceTensor &tensor) {
  if (tensor.dimensions.empty() ||
      tensor.dimensions.size() != tensor.strides.size()) {
    throw std::invalid_argument("CoreX cuDNN tensor descriptor is invalid");
  }
  check_cudnn(cudnnSetTensorNdDescriptor(
                  descriptor_, tensor.data_type,
                  static_cast<int>(tensor.dimensions.size()),
                  tensor.dimensions.data(), tensor.strides.data()),
              "cudnnSetTensorNdDescriptor");
}

CorexCudnnFilterDescriptor::CorexCudnnFilterDescriptor() {
  check_cudnn(cudnnCreateFilterDescriptor(&descriptor_),
              "cudnnCreateFilterDescriptor");
}

CorexCudnnFilterDescriptor::~CorexCudnnFilterDescriptor() noexcept {
  destroy_noexcept<cudnnFilterDescriptor_t, cudnnDestroyFilterDescriptor>(
      descriptor_);
}

CorexCudnnFilterDescriptor::CorexCudnnFilterDescriptor(
    CorexCudnnFilterDescriptor &&other) noexcept
    : descriptor_(std::exchange(other.descriptor_, nullptr)) {}

CorexCudnnFilterDescriptor &CorexCudnnFilterDescriptor::operator=(
    CorexCudnnFilterDescriptor &&other) noexcept {
  if (this != &other) {
    move_assign<cudnnFilterDescriptor_t, cudnnDestroyFilterDescriptor>(
        descriptor_, other.descriptor_);
  }
  return *this;
}

void CorexCudnnFilterDescriptor::set(cudnnDataType_t data_type,
                                     std::span<const int> dimensions) {
  if (dimensions.empty() || dimensions.size() > CUDNN_DIM_MAX) {
    throw std::invalid_argument("CoreX cuDNN filter dimensions are invalid");
  }
  check_cudnn(cudnnSetFilterNdDescriptor(
                  descriptor_, data_type, CUDNN_TENSOR_NCHW,
                  static_cast<int>(dimensions.size()), dimensions.data()),
              "cudnnSetFilterNdDescriptor");
}

CorexCudnnConvolutionDescriptor::CorexCudnnConvolutionDescriptor() {
  check_cudnn(cudnnCreateConvolutionDescriptor(&descriptor_),
              "cudnnCreateConvolutionDescriptor");
}

CorexCudnnConvolutionDescriptor::~CorexCudnnConvolutionDescriptor() noexcept {
  destroy_noexcept<cudnnConvolutionDescriptor_t,
                   cudnnDestroyConvolutionDescriptor>(descriptor_);
}

CorexCudnnConvolutionDescriptor::CorexCudnnConvolutionDescriptor(
    CorexCudnnConvolutionDescriptor &&other) noexcept
    : descriptor_(std::exchange(other.descriptor_, nullptr)) {}

CorexCudnnConvolutionDescriptor &CorexCudnnConvolutionDescriptor::operator=(
    CorexCudnnConvolutionDescriptor &&other) noexcept {
  if (this != &other) {
    move_assign<cudnnConvolutionDescriptor_t,
                cudnnDestroyConvolutionDescriptor>(descriptor_,
                                                   other.descriptor_);
  }
  return *this;
}

CorexCudnnOpTensorDescriptor::CorexCudnnOpTensorDescriptor() {
  check_cudnn(cudnnCreateOpTensorDescriptor(&descriptor_),
              "cudnnCreateOpTensorDescriptor");
}

CorexCudnnOpTensorDescriptor::~CorexCudnnOpTensorDescriptor() noexcept {
  destroy_noexcept<cudnnOpTensorDescriptor_t, cudnnDestroyOpTensorDescriptor>(
      descriptor_);
}

CorexCudnnOpTensorDescriptor::CorexCudnnOpTensorDescriptor(
    CorexCudnnOpTensorDescriptor &&other) noexcept
    : descriptor_(std::exchange(other.descriptor_, nullptr)) {}

CorexCudnnOpTensorDescriptor &CorexCudnnOpTensorDescriptor::operator=(
    CorexCudnnOpTensorDescriptor &&other) noexcept {
  if (this != &other) {
    move_assign<cudnnOpTensorDescriptor_t, cudnnDestroyOpTensorDescriptor>(
        descriptor_, other.descriptor_);
  }
  return *this;
}

CorexCudnnReductionDescriptor::CorexCudnnReductionDescriptor() {
  check_cudnn(cudnnCreateReduceTensorDescriptor(&descriptor_),
              "cudnnCreateReduceTensorDescriptor");
}

CorexCudnnReductionDescriptor::~CorexCudnnReductionDescriptor() noexcept {
  destroy_noexcept<cudnnReduceTensorDescriptor_t,
                   cudnnDestroyReduceTensorDescriptor>(descriptor_);
}

CorexCudnnReductionDescriptor::CorexCudnnReductionDescriptor(
    CorexCudnnReductionDescriptor &&other) noexcept
    : descriptor_(std::exchange(other.descriptor_, nullptr)) {}

CorexCudnnReductionDescriptor &CorexCudnnReductionDescriptor::operator=(
    CorexCudnnReductionDescriptor &&other) noexcept {
  if (this != &other) {
    move_assign<cudnnReduceTensorDescriptor_t,
                cudnnDestroyReduceTensorDescriptor>(descriptor_,
                                                    other.descriptor_);
  }
  return *this;
}

CorexCudnnActivationDescriptor::CorexCudnnActivationDescriptor() {
  check_cudnn(cudnnCreateActivationDescriptor(&descriptor_),
              "cudnnCreateActivationDescriptor");
}

CorexCudnnActivationDescriptor::~CorexCudnnActivationDescriptor() noexcept {
  destroy_noexcept<cudnnActivationDescriptor_t,
                   cudnnDestroyActivationDescriptor>(descriptor_);
}

CorexCudnnActivationDescriptor::CorexCudnnActivationDescriptor(
    CorexCudnnActivationDescriptor &&other) noexcept
    : descriptor_(std::exchange(other.descriptor_, nullptr)) {}

CorexCudnnActivationDescriptor &CorexCudnnActivationDescriptor::operator=(
    CorexCudnnActivationDescriptor &&other) noexcept {
  if (this != &other) {
    move_assign<cudnnActivationDescriptor_t, cudnnDestroyActivationDescriptor>(
        descriptor_, other.descriptor_);
  }
  return *this;
}

bool CorexCudnnFlashAttentionDescriptor::symbols_available() noexcept {
  return dlsym(RTLD_DEFAULT, "cudnnCreateFlashAttnDescriptor") != nullptr &&
         dlsym(RTLD_DEFAULT, "cudnnDestroyFlashAttnDescriptor") != nullptr &&
         dlsym(RTLD_DEFAULT, "cudnnGetFlashAttnBuffers") != nullptr &&
         dlsym(RTLD_DEFAULT, "cudnnFlashAttnForward") != nullptr &&
         dlsym(RTLD_DEFAULT, "cudnnFlashAttnBackward") != nullptr;
}

CorexCudnnFlashAttentionDescriptor::CorexCudnnFlashAttentionDescriptor() {
  using Create = cudnnStatus_t (*)(cudnnFlashAttnDescriptor_t *);
  const Create create =
      dynamic_function<Create>("cudnnCreateFlashAttnDescriptor");
  if (create == nullptr) {
    throw std::runtime_error(
        "selected CoreX cuDNN has no Flash Attention descriptor ABI");
  }
  check_cudnn(create(&descriptor_), "cudnnCreateFlashAttnDescriptor");
}

CorexCudnnFlashAttentionDescriptor::
    ~CorexCudnnFlashAttentionDescriptor() noexcept {
  using Destroy = cudnnStatus_t (*)(cudnnFlashAttnDescriptor_t);
  const Destroy destroy =
      dynamic_function<Destroy>("cudnnDestroyFlashAttnDescriptor");
  if (descriptor_ != nullptr && destroy != nullptr) {
    (void)destroy(descriptor_);
  }
  descriptor_ = nullptr;
}

CorexCudnnFlashAttentionDescriptor::CorexCudnnFlashAttentionDescriptor(
    CorexCudnnFlashAttentionDescriptor &&other) noexcept
    : descriptor_(std::exchange(other.descriptor_, nullptr)) {}

CorexCudnnFlashAttentionDescriptor &
CorexCudnnFlashAttentionDescriptor::operator=(
    CorexCudnnFlashAttentionDescriptor &&other) noexcept {
  if (this != &other) {
    using Destroy = cudnnStatus_t (*)(cudnnFlashAttnDescriptor_t);
    const Destroy destroy =
        dynamic_function<Destroy>("cudnnDestroyFlashAttnDescriptor");
    if (descriptor_ != nullptr && destroy != nullptr) {
      (void)destroy(descriptor_);
    }
    descriptor_ = std::exchange(other.descriptor_, nullptr);
  }
  return *this;
}

CorexDeviceWorkspace::CorexDeviceWorkspace(std::size_t size) : size_(size) {
  if (size_ != 0) {
    check_cuda_runtime(cudaMalloc(&data_, size_), "cudaMalloc(reference)");
  }
}

CorexDeviceWorkspace::~CorexDeviceWorkspace() noexcept {
  if (data_ != nullptr) {
    (void)cudaFree(data_);
  }
}

CorexDeviceWorkspace::CorexDeviceWorkspace(
    CorexDeviceWorkspace &&other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)) {}

CorexDeviceWorkspace &
CorexDeviceWorkspace::operator=(CorexDeviceWorkspace &&other) noexcept {
  if (this != &other) {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
  }
  return *this;
}

} // namespace flagdnn::iluvatar::validation
