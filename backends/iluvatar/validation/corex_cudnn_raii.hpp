// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_COREX_CUDNN_RAII_HPP_
#define FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_COREX_CUDNN_RAII_HPP_

#include "common/common.hpp"

#include <cudnn.h>

#include <cstddef>
#include <span>

namespace flagdnn::iluvatar::validation {

struct CorexCudnnCapability;
struct ReferenceTensor;

class CorexCudnnHandle final {
public:
  CorexCudnnHandle();
  ~CorexCudnnHandle() noexcept;
  CorexCudnnHandle(const CorexCudnnHandle &) = delete;
  CorexCudnnHandle &operator=(const CorexCudnnHandle &) = delete;
  CorexCudnnHandle(CorexCudnnHandle &&other) noexcept;
  CorexCudnnHandle &operator=(CorexCudnnHandle &&other) noexcept;

  [[nodiscard]] cudnnHandle_t get() const noexcept { return handle_; }
  void bind_stream(flagdnnStream_t stream);

private:
  cudnnHandle_t handle_ = nullptr;
};

class CorexCudnnTensorDescriptor final {
public:
  CorexCudnnTensorDescriptor();
  explicit CorexCudnnTensorDescriptor(const ReferenceTensor &tensor);
  ~CorexCudnnTensorDescriptor() noexcept;
  CorexCudnnTensorDescriptor(const CorexCudnnTensorDescriptor &) = delete;
  CorexCudnnTensorDescriptor &
  operator=(const CorexCudnnTensorDescriptor &) = delete;
  CorexCudnnTensorDescriptor(CorexCudnnTensorDescriptor &&other) noexcept;
  CorexCudnnTensorDescriptor &
  operator=(CorexCudnnTensorDescriptor &&other) noexcept;

  void set(const ReferenceTensor &tensor);
  [[nodiscard]] cudnnTensorDescriptor_t get() const noexcept {
    return descriptor_;
  }

private:
  cudnnTensorDescriptor_t descriptor_ = nullptr;
};

class CorexCudnnFilterDescriptor final {
public:
  CorexCudnnFilterDescriptor();
  ~CorexCudnnFilterDescriptor() noexcept;
  CorexCudnnFilterDescriptor(const CorexCudnnFilterDescriptor &) = delete;
  CorexCudnnFilterDescriptor &
  operator=(const CorexCudnnFilterDescriptor &) = delete;
  CorexCudnnFilterDescriptor(CorexCudnnFilterDescriptor &&other) noexcept;
  CorexCudnnFilterDescriptor &
  operator=(CorexCudnnFilterDescriptor &&other) noexcept;
  void set(cudnnDataType_t data_type, std::span<const int> dimensions);
  [[nodiscard]] cudnnFilterDescriptor_t get() const noexcept {
    return descriptor_;
  }

private:
  cudnnFilterDescriptor_t descriptor_ = nullptr;
};

class CorexCudnnConvolutionDescriptor final {
public:
  CorexCudnnConvolutionDescriptor();
  ~CorexCudnnConvolutionDescriptor() noexcept;
  CorexCudnnConvolutionDescriptor(const CorexCudnnConvolutionDescriptor &) =
      delete;
  CorexCudnnConvolutionDescriptor &
  operator=(const CorexCudnnConvolutionDescriptor &) = delete;
  CorexCudnnConvolutionDescriptor(
      CorexCudnnConvolutionDescriptor &&other) noexcept;
  CorexCudnnConvolutionDescriptor &
  operator=(CorexCudnnConvolutionDescriptor &&other) noexcept;
  [[nodiscard]] cudnnConvolutionDescriptor_t get() const noexcept {
    return descriptor_;
  }

private:
  cudnnConvolutionDescriptor_t descriptor_ = nullptr;
};

class CorexCudnnOpTensorDescriptor final {
public:
  CorexCudnnOpTensorDescriptor();
  ~CorexCudnnOpTensorDescriptor() noexcept;
  CorexCudnnOpTensorDescriptor(const CorexCudnnOpTensorDescriptor &) = delete;
  CorexCudnnOpTensorDescriptor &
  operator=(const CorexCudnnOpTensorDescriptor &) = delete;
  CorexCudnnOpTensorDescriptor(CorexCudnnOpTensorDescriptor &&other) noexcept;
  CorexCudnnOpTensorDescriptor &
  operator=(CorexCudnnOpTensorDescriptor &&other) noexcept;
  [[nodiscard]] cudnnOpTensorDescriptor_t get() const noexcept {
    return descriptor_;
  }

private:
  cudnnOpTensorDescriptor_t descriptor_ = nullptr;
};

class CorexCudnnReductionDescriptor final {
public:
  CorexCudnnReductionDescriptor();
  ~CorexCudnnReductionDescriptor() noexcept;
  CorexCudnnReductionDescriptor(const CorexCudnnReductionDescriptor &) = delete;
  CorexCudnnReductionDescriptor &
  operator=(const CorexCudnnReductionDescriptor &) = delete;
  CorexCudnnReductionDescriptor(CorexCudnnReductionDescriptor &&other) noexcept;
  CorexCudnnReductionDescriptor &
  operator=(CorexCudnnReductionDescriptor &&other) noexcept;
  [[nodiscard]] cudnnReduceTensorDescriptor_t get() const noexcept {
    return descriptor_;
  }

private:
  cudnnReduceTensorDescriptor_t descriptor_ = nullptr;
};

class CorexCudnnActivationDescriptor final {
public:
  CorexCudnnActivationDescriptor();
  ~CorexCudnnActivationDescriptor() noexcept;
  CorexCudnnActivationDescriptor(const CorexCudnnActivationDescriptor &) =
      delete;
  CorexCudnnActivationDescriptor &
  operator=(const CorexCudnnActivationDescriptor &) = delete;
  CorexCudnnActivationDescriptor(
      CorexCudnnActivationDescriptor &&other) noexcept;
  CorexCudnnActivationDescriptor &
  operator=(CorexCudnnActivationDescriptor &&other) noexcept;
  [[nodiscard]] cudnnActivationDescriptor_t get() const noexcept {
    return descriptor_;
  }

private:
  cudnnActivationDescriptor_t descriptor_ = nullptr;
};

class CorexCudnnFlashAttentionDescriptor final {
public:
  CorexCudnnFlashAttentionDescriptor();
  ~CorexCudnnFlashAttentionDescriptor() noexcept;
  CorexCudnnFlashAttentionDescriptor(
      const CorexCudnnFlashAttentionDescriptor &) = delete;
  CorexCudnnFlashAttentionDescriptor &
  operator=(const CorexCudnnFlashAttentionDescriptor &) = delete;
  CorexCudnnFlashAttentionDescriptor(
      CorexCudnnFlashAttentionDescriptor &&other) noexcept;
  CorexCudnnFlashAttentionDescriptor &
  operator=(CorexCudnnFlashAttentionDescriptor &&other) noexcept;
  [[nodiscard]] cudnnFlashAttnDescriptor_t get() const noexcept {
    return descriptor_;
  }
  [[nodiscard]] static bool symbols_available() noexcept;

private:
  cudnnFlashAttnDescriptor_t descriptor_ = nullptr;
};

class CorexDeviceWorkspace final {
public:
  explicit CorexDeviceWorkspace(std::size_t size);
  ~CorexDeviceWorkspace() noexcept;
  CorexDeviceWorkspace(const CorexDeviceWorkspace &) = delete;
  CorexDeviceWorkspace &operator=(const CorexDeviceWorkspace &) = delete;
  CorexDeviceWorkspace(CorexDeviceWorkspace &&other) noexcept;
  CorexDeviceWorkspace &operator=(CorexDeviceWorkspace &&other) noexcept;

  [[nodiscard]] void *data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

private:
  void *data_ = nullptr;
  std::size_t size_ = 0;
};

class CorexCudnnExecutable : public flagdnn::testing::TestExecutable {
public:
  [[nodiscard]] virtual CorexCudnnCapability
  probe(std::span<const flagdnnBinding_t> bindings, void *workspace,
        std::size_t workspace_size, flagdnnStream_t stream) = 0;
};

} // namespace flagdnn::iluvatar::validation

#endif // FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_COREX_CUDNN_RAII_HPP_
