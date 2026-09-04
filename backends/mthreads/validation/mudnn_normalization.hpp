/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_NORMALIZATION_HPP_
#define FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_NORMALIZATION_HPP_

#include "backends/mthreads/validation/tensor.hpp"

#include <cstddef>
#include <memory>
#include <span>

namespace flagdnn::validation::mthreads {

struct MudnnLayernormDescriptor {
  TensorDescriptor x;
  TensorDescriptor scale;
  TensorDescriptor bias;
  TensorDescriptor y;
  TensorDescriptor mean;
  TensorDescriptor inv_variance;
  double epsilon = 1.0e-3;
};

struct MudnnRmsnormDescriptor {
  TensorDescriptor x;
  TensorDescriptor scale;
  TensorDescriptor bias;
  TensorDescriptor y;
  TensorDescriptor inv_variance;
  double epsilon = 1.0e-3;
};

struct MudnnBatchnormDescriptor {
  TensorDescriptor x;
  TensorDescriptor scale;
  TensorDescriptor bias;
  TensorDescriptor previous_running_mean;
  TensorDescriptor previous_running_variance;
  TensorDescriptor y;
  TensorDescriptor mean;
  TensorDescriptor inv_variance;
  TensorDescriptor next_running_mean;
  TensorDescriptor next_running_variance;
  double epsilon = 1.0e-3;
  double momentum = 0.1;
};

struct MudnnBatchnormInferenceDescriptor {
  TensorDescriptor x;
  TensorDescriptor mean;
  TensorDescriptor inv_variance;
  TensorDescriptor scale;
  TensorDescriptor bias;
  TensorDescriptor y;
};

// muDNN exposes C++ normalization operators but no Frontend Graph API.
// These validation-only wrappers preserve FlagDNN's external binding ABI and
// always enqueue the direct muDNN reference on the caller-provided stream.
class MudnnLayernormOperation final {
 public:
  explicit MudnnLayernormOperation(MudnnLayernormDescriptor descriptor);
  ~MudnnLayernormOperation();

  MudnnLayernormOperation(const MudnnLayernormOperation&) = delete;
  MudnnLayernormOperation& operator=(const MudnnLayernormOperation&) = delete;

  [[nodiscard]] std::size_t workspace_size() const noexcept;
  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream);

 private:
  struct Impl;
  std::unique_ptr<Impl> implementation_;
};

class MudnnRmsnormOperation final {
 public:
  explicit MudnnRmsnormOperation(MudnnRmsnormDescriptor descriptor);
  ~MudnnRmsnormOperation();

  MudnnRmsnormOperation(const MudnnRmsnormOperation&) = delete;
  MudnnRmsnormOperation& operator=(const MudnnRmsnormOperation&) = delete;

  [[nodiscard]] std::size_t workspace_size() const noexcept;
  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream);

 private:
  struct Impl;
  std::unique_ptr<Impl> implementation_;
};

class MudnnBatchnormOperation final {
 public:
  explicit MudnnBatchnormOperation(MudnnBatchnormDescriptor descriptor);
  ~MudnnBatchnormOperation();

  MudnnBatchnormOperation(const MudnnBatchnormOperation&) = delete;
  MudnnBatchnormOperation& operator=(const MudnnBatchnormOperation&) = delete;

  [[nodiscard]] std::size_t workspace_size() const noexcept;
  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream);

 private:
  struct Impl;
  std::unique_ptr<Impl> implementation_;
};

class MudnnBatchnormInferenceOperation final {
 public:
  explicit MudnnBatchnormInferenceOperation(
      MudnnBatchnormInferenceDescriptor descriptor);
  ~MudnnBatchnormInferenceOperation();

  MudnnBatchnormInferenceOperation(
      const MudnnBatchnormInferenceOperation&) = delete;
  MudnnBatchnormInferenceOperation& operator=(
      const MudnnBatchnormInferenceOperation&) = delete;

  [[nodiscard]] std::size_t workspace_size() const noexcept;
  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream);

 private:
  struct Impl;
  std::unique_ptr<Impl> implementation_;
};

}  // namespace flagdnn::validation::mthreads

#endif  // FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_NORMALIZATION_HPP_
