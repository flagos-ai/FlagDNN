/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_ATTENTION_HPP_
#define FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_ATTENTION_HPP_

#include "backends/mthreads/validation/tensor.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <span>

namespace flagdnn::validation::mthreads {

struct MudnnSdpaDescriptor {
  TensorDescriptor q;
  TensorDescriptor k;
  TensorDescriptor v;
  std::optional<TensorDescriptor> bias;
  TensorDescriptor output;
  std::optional<TensorDescriptor> stats;
  double attention_scale = 1.0;
  bool causal = false;
};

struct MudnnSdpaBackwardDescriptor {
  TensorDescriptor q;
  TensorDescriptor k;
  TensorDescriptor v;
  std::optional<TensorDescriptor> bias;
  TensorDescriptor output;
  TensorDescriptor doutput;
  TensorDescriptor stats;
  TensorDescriptor dq;
  TensorDescriptor dk;
  TensorDescriptor dv;
  std::optional<TensorDescriptor> dbias;
  double attention_scale = 1.0;
  bool causal = false;
  bool deterministic = false;
};

// muDNN 3.1.5 exposes ScaledDotProductAttention as a direct C++ operator,
// but does not expose a frontend Graph API.  These validation-only wrappers
// intentionally preserve the public FlagDNN binding and stream contracts so
// functional tests compare the public Graph with that direct operator.
class MudnnSdpaOperation final {
 public:
  explicit MudnnSdpaOperation(MudnnSdpaDescriptor descriptor);
  ~MudnnSdpaOperation();

  MudnnSdpaOperation(const MudnnSdpaOperation&) = delete;
  MudnnSdpaOperation& operator=(const MudnnSdpaOperation&) = delete;

  [[nodiscard]] std::size_t workspace_size() const noexcept;
  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream);

 private:
  struct Impl;
  std::unique_ptr<Impl> implementation_;
};

class MudnnSdpaBackwardOperation final {
 public:
  explicit MudnnSdpaBackwardOperation(
      MudnnSdpaBackwardDescriptor descriptor);
  ~MudnnSdpaBackwardOperation();

  MudnnSdpaBackwardOperation(const MudnnSdpaBackwardOperation&) = delete;
  MudnnSdpaBackwardOperation& operator=(
      const MudnnSdpaBackwardOperation&) = delete;

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

#endif  // FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_ATTENTION_HPP_
