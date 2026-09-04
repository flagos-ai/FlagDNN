/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_POINTWISE_HPP_
#define FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_POINTWISE_HPP_

#include "backends/mthreads/validation/tensor.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace flagdnn::validation::mthreads {

struct MudnnPointwiseDescriptor {
  flagdnnPointwiseMode_t mode = FLAGDNN_POINTWISE_NOT_SET;
  std::vector<TensorDescriptor> inputs;
  TensorDescriptor output;
  flagdnnPointwiseAttributes_t attributes =
      FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  double alpha = 1.0;
};

// Direct muDNN tensor-operator execution. MUSA Graph is intentionally not
// modeled here; the benchmark protocol captures this operator and FlagDNN on
// the same caller stream.
class MudnnPointwiseOperation final {
 public:
  explicit MudnnPointwiseOperation(MudnnPointwiseDescriptor descriptor);
  ~MudnnPointwiseOperation();

  MudnnPointwiseOperation(const MudnnPointwiseOperation&) = delete;
  MudnnPointwiseOperation& operator=(const MudnnPointwiseOperation&) = delete;

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

#endif  // FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_POINTWISE_HPP_
