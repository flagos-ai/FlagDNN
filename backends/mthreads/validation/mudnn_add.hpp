/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_ADD_HPP_
#define FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_ADD_HPP_

#include "backends/mthreads/validation/tensor.hpp"

#include <cstddef>
#include <memory>
#include <span>

namespace flagdnn::validation::mthreads {

struct MudnnAddDescriptor {
  TensorDescriptor left;
  TensorDescriptor right;
  TensorDescriptor output;
  double alpha = 1.0;
};

// Direct muDNN Binary execution.  This is intentionally an operator API,
// not a graph emulation layer: MUSA Graph capture belongs to the benchmark
// execution protocol and is applied equally to both providers.
class MudnnAddOperation final {
 public:
  explicit MudnnAddOperation(MudnnAddDescriptor descriptor);
  ~MudnnAddOperation();

  MudnnAddOperation(const MudnnAddOperation&) = delete;
  MudnnAddOperation& operator=(const MudnnAddOperation&) = delete;

  [[nodiscard]] std::size_t workspace_size() const noexcept { return 0; }
  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream);

 private:
  struct Impl;
  std::unique_ptr<Impl> implementation_;
};

}  // namespace flagdnn::validation::mthreads

#endif  // FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_ADD_HPP_
