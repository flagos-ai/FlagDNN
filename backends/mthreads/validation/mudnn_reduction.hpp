/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_REDUCTION_HPP_
#define FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_REDUCTION_HPP_

#include "backends/mthreads/validation/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace flagdnn::validation::mthreads {

struct MudnnReductionDescriptor {
  flagdnnReductionMode_t mode = FLAGDNN_REDUCTION_ADD;
  TensorDescriptor input;
  TensorDescriptor output;
  std::int32_t axis = 0;
  bool keep_dimensions = false;
};

// muDNN has no Frontend Graph API. Functional and performance validation
// therefore compares FlagDNN Graph execution with this independently
// configured muDNN Reduce operator on the exact caller stream.
class MudnnReductionOperation final {
 public:
  explicit MudnnReductionOperation(MudnnReductionDescriptor descriptor);
  ~MudnnReductionOperation();

  MudnnReductionOperation(const MudnnReductionOperation&) = delete;
  MudnnReductionOperation& operator=(const MudnnReductionOperation&) = delete;

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

#endif  // FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_REDUCTION_HPP_
