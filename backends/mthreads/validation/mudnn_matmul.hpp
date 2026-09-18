/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_MATMUL_HPP_
#define FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_MATMUL_HPP_

#include "backends/mthreads/validation/tensor.hpp"

#include <cstddef>
#include <memory>
#include <span>

namespace flagdnn::validation::mthreads {

struct MudnnMatmulDescriptor {
  TensorDescriptor a;
  TensorDescriptor b;
  TensorDescriptor output;
  int input_precision = 0;
};

// muDNN exposes MatMul and BatchMatMul C++ operators, but no Frontend Graph
// API. Validation therefore compares the public FlagDNN Graph with this
// independently configured direct operator on the exact caller stream.
class MudnnMatmulOperation final {
 public:
  explicit MudnnMatmulOperation(MudnnMatmulDescriptor descriptor);
  ~MudnnMatmulOperation();

  MudnnMatmulOperation(const MudnnMatmulOperation&) = delete;
  MudnnMatmulOperation& operator=(const MudnnMatmulOperation&) = delete;

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

#endif  // FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_MATMUL_HPP_
