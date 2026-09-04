/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_LAYOUT_HPP_
#define FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_LAYOUT_HPP_

#include "backends/mthreads/validation/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace flagdnn::validation::mthreads {

enum class MudnnLayoutMode { kReshape, kTranspose, kSlice };

struct MudnnLayoutDescriptor {
  MudnnLayoutMode mode = MudnnLayoutMode::kReshape;
  TensorDescriptor input;
  TensorDescriptor output;
  std::vector<std::int64_t> permutation;
  std::vector<std::pair<std::int64_t, std::int64_t>> slices;
  std::vector<std::int64_t> slice_strides;
};

// Direct muDNN tensor-operator execution.  muDNN has no Frontend Graph API,
// so functional and performance validation compare FlagDNN Graph execution
// with this independently configured Permute operator on the caller stream.
class MudnnLayoutOperation final {
 public:
  explicit MudnnLayoutOperation(MudnnLayoutDescriptor descriptor);
  ~MudnnLayoutOperation();

  MudnnLayoutOperation(const MudnnLayoutOperation&) = delete;
  MudnnLayoutOperation& operator=(const MudnnLayoutOperation&) = delete;

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

#endif  // FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_LAYOUT_HPP_
