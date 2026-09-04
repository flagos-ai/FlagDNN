/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_COMPOSITE_HPP_
#define FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_COMPOSITE_HPP_

#include "backends/mthreads/validation/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace flagdnn::validation::mthreads {

struct MudnnAddSquareDescriptor {
  TensorDescriptor left;
  TensorDescriptor right;
  TensorDescriptor output;
};

// Direct muDNN operator sequence used only as validation reference:
// Binary(MUL) writes the validation-owned intermediate, then Binary(ADD)
// consumes it.  It intentionally does not emulate a muDNN graph frontend.
class MudnnAddSquareOperation final {
 public:
  explicit MudnnAddSquareOperation(MudnnAddSquareDescriptor descriptor);
  ~MudnnAddSquareOperation();

  MudnnAddSquareOperation(const MudnnAddSquareOperation&) = delete;
  MudnnAddSquareOperation& operator=(const MudnnAddSquareOperation&) = delete;

  [[nodiscard]] std::size_t workspace_size() const noexcept;
  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream);

 private:
  struct Impl;
  std::unique_ptr<Impl> implementation_;
};

struct MudnnConvBiasReluDescriptor {
  TensorDescriptor input;
  TensorDescriptor filter;
  TensorDescriptor bias;
  TensorDescriptor output;
  std::vector<std::int64_t> pre_padding;
  std::vector<std::int64_t> post_padding;
  std::vector<std::int64_t> stride;
  std::vector<std::int64_t> dilation;
  std::int64_t groups = 1;
};

// muDNN has no cudnn-frontend-style graph API.  This validation reference
// therefore invokes three independent C++ operators on the caller stream:
// Convolution(FProp), Binary(ADD with channel-bias broadcast), then
// Unary(ReLU).  Both intermediate tensors and the convolution workspace are
// carved from the validation-owned workspace supplied to execute().
class MudnnConvBiasReluOperation final {
 public:
  explicit MudnnConvBiasReluOperation(
      MudnnConvBiasReluDescriptor descriptor);
  ~MudnnConvBiasReluOperation();

  MudnnConvBiasReluOperation(const MudnnConvBiasReluOperation&) = delete;
  MudnnConvBiasReluOperation& operator=(
      const MudnnConvBiasReluOperation&) = delete;

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

#endif  // FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_COMPOSITE_HPP_
