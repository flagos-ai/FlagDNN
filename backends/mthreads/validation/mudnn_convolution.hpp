/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_CONVOLUTION_HPP_
#define FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_CONVOLUTION_HPP_

#include "backends/mthreads/validation/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace flagdnn::validation::mthreads {

enum class MudnnConvolutionDirection {
  kFprop,
  kDgrad,
  kWgrad,
};

enum class MudnnConvolutionMode {
  kCrossCorrelation,
  kConvolution,
};

// The tensor roles are mathematical and remain stable for every direction:
//   result = convolution(image, filter)
// FProp consumes image/filter, Dgrad consumes result/filter, and Wgrad
// consumes result/image.  This mirrors the public FlagDNN Graph contract while
// the implementation below invokes independent muDNN C++ primitives directly.
struct MudnnConvolutionDescriptor {
  MudnnConvolutionDirection direction =
      MudnnConvolutionDirection::kFprop;
  MudnnConvolutionMode mode =
      MudnnConvolutionMode::kCrossCorrelation;
  TensorDescriptor image;
  TensorDescriptor filter;
  TensorDescriptor result;
  std::vector<std::int64_t> pre_padding;
  std::vector<std::int64_t> post_padding;
  std::vector<std::int64_t> stride;
  std::vector<std::int64_t> dilation;
  std::int64_t groups = 1;
  int input_precision = 0;
};

// muDNN has no cudnn-frontend-style Graph API.  Validation therefore compares
// the public FlagDNN Graph with this independently configured direct
// Convolution operator.  Layout conversion, asymmetric padding, 1D promotion,
// and explicit filter reversal are implemented only inside the validation
// adapter and are all enqueued on the exact caller stream.
class MudnnConvolutionOperation final {
 public:
  explicit MudnnConvolutionOperation(
      MudnnConvolutionDescriptor descriptor);
  ~MudnnConvolutionOperation();

  MudnnConvolutionOperation(const MudnnConvolutionOperation&) = delete;
  MudnnConvolutionOperation& operator=(
      const MudnnConvolutionOperation&) = delete;

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

#endif  // FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_CONVOLUTION_HPP_
