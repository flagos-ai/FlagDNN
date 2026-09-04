/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/convolution.hpp"

#include "backends/mthreads/validation/functional/tensor_io_adapter.hpp"
#include "backends/mthreads/validation/mudnn_convolution.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>

namespace flagdnn::testing {
namespace {

namespace mv = validation::mthreads;

mv::MudnnConvolutionDirection mudnn_direction(
    ConvolutionDirection direction) {
  switch (direction) {
    case ConvolutionDirection::kFprop:
      return mv::MudnnConvolutionDirection::kFprop;
    case ConvolutionDirection::kDgrad:
      return mv::MudnnConvolutionDirection::kDgrad;
    case ConvolutionDirection::kWgrad:
      return mv::MudnnConvolutionDirection::kWgrad;
  }
  throw std::invalid_argument(
      "unsupported MThreads functional Convolution direction");
}

mv::MudnnConvolutionMode mudnn_mode(ConvolutionMode mode) {
  switch (mode) {
    case ConvolutionMode::kCrossCorrelation:
      return mv::MudnnConvolutionMode::kCrossCorrelation;
    case ConvolutionMode::kConvolution:
      return mv::MudnnConvolutionMode::kConvolution;
  }
  throw std::invalid_argument(
      "unsupported MThreads functional Convolution mode");
}

class MudnnFunctionalConvolutionExecutable final
    : public ConvolutionExecutable {
 public:
  explicit MudnnFunctionalConvolutionExecutable(
      const ConvolutionTestCase& test_case)
      : operation_({
            mudnn_direction(test_case.direction),
            mudnn_mode(test_case.mode),
            mv::describe_tensor(test_case.x),
            mv::describe_tensor(test_case.w),
            mv::describe_tensor(test_case.y),
            test_case.pre_padding,
            test_case.post_padding,
            test_case.stride,
            test_case.dilation,
            test_case.groups,
        }) {
    validate_convolution_case(test_case);
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return operation_.workspace_size();
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    operation_.execute(
        bindings, workspace, workspace_size, stream);
  }

 private:
  mv::MudnnConvolutionOperation operation_;
};

}  // namespace

std::unique_ptr<ConvolutionExecutable>
build_convolution_reference(const ConvolutionTestCase& test_case) {
  return std::make_unique<MudnnFunctionalConvolutionExecutable>(
      test_case);
}

}  // namespace flagdnn::testing
