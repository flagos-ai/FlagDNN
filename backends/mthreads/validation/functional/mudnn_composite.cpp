/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/composite.hpp"

#include "backends/mthreads/validation/functional/tensor_io_adapter.hpp"
#include "backends/mthreads/validation/mudnn_composite.hpp"

#include <cstddef>
#include <memory>
#include <span>

namespace flagdnn::testing {
namespace {

namespace mv = validation::mthreads;

class MudnnFunctionalAddSquareExecutable final
    : public CompositeExecutable {
 public:
  explicit MudnnFunctionalAddSquareExecutable(
      const AddSquareTestCase& test_case)
      : operation_({
            mv::describe_tensor(test_case.left),
            mv::describe_tensor(test_case.right),
            mv::describe_tensor(test_case.output),
        }) {
    validate_composite_case(test_case);
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return operation_.workspace_size();
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    operation_.execute(bindings, workspace, workspace_size, stream);
  }

 private:
  mv::MudnnAddSquareOperation operation_;
};

class MudnnFunctionalConvBiasReluExecutable final
    : public CompositeExecutable {
 public:
  explicit MudnnFunctionalConvBiasReluExecutable(
      const ConvBiasReluTestCase& test_case)
      : operation_({
            mv::describe_tensor(test_case.x),
            mv::describe_tensor(test_case.w),
            mv::describe_tensor(test_case.bias),
            mv::describe_tensor(test_case.output),
            test_case.padding,
            test_case.padding,
            test_case.stride,
            test_case.dilation,
            1,
        }) {
    validate_composite_case(test_case);
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return operation_.workspace_size();
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    operation_.execute(bindings, workspace, workspace_size, stream);
  }

 private:
  mv::MudnnConvBiasReluOperation operation_;
};

}  // namespace

std::unique_ptr<CompositeExecutable> build_add_square_reference(
    const AddSquareTestCase& test_case) {
  return std::make_unique<MudnnFunctionalAddSquareExecutable>(test_case);
}

std::unique_ptr<CompositeExecutable> build_conv_bias_relu_reference(
    const ConvBiasReluTestCase& test_case) {
  return std::make_unique<MudnnFunctionalConvBiasReluExecutable>(test_case);
}

}  // namespace flagdnn::testing
