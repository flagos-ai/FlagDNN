/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/pointwise.hpp"

#include "backends/mthreads/validation/functional/tensor_io_adapter.hpp"
#include "backends/mthreads/validation/mudnn_pointwise.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace flagdnn::testing {
namespace {

namespace mv = validation::mthreads;

class MudnnFunctionalPointwiseExecutable final
    : public PointwiseExecutable {
 public:
  explicit MudnnFunctionalPointwiseExecutable(
      const PointwiseTestCase& test_case)
      : operation_(describe(test_case)) {
    validate_pointwise_case(test_case);
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
  static mv::MudnnPointwiseDescriptor describe(
      const PointwiseTestCase& test_case) {
    std::vector<mv::TensorDescriptor> inputs;
    inputs.reserve(test_case.inputs.size());
    for (const TestTensor& input : test_case.inputs) {
      inputs.push_back(mv::describe_tensor(input));
    }
    return {
        test_case.mode,
        std::move(inputs),
        mv::describe_tensor(test_case.output),
        test_case.attributes,
        test_case.alpha,
    };
  }

  mv::MudnnPointwiseOperation operation_;
};

}  // namespace

std::unique_ptr<PointwiseExecutable> build_pointwise_reference(
    const PointwiseTestCase& test_case) {
  return std::make_unique<MudnnFunctionalPointwiseExecutable>(test_case);
}

}  // namespace flagdnn::testing
