/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/reduction.hpp"

#include "backends/mthreads/validation/functional/tensor_io_adapter.hpp"
#include "backends/mthreads/validation/mudnn_reduction.hpp"

#include <cstddef>
#include <memory>
#include <span>

namespace flagdnn::testing {
namespace {

namespace mv = validation::mthreads;

class MudnnFunctionalReductionExecutable final
    : public ReductionExecutable {
 public:
  explicit MudnnFunctionalReductionExecutable(
      const ReductionTestCase& test_case)
      : operation_(describe(test_case)) {
    validate_reduction_case(test_case);
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
  static mv::MudnnReductionDescriptor describe(
      const ReductionTestCase& test_case) {
    return {
        test_case.mode,
        mv::describe_tensor(test_case.input),
        mv::describe_tensor(test_case.output),
        test_case.axis,
        test_case.keep_dimensions,
    };
  }

  mv::MudnnReductionOperation operation_;
};

}  // namespace

TestTensor reduction_reference_input_tensor(
    const ReductionTestCase& test_case) {
  validate_reduction_case(test_case);
  return test_case.input;
}

std::unique_ptr<ReductionExecutable> build_reduction_reference(
    const ReductionTestCase& test_case) {
  return std::make_unique<MudnnFunctionalReductionExecutable>(test_case);
}

}  // namespace flagdnn::testing
