/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/mthreads/validation/mudnn_reference.hpp"

#include "backends/mthreads/validation/functional/tensor_io_adapter.hpp"
#include "backends/mthreads/validation/mudnn_add.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <utility>

namespace flagdnn::validation::mthreads {
namespace {

class MudnnFunctionalAddExecutable final
    : public flagdnn::testing::AddExecutable {
 public:
  explicit MudnnFunctionalAddExecutable(
      const flagdnn::testing::AddTestCase& test_case)
      : operation_(MudnnAddDescriptor{
            describe_tensor(test_case.left),
            describe_tensor(test_case.right),
            describe_tensor(test_case.output),
            test_case.alpha,
        }) {
    flagdnn::testing::validate_add_case(test_case);
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
  MudnnAddOperation operation_;
};

}  // namespace

std::unique_ptr<flagdnn::testing::AddExecutable>
build_mudnn_add_reference(
    const flagdnn::testing::AddTestCase& test_case) {
  return std::make_unique<MudnnFunctionalAddExecutable>(test_case);
}

}  // namespace flagdnn::validation::mthreads
