/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/matmul.hpp"

#include "backends/mthreads/validation/functional/tensor_io_adapter.hpp"
#include "backends/mthreads/validation/mudnn_matmul.hpp"

#include <cstddef>
#include <memory>
#include <span>

namespace flagdnn::testing {
namespace {

namespace mv = validation::mthreads;

class MudnnFunctionalMatmulExecutable final : public MatmulExecutable {
 public:
  explicit MudnnFunctionalMatmulExecutable(
      const MatmulTestCase& test_case)
      : operation_({mv::describe_tensor(test_case.a),
                    mv::describe_tensor(test_case.b),
                    mv::describe_tensor(test_case.output)}) {
    validate_matmul_case(test_case);
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
  mv::MudnnMatmulOperation operation_;
};

}  // namespace

std::unique_ptr<MatmulExecutable> build_matmul_reference(
    const MatmulTestCase& test_case) {
  return std::make_unique<MudnnFunctionalMatmulExecutable>(test_case);
}

}  // namespace flagdnn::testing
