/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/layout.hpp"

#include "backends/mthreads/validation/functional/tensor_io_adapter.hpp"
#include "backends/mthreads/validation/mudnn_layout.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>

namespace flagdnn::testing {
namespace {

namespace mv = validation::mthreads;

mv::MudnnLayoutMode mudnn_layout_mode(LayoutOperation operation) {
  switch (operation) {
    case LayoutOperation::kReshape:
      return mv::MudnnLayoutMode::kReshape;
    case LayoutOperation::kTranspose:
      return mv::MudnnLayoutMode::kTranspose;
    case LayoutOperation::kSlice:
      return mv::MudnnLayoutMode::kSlice;
  }
  throw std::invalid_argument("unsupported MThreads layout operation");
}

class MudnnFunctionalLayoutExecutable final : public LayoutExecutable {
 public:
  explicit MudnnFunctionalLayoutExecutable(const LayoutTestCase& test_case)
      : operation_(describe(test_case)) {
    validate_layout_case(test_case);
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
  static mv::MudnnLayoutDescriptor describe(
      const LayoutTestCase& test_case) {
    return {
        mudnn_layout_mode(test_case.operation),
        mv::describe_tensor(test_case.input),
        mv::describe_tensor(test_case.output),
        test_case.permutation,
        test_case.slices,
        test_case.slice_strides,
    };
  }

  mv::MudnnLayoutOperation operation_;
};

}  // namespace

std::unique_ptr<LayoutExecutable> build_layout_reference(
    const LayoutTestCase& test_case) {
  return std::make_unique<MudnnFunctionalLayoutExecutable>(test_case);
}

}  // namespace flagdnn::testing
