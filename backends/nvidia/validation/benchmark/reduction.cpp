/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/reduction.hpp"

#include "cudnn_common.hpp"

namespace flagdnn::benchmarking::cudnn_detail {
namespace {
class ReductionExecutable final : public BenchmarkExecutable {
 public:
  explicit ReductionExecutable(const BenchmarkCase& specification) {
    const auto tensor = [](const TensorSpec& value) {
      return testing::TestTensor{value.uid, value.data_type, value.dimensions,
                                 value.strides, value.binding_byte_offset};
    };
    testing::ReductionTestCase value;
    value.name = specification.name;
    value.input = tensor(specification.tensors.at(0));
    value.output = tensor(specification.tensors.at(1));
    value.mode = specification.reduction_mode;
    value.axis = specification.reduction_axis;
    value.keep_dimensions = specification.keep_dimensions;
    executable_ = testing::build_reduction_reference(value);
  }
  std::size_t workspace_size() const noexcept override {
    return executable_->workspace_size();
  }
  void execute(std::span<const flagdnnBinding_t> bindings, void* workspace,
               std::size_t size, flagdnnStream_t stream) override {
    executable_->execute(bindings, workspace, size, stream);
  }

 private:
  std::unique_ptr<testing::TestExecutable> executable_;
};
}  // namespace
std::unique_ptr<BenchmarkExecutable> build_reduction(
    const BenchmarkCase& specification) {
  return std::make_unique<ReductionExecutable>(specification);
}
}  // namespace flagdnn::benchmarking::cudnn_detail
