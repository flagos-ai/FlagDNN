/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_NVIDIA_BENCHMARK_LAYOUT_REFERENCE_HPP_
#define FLAGDNN_NVIDIA_BENCHMARK_LAYOUT_REFERENCE_HPP_

#include "common/layout.hpp"
#include "validation/benchmark/cudnn_common.hpp"

namespace flagdnn::benchmarking::cudnn_detail {

// Reuse the functional adapter's native cuDNN transform when a standalone
// layout graph has no execution plan. Tensor metadata and bindings stay exact.
class LayoutReferenceExecutable final : public BenchmarkExecutable {
 public:
  LayoutReferenceExecutable(const BenchmarkCase& specification,
                            testing::LayoutOperation operation) {
    require_tensor_count(specification, 2);
    const auto tensor = [](const TensorSpec& value) {
      return testing::TestTensor{value.uid, value.data_type, value.dimensions,
                                 value.strides, value.binding_byte_offset};
    };
    testing::LayoutTestCase value;
    value.name = specification.name;
    value.operation = operation;
    value.input = tensor(specification.tensors.at(0));
    value.output = tensor(specification.tensors.at(1));
    value.permutation = specification.transpose.permutation;
    value.slices = specification.slice.slices;
    value.slice_strides = specification.slice.strides;
    executable_ = testing::build_layout_reference(value);
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
}  // namespace flagdnn::benchmarking::cudnn_detail

#endif  // FLAGDNN_NVIDIA_BENCHMARK_LAYOUT_REFERENCE_HPP_
