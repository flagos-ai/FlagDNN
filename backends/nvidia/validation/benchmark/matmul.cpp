/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "ops.hpp"

#include "validation/benchmark/cudnn_common.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>

namespace flagdnn::benchmarking::cudnn_detail {
namespace {

TensorSpec padded_matmul_tensor(const TensorSpec& specification,
                                std::size_t rank) {
  if (specification.dimensions.size() > rank) {
    throw std::invalid_argument("cuDNN MatMul tensor rank is invalid");
  }
  TensorSpec result = specification;
  std::int64_t storage_span = 1;
  for (std::size_t axis = 0; axis < specification.dimensions.size(); ++axis) {
    storage_span += (specification.dimensions[axis] - 1) *
                    specification.strides[axis];
  }
  const std::size_t leading = rank - specification.dimensions.size();
  result.dimensions.insert(result.dimensions.begin(), leading, 1);
  result.strides.insert(result.strides.begin(), leading, storage_span);
  return result;
}

class MatmulExecutable final : public ExecutableBase {
 public:
  explicit MatmulExecutable(const BenchmarkCase& specification) {
    require_tensor_count(specification, 3);
    const TensorSpec& expected = specification.tensors[2];

    const std::size_t cudnn_rank =
        std::max<std::size_t>(3, expected.dimensions.size());
    const TensorSpec cudnn_a =
        padded_matmul_tensor(specification.tensors[0], cudnn_rank);
    const TensorSpec cudnn_b =
        padded_matmul_tensor(specification.tensors[1], cudnn_rank);
    const TensorSpec cudnn_output =
        padded_matmul_tensor(expected, cudnn_rank);

    graph_ = std::make_shared<fe::graph::Graph>();
    graph_->set_name(specification.name)
        .set_io_data_type(data_type(specification.tensors[0].data_type))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto a =
        make_tensor(graph_, cudnn_a, "a", false);
    auto b =
        make_tensor(graph_, cudnn_b, "b", false);
    auto output = graph_->matmul(
        a,
        b,
        fe::graph::Matmul_attributes()
            .set_name("matmul")
            .set_compute_data_type(fe::DataType_t::FLOAT));
    output->set_name("output")
        .set_uid(expected.uid)
        .set_data_type(data_type(expected.data_type))
        .set_dim(cudnn_output.dimensions)
        .set_stride(cudnn_output.strides)
        .set_output(true);

    check_frontend(
        graph_->build(handle(), {fe::HeurMode_t::A}),
        "cuDNN graph build");
    std::int64_t workspace = 0;
    check_frontend(
        graph_->get_workspace_size(workspace), "cuDNN workspace query");
    set_workspace_size(workspace);
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    begin_execute(workspace, workspace_size, stream);
    BindingMap pointers = make_binding_map(bindings);
    check_frontend(graph_->execute(handle(), pointers, workspace),
                   "cuDNN MatMul graph execute");
  }

 private:
  std::shared_ptr<fe::graph::Graph> graph_;
};

}  // namespace

std::unique_ptr<BenchmarkExecutable> build_matmul(
    const BenchmarkCase& specification) {
  return std::make_unique<MatmulExecutable>(specification);
}

}  // namespace flagdnn::benchmarking::cudnn_detail
