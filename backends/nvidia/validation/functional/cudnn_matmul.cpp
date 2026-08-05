/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/matmul.hpp"
#include "validation/functional/cudnn_graph.hpp"
#include "validation/tensor_io.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>

namespace flagdnn::testing {
namespace {

namespace cfe = cuda::cfe;

TestTensor padded_matmul_tensor(const TestTensor& tensor,
                                std::size_t rank) {
  if (tensor.dimensions.size() > rank) {
    throw std::invalid_argument("cuDNN MatMul tensor rank is invalid");
  }
  TestTensor result = tensor;
  const std::int64_t storage_span =
      static_cast<std::int64_t>(cuda::storage_element_count(tensor));
  const std::size_t leading = rank - tensor.dimensions.size();
  result.dimensions.insert(result.dimensions.begin(), leading, 1);
  result.strides.insert(result.strides.begin(), leading, storage_span);
  return result;
}

class CudnnMatmulExecutable final : public cuda::CudnnGraphExecutable {
 public:
  explicit CudnnMatmulExecutable(const MatmulTestCase& test_case)
      : graph_(std::make_shared<cfe::graph::Graph>()) {
    validate_matmul_case(test_case);
    const std::size_t rank =
        std::max<std::size_t>(3, test_case.output.dimensions.size());
    const TestTensor a_specification = padded_matmul_tensor(test_case.a, rank);
    const TestTensor b_specification = padded_matmul_tensor(test_case.b, rank);
    const TestTensor output_specification =
        padded_matmul_tensor(test_case.output, rank);

    graph_->set_name(test_case.name + "::cudnn")
        .set_io_data_type(cuda::cudnn_frontend_data_type(test_case.a.data_type))
        .set_intermediate_data_type(cfe::DataType_t::FLOAT)
        .set_compute_data_type(cfe::DataType_t::FLOAT);
    const auto a = cuda::make_cudnn_tensor(graph_, a_specification, "a");
    const auto b = cuda::make_cudnn_tensor(graph_, b_specification, "b");
    auto output = graph_->matmul(
        a,
        b,
        cfe::graph::Matmul_attributes()
            .set_name("matmul")
            .set_compute_data_type(cfe::DataType_t::FLOAT));
    output->set_name("output")
        .set_uid(output_specification.uid)
        .set_data_type(
            cuda::cudnn_frontend_data_type(output_specification.data_type))
        .set_dim(output_specification.dimensions)
        .set_stride(output_specification.strides)
        .set_output(true);

    cuda::check_cudnn_frontend(
        graph_->build(handle(), {cfe::HeurMode_t::A}),
        "cuDNN MatMul graph build");
    std::int64_t workspace_size = 0;
    cuda::check_cudnn_frontend(graph_->get_workspace_size(workspace_size),
                               "cuDNN MatMul workspace query");
    set_workspace_size(workspace_size);
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    begin_execute(workspace, workspace_size, stream);
    cuda::CudnnBindingMap pointers =
        cuda::make_cudnn_binding_map(bindings);
    cuda::check_cudnn_frontend(graph_->execute(handle(), pointers, workspace),
                               "cuDNN MatMul graph execute");
  }

 private:
  std::shared_ptr<cfe::graph::Graph> graph_;
};

}  // namespace

std::unique_ptr<MatmulExecutable> build_matmul_reference(
    const MatmulTestCase& test_case) {
  return std::make_unique<CudnnMatmulExecutable>(test_case);
}

}  // namespace flagdnn::testing
