/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/layout.hpp"
#include "validation/functional/cudnn_graph.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace flagdnn::testing {
namespace {

namespace cfe = cuda::cfe;

std::string_view operation_name(LayoutOperation operation) {
  switch (operation) {
    case LayoutOperation::kReshape:
      return "reshape";
    case LayoutOperation::kTranspose:
      return "transpose";
    case LayoutOperation::kSlice:
      return "slice";
  }
  throw std::invalid_argument("unsupported cuDNN Layout operation");
}

void require_plan_stage(cfe::error_t status,
                        std::string_view operation,
                        std::string_view stage) {
  if (status.is_good()) {
    return;
  }
  const cfe::error_code_t code = status.get_code();
  if (code == cfe::error_code_t::HEURISTIC_QUERY_FAILED ||
      code == cfe::error_code_t::GRAPH_NOT_SUPPORTED ||
      code == cfe::error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED) {
    throw std::runtime_error(
        "cuDNN Frontend native " + std::string(operation) +
        " graph validated and lowered, but the backend has no standalone "
        "execution plan during " + std::string(stage));
  }
  throw std::runtime_error(
      "cuDNN Frontend native " + std::string(operation) + " " +
      std::string(stage) + " failed: " + status.get_message());
}

class CudnnLayoutExecutable final : public cuda::CudnnGraphExecutable {
 public:
  explicit CudnnLayoutExecutable(const LayoutTestCase& test_case)
      : graph_(std::make_shared<cfe::graph::Graph>()) {
    validate_layout_case(test_case);
    const std::string_view operation = operation_name(test_case.operation);
    graph_->set_name(test_case.name + "::cudnn")
        .set_io_data_type(
            cuda::cudnn_frontend_data_type(test_case.input.data_type))
        .set_intermediate_data_type(cfe::DataType_t::FLOAT)
        .set_compute_data_type(cfe::DataType_t::FLOAT);
    const auto input = cuda::make_cudnn_tensor(graph_, test_case.input, "input");

    std::shared_ptr<cfe::graph::Tensor_attributes> output;
    switch (test_case.operation) {
      case LayoutOperation::kReshape:
        output = graph_->reshape(
            input,
            cfe::graph::Reshape_attributes()
                .set_name("reshape")
                .set_compute_data_type(cfe::DataType_t::FLOAT)
                .set_dim(test_case.output.dimensions)
                .set_stride(test_case.output.strides)
                .set_reshape_mode(cfe::ReshapeMode_t::LOGICAL));
        break;
      case LayoutOperation::kTranspose:
        output = graph_->transpose(
            input,
            cfe::graph::Transpose_attributes()
                .set_name("transpose")
                .set_permutation(test_case.permutation));
        break;
      case LayoutOperation::kSlice:
        output = graph_->slice(
            input,
            cfe::graph::Slice_attributes()
                .set_name("slice")
                .set_compute_data_type(cfe::DataType_t::FLOAT)
                .set_slices(test_case.slices)
                .set_strides(test_case.slice_strides));
        break;
    }
    output->set_name("output")
        .set_uid(test_case.output.uid)
        .set_data_type(
            cuda::cudnn_frontend_data_type(test_case.output.data_type))
        .set_dim(test_case.output.dimensions)
        .set_stride(test_case.output.strides)
        .set_output(true);

    cuda::check_cudnn_frontend(graph_->validate(),
                               "cuDNN Layout graph validation");
    cuda::check_cudnn_frontend(graph_->build_operation_graph(handle()),
                               "cuDNN Layout operation graph lowering");
    require_plan_stage(
        graph_->create_execution_plans(
            {cfe::HeurMode_t::A, cfe::HeurMode_t::FALLBACK}),
        operation,
        "execution-plan discovery");
    require_plan_stage(graph_->check_support(handle()), operation, "support check");
    require_plan_stage(graph_->build_plans(handle()), operation, "plan build");

    std::int64_t workspace_size = 0;
    cuda::check_cudnn_frontend(graph_->get_workspace_size(workspace_size),
                               "cuDNN Layout workspace query");
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
                               "cuDNN Layout graph execute");
  }

 private:
  std::shared_ptr<cfe::graph::Graph> graph_;
};

}  // namespace

std::unique_ptr<LayoutExecutable> build_layout_reference(
    const LayoutTestCase& test_case) {
  return std::make_unique<CudnnLayoutExecutable>(test_case);
}

}  // namespace flagdnn::testing
