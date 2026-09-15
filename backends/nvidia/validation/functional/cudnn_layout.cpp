/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

#include "common/layout.hpp"
#include "validation/cuda_driver.hpp"
#include "validation/functional/cudnn_extended.hpp"
#include "validation/functional/cudnn_graph.hpp"
#include "validation/functional/cudnn_tensor.hpp"
#include "validation/tensor_io.hpp"

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

void require_plan_stage(cfe::error_t status, std::string_view operation,
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
        "execution plan during " +
        std::string(stage));
  }
  throw std::runtime_error("cuDNN Frontend native " + std::string(operation) +
                           " " + std::string(stage) +
                           " failed: " + status.get_message());
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
    const auto input =
        cuda::make_cudnn_tensor(graph_, test_case.input, "input");

    std::shared_ptr<cfe::graph::Tensor_attributes> output;
    switch (test_case.operation) {
      case LayoutOperation::kReshape:
        output = graph_->reshape(
            input, cfe::graph::Reshape_attributes()
                       .set_name("reshape")
                       .set_compute_data_type(cfe::DataType_t::FLOAT)
                       .set_dim(test_case.output.dimensions)
                       .set_stride(test_case.output.strides)
                       .set_reshape_mode(cfe::ReshapeMode_t::LOGICAL));
        break;
      case LayoutOperation::kTranspose:
        output = graph_->transpose(input,
                                   cfe::graph::Transpose_attributes()
                                       .set_name("transpose")
                                       .set_permutation(test_case.permutation));
        break;
      case LayoutOperation::kSlice:
        output = graph_->slice(
            input, cfe::graph::Slice_attributes()
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
    require_plan_stage(graph_->create_execution_plans(
                           {cfe::HeurMode_t::A, cfe::HeurMode_t::FALLBACK}),
                       operation, "execution-plan discovery");
    require_plan_stage(graph_->check_support(handle()), operation,
                       "support check");
    require_plan_stage(graph_->build_plans(handle()), operation, "plan build");

    std::int64_t workspace_size = 0;
    cuda::check_cudnn_frontend(graph_->get_workspace_size(workspace_size),
                               "cuDNN Layout workspace query");
    set_workspace_size(workspace_size);
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void* workspace,
               std::size_t workspace_size, flagdnnStream_t stream) override {
    begin_execute(workspace, workspace_size, stream);
    cuda::CudnnBindingMap pointers = cuda::make_cudnn_binding_map(bindings);
    cuda::check_cudnn_frontend(graph_->execute(handle(), pointers, workspace),
                               "cuDNN Layout graph execute");
  }

 private:
  std::shared_ptr<cfe::graph::Graph> graph_;
};

// These legacy cuDNN entry points supply an executable tensor transform when
// the standalone Frontend reshape/transpose/slice graph has no engine.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
using cuda::TensorDescriptor;

class CudnnLayoutTransform final : public cuda::CudnnGraphExecutable {
 public:
  explicit CudnnLayoutTransform(const LayoutTestCase& test_case)
      : input_uid_(test_case.input.uid), output_uid_(test_case.output.uid) {
    auto input = test_case.input;
    auto output = test_case.output;
    if (test_case.operation == LayoutOperation::kReshape) {
      for (const auto* tensor : {&input, &output}) {
        std::int64_t stride = 1;
        for (std::size_t axis = tensor->dimensions.size(); axis > 0; --axis) {
          if (tensor->dimensions[axis - 1] != 1 &&
              tensor->strides[axis - 1] != stride)
            throw std::invalid_argument(
                "cuDNN reshape transform requires compact tensors");
          stride *= tensor->dimensions[axis - 1];
        }
      }
      input = cuda::flatten_compact_tensor(input);
      output = cuda::flatten_compact_tensor(output);
    } else if (test_case.operation == LayoutOperation::kTranspose) {
      input.dimensions = output.dimensions;
      for (std::size_t axis = 0; axis < input.strides.size(); ++axis)
        input.strides[axis] =
            test_case.input.strides.at(test_case.permutation.at(axis));
    } else {
      input.dimensions = output.dimensions;
      for (std::size_t axis = 0; axis < input.strides.size(); ++axis) {
        input_offset_ += test_case.slices.at(axis).first * input.strides[axis];
        input.strides[axis] *= test_case.slice_strides.at(axis);
      }
      input_offset_ *=
          test_case.input.data_type == FLAGDNN_DATA_FLOAT32 ? 4 : 2;
    }
    input_ = std::make_unique<TensorDescriptor>(input);
    output_ = std::make_unique<TensorDescriptor>(output);
  }
  void execute(std::span<const flagdnnBinding_t> bindings, void* workspace,
               std::size_t size, flagdnnStream_t stream) override {
    begin_execute(workspace, size, stream);
    const auto pointers = cuda::make_cudnn_binding_map(bindings);
    const float alpha = 1.0F, beta = 0.0F;
    cuda::check_cudnn(
        cudnnTransformTensor(
            handle(), &alpha, input_->get(),
            static_cast<const std::uint8_t*>(pointers.at(input_uid_)) +
                input_offset_,
            &beta, output_->get(), pointers.at(output_uid_)),
        "cudnnTransformTensor");
  }

 private:
  std::int64_t input_uid_, output_uid_;
  std::size_t input_offset_ = 0;
  std::unique_ptr<TensorDescriptor> input_, output_;
};
class CudnnConcatenateTransform final : public cuda::CudnnGraphExecutable {
 public:
  explicit CudnnConcatenateTransform(const IndexTestCase& test_case)
      : output_uid_(test_case.output.uid) {
    if (test_case.operation != "concatenate")
      throw std::invalid_argument(
          "cuDNN concatenation transform requires concatenate");
    const auto rank =
        static_cast<std::int64_t>(test_case.output.dimensions.size());
    const auto axis =
        test_case.axis < 0 ? test_case.axis + rank : test_case.axis;
    const auto width = cuda::data_type_size(test_case.output.data_type);
    std::size_t offset = 0;
    for (const auto& input : test_case.inputs) {
      auto output_view = test_case.output;
      output_view.dimensions = input.dimensions;
      inputs_.push_back(std::make_unique<TensorDescriptor>(input));
      outputs_.push_back(std::make_unique<TensorDescriptor>(output_view));
      uids_.push_back(input.uid);
      offsets_.push_back(offset);
      offset +=
          input.dimensions.at(axis) * output_view.strides.at(axis) * width;
    }
  }
  void execute(std::span<const flagdnnBinding_t> bindings, void* workspace,
               std::size_t size, flagdnnStream_t stream) override {
    begin_execute(workspace, size, stream);
    const auto pointers = cuda::make_cudnn_binding_map(bindings);
    const float alpha = 1.0F, beta = 0.0F;
    for (std::size_t index = 0; index < inputs_.size(); ++index)
      cuda::check_cudnn(
          cudnnTransformTensor(
              handle(), &alpha, inputs_[index]->get(),
              pointers.at(uids_[index]), &beta, outputs_[index]->get(),
              static_cast<std::uint8_t*>(pointers.at(output_uid_)) +
                  offsets_[index]),
          "cudnnTransformTensor(concatenate)");
  }

 private:
  std::int64_t output_uid_;
  std::vector<std::int64_t> uids_;
  std::vector<std::size_t> offsets_;
  std::vector<std::unique_ptr<TensorDescriptor>> inputs_, outputs_;
};
#pragma GCC diagnostic pop

}  // namespace

std::unique_ptr<LayoutExecutable> build_layout_reference(
    const LayoutTestCase& test_case) {
  validate_layout_case(test_case);
  try {
    return std::make_unique<CudnnLayoutExecutable>(test_case);
  } catch (const std::runtime_error& graph_error) {
    try {
      return std::make_unique<CudnnLayoutTransform>(test_case);
    } catch (const std::exception& transform_error) {
      throw std::runtime_error(std::string(graph_error.what()) +
                               "; cuDNN transform: " + transform_error.what());
    }
  }
}

std::unique_ptr<TestExecutable> build_cudnn_concatenate_transform(
    const IndexTestCase& test_case) {
  return std::make_unique<CudnnConcatenateTransform>(test_case);
}

}  // namespace flagdnn::testing
