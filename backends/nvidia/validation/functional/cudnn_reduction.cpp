/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/reduction.hpp"
#include "validation/functional/cudnn_graph.hpp"
#include "validation/functional/cudnn_tensor.hpp"
#include "validation/tensor_io.hpp"

namespace flagdnn::testing {
namespace {

namespace cfe = cuda::cfe;

cfe::ReductionMode_t cudnn_reduction_mode(flagdnnReductionMode_t mode) {
  switch (mode) {
    case FLAGDNN_REDUCTION_ADD:
      return cfe::ReductionMode_t::ADD;
    case FLAGDNN_REDUCTION_AVG:
      return cfe::ReductionMode_t::AVG;
    case FLAGDNN_REDUCTION_MUL:
      return cfe::ReductionMode_t::MUL;
  }
  throw std::invalid_argument("unsupported cuDNN Reduction mode");
}

std::int32_t normalized_axis(const ReductionTestCase& test_case) {
  std::int32_t axis = test_case.axis;
  const std::int32_t rank =
      static_cast<std::int32_t>(test_case.input.dimensions.size());
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= rank) {
    throw std::invalid_argument("cuDNN Reduction axis is out of range");
  }
  return axis;
}

TestTensor full_rank_output(const ReductionTestCase& test_case) {
  TestTensor result = test_case.output;
  if (!test_case.keep_dimensions) {
    const std::int32_t axis = normalized_axis(test_case);
    result.dimensions.insert(result.dimensions.begin() + axis, 1);
    result.strides.insert(
        result.strides.begin() + axis,
        test_case.input.strides[static_cast<std::size_t>(axis)]);
  }
  return result;
}

using AxisOrder = std::array<int, 4>;

AxisOrder nhwc_axis_order(const TestTensor& input) {
  if (input.dimensions.empty() || input.dimensions.size() > 4) {
    throw std::invalid_argument(
        "cuDNN Graph Reduction adapter currently supports rank 1-4");
  }
  const auto channel = static_cast<std::size_t>(std::distance(
      input.strides.begin(),
      std::min_element(input.strides.begin(), input.strides.end())));
  AxisOrder result = {-1, static_cast<int>(channel), -1, -1};
  constexpr std::array<std::size_t, 3> kRemainingSlots = {0, 2, 3};
  std::size_t next_slot = 0;
  for (std::size_t axis = 0; axis < input.dimensions.size(); ++axis) {
    if (axis != channel) {
      result[kRemainingSlots[next_slot++]] = static_cast<int>(axis);
    }
  }
  return result;
}

TestTensor permute_to_nhwc(const TestTensor& tensor, const AxisOrder& order) {
  const std::int64_t storage_span =
      static_cast<std::int64_t>(cuda::storage_element_count(tensor));
  TestTensor result{tensor.uid,
                    tensor.data_type,
                    {1, 1, 1, 1},
                    {storage_span, storage_span, storage_span, storage_span},
                    tensor.binding_byte_offset};
  for (std::size_t slot = 0; slot < order.size(); ++slot) {
    if (order[slot] >= 0) {
      const std::size_t axis = static_cast<std::size_t>(order[slot]);
      result.dimensions[slot] = tensor.dimensions[axis];
      result.strides[slot] = tensor.strides[axis];
    }
  }
  const std::array<std::int64_t, 4> compact_nhwc_strides = {
      result.dimensions[1] * result.dimensions[2] * result.dimensions[3],
      1,
      result.dimensions[1] * result.dimensions[3],
      result.dimensions[1],
  };
  for (std::size_t slot = 0; slot < order.size(); ++slot) {
    if (order[slot] < 0 || result.dimensions[slot] == 1) {
      result.strides[slot] = compact_nhwc_strides[slot];
    }
  }
  return result;
}

class CudnnReductionExecutable final : public cuda::CudnnGraphExecutable {
 public:
  explicit CudnnReductionExecutable(const ReductionTestCase& test_case)
      : graph_(std::make_shared<cfe::graph::Graph>()) {
    validate_reduction_case(test_case);
    // The cuDNN Graph reduction engine writes packed outputs on this stack.
    // Use cudnnReduceTensor when the public output descriptor has holes.
    if (cuda::storage_element_count(test_case.output) !=
        cuda::element_count(test_case.output))
      throw std::runtime_error("cuDNN Graph reduction requires a dense output");
    const TestTensor reference_input =
        reduction_reference_input_tensor(test_case);
    const AxisOrder order = nhwc_axis_order(reference_input);
    const TestTensor input_specification =
        permute_to_nhwc(reference_input, order);
    const TestTensor output_specification =
        permute_to_nhwc(full_rank_output(test_case), order);

    graph_->set_name(test_case.name + "::cudnn")
        .set_io_data_type(
            cuda::cudnn_frontend_data_type(test_case.input.data_type))
        .set_intermediate_data_type(cfe::DataType_t::FLOAT)
        .set_compute_data_type(cfe::DataType_t::FLOAT);
    const auto input =
        cuda::make_cudnn_tensor(graph_, input_specification, "input");
    auto output = graph_->reduction(
        input, cfe::graph::Reduction_attributes()
                   .set_name("reduction")
                   .set_mode(cudnn_reduction_mode(test_case.mode))
                   .set_compute_data_type(cfe::DataType_t::FLOAT));
    output->set_name("output")
        .set_uid(output_specification.uid)
        .set_data_type(
            cuda::cudnn_frontend_data_type(output_specification.data_type))
        .set_dim(output_specification.dimensions)
        .set_stride(output_specification.strides)
        .set_output(true);

    cuda::check_cudnn_frontend(graph_->build(handle(), {cfe::HeurMode_t::A}),
                               "cuDNN Reduction graph build");
    std::int64_t workspace_size = 0;
    cuda::check_cudnn_frontend(graph_->get_workspace_size(workspace_size),
                               "cuDNN Reduction workspace query");
    set_workspace_size(workspace_size);
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void* workspace,
               std::size_t workspace_size, flagdnnStream_t stream) override {
    begin_execute(workspace, workspace_size, stream);
    cuda::CudnnBindingMap pointers = cuda::make_cudnn_binding_map(bindings);
    cuda::check_cudnn_frontend(graph_->execute(handle(), pointers, workspace),
                               "cuDNN Reduction graph execute");
  }

 private:
  std::shared_ptr<cfe::graph::Graph> graph_;
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
class CudnnLegacyReduction final : public cuda::CudnnGraphExecutable {
 public:
  explicit CudnnLegacyReduction(const ReductionTestCase& test_case)
      : input_(reduction_reference_input_tensor(test_case)),
        output_(full_rank_output(test_case)),
        input_uid_(test_case.input.uid),
        output_uid_(test_case.output.uid) {
    cuda::check_cudnn(cudnnCreateReduceTensorDescriptor(&reduction_),
                      "cudnnCreateReduceTensorDescriptor");
    try {
      const auto mode =
          test_case.mode == FLAGDNN_REDUCTION_MUL   ? CUDNN_REDUCE_TENSOR_MUL
          : test_case.mode == FLAGDNN_REDUCTION_AVG ? CUDNN_REDUCE_TENSOR_AVG
                                                    : CUDNN_REDUCE_TENSOR_ADD;
      cuda::check_cudnn(
          cudnnSetReduceTensorDescriptor(
              reduction_, mode, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN,
              CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES),
          "cudnnSetReduceTensorDescriptor");
      std::size_t size = 0;
      cuda::check_cudnn(
          cudnnGetReductionWorkspaceSize(handle(), reduction_, input_.get(),
                                         output_.get(), &size),
          "cudnnGetReductionWorkspaceSize");
      set_workspace_size(static_cast<std::int64_t>(size));
    } catch (...) {
      (void)cudnnDestroyReduceTensorDescriptor(reduction_);
      throw;
    }
  }
  ~CudnnLegacyReduction() override {
    (void)cudnnDestroyReduceTensorDescriptor(reduction_);
  }
  void execute(std::span<const flagdnnBinding_t> bindings, void* workspace,
               std::size_t size, flagdnnStream_t stream) override {
    begin_execute(workspace, size, stream);
    const auto pointers = cuda::make_cudnn_binding_map(bindings);
    const float alpha = 1.0F, beta = 0.0F;
    cuda::check_cudnn(
        cudnnReduceTensor(handle(), reduction_, nullptr, 0, workspace,
                          workspace_size(), &alpha, input_.get(),
                          pointers.at(input_uid_), &beta, output_.get(),
                          pointers.at(output_uid_)),
        "cudnnReduceTensor");
  }

 private:
  cuda::TensorDescriptor input_, output_;
  std::int64_t input_uid_, output_uid_;
  cudnnReduceTensorDescriptor_t reduction_ = nullptr;
};
#pragma GCC diagnostic pop

}  // namespace

TestTensor reduction_reference_input_tensor(
    const ReductionTestCase& test_case) {
  validate_reduction_case(test_case);
  TestTensor result = test_case.input;
  result.binding_byte_offset = 0;
  return result;
}

std::unique_ptr<ReductionExecutable> build_reduction_reference(
    const ReductionTestCase& test_case) {
  validate_reduction_case(test_case);
  try {
    return std::make_unique<CudnnReductionExecutable>(test_case);
  } catch (const std::runtime_error&) {
    return std::make_unique<CudnnLegacyReduction>(test_case);
  }
}

}  // namespace flagdnn::testing
