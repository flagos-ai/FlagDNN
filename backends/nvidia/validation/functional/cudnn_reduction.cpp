/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/reduction.hpp"
#include "validation/functional/cudnn_graph.hpp"
#include "validation/tensor_io.hpp"

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

TestTensor permute_to_nhwc(const TestTensor& tensor,
                           const AxisOrder& order) {
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
        input,
        cfe::graph::Reduction_attributes()
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

    cuda::check_cudnn_frontend(
        graph_->build(handle(), {cfe::HeurMode_t::A}),
        "cuDNN Reduction graph build");
    std::int64_t workspace_size = 0;
    cuda::check_cudnn_frontend(graph_->get_workspace_size(workspace_size),
                               "cuDNN Reduction workspace query");
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
                               "cuDNN Reduction graph execute");
  }

 private:
  std::shared_ptr<cfe::graph::Graph> graph_;
};

class CudnnReductionPointwiseTreeExecutable final
    : public cuda::CudnnGraphExecutable {
 public:
  explicit CudnnReductionPointwiseTreeExecutable(
      const ReductionTestCase& test_case)
      : graph_(std::make_shared<cfe::graph::Graph>()),
        input_uid_(test_case.input.uid),
        output_uid_(test_case.output.uid) {
    validate_reduction_case(test_case);
    const std::int32_t axis = normalized_axis(test_case);
    const std::int64_t extent =
        test_case.input.dimensions[static_cast<std::size_t>(axis)];
    if (test_case.input.dimensions.size() <= 1 || extent <= 0) {
      throw std::invalid_argument(
          "cuDNN pointwise-tree Reduction requires rank at least two");
    }

    const TestTensor reference_input =
        reduction_reference_input_tensor(test_case);
    TestTensor slice = reference_input;
    slice.dimensions.erase(slice.dimensions.begin() + axis);
    slice.strides.erase(slice.strides.begin() + axis);
    TestTensor output_specification = test_case.output;
    if (test_case.keep_dimensions) {
      output_specification.dimensions.erase(
          output_specification.dimensions.begin() + axis);
      output_specification.strides.erase(
          output_specification.strides.begin() + axis);
    }
    if (slice.dimensions != output_specification.dimensions) {
      throw std::invalid_argument(
          "cuDNN pointwise-tree Reduction output shape is invalid");
    }

    graph_->set_name(test_case.name + "::cudnn_pointwise_tree")
        .set_io_data_type(
            cuda::cudnn_frontend_data_type(test_case.input.data_type))
        .set_intermediate_data_type(cfe::DataType_t::FLOAT)
        .set_compute_data_type(cfe::DataType_t::FLOAT);

    std::int64_t next_uid = std::numeric_limits<std::int64_t>::max();
    const auto take_uid = [&] {
      while (next_uid == input_uid_ || next_uid == output_uid_) {
        --next_uid;
      }
      return next_uid--;
    };
    const std::size_t element_size =
        cuda::data_type_size(test_case.input.data_type);
    const std::size_t slice_stride = static_cast<std::size_t>(
        reference_input.strides[static_cast<std::size_t>(axis)]);
    std::vector<std::shared_ptr<cfe::graph::Tensor_attributes>> values;
    values.reserve(static_cast<std::size_t>(extent));
    leaf_bindings_.reserve(static_cast<std::size_t>(extent));
    for (std::int64_t index = 0; index < extent; ++index) {
      slice.uid = take_uid();
      const TestTensor leaf = slice;
      values.push_back(cuda::make_cudnn_tensor(
          graph_, leaf, "slice_" + std::to_string(index)));
      leaf_bindings_.emplace_back(
          leaf.uid,
          static_cast<std::size_t>(index) * slice_stride * element_size);
    }

    int level = 0;
    while (values.size() > 1) {
      std::vector<std::shared_ptr<cfe::graph::Tensor_attributes>> next;
      next.reserve((values.size() + 1) / 2);
      for (std::size_t index = 0; index < values.size(); index += 2) {
        if (index + 1 == values.size()) {
          next.push_back(values[index]);
          continue;
        }
        const cfe::PointwiseMode_t pointwise_mode =
            test_case.mode == FLAGDNN_REDUCTION_MUL
                ? cfe::PointwiseMode_t::MUL
                : cfe::PointwiseMode_t::ADD;
        next.push_back(graph_->pointwise(
            values[index],
            values[index + 1],
            cfe::graph::Pointwise_attributes()
                .set_name("combine_" + std::to_string(level) + "_" +
                          std::to_string(index / 2))
                .set_mode(pointwise_mode)
                .set_compute_data_type(cfe::DataType_t::FLOAT)));
      }
      values = std::move(next);
      ++level;
    }
    if (extent == 1) {
      values[0] = graph_->pointwise(
          values[0],
          cfe::graph::Pointwise_attributes()
              .set_name("identity")
              .set_mode(cfe::PointwiseMode_t::IDENTITY)
              .set_compute_data_type(cfe::DataType_t::FLOAT));
    }
    if (test_case.mode == FLAGDNN_REDUCTION_AVG) {
      cfe::graph::Tensor_attributes scale_attributes(
          1.0F / static_cast<float>(extent));
      scale_attributes.set_name("average_scale")
          .set_dim(std::vector<std::int64_t>(
              output_specification.dimensions.size(), 1))
          .set_stride(std::vector<std::int64_t>(
              output_specification.dimensions.size(), 1));
      const auto scale = graph_->tensor(scale_attributes);
      values[0] = graph_->pointwise(
          values[0],
          scale,
          cfe::graph::Pointwise_attributes()
              .set_name("average")
              .set_mode(cfe::PointwiseMode_t::MUL)
              .set_compute_data_type(cfe::DataType_t::FLOAT));
    }
    values[0]->set_name("output")
        .set_uid(output_uid_)
        .set_data_type(
            cuda::cudnn_frontend_data_type(test_case.output.data_type))
        .set_dim(output_specification.dimensions)
        .set_stride(output_specification.strides)
        .set_output(true);

    cuda::check_cudnn_frontend(
        graph_->build(handle(), {cfe::HeurMode_t::A}),
        "cuDNN pointwise-tree Reduction graph build");
    std::int64_t workspace_size = 0;
    cuda::check_cudnn_frontend(graph_->get_workspace_size(workspace_size),
                               "cuDNN pointwise-tree workspace query");
    set_workspace_size(workspace_size);
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    begin_execute(workspace, workspace_size, stream);
    cuda::CudnnBindingMap caller = cuda::make_cudnn_binding_map(bindings);
    const auto input = caller.find(input_uid_);
    const auto output = caller.find(output_uid_);
    if (input == caller.end() || output == caller.end()) {
      throw std::invalid_argument(
          "cuDNN pointwise-tree Reduction bindings are incomplete");
    }
    auto* input_bytes = static_cast<std::byte*>(input->second);
    cuda::CudnnBindingMap pointers;
    pointers.reserve(leaf_bindings_.size() + 1);
    for (const auto& [uid, offset] : leaf_bindings_) {
      pointers.emplace(uid, input_bytes + offset);
    }
    pointers.emplace(output_uid_, output->second);
    cuda::check_cudnn_frontend(graph_->execute(handle(), pointers, workspace),
                               "cuDNN pointwise-tree Reduction execute");
  }

 private:
  std::shared_ptr<cfe::graph::Graph> graph_;
  std::vector<std::pair<std::int64_t, std::size_t>> leaf_bindings_;
  std::int64_t input_uid_ = 0;
  std::int64_t output_uid_ = 0;
};

}  // namespace

TestTensor reduction_reference_input_tensor(
    const ReductionTestCase& test_case) {
  validate_reduction_case(test_case);
  TestTensor result = test_case.input;
  result.binding_byte_offset = 0;
  std::int64_t stride = 1;
  for (std::size_t axis = result.dimensions.size(); axis != 0; --axis) {
    result.strides[axis - 1] = stride;
    stride *= result.dimensions[axis - 1];
  }
  return result;
}

std::unique_ptr<ReductionExecutable> build_reduction_reference(
    const ReductionTestCase& test_case) {
  validate_reduction_case(test_case);
  const std::int32_t axis = normalized_axis(test_case);
  const TestTensor reference_input =
      reduction_reference_input_tensor(test_case);
  if (test_case.input.dimensions.size() == 1 ||
      (test_case.input.data_type == FLAGDNN_DATA_FLOAT32 &&
       reference_input.strides[static_cast<std::size_t>(axis)] == 1)) {
    return std::make_unique<CudnnReductionExecutable>(test_case);
  }
  return std::make_unique<CudnnReductionPointwiseTreeExecutable>(test_case);
}

}  // namespace flagdnn::testing
