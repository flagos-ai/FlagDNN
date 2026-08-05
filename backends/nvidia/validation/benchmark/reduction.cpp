/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "ops.hpp"

#include "validation/benchmark/cudnn_common.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flagdnn::benchmarking::cudnn_detail {
namespace {

bool has_non_overlapping_strides(const TensorSpec& tensor) {
  if (tensor.dimensions.size() != tensor.strides.size()) {
    return false;
  }
  std::vector<std::size_t> axes;
  axes.reserve(tensor.dimensions.size());
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] > 1) {
      axes.push_back(axis);
    }
  }
  std::sort(axes.begin(), axes.end(), [&](std::size_t left, std::size_t right) {
    return tensor.strides[left] < tensor.strides[right];
  });
  std::uint64_t required_span = 1;
  for (const std::size_t axis : axes) {
    const std::uint64_t stride =
        static_cast<std::uint64_t>(tensor.strides[axis]);
    if (stride < required_span) {
      return false;
    }
    const std::uint64_t extent =
        static_cast<std::uint64_t>(tensor.dimensions[axis] - 1);
    if (extent != 0 &&
        stride > (std::numeric_limits<std::uint64_t>::max() - required_span) /
                     extent) {
      return false;
    }
    required_span += extent * stride;
  }
  return true;
}

std::int64_t storage_element_count(const TensorSpec& tensor) {
  std::int64_t maximum_offset = 0;
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    maximum_offset +=
        (tensor.dimensions[axis] - 1) * tensor.strides[axis];
  }
  return maximum_offset + 1;
}

std::size_t checked_size(std::int64_t value, const char* message) {
  if (value < 0 ||
      static_cast<std::uint64_t>(value) >
          std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error(message);
  }
  return static_cast<std::size_t>(value);
}

std::size_t checked_add(std::size_t left,
                        std::size_t right,
                        const char* message) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::runtime_error(message);
  }
  return left + right;
}

std::size_t align_up(std::size_t value,
                     std::size_t alignment,
                     const char* message) {
  const std::size_t remainder = value % alignment;
  return remainder == 0
             ? value
             : checked_add(value, alignment - remainder, message);
}

cudnnReduceTensorOp_t cudnn_reduction_mode(flagdnnReductionMode_t mode) {
  switch (mode) {
    case FLAGDNN_REDUCTION_ADD:
      return CUDNN_REDUCE_TENSOR_ADD;
    case FLAGDNN_REDUCTION_AVG:
      return CUDNN_REDUCE_TENSOR_AVG;
    case FLAGDNN_REDUCTION_MUL:
      return CUDNN_REDUCE_TENSOR_MUL;
  }
  throw std::invalid_argument("cuDNN Reduction mode is invalid");
}

class ReductionExecutable final : public ExecutableBase {
 public:
  explicit ReductionExecutable(const BenchmarkCase& specification) {
    try {
      build(specification);
    } catch (...) {
      destroy();
      throw;
    }
  }

  ~ReductionExecutable() override { destroy(); }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    begin_execute(workspace, workspace_size, stream);
    const BindingMap pointers = make_binding_map(bindings);

    if (postprocess_graph_ != nullptr) {
      auto* workspace_bytes = static_cast<std::byte*>(workspace);
      void* intermediate = workspace_bytes + intermediate_offset_;
      BindingMap reduction_pointers;
      reduction_pointers.emplace(
          input_uid_, pointer_for(pointers, input_uid_, "Reduction"));
      reduction_pointers.emplace(intermediate_uid_, intermediate);
      check_frontend(frontend_graph_->execute(
                         handle(),
                         reduction_pointers,
                         workspace_bytes + frontend_workspace_offset_),
                     "cuDNN BF16 native reduction execute");

      BindingMap postprocess_pointers;
      postprocess_pointers.emplace(intermediate_uid_, intermediate);
      postprocess_pointers.emplace(
          output_uid_, pointer_for(pointers, output_uid_, "Reduction"));
      check_frontend(postprocess_graph_->execute(
                         handle(),
                         postprocess_pointers,
                         workspace_bytes + postprocess_workspace_offset_),
                     "cuDNN BF16 reduction cast execute");
      return;
    }

    if (frontend_graph_ != nullptr) {
      BindingMap graph_pointers;
      graph_pointers.reserve(
          leaf_bindings_.empty() ? 2 : leaf_bindings_.size() + 1);
      if (leaf_bindings_.empty()) {
        graph_pointers.emplace(
            input_uid_, pointer_for(pointers, input_uid_, "Reduction"));
      } else {
        auto* input_bytes = static_cast<std::byte*>(
            pointer_for(pointers, input_uid_, "Reduction"));
        for (const auto& [uid, offset] : leaf_bindings_) {
          graph_pointers.emplace(uid, input_bytes + offset);
        }
      }
      graph_pointers.emplace(
          output_uid_, pointer_for(pointers, output_uid_, "Reduction"));
      check_frontend(
          frontend_graph_->execute(handle(), graph_pointers, workspace),
          "cuDNN BF16 reduction composite execute");
      return;
    }

    const float alpha = 1.0F;
    const float beta = 0.0F;
    check_cudnn(
        cudnnReduceTensor(handle(),
                          reduction_descriptor_,
                          nullptr,
                          0,
                          workspace,
                          reduction_workspace_size_,
                          &alpha,
                          input_descriptor_,
                          pointer_for(pointers, input_uid_, "Reduction"),
                          &beta,
                          output_descriptor_,
                          pointer_for(pointers, output_uid_, "Reduction")),
        "cudnnReduceTensor");
  }

 private:
  void build(const BenchmarkCase& specification) {
    require_tensor_count(specification, 2);
    const TensorSpec& logical_input = specification.tensors[0];
    const TensorSpec& logical_output = specification.tensors[1];
    if (logical_input.dimensions.empty() ||
        logical_input.dimensions.size() > 8 ||
        !has_non_overlapping_strides(logical_input)) {
      throw std::invalid_argument(
          "cuDNN Reduction requires a non-overlapping rank 1-8 input");
    }
    if (logical_input.data_type != logical_output.data_type) {
      throw std::invalid_argument(
          "cuDNN Reduction input/output data types must match");
    }
    if (specification.reduction_mode != FLAGDNN_REDUCTION_ADD &&
        specification.reduction_mode != FLAGDNN_REDUCTION_AVG &&
        specification.reduction_mode != FLAGDNN_REDUCTION_MUL) {
      throw std::invalid_argument("cuDNN Reduction mode is invalid");
    }

    std::int32_t axis = specification.reduction_axis;
    const std::int32_t original_rank =
        static_cast<std::int32_t>(logical_input.dimensions.size());
    if (axis < 0) {
      axis += original_rank;
    }
    if (axis < 0 || axis >= original_rank) {
      throw std::invalid_argument("reduction axis is out of range");
    }

    std::vector<std::int64_t> expected = logical_input.dimensions;
    if (specification.keep_dimensions) {
      expected[static_cast<std::size_t>(axis)] = 1;
    } else {
      expected.erase(expected.begin() + axis);
    }
    if (logical_output.dimensions != expected ||
        !has_non_overlapping_strides(logical_output)) {
      throw std::invalid_argument(
          "cuDNN Reduction output shape or strides are invalid");
    }

    input_uid_ = logical_input.uid;
    output_uid_ = logical_output.uid;
    TensorSpec full_output = logical_output;
    if (!specification.keep_dimensions) {
      full_output.dimensions.insert(full_output.dimensions.begin() + axis, 1);
      full_output.strides.insert(full_output.strides.begin() + axis,
                                 storage_element_count(logical_output));
    }
    if (logical_input.data_type == FLAGDNN_DATA_BFLOAT16) {
      const std::uint64_t slice_stride_bytes =
          static_cast<std::uint64_t>(
              logical_input.strides[static_cast<std::size_t>(axis)]) *
          2U;
      const std::int64_t extent =
          logical_input.dimensions[static_cast<std::size_t>(axis)];
      if (extent > 1 && slice_stride_bytes % 16U != 0U) {
        if (specification.reduction_mode == FLAGDNN_REDUCTION_MUL) {
          throw std::invalid_argument(
              "cuDNN has no exact unaligned BF16 MUL reduction fallback");
        }
        build_bfloat16_native_composite(logical_input,
                                        full_output,
                                        axis,
                                        specification.reduction_mode);
        return;
      }
      build_bfloat16_composite(logical_input,
                               logical_output,
                               axis,
                               specification.keep_dimensions,
                               specification.reduction_mode);
      return;
    }

    const TensorSpec external_input =
        padded_to_minimum_rank_four(logical_input);
    const TensorSpec external_output =
        padded_to_minimum_rank_four(full_output);

    check_cudnn(cudnnCreateTensorDescriptor(&input_descriptor_),
                "cudnnCreateTensorDescriptor(reduction input)");
    check_cudnn(cudnnCreateTensorDescriptor(&output_descriptor_),
                "cudnnCreateTensorDescriptor(reduction output)");
    set_tensor_descriptor(input_descriptor_, external_input);
    set_tensor_descriptor(output_descriptor_, external_output);
    check_cudnn(cudnnCreateReduceTensorDescriptor(&reduction_descriptor_),
                "cudnnCreateReduceTensorDescriptor");
    check_cudnn(
        cudnnSetReduceTensorDescriptor(reduction_descriptor_,
                                       cudnn_reduction_mode(
                                           specification.reduction_mode),
                                       CUDNN_DATA_FLOAT,
                                       CUDNN_PROPAGATE_NAN,
                                       CUDNN_REDUCE_TENSOR_NO_INDICES,
                                       CUDNN_32BIT_INDICES),
        "cudnnSetReduceTensorDescriptor");
    check_cudnn(
        cudnnGetReductionWorkspaceSize(handle(),
                                       reduction_descriptor_,
                                       input_descriptor_,
                                       output_descriptor_,
                                       &reduction_workspace_size_),
        "cudnnGetReductionWorkspaceSize");
    set_workspace_size(reduction_workspace_size_);
  }

  void build_bfloat16_native_composite(
      const TensorSpec& logical_input,
      const TensorSpec& full_output,
      std::int32_t axis,
      flagdnnReductionMode_t mode) {
    frontend_graph_ = std::make_shared<fe::graph::Graph>();
    frontend_graph_->set_name("reduction_bfloat16_native_composite")
        .set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    const auto input =
        make_tensor(frontend_graph_, logical_input, "input", false);
    auto reduced = frontend_graph_->reduction(
        input,
        fe::graph::Reduction_attributes()
            .set_name("reduction_add_to_float")
            .set_mode(fe::ReductionMode_t::ADD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    reduced->set_name("reduced_float")
        .set_uid(intermediate_uid_)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim(full_output.dimensions)
        .set_stride(full_output.strides)
        .set_output(true);

    check_frontend(
        frontend_graph_->build(handle(), {fe::HeurMode_t::A}),
        "cuDNN BF16 native reduction build");
    std::int64_t reduction_workspace = 0;
    check_frontend(
        frontend_graph_->get_workspace_size(reduction_workspace),
        "cuDNN BF16 native reduction workspace query");

    TensorSpec intermediate = full_output;
    intermediate.dimensions.erase(intermediate.dimensions.begin() + axis);
    intermediate.strides.erase(intermediate.strides.begin() + axis);
    intermediate.uid = intermediate_uid_;
    intermediate.data_type = FLAGDNN_DATA_FLOAT32;
    postprocess_graph_ = std::make_shared<fe::graph::Graph>();
    postprocess_graph_->set_name("reduction_bfloat16_cast")
        .set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    const auto postprocess_input =
        make_tensor(postprocess_graph_, intermediate, "reduced_float", false);
    std::shared_ptr<fe::graph::Tensor_attributes> output;
    if (mode == FLAGDNN_REDUCTION_AVG) {
      fe::graph::Tensor_attributes scale_attributes(
          1.0F / static_cast<float>(
                     logical_input.dimensions[static_cast<std::size_t>(axis)]));
      scale_attributes.set_name("reduction_average_scale")
          .set_dim(std::vector<std::int64_t>(
              intermediate.dimensions.size(), 1))
          .set_stride(std::vector<std::int64_t>(
              intermediate.dimensions.size(), 1));
      const auto scale = postprocess_graph_->tensor(scale_attributes);
      output = postprocess_graph_->pointwise(
          postprocess_input,
          scale,
          fe::graph::Pointwise_attributes()
              .set_name("reduction_average")
              .set_mode(fe::PointwiseMode_t::MUL));
    } else {
      output = postprocess_graph_->pointwise(
          postprocess_input,
          fe::graph::Pointwise_attributes()
              .set_name("reduction_cast")
              .set_mode(fe::PointwiseMode_t::IDENTITY));
    }
    output->set_name("output")
        .set_uid(output_uid_)
        .set_data_type(fe::DataType_t::BFLOAT16)
        .set_dim(intermediate.dimensions)
        .set_stride(intermediate.strides)
        .set_output(true);

    check_frontend(
        postprocess_graph_->build(handle(), {fe::HeurMode_t::A}),
        "cuDNN BF16 reduction cast build");
    std::int64_t postprocess_workspace = 0;
    check_frontend(
        postprocess_graph_->get_workspace_size(postprocess_workspace),
        "cuDNN BF16 reduction cast workspace query");

    constexpr std::size_t kWorkspaceAlignment = 256;
    const std::size_t reduction_workspace_size = checked_size(
        reduction_workspace, "cuDNN BF16 reduction workspace is invalid");
    const std::size_t postprocess_workspace_size = checked_size(
        postprocess_workspace, "cuDNN BF16 cast workspace is invalid");
    frontend_workspace_offset_ = 0;
    postprocess_workspace_offset_ = align_up(reduction_workspace_size,
                                             kWorkspaceAlignment,
                                             "cuDNN BF16 workspace overflows");
    intermediate_offset_ = align_up(
        checked_add(postprocess_workspace_offset_,
                    postprocess_workspace_size,
                    "cuDNN BF16 workspace overflows"),
        kWorkspaceAlignment,
        "cuDNN BF16 workspace overflows");
    const std::size_t intermediate_elements = checked_size(
        storage_element_count(full_output),
        "cuDNN BF16 intermediate extent is invalid");
    if (intermediate_elements >
        std::numeric_limits<std::size_t>::max() / sizeof(float)) {
      throw std::runtime_error("cuDNN BF16 intermediate size overflows");
    }
    set_workspace_size(checked_add(intermediate_offset_,
                                   intermediate_elements * sizeof(float),
                                   "cuDNN BF16 workspace overflows"));
  }

  void build_bfloat16_composite(const TensorSpec& logical_input,
                                const TensorSpec& logical_output,
                                std::int32_t axis,
                                bool keep_dimensions,
                                flagdnnReductionMode_t mode) {
    const std::int64_t extent =
        logical_input.dimensions[static_cast<std::size_t>(axis)];
    if (extent <= 0 || extent > 65536) {
      throw std::invalid_argument(
          "cuDNN BF16 Reduction extent is out of range");
    }

    TensorSpec slice_spec = logical_input;
    slice_spec.dimensions.erase(slice_spec.dimensions.begin() + axis);
    slice_spec.strides.erase(slice_spec.strides.begin() + axis);
    if (slice_spec.dimensions.empty()) {
      throw std::invalid_argument(
          "cuDNN BF16 add-tree Reduction requires rank at least two");
    }
    // Match a native select(axis, index) view exactly. Padding this view to
    // rank four changes cuDNN's layout classification and rejects the graph.
    TensorSpec output_spec = logical_output;
    if (keep_dimensions) {
      output_spec.dimensions.erase(output_spec.dimensions.begin() + axis);
      output_spec.strides.erase(output_spec.strides.begin() + axis);
    }
    if (output_spec.dimensions != slice_spec.dimensions) {
      throw std::invalid_argument(
          "cuDNN BF16 Reduction output shape is invalid");
    }

    frontend_graph_ = std::make_shared<fe::graph::Graph>();
    frontend_graph_->set_name("reduction_bfloat16_pointwise_tree")
        .set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    std::int64_t next_uid = std::numeric_limits<std::int64_t>::max();
    const auto take_uid = [&] {
      while (next_uid == input_uid_ || next_uid == output_uid_) {
        --next_uid;
      }
      return next_uid--;
    };
    const std::uint64_t slice_stride = static_cast<std::uint64_t>(
        logical_input.strides[static_cast<std::size_t>(axis)]);
    std::vector<std::shared_ptr<fe::graph::Tensor_attributes>> values;
    values.reserve(static_cast<std::size_t>(extent));
    leaf_bindings_.reserve(static_cast<std::size_t>(extent));
    for (std::int64_t index = 0; index < extent; ++index) {
      const std::int64_t uid = take_uid();
      slice_spec.uid = uid;
      const std::string slice_name =
          "reduction_slice_" + std::to_string(index);
      values.push_back(
          make_tensor(frontend_graph_, slice_spec, slice_name, false));

      const std::uint64_t slice_index =
          static_cast<std::uint64_t>(index);
      if (slice_index != 0 &&
          slice_stride >
              std::numeric_limits<std::size_t>::max() /
                  (slice_index * 2U)) {
        throw std::invalid_argument(
            "cuDNN BF16 Reduction slice offset overflows");
      }
      const std::size_t byte_offset = static_cast<std::size_t>(
          slice_index * slice_stride * 2U);
      leaf_bindings_.emplace_back(uid, byte_offset);
    }

    int level = 0;
    while (values.size() > 1) {
      std::vector<std::shared_ptr<fe::graph::Tensor_attributes>> next;
      next.reserve((values.size() + 1) / 2);
      for (std::size_t index = 0; index < values.size(); index += 2) {
        if (index + 1 == values.size()) {
          next.push_back(values[index]);
          continue;
        }
        const fe::PointwiseMode_t pointwise_mode =
            mode == FLAGDNN_REDUCTION_MUL ? fe::PointwiseMode_t::MUL
                                          : fe::PointwiseMode_t::ADD;
        next.push_back(frontend_graph_->pointwise(
            values[index],
            values[index + 1],
            fe::graph::Pointwise_attributes()
                .set_name("reduction_combine_" + std::to_string(level) +
                          "_" + std::to_string(index / 2))
                .set_mode(pointwise_mode)));
      }
      values = std::move(next);
      ++level;
    }
    if (extent == 1) {
      values[0] = frontend_graph_->pointwise(
          values[0],
          fe::graph::Pointwise_attributes()
              .set_name("reduction_identity")
              .set_mode(fe::PointwiseMode_t::IDENTITY));
    }
    if (mode == FLAGDNN_REDUCTION_AVG) {
      fe::graph::Tensor_attributes scale_attributes(
          1.0F / static_cast<float>(extent));
      scale_attributes.set_name("reduction_average_scale")
          .set_dim(std::vector<std::int64_t>(
              output_spec.dimensions.size(), 1))
          .set_stride(std::vector<std::int64_t>(
              output_spec.dimensions.size(), 1));
      const auto scale = frontend_graph_->tensor(scale_attributes);
      values[0] = frontend_graph_->pointwise(
          values[0],
          scale,
          fe::graph::Pointwise_attributes()
              .set_name("reduction_average")
              .set_mode(fe::PointwiseMode_t::MUL));
    }
    values[0]->set_name("output")
        .set_uid(output_spec.uid)
        .set_data_type(fe::DataType_t::BFLOAT16)
        .set_dim(output_spec.dimensions)
        .set_stride(output_spec.strides)
        .set_output(true);
    check_frontend(
        frontend_graph_->build(handle(), {fe::HeurMode_t::A}),
        "cuDNN BF16 reduction add-tree build");

    std::int64_t graph_workspace = 0;
    check_frontend(
        frontend_graph_->get_workspace_size(graph_workspace),
        "cuDNN BF16 reduction add-tree workspace query");
    set_workspace_size(graph_workspace);
  }

  void destroy() noexcept {
    if (reduction_descriptor_ != nullptr) {
      (void)cudnnDestroyReduceTensorDescriptor(reduction_descriptor_);
      reduction_descriptor_ = nullptr;
    }
    if (output_descriptor_ != nullptr) {
      (void)cudnnDestroyTensorDescriptor(output_descriptor_);
      output_descriptor_ = nullptr;
    }
    if (input_descriptor_ != nullptr) {
      (void)cudnnDestroyTensorDescriptor(input_descriptor_);
      input_descriptor_ = nullptr;
    }
  }

  std::shared_ptr<fe::graph::Graph> frontend_graph_;
  std::shared_ptr<fe::graph::Graph> postprocess_graph_;
  cudnnTensorDescriptor_t input_descriptor_ = nullptr;
  cudnnTensorDescriptor_t output_descriptor_ = nullptr;
  cudnnReduceTensorDescriptor_t reduction_descriptor_ = nullptr;
  std::int64_t input_uid_ = 0;
  std::int64_t output_uid_ = 0;
  std::int64_t intermediate_uid_ =
      std::numeric_limits<std::int64_t>::max();
  std::vector<std::pair<std::int64_t, std::size_t>> leaf_bindings_;
  std::size_t reduction_workspace_size_ = 0;
  std::size_t frontend_workspace_offset_ = 0;
  std::size_t postprocess_workspace_offset_ = 0;
  std::size_t intermediate_offset_ = 0;
};

}  // namespace

std::unique_ptr<BenchmarkExecutable> build_reduction(
    const BenchmarkCase& specification) {
  return std::make_unique<ReductionExecutable>(specification);
}

}  // namespace flagdnn::benchmarking::cudnn_detail
