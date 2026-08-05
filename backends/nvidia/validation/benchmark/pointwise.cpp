/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "ops.hpp"

#include "validation/benchmark/cudnn_common.hpp"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <string>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace flagdnn::benchmarking::cudnn_detail {
namespace {

void check_cuda_runtime(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + cudaGetErrorString(status));
  }
}

class DeviceScalar {
 public:
  DeviceScalar(flagdnnDataType_t data_type, double value) {
    switch (data_type) {
      case FLAGDNN_DATA_FLOAT32:
        allocate_and_copy(static_cast<float>(value));
        return;
      case FLAGDNN_DATA_FLOAT16:
        allocate_and_copy(__float2half(static_cast<float>(value)));
        return;
      case FLAGDNN_DATA_BFLOAT16:
        allocate_and_copy(__float2bfloat16(static_cast<float>(value)));
        return;
      case FLAGDNN_DATA_FP8_E4M3:
      case FLAGDNN_DATA_FP8_E5M2:
        break;
      case FLAGDNN_DATA_BOOLEAN:
        break;
    }
    throw std::invalid_argument(
        "cuDNN Add alpha scalar requires a floating data type");
  }

  ~DeviceScalar() {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
    }
  }

  DeviceScalar(const DeviceScalar&) = delete;
  DeviceScalar& operator=(const DeviceScalar&) = delete;

  [[nodiscard]] void* get() const noexcept { return pointer_; }

 private:
  template <typename Value>
  void allocate_and_copy(const Value& value) {
    check_cuda_runtime(cudaMalloc(&pointer_, sizeof(Value)), "cudaMalloc");
    try {
      check_cuda_runtime(
          cudaMemcpy(
              pointer_, &value, sizeof(Value), cudaMemcpyHostToDevice),
          "cudaMemcpy(Add alpha)");
    } catch (...) {
      (void)cudaFree(pointer_);
      pointer_ = nullptr;
      throw;
    }
  }

  void* pointer_ = nullptr;
};

std::int64_t internal_scalar_uid(const BenchmarkCase& specification) {
  std::int64_t candidate = std::numeric_limits<std::int64_t>::max();
  for (;;) {
    bool used = false;
    for (const TensorSpec& tensor : specification.tensors) {
      used = used || tensor.uid == candidate;
    }
    if (!used) {
      return candidate;
    }
    --candidate;
  }
}

std::int64_t element_count(const TensorSpec& specification) {
  std::int64_t result = 1;
  for (const std::int64_t dimension : specification.dimensions) {
    result *= dimension;
  }
  return result;
}

std::int64_t storage_element_count(const TensorSpec& specification) {
  std::int64_t result = 1;
  for (std::size_t axis = 0; axis < specification.dimensions.size(); ++axis) {
    result += (specification.dimensions[axis] - 1) *
              specification.strides[axis];
  }
  return result;
}

TensorSpec canonicalize_binary_pointwise_tensor(
    const TensorSpec& specification,
    std::size_t logical_rank) {
  if (logical_rank == 0 || logical_rank > 4 ||
      specification.dimensions.size() > logical_rank ||
      specification.dimensions.size() != specification.strides.size()) {
    throw std::invalid_argument(
        "cuDNN binary pointwise tensor rank is invalid");
  }
  if (logical_rank == 4) {
    return padded_to_rank_four(specification, logical_rank);
  }

  TensorSpec aligned = specification;
  const std::size_t leading =
      logical_rank - specification.dimensions.size();
  const std::int64_t storage_span =
      storage_element_count(specification);
  aligned.dimensions.insert(aligned.dimensions.begin(), leading, 1);
  aligned.strides.insert(aligned.strides.begin(), leading, storage_span);

  TensorSpec result = aligned;
  if (logical_rank == 1) {
    result.dimensions = {1, aligned.dimensions[0], 1, 1};
    result.strides = {storage_span,
                      aligned.strides[0],
                      storage_span,
                      storage_span};
  } else if (logical_rank == 2) {
    result.dimensions = {
        aligned.dimensions[0], aligned.dimensions[1], 1, 1};
    result.strides = {aligned.strides[0],
                      aligned.strides[1],
                      aligned.strides[0],
                      aligned.strides[0]};
  } else {
    result.dimensions = {aligned.dimensions[0],
                         aligned.dimensions[2],
                         aligned.dimensions[1],
                         1};
    result.strides = {aligned.strides[0],
                      aligned.strides[2],
                      aligned.strides[1],
                      aligned.strides[1]};
  }
  return result;
}

bool has_same_physical_mapping(const TensorSpec& input,
                               const TensorSpec& output) {
  if (input.dimensions != output.dimensions) {
    return false;
  }
  for (std::size_t axis = 0; axis < input.dimensions.size(); ++axis) {
    if (input.dimensions[axis] > 1 &&
        input.strides[axis] != output.strides[axis]) {
      return false;
    }
  }
  return true;
}

fe::PointwiseMode_t pointwise_mode(flagdnnPointwiseMode_t mode) {
  switch (mode) {
    case FLAGDNN_POINTWISE_SQRT:
      return fe::PointwiseMode_t::SQRT;
    case FLAGDNN_POINTWISE_ERF:
      return fe::PointwiseMode_t::ERF;
    case FLAGDNN_POINTWISE_IDENTITY:
      return fe::PointwiseMode_t::IDENTITY;
    case FLAGDNN_POINTWISE_EXP:
      return fe::PointwiseMode_t::EXP;
    case FLAGDNN_POINTWISE_LOG:
      return fe::PointwiseMode_t::LOG;
    case FLAGDNN_POINTWISE_NEG:
      return fe::PointwiseMode_t::NEG;
    case FLAGDNN_POINTWISE_ABS:
      return fe::PointwiseMode_t::ABS;
    case FLAGDNN_POINTWISE_CEIL:
      return fe::PointwiseMode_t::CEIL;
    case FLAGDNN_POINTWISE_COS:
      return fe::PointwiseMode_t::COS;
    case FLAGDNN_POINTWISE_FLOOR:
      return fe::PointwiseMode_t::FLOOR;
    case FLAGDNN_POINTWISE_RSQRT:
      return fe::PointwiseMode_t::RSQRT;
    case FLAGDNN_POINTWISE_SIN:
      return fe::PointwiseMode_t::SIN;
    case FLAGDNN_POINTWISE_TAN:
      return fe::PointwiseMode_t::TAN;
    case FLAGDNN_POINTWISE_RECIPROCAL:
      return fe::PointwiseMode_t::RECIPROCAL;
    case FLAGDNN_POINTWISE_RELU_FWD:
      return fe::PointwiseMode_t::RELU_FWD;
    case FLAGDNN_POINTWISE_ADD:
      return fe::PointwiseMode_t::ADD;
    case FLAGDNN_POINTWISE_SUB:
      return fe::PointwiseMode_t::SUB;
    case FLAGDNN_POINTWISE_MUL:
      return fe::PointwiseMode_t::MUL;
    case FLAGDNN_POINTWISE_DIV:
      return fe::PointwiseMode_t::DIV;
    case FLAGDNN_POINTWISE_MIN:
      return fe::PointwiseMode_t::MIN;
    case FLAGDNN_POINTWISE_MAX:
      return fe::PointwiseMode_t::MAX;
    case FLAGDNN_POINTWISE_MOD:
      return fe::PointwiseMode_t::MOD;
    case FLAGDNN_POINTWISE_POW:
      return fe::PointwiseMode_t::POW;
    case FLAGDNN_POINTWISE_LOGICAL_NOT:
      return fe::PointwiseMode_t::LOGICAL_NOT;
    case FLAGDNN_POINTWISE_CMP_EQ:
      return fe::PointwiseMode_t::CMP_EQ;
    case FLAGDNN_POINTWISE_CMP_NEQ:
      return fe::PointwiseMode_t::CMP_NEQ;
    case FLAGDNN_POINTWISE_CMP_GT:
      return fe::PointwiseMode_t::CMP_GT;
    case FLAGDNN_POINTWISE_CMP_GE:
      return fe::PointwiseMode_t::CMP_GE;
    case FLAGDNN_POINTWISE_CMP_LT:
      return fe::PointwiseMode_t::CMP_LT;
    case FLAGDNN_POINTWISE_CMP_LE:
      return fe::PointwiseMode_t::CMP_LE;
    case FLAGDNN_POINTWISE_LOGICAL_AND:
      return fe::PointwiseMode_t::LOGICAL_AND;
    case FLAGDNN_POINTWISE_LOGICAL_OR:
      return fe::PointwiseMode_t::LOGICAL_OR;
    case FLAGDNN_POINTWISE_SIGMOID_BWD:
      return fe::PointwiseMode_t::SIGMOID_BWD;
    case FLAGDNN_POINTWISE_BINARY_SELECT:
      return fe::PointwiseMode_t::BINARY_SELECT;
    case FLAGDNN_POINTWISE_SIGMOID_FWD:
      return fe::PointwiseMode_t::SIGMOID_FWD;
    case FLAGDNN_POINTWISE_TANH_FWD:
      return fe::PointwiseMode_t::TANH_FWD;
    case FLAGDNN_POINTWISE_ELU_FWD:
      return fe::PointwiseMode_t::ELU_FWD;
    case FLAGDNN_POINTWISE_GELU_FWD:
      return fe::PointwiseMode_t::GELU_FWD;
    case FLAGDNN_POINTWISE_SOFTPLUS_FWD:
      return fe::PointwiseMode_t::SOFTPLUS_FWD;
    case FLAGDNN_POINTWISE_SWISH_FWD:
      return fe::PointwiseMode_t::SWISH_FWD;
    case FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD:
      return fe::PointwiseMode_t::GELU_APPROX_TANH_FWD;
    case FLAGDNN_POINTWISE_NOT_SET:
      break;
  }
  throw std::invalid_argument("unsupported cuDNN pointwise mode");
}

void apply_pointwise_attributes(
    fe::graph::Pointwise_attributes& output,
    const flagdnnPointwiseAttributes_t& input) {
  if ((input.flags & FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP) != 0U) {
    output.set_relu_lower_clip(
        static_cast<float>(input.relu_lower_clip));
  }
  if ((input.flags & FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP) != 0U) {
    output.set_relu_upper_clip(
        static_cast<float>(input.relu_upper_clip));
  }
  if ((input.flags &
       FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE) != 0U) {
    output.set_relu_lower_clip_slope(
        static_cast<float>(input.relu_lower_clip_slope));
  }
  if ((input.flags & FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA) != 0U) {
    output.set_swish_beta(static_cast<float>(input.swish_beta));
  }
  if ((input.flags & FLAGDNN_POINTWISE_ATTRIBUTE_ELU_ALPHA) != 0U) {
    output.set_elu_alpha(static_cast<float>(input.elu_alpha));
  }
  if ((input.flags & FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA) != 0U) {
    output.set_softplus_beta(static_cast<float>(input.softplus_beta));
  }
}

bool uses_boolean_compute(flagdnnPointwiseMode_t mode) {
  // cuDNN 9.24 standalone floating-input comparison engines require FLOAT
  // math precision even though their output tensor is packed BOOLEAN. Fused
  // logical graphs use BOOLEAN compute precision as documented by Frontend.
  return mode == FLAGDNN_POINTWISE_LOGICAL_NOT ||
         mode == FLAGDNN_POINTWISE_LOGICAL_AND ||
         mode == FLAGDNN_POINTWISE_LOGICAL_OR;
}

class PointwiseExecutable final : public ExecutableBase {
 public:
  explicit PointwiseExecutable(const BenchmarkCase& specification) {
    if (specification.tensors.size() != 2 &&
        specification.tensors.size() != 3 &&
        specification.tensors.size() != 4) {
      throw std::invalid_argument(
          "cuDNN pointwise case must have one, two, or three inputs");
    }

    graph_ = std::make_shared<fe::graph::Graph>();
    graph_->set_name(specification.name)
        .set_io_data_type(data_type(specification.tensors[0].data_type))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    fe::graph::Pointwise_attributes attributes;
    attributes.set_name(specification.name)
        .set_mode(pointwise_mode(specification.pointwise_mode))
        .set_compute_data_type(
            uses_boolean_compute(specification.pointwise_mode)
                ? fe::DataType_t::BOOLEAN
                : fe::DataType_t::FLOAT);
    apply_pointwise_attributes(
        attributes, specification.pointwise_attributes);

    TensorSpec output_spec;
    std::shared_ptr<fe::graph::Tensor_attributes> output;
    if (specification.tensors.size() == 2) {
      const std::size_t logical_rank =
          specification.tensors[1].dimensions.size();
      const bool compact_pair =
          storage_element_count(specification.tensors[0]) ==
              element_count(specification.tensors[0]) &&
          storage_element_count(specification.tensors[1]) ==
              element_count(specification.tensors[1]) &&
          has_same_physical_mapping(specification.tensors[0],
                                    specification.tensors[1]);
      const TensorSpec input_spec =
          compact_pair
              ? compact_unary_tensor(specification.tensors[0], logical_rank)
              : padded_to_rank_four(specification.tensors[0], logical_rank);
      output_spec =
          compact_pair
              ? compact_unary_tensor(specification.tensors[1], logical_rank)
              : padded_to_rank_four(specification.tensors[1], logical_rank);
      const auto input = make_tensor(graph_, input_spec, "x", false);
      output = graph_->pointwise(input, attributes);
    } else if (specification.tensors.size() == 3) {
      const std::size_t logical_rank =
          specification.tensors[2].dimensions.size();
      const TensorSpec left_spec =
          canonicalize_binary_pointwise_tensor(
              specification.tensors[0], logical_rank);
      const TensorSpec right_spec =
          canonicalize_binary_pointwise_tensor(
              specification.tensors[1], logical_rank);
      output_spec =
          canonicalize_binary_pointwise_tensor(
              specification.tensors[2], logical_rank);
      const auto left = make_tensor(graph_, left_spec, "left", false);
      const auto right = make_tensor(graph_, right_spec, "right", false);
      const bool scale_right =
          specification.pointwise_mode == FLAGDNN_POINTWISE_ADD &&
          specification.add_alpha != 1.0;
      if (scale_right) {
        alpha_uid_ = internal_scalar_uid(specification);
        alpha_scalar_ = std::make_unique<DeviceScalar>(
            right_spec.data_type, specification.add_alpha);
        TensorSpec alpha_spec;
        alpha_spec.uid = alpha_uid_;
        alpha_spec.data_type = right_spec.data_type;
        alpha_spec.dimensions = {1, 1, 1, 1};
        alpha_spec.strides = {1, 1, 1, 1};
        const auto alpha =
            make_tensor(graph_, alpha_spec, "alpha", false);
        auto scaled_right = graph_->pointwise(
            right,
            alpha,
            fe::graph::Pointwise_attributes()
                .set_name(specification.name + "::scale_right")
                .set_mode(fe::PointwiseMode_t::MUL)
                .set_compute_data_type(fe::DataType_t::FLOAT));
        scaled_right->set_name("scaled_right")
            .set_data_type(data_type(right_spec.data_type))
            .set_dim(right_spec.dimensions)
            .set_stride(right_spec.strides);
        output = graph_->pointwise(left, scaled_right, attributes);
      } else {
        output = graph_->pointwise(left, right, attributes);
      }
    } else {
      const std::size_t logical_rank =
          specification.tensors[3].dimensions.size();
      const TensorSpec a_spec = canonicalize_binary_pointwise_tensor(
          specification.tensors[0], logical_rank);
      const TensorSpec b_spec = canonicalize_binary_pointwise_tensor(
          specification.tensors[1], logical_rank);
      const TensorSpec t_spec = canonicalize_binary_pointwise_tensor(
          specification.tensors[2], logical_rank);
      output_spec = canonicalize_binary_pointwise_tensor(
          specification.tensors[3], logical_rank);
      const auto a = make_tensor(graph_, a_spec, "a", false);
      const auto b = make_tensor(graph_, b_spec, "b", false);
      const auto t = make_tensor(graph_, t_spec, "t", false);
      output = graph_->pointwise(a, b, t, attributes);
    }
    output->set_name("y")
        .set_uid(output_spec.uid)
        .set_data_type(data_type(output_spec.data_type))
        .set_dim(output_spec.dimensions)
        .set_stride(output_spec.strides)
        .set_output(true);

    check_frontend(
        graph_->build(handle(), {fe::HeurMode_t::A}),
        "cuDNN pointwise graph build");
    std::int64_t workspace = 0;
    check_frontend(
        graph_->get_workspace_size(workspace),
        "cuDNN pointwise workspace query");
    set_workspace_size(workspace);
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    begin_execute(workspace, workspace_size, stream);
    BindingMap pointers = make_binding_map(bindings);
    if (alpha_scalar_ != nullptr &&
        !pointers.emplace(alpha_uid_, alpha_scalar_->get()).second) {
      throw std::invalid_argument(
          "cuDNN Add alpha UID collides with a caller binding");
    }
    check_frontend(graph_->execute(handle(), pointers, workspace),
                   "cuDNN pointwise graph execute");
  }

 private:
  std::unique_ptr<DeviceScalar> alpha_scalar_;
  std::int64_t alpha_uid_ = 0;
  std::shared_ptr<fe::graph::Graph> graph_;
};

}  // namespace

std::unique_ptr<BenchmarkExecutable> build_pointwise(
    const BenchmarkCase& specification) {
  return std::make_unique<PointwiseExecutable>(specification);
}

}  // namespace flagdnn::benchmarking::cudnn_detail
