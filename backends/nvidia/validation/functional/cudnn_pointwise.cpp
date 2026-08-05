/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/pointwise.hpp"
#include "validation/functional/cudnn_graph.hpp"
#include "validation/tensor_io.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

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

cfe::PointwiseMode_t cudnn_pointwise_mode(flagdnnPointwiseMode_t mode) {
  switch (mode) {
    case FLAGDNN_POINTWISE_RELU_FWD:
      return cfe::PointwiseMode_t::RELU_FWD;
    case FLAGDNN_POINTWISE_ADD:
      return cfe::PointwiseMode_t::ADD;
    case FLAGDNN_POINTWISE_SQRT:
      return cfe::PointwiseMode_t::SQRT;
    case FLAGDNN_POINTWISE_ERF:
      return cfe::PointwiseMode_t::ERF;
    case FLAGDNN_POINTWISE_IDENTITY:
      return cfe::PointwiseMode_t::IDENTITY;
    case FLAGDNN_POINTWISE_EXP:
      return cfe::PointwiseMode_t::EXP;
    case FLAGDNN_POINTWISE_LOG:
      return cfe::PointwiseMode_t::LOG;
    case FLAGDNN_POINTWISE_NEG:
      return cfe::PointwiseMode_t::NEG;
    case FLAGDNN_POINTWISE_ABS:
      return cfe::PointwiseMode_t::ABS;
    case FLAGDNN_POINTWISE_CEIL:
      return cfe::PointwiseMode_t::CEIL;
    case FLAGDNN_POINTWISE_COS:
      return cfe::PointwiseMode_t::COS;
    case FLAGDNN_POINTWISE_FLOOR:
      return cfe::PointwiseMode_t::FLOOR;
    case FLAGDNN_POINTWISE_RSQRT:
      return cfe::PointwiseMode_t::RSQRT;
    case FLAGDNN_POINTWISE_SIN:
      return cfe::PointwiseMode_t::SIN;
    case FLAGDNN_POINTWISE_TAN:
      return cfe::PointwiseMode_t::TAN;
    case FLAGDNN_POINTWISE_RECIPROCAL:
      return cfe::PointwiseMode_t::RECIPROCAL;
    case FLAGDNN_POINTWISE_SUB:
      return cfe::PointwiseMode_t::SUB;
    case FLAGDNN_POINTWISE_MUL:
      return cfe::PointwiseMode_t::MUL;
    case FLAGDNN_POINTWISE_DIV:
      return cfe::PointwiseMode_t::DIV;
    case FLAGDNN_POINTWISE_MIN:
      return cfe::PointwiseMode_t::MIN;
    case FLAGDNN_POINTWISE_MAX:
      return cfe::PointwiseMode_t::MAX;
    case FLAGDNN_POINTWISE_MOD:
      return cfe::PointwiseMode_t::MOD;
    case FLAGDNN_POINTWISE_POW:
      return cfe::PointwiseMode_t::POW;
    case FLAGDNN_POINTWISE_LOGICAL_NOT:
      return cfe::PointwiseMode_t::LOGICAL_NOT;
    case FLAGDNN_POINTWISE_CMP_EQ:
      return cfe::PointwiseMode_t::CMP_EQ;
    case FLAGDNN_POINTWISE_CMP_NEQ:
      return cfe::PointwiseMode_t::CMP_NEQ;
    case FLAGDNN_POINTWISE_CMP_GT:
      return cfe::PointwiseMode_t::CMP_GT;
    case FLAGDNN_POINTWISE_CMP_GE:
      return cfe::PointwiseMode_t::CMP_GE;
    case FLAGDNN_POINTWISE_CMP_LT:
      return cfe::PointwiseMode_t::CMP_LT;
    case FLAGDNN_POINTWISE_CMP_LE:
      return cfe::PointwiseMode_t::CMP_LE;
    case FLAGDNN_POINTWISE_LOGICAL_AND:
      return cfe::PointwiseMode_t::LOGICAL_AND;
    case FLAGDNN_POINTWISE_LOGICAL_OR:
      return cfe::PointwiseMode_t::LOGICAL_OR;
    case FLAGDNN_POINTWISE_SIGMOID_BWD:
      return cfe::PointwiseMode_t::SIGMOID_BWD;
    case FLAGDNN_POINTWISE_BINARY_SELECT:
      return cfe::PointwiseMode_t::BINARY_SELECT;
    case FLAGDNN_POINTWISE_SIGMOID_FWD:
      return cfe::PointwiseMode_t::SIGMOID_FWD;
    case FLAGDNN_POINTWISE_TANH_FWD:
      return cfe::PointwiseMode_t::TANH_FWD;
    case FLAGDNN_POINTWISE_ELU_FWD:
      return cfe::PointwiseMode_t::ELU_FWD;
    case FLAGDNN_POINTWISE_GELU_FWD:
      return cfe::PointwiseMode_t::GELU_FWD;
    case FLAGDNN_POINTWISE_SOFTPLUS_FWD:
      return cfe::PointwiseMode_t::SOFTPLUS_FWD;
    case FLAGDNN_POINTWISE_SWISH_FWD:
      return cfe::PointwiseMode_t::SWISH_FWD;
    case FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD:
      return cfe::PointwiseMode_t::GELU_APPROX_TANH_FWD;
    case FLAGDNN_POINTWISE_NOT_SET:
      break;
  }
  throw std::invalid_argument("unsupported cuDNN pointwise mode");
}

bool uses_boolean_compute(flagdnnPointwiseMode_t mode) {
  // cuDNN 9.24 floating-input comparisons require FLOAT compute even though
  // their result is packed BOOLEAN.  Logical operators use BOOLEAN compute.
  return mode == FLAGDNN_POINTWISE_LOGICAL_NOT ||
         mode == FLAGDNN_POINTWISE_LOGICAL_AND ||
         mode == FLAGDNN_POINTWISE_LOGICAL_OR;
}

void apply_pointwise_attributes(
    cfe::graph::Pointwise_attributes& output,
    const flagdnnPointwiseAttributes_t& input) {
  if ((input.flags & FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP) != 0U) {
    output.set_relu_lower_clip(static_cast<float>(input.relu_lower_clip));
  }
  if ((input.flags & FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP) != 0U) {
    output.set_relu_upper_clip(static_cast<float>(input.relu_upper_clip));
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

class DeviceScalar {
 public:
  DeviceScalar(flagdnnDataType_t data_type, double value) {
    switch (data_type) {
      case FLAGDNN_DATA_FLOAT32:
        allocate_and_copy(static_cast<float>(value));
        return;
      case FLAGDNN_DATA_FLOAT16:
        allocate_and_copy(__float2half_rn(static_cast<float>(value)));
        return;
      case FLAGDNN_DATA_BFLOAT16:
        allocate_and_copy(__float2bfloat16_rn(static_cast<float>(value)));
        return;
      case FLAGDNN_DATA_FP8_E4M3:
      case FLAGDNN_DATA_FP8_E5M2:
        break;
      case FLAGDNN_DATA_BOOLEAN:
        break;
    }
    throw std::invalid_argument("cuDNN pointwise scalar type is invalid");
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
    cuda::check_cuda_runtime(cudaMalloc(&pointer_, sizeof(Value)),
                             "cudaMalloc(pointwise scalar)");
    try {
      cuda::check_cuda_runtime(
          cudaMemcpy(pointer_, &value, sizeof(Value), cudaMemcpyHostToDevice),
          "cudaMemcpy(pointwise scalar)");
    } catch (...) {
      (void)cudaFree(pointer_);
      pointer_ = nullptr;
      throw;
    }
  }

  void* pointer_ = nullptr;
};

std::int64_t internal_scalar_uid(const PointwiseTestCase& test_case) {
  std::int64_t candidate = std::numeric_limits<std::int64_t>::max();
  for (;;) {
    bool used = candidate == test_case.output.uid;
    for (const TestTensor& input : test_case.inputs) {
      used = used || candidate == input.uid;
    }
    if (!used) {
      return candidate;
    }
    --candidate;
  }
}

class CudnnPointwiseExecutable final : public cuda::CudnnGraphExecutable {
 public:
  explicit CudnnPointwiseExecutable(const PointwiseTestCase& test_case)
      : graph_(std::make_shared<cfe::graph::Graph>()) {
    validate_pointwise_case(test_case);
    const std::size_t logical_rank = test_case.output.dimensions.size();

    graph_->set_name(test_case.name + "::cudnn")
        .set_io_data_type(
            cuda::cudnn_frontend_data_type(test_case.inputs.front().data_type))
        .set_intermediate_data_type(cfe::DataType_t::FLOAT)
        .set_compute_data_type(cfe::DataType_t::FLOAT);

    cfe::graph::Pointwise_attributes attributes;
    attributes.set_name(test_case.name)
        .set_mode(cudnn_pointwise_mode(test_case.mode))
        .set_compute_data_type(uses_boolean_compute(test_case.mode)
                                   ? cfe::DataType_t::BOOLEAN
                                   : cfe::DataType_t::FLOAT);
    apply_pointwise_attributes(attributes, test_case.attributes);

    TestTensor output_spec;
    std::shared_ptr<cfe::graph::Tensor_attributes> output;
    if (test_case.inputs.size() == 1) {
      const bool compact_pair =
          cuda::storage_element_count(test_case.inputs[0]) ==
              cuda::element_count(test_case.inputs[0]) &&
          cuda::storage_element_count(test_case.output) ==
              cuda::element_count(test_case.output) &&
          cuda::has_same_physical_mapping(test_case.inputs[0],
                                          test_case.output);
      const TestTensor input_spec =
          compact_pair
              ? cuda::flatten_compact_tensor(test_case.inputs[0])
              : cuda::padded_to_rank_four(test_case.inputs[0], logical_rank);
      output_spec =
          compact_pair
              ? cuda::flatten_compact_tensor(test_case.output)
              : cuda::padded_to_rank_four(test_case.output, logical_rank);
      const auto input = cuda::make_cudnn_tensor(graph_, input_spec, "input");
      output = graph_->pointwise(input, attributes);
    } else if (test_case.inputs.size() == 2) {
      const TestTensor left_spec = cuda::canonicalize_pointwise_tensor(
          test_case.inputs[0], logical_rank);
      const TestTensor right_spec = cuda::canonicalize_pointwise_tensor(
          test_case.inputs[1], logical_rank);
      output_spec =
          cuda::canonicalize_pointwise_tensor(test_case.output, logical_rank);
      const auto left = cuda::make_cudnn_tensor(graph_, left_spec, "left");
      const auto right = cuda::make_cudnn_tensor(graph_, right_spec, "right");

      std::shared_ptr<cfe::graph::Tensor_attributes> right_operand = right;
      const bool scale_right =
          (test_case.mode == FLAGDNN_POINTWISE_ADD ||
           test_case.mode == FLAGDNN_POINTWISE_SUB) &&
          test_case.alpha != 1.0;
      if (scale_right) {
        scalar_uid_ = internal_scalar_uid(test_case);
        scalar_ = std::make_unique<DeviceScalar>(right_spec.data_type,
                                                 test_case.alpha);
        const TestTensor scalar_spec{
            scalar_uid_, right_spec.data_type, {1, 1, 1, 1}, {1, 1, 1, 1}};
        const auto scalar =
            cuda::make_cudnn_tensor(graph_, scalar_spec, "alpha");
        auto scaled_right = graph_->pointwise(
            right,
            scalar,
            cfe::graph::Pointwise_attributes()
                .set_name(test_case.name + "::scale_right")
                .set_mode(cfe::PointwiseMode_t::MUL)
                .set_compute_data_type(cfe::DataType_t::FLOAT));
        scaled_right->set_name("scaled_right")
            .set_data_type(
                cuda::cudnn_frontend_data_type(right_spec.data_type))
            .set_dim(right_spec.dimensions)
            .set_stride(right_spec.strides);
        right_operand = std::move(scaled_right);
      }
      output = graph_->pointwise(left, right_operand, attributes);
    } else {
      const TestTensor a_spec = cuda::canonicalize_pointwise_tensor(
          test_case.inputs[0], logical_rank);
      const TestTensor b_spec = cuda::canonicalize_pointwise_tensor(
          test_case.inputs[1], logical_rank);
      const TestTensor t_spec = cuda::canonicalize_pointwise_tensor(
          test_case.inputs[2], logical_rank);
      output_spec =
          cuda::canonicalize_pointwise_tensor(test_case.output, logical_rank);
      const auto a = cuda::make_cudnn_tensor(graph_, a_spec, "a");
      const auto b = cuda::make_cudnn_tensor(graph_, b_spec, "b");
      const auto t = cuda::make_cudnn_tensor(graph_, t_spec, "predicate");
      output = graph_->pointwise(a, b, t, attributes);
    }

    output->set_name("output")
        .set_uid(output_spec.uid)
        .set_data_type(
            cuda::cudnn_frontend_data_type(output_spec.data_type))
        .set_dim(output_spec.dimensions)
        .set_stride(output_spec.strides)
        .set_output(true);

    cuda::check_cudnn_frontend(
        graph_->build(handle(), {cfe::HeurMode_t::A}),
        "cuDNN pointwise graph build");
    std::int64_t workspace_size = 0;
    cuda::check_cudnn_frontend(graph_->get_workspace_size(workspace_size),
                               "cuDNN pointwise workspace query");
    set_workspace_size(workspace_size);
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    begin_execute(workspace, workspace_size, stream);
    cuda::CudnnBindingMap pointers = cuda::make_cudnn_binding_map(bindings);
    if (scalar_ != nullptr &&
        !pointers.emplace(scalar_uid_, scalar_->get()).second) {
      throw std::invalid_argument(
          "cuDNN pointwise scalar UID collides with a caller binding");
    }
    cuda::check_cudnn_frontend(graph_->execute(handle(), pointers, workspace),
                               "cuDNN pointwise graph execute");
  }

 private:
  std::unique_ptr<DeviceScalar> scalar_;
  std::int64_t scalar_uid_ = 0;
  std::shared_ptr<cfe::graph::Graph> graph_;
};

}  // namespace

std::unique_ptr<PointwiseExecutable> build_pointwise_reference(
    const PointwiseTestCase& test_case) {
  return std::make_unique<CudnnPointwiseExecutable>(test_case);
}

}  // namespace flagdnn::testing
