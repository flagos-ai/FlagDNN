/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/attention.hpp"

#include <flagdnn/flagdnn.hpp>
#include <flagdnn_frontend.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace flagdnn::testing {
namespace {

namespace fe = ::flagdnn_frontend;
using Shape = std::vector<std::int64_t>;

Shape contiguous_strides(const Shape& dimensions) {
  Shape result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    const std::int64_t dimension = dimensions[axis - 1];
    if (dimension <= 0 ||
        stride > std::numeric_limits<std::int64_t>::max() / dimension) {
      throw std::invalid_argument("SDPA shape is invalid or too large");
    }
    result[axis - 1] = stride;
    stride *= dimension;
  }
  return result;
}

TestTensor tensor(std::int64_t uid,
                  Shape dimensions,
                  flagdnnDataType_t data_type) {
  Shape strides = contiguous_strides(dimensions);
  return {uid, data_type, std::move(dimensions), std::move(strides)};
}

TestTensor stats_tensor(std::int64_t uid,
                        std::int64_t batch,
                        std::int64_t heads,
                        std::int64_t sequence) {
  return {uid,
          FLAGDNN_DATA_FLOAT32,
          {batch, heads, sequence, 1},
          {heads * sequence, sequence, 1, 1}};
}

fe::DataType_t frontend_data_type(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_FLOAT32:
      return fe::DataType_t::FLOAT;
    case FLAGDNN_DATA_FLOAT16:
      return fe::DataType_t::HALF;
    case FLAGDNN_DATA_BFLOAT16:
      return fe::DataType_t::BFLOAT16;
    case FLAGDNN_DATA_BOOLEAN:
      break;
    case FLAGDNN_DATA_FP8_E4M3:
      return fe::DataType_t::FP8_E4M3;
    case FLAGDNN_DATA_FP8_E5M2:
      return fe::DataType_t::FP8_E5M2;
  }
  throw std::invalid_argument("unsupported SDPA data type");
}

void check_frontend(fe::error_t status, std::string_view operation) {
  if (status.is_bad()) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + status.get_message());
  }
}

void validate_tensor(const TestTensor& specification,
                     std::string_view name,
                     flagdnnDataType_t expected_data_type) {
  if (specification.uid <= 0 || specification.dimensions.size() != 4 ||
      specification.strides.size() != 4 ||
      specification.data_type != expected_data_type) {
    throw std::invalid_argument(std::string(name) + " metadata is invalid");
  }
  for (std::size_t axis = 0; axis < 4; ++axis) {
    if (specification.dimensions[axis] <= 0 ||
        specification.strides[axis] <= 0) {
      throw std::invalid_argument(
          std::string(name) + " dimensions and strides must be positive");
    }
  }
}

void validate_io_data_type(flagdnnDataType_t data_type) {
  if (data_type != FLAGDNN_DATA_FLOAT16 &&
      data_type != FLAGDNN_DATA_BFLOAT16 &&
      data_type != FLAGDNN_DATA_FLOAT32) {
    throw std::invalid_argument("SDPA requires a floating-point data type");
  }
}

void validate_options(const AttentionOptions& options) {
  if (options.attention_scale.has_value() &&
      (!std::isfinite(*options.attention_scale) ||
       *options.attention_scale <= 0.0F)) {
    throw std::invalid_argument(
        "SDPA attention scale must be positive and finite");
  }
  if (options.diagonal_band_left_bound.has_value() &&
      *options.diagonal_band_left_bound < 1) {
    throw std::invalid_argument("SDPA left diagonal bound is invalid");
  }
  if (options.diagonal_band_right_bound.has_value() &&
      *options.diagonal_band_right_bound < 0) {
    throw std::invalid_argument("SDPA right diagonal bound is invalid");
  }
}

void validate_unique_uids(std::span<const TestTensor* const> tensors) {
  std::set<std::int64_t> uids;
  for (const TestTensor* tensor_specification : tensors) {
    if (tensor_specification != nullptr &&
        !uids.insert(tensor_specification->uid).second) {
      throw std::invalid_argument("SDPA tensor UIDs must be unique");
    }
  }
}

struct AttentionShape {
  std::int64_t batch;
  std::int64_t query_heads;
  std::int64_t key_heads;
  std::int64_t value_heads;
  std::int64_t sequence_q;
  std::int64_t sequence_kv;
  std::int64_t head_dimension;
  std::int64_t value_dimension;
};

AttentionShape validate_qkv(const TestTensor& q,
                            const TestTensor& k,
                            const TestTensor& v) {
  validate_io_data_type(q.data_type);
  validate_tensor(q, "SDPA Q", q.data_type);
  validate_tensor(k, "SDPA K", q.data_type);
  validate_tensor(v, "SDPA V", q.data_type);
  if (q.dimensions[0] != k.dimensions[0] ||
      q.dimensions[0] != v.dimensions[0] ||
      q.dimensions[3] != k.dimensions[3] ||
      k.dimensions[2] != v.dimensions[2] ||
      q.dimensions[1] % k.dimensions[1] != 0 ||
      q.dimensions[1] % v.dimensions[1] != 0) {
    throw std::invalid_argument("SDPA Q/K/V shapes are inconsistent");
  }
  return {q.dimensions[0],
          q.dimensions[1],
          k.dimensions[1],
          v.dimensions[1],
          q.dimensions[2],
          k.dimensions[2],
          q.dimensions[3],
          v.dimensions[3]};
}

bool is_fp8_data_type(flagdnnDataType_t data_type) {
  return data_type == FLAGDNN_DATA_FP8_E4M3 ||
         data_type == FLAGDNN_DATA_FP8_E5M2;
}

AttentionShape validate_fp8_qkv(const TestTensor& q,
                                const TestTensor& k,
                                const TestTensor& v) {
  if (!is_fp8_data_type(q.data_type)) {
    throw std::invalid_argument("FP8 SDPA requires an FP8 data type");
  }
  validate_tensor(q, "FP8 SDPA Q", q.data_type);
  validate_tensor(k, "FP8 SDPA K", q.data_type);
  validate_tensor(v, "FP8 SDPA V", q.data_type);
  if (q.dimensions[0] != k.dimensions[0] ||
      q.dimensions[0] != v.dimensions[0] ||
      q.dimensions[3] != k.dimensions[3] ||
      k.dimensions[2] != v.dimensions[2] ||
      q.dimensions[1] % k.dimensions[1] != 0 ||
      q.dimensions[1] % v.dimensions[1] != 0) {
    throw std::invalid_argument("FP8 SDPA Q/K/V shapes are inconsistent");
  }
  return {q.dimensions[0],
          q.dimensions[1],
          k.dimensions[1],
          v.dimensions[1],
          q.dimensions[2],
          k.dimensions[2],
          q.dimensions[3],
          v.dimensions[3]};
}

Fp8Scalar fp8_scalar(std::int64_t uid, float value) {
  if (!std::isfinite(value) || value <= 0.0F) {
    throw std::invalid_argument("FP8 scale must be positive and finite");
  }
  return {tensor(uid, {1, 1, 1, 1}, FLAGDNN_DATA_FLOAT32), value};
}

TestTensor fp8_amax(std::int64_t uid) {
  return tensor(uid, {1, 1, 1, 1}, FLAGDNN_DATA_FLOAT32);
}

void validate_fp8_scalar(const Fp8Scalar& scalar_value,
                         std::string_view name) {
  validate_tensor(
      scalar_value.tensor, name, FLAGDNN_DATA_FLOAT32);
  if (scalar_value.tensor.dimensions != Shape{1, 1, 1, 1} ||
      !std::isfinite(scalar_value.value) || scalar_value.value <= 0.0F) {
    throw std::invalid_argument(std::string(name) + " is invalid");
  }
}

bool valid_tolerance(double value) {
  return std::isfinite(value) && value >= 0.0;
}

void validate_bias(const TestTensor& bias,
                   flagdnnDataType_t data_type,
                   const AttentionShape& shape,
                   std::string_view name) {
  validate_tensor(bias, name, data_type);
  if ((bias.dimensions[0] != 1 &&
       bias.dimensions[0] != shape.batch) ||
      (bias.dimensions[1] != 1 &&
       bias.dimensions[1] != shape.query_heads) ||
      bias.dimensions[2] != shape.sequence_q ||
      bias.dimensions[3] != shape.sequence_kv) {
    throw std::invalid_argument(std::string(name) + " shape is invalid");
  }
}

std::shared_ptr<fe::graph::Tensor_attributes> make_tensor(
    const std::shared_ptr<fe::graph::Graph>& graph,
    const TestTensor& specification,
    std::string_view name) {
  return graph->tensor(
      fe::graph::Tensor_attributes()
          .set_name(std::string(name))
          .set_uid(specification.uid)
          .set_data_type(frontend_data_type(specification.data_type))
          .set_dim(specification.dimensions)
          .set_stride(specification.strides));
}

template <typename Attributes>
void apply_options(Attributes& attributes,
                   const AttentionOptions& options) {
  if (options.attention_scale.has_value()) {
    attributes.set_attn_scale(*options.attention_scale);
  }
  attributes.set_diagonal_alignment(
      options.diagonal_alignment == AttentionDiagonalAlignment::kTopLeft
          ? fe::DiagonalAlignment_t::TOP_LEFT
          : fe::DiagonalAlignment_t::BOTTOM_RIGHT);
  if (options.diagonal_band_left_bound.has_value()) {
    attributes.set_diagonal_band_left_bound(
        *options.diagonal_band_left_bound);
  }
  if (options.diagonal_band_right_bound.has_value()) {
    attributes.set_diagonal_band_right_bound(
        *options.diagonal_band_right_bound);
  }
}

void set_output(std::shared_ptr<fe::graph::Tensor_attributes>& tensor_value,
                const TestTensor& specification,
                std::string_view name) {
  tensor_value->set_name(std::string(name))
      .set_uid(specification.uid)
      .set_data_type(frontend_data_type(specification.data_type))
      .set_dim(specification.dimensions)
      .set_stride(specification.strides)
      .set_output(true);
}

class FlagdnnAttentionExecutable final : public AttentionExecutable {
 public:
  FlagdnnAttentionExecutable(
      flagdnn::Handle& handle,
      std::shared_ptr<fe::graph::Graph> graph,
      std::string operation)
      : handle_(handle),
        graph_(std::move(graph)),
        operation_(std::move(operation)) {
    check_frontend(graph_->build(handle_, {fe::HeurMode_t::A}),
                   "FlagDNN " + operation_ + " graph build");
    std::int64_t workspace_size = 0;
    check_frontend(graph_->get_workspace_size(workspace_size),
                   "FlagDNN " + operation_ + " workspace query");
    if (workspace_size < 0) {
      throw std::runtime_error("FlagDNN returned a negative workspace size");
    }
    workspace_size_ = static_cast<std::size_t>(workspace_size);
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    if (workspace_size < workspace_size_ ||
        (workspace_size_ != 0 && workspace == nullptr)) {
      throw std::invalid_argument(
          "FlagDNN " + operation_ + " workspace is too small");
    }
    check_frontend(
        graph_->execute(handle_, bindings, workspace, workspace_size, stream),
        "FlagDNN " + operation_ + " graph execute");
  }

 private:
  flagdnn::Handle& handle_;
  std::shared_ptr<fe::graph::Graph> graph_;
  std::string operation_;
  std::size_t workspace_size_ = 0;
};

SdpaTestCase make_forward_case(std::string name,
                               std::int64_t uid,
                               flagdnnDataType_t data_type,
                               Shape q_shape,
                               Shape k_shape,
                               Shape v_shape) {
  SdpaTestCase result;
  result.name = std::move(name);
  result.q = tensor(uid, std::move(q_shape), data_type);
  result.k = tensor(uid + 1, std::move(k_shape), data_type);
  result.v = tensor(uid + 2, std::move(v_shape), data_type);
  result.output = tensor(
      uid + 3,
      {result.q.dimensions[0],
       result.q.dimensions[1],
       result.q.dimensions[2],
       result.v.dimensions[3]},
      data_type);
  result.stats = stats_tensor(uid + 4,
                              result.q.dimensions[0],
                              result.q.dimensions[1],
                              result.q.dimensions[2]);
  if (data_type == FLAGDNN_DATA_BFLOAT16) {
    result.output_absolute_tolerance = 1.0e-1;
    result.output_relative_tolerance = 5.0e-2;
    result.stats_absolute_tolerance = 2.0e-2;
    result.stats_relative_tolerance = 2.0e-2;
  } else if (data_type == FLAGDNN_DATA_FLOAT16) {
    result.output_absolute_tolerance = 5.0e-2;
    result.output_relative_tolerance = 5.0e-2;
    result.stats_absolute_tolerance = 2.0e-2;
    result.stats_relative_tolerance = 2.0e-2;
  } else {
    result.output_absolute_tolerance = 1.0e-3;
    result.output_relative_tolerance = 1.0e-3;
    result.stats_absolute_tolerance = 5.0e-3;
    result.stats_relative_tolerance = 5.0e-3;
  }
  return result;
}

SdpaBackwardTestCase make_backward_case(
    std::string name,
    std::int64_t uid,
    flagdnnDataType_t data_type,
    Shape q_shape,
    Shape k_shape,
    Shape v_shape) {
  SdpaBackwardTestCase result;
  result.name = std::move(name);
  result.q = tensor(uid, std::move(q_shape), data_type);
  result.k = tensor(uid + 1, std::move(k_shape), data_type);
  result.v = tensor(uid + 2, std::move(v_shape), data_type);
  const Shape output_shape{result.q.dimensions[0],
                           result.q.dimensions[1],
                           result.q.dimensions[2],
                           result.v.dimensions[3]};
  result.output = tensor(uid + 3, output_shape, data_type);
  result.doutput = tensor(uid + 4, output_shape, data_type);
  result.stats = stats_tensor(uid + 5,
                              result.q.dimensions[0],
                              result.q.dimensions[1],
                              result.q.dimensions[2]);
  result.dq = tensor(uid + 6, result.q.dimensions, data_type);
  result.dk = tensor(uid + 7, result.k.dimensions, data_type);
  result.dv = tensor(uid + 8, result.v.dimensions, data_type);
  if (data_type == FLAGDNN_DATA_BFLOAT16) {
    result.absolute_tolerance = 8.0e-2;
    result.relative_tolerance = 3.0e-2;
  } else if (data_type == FLAGDNN_DATA_FLOAT16) {
    result.absolute_tolerance = 4.0e-2;
    result.relative_tolerance = 2.0e-2;
  } else {
    result.absolute_tolerance = 3.0e-2;
    result.relative_tolerance = 3.0e-2;
  }
  return result;
}

SdpaFp8TestCase make_fp8_forward_case(
    std::string name,
    std::int64_t uid,
    flagdnnDataType_t data_type,
    Shape q_shape,
    Shape k_shape,
    Shape v_shape) {
  SdpaFp8TestCase result;
  result.name = std::move(name);
  result.q = tensor(uid, std::move(q_shape), data_type);
  result.k = tensor(uid + 1, std::move(k_shape), data_type);
  result.v = tensor(uid + 2, std::move(v_shape), data_type);
  result.descale_q = fp8_scalar(uid + 3, 0.5F);
  result.descale_k = fp8_scalar(uid + 4, 0.5F);
  result.descale_v = fp8_scalar(uid + 5, 0.5F);
  result.descale_s = fp8_scalar(uid + 6, 1.0F / 32.0F);
  result.scale_s = fp8_scalar(uid + 7, 32.0F);
  result.scale_o = fp8_scalar(uid + 8, 4.0F);
  const Shape output_shape{result.q.dimensions[0],
                           result.q.dimensions[1],
                           result.q.dimensions[2],
                           result.v.dimensions[3]};
  result.output = tensor(uid + 9, output_shape, data_type);
  result.stats = stats_tensor(uid + 10,
                              result.q.dimensions[0],
                              result.q.dimensions[1],
                              result.q.dimensions[2]);
  result.amax_s = fp8_amax(uid + 11);
  result.amax_o = fp8_amax(uid + 12);
  result.options.attention_scale =
      1.0F / std::sqrt(static_cast<float>(result.q.dimensions[3]));
  result.output_absolute_tolerance = 0.5;
  result.output_relative_tolerance = 0.35;
  result.stats_absolute_tolerance = 0.08;
  result.stats_relative_tolerance = 0.08;
  result.amax_absolute_tolerance = 0.15;
  result.amax_relative_tolerance = 0.25;
  return result;
}

SdpaFp8BackwardTestCase make_fp8_backward_case(
    std::string name,
    std::int64_t uid,
    flagdnnDataType_t data_type,
    Shape q_shape,
    Shape k_shape,
    Shape v_shape) {
  SdpaFp8BackwardTestCase result;
  result.name = std::move(name);
  result.q = tensor(uid, std::move(q_shape), data_type);
  result.k = tensor(uid + 1, std::move(k_shape), data_type);
  result.v = tensor(uid + 2, std::move(v_shape), data_type);
  const Shape output_shape{result.q.dimensions[0],
                           result.q.dimensions[1],
                           result.q.dimensions[2],
                           result.v.dimensions[3]};
  result.output = tensor(uid + 3, output_shape, data_type);
  result.doutput = tensor(uid + 4, output_shape, data_type);
  result.stats = stats_tensor(uid + 5,
                              result.q.dimensions[0],
                              result.q.dimensions[1],
                              result.q.dimensions[2]);
  result.descale_q = fp8_scalar(uid + 6, 0.5F);
  result.descale_k = fp8_scalar(uid + 7, 0.5F);
  result.descale_v = fp8_scalar(uid + 8, 0.5F);
  result.descale_o = fp8_scalar(uid + 9, 0.25F);
  result.descale_doutput = fp8_scalar(uid + 10, 0.5F);
  result.descale_s = fp8_scalar(uid + 11, 1.0F / 32.0F);
  result.descale_dp = fp8_scalar(uid + 12, 1.0F / 32.0F);
  result.scale_s = fp8_scalar(uid + 13, 32.0F);
  result.scale_dq = fp8_scalar(uid + 14, 4.0F);
  result.scale_dk = fp8_scalar(uid + 15, 4.0F);
  result.scale_dv = fp8_scalar(uid + 16, 4.0F);
  result.scale_dp = fp8_scalar(uid + 17, 32.0F);
  result.dq = tensor(uid + 18, result.q.dimensions, data_type);
  result.dk = tensor(uid + 19, result.k.dimensions, data_type);
  result.dv = tensor(uid + 20, result.v.dimensions, data_type);
  result.amax_dq = fp8_amax(uid + 21);
  result.amax_dk = fp8_amax(uid + 22);
  result.amax_dv = fp8_amax(uid + 23);
  result.amax_dp = fp8_amax(uid + 24);
  result.options.attention_scale =
      1.0F / std::sqrt(static_cast<float>(result.q.dimensions[3]));
  result.gradient_absolute_tolerance = 0.75;
  result.gradient_relative_tolerance = 0.5;
  result.amax_absolute_tolerance = 0.25;
  result.amax_relative_tolerance = 0.5;
  return result;
}

}  // namespace

std::vector<SdpaTestCase> make_sdpa_cases() {
  std::vector<SdpaTestCase> result;

  SdpaTestCase dense = make_forward_case(
      "sdpa_fp16_dense_autotune",
      70001,
      FLAGDNN_DATA_FLOAT16,
      {1, 2, 64, 64},
      {1, 2, 64, 64},
      {1, 2, 64, 64});
  dense.options.attention_scale = 0.125F;
  dense.autotune = true;
  result.push_back(std::move(dense));

  SdpaTestCase causal_gqa = make_forward_case(
      "sdpa_bfloat16_causal_gqa",
      70011,
      FLAGDNN_DATA_BFLOAT16,
      {1, 4, 48, 64},
      {1, 2, 48, 64},
      {1, 2, 48, 64});
  causal_gqa.options.attention_scale = 0.125F;
  causal_gqa.options.diagonal_band_right_bound = 0;
  result.push_back(std::move(causal_gqa));

  SdpaTestCase biased = make_forward_case(
      "sdpa_fp16_broadcast_bias",
      70021,
      FLAGDNN_DATA_FLOAT16,
      {2, 2, 32, 64},
      {2, 2, 40, 64},
      {2, 2, 40, 64});
  biased.options.attention_scale = 0.125F;
  biased.bias = tensor(70026, {1, 2, 32, 40}, FLAGDNN_DATA_FLOAT16);
  result.push_back(std::move(biased));

  SdpaTestCase inference = make_forward_case(
      "sdpa_fp16_inference_without_stats",
      70031,
      FLAGDNN_DATA_FLOAT16,
      {1, 2, 16, 64},
      {1, 2, 24, 64},
      {1, 2, 24, 64});
  inference.options.attention_scale = 0.125F;
  inference.stats.reset();
  result.push_back(std::move(inference));

  for (const SdpaTestCase& test_case : result) {
    validate_sdpa_case(test_case);
  }
  return result;
}

std::vector<SdpaBackwardTestCase> make_sdpa_backward_cases() {
  std::vector<SdpaBackwardTestCase> result;

  SdpaBackwardTestCase dense = make_backward_case(
      "sdpa_backward_fp16_dense_autotune",
      71001,
      FLAGDNN_DATA_FLOAT16,
      {1, 2, 32, 32},
      {1, 2, 32, 32},
      {1, 2, 32, 32});
  dense.options.attention_scale = 0.176776695F;
  dense.autotune = true;
  result.push_back(std::move(dense));

  SdpaBackwardTestCase causal_gqa = make_backward_case(
      "sdpa_backward_bfloat16_causal_gqa",
      71021,
      FLAGDNN_DATA_BFLOAT16,
      {1, 4, 32, 64},
      {1, 2, 32, 64},
      {1, 2, 32, 64});
  causal_gqa.options.attention_scale = 0.125F;
  causal_gqa.options.diagonal_band_right_bound = 0;
  causal_gqa.autotune = true;
  result.push_back(std::move(causal_gqa));

  SdpaBackwardTestCase different_v_dimension = make_backward_case(
      "sdpa_backward_fp16_different_value_dimension",
      71041,
      FLAGDNN_DATA_FLOAT16,
      {1, 2, 24, 32},
      {1, 2, 32, 32},
      {1, 2, 32, 64});
  different_v_dimension.options.attention_scale = 0.2F;
  different_v_dimension.autotune = true;
  result.push_back(std::move(different_v_dimension));

  SdpaBackwardTestCase biased = make_backward_case(
      "sdpa_backward_fp16_broadcast_dbias",
      71061,
      FLAGDNN_DATA_FLOAT16,
      {2, 4, 32, 64},
      {2, 4, 40, 64},
      {2, 4, 40, 64});
  biased.options.attention_scale = 0.125F;
  biased.bias = tensor(71070, {1, 4, 32, 40}, FLAGDNN_DATA_FLOAT16);
  biased.dbias = tensor(71071, {1, 4, 32, 40}, FLAGDNN_DATA_FLOAT16);
  biased.autotune = true;
  result.push_back(std::move(biased));

  for (const SdpaBackwardTestCase& test_case : result) {
    validate_sdpa_backward_case(test_case);
  }
  return result;
}

std::vector<SdpaFp8TestCase> make_sdpa_fp8_cases() {
  std::vector<SdpaFp8TestCase> result;

  SdpaFp8TestCase dense = make_fp8_forward_case(
      "sdpa_fp8_e4m3_dense_autotune",
      72001,
      FLAGDNN_DATA_FP8_E4M3,
      {1, 2, 64, 128},
      {1, 2, 64, 128},
      {1, 2, 64, 128});
  dense.autotune = true;
  result.push_back(std::move(dense));

  SdpaFp8TestCase causal_gqa = make_fp8_forward_case(
      "sdpa_fp8_e5m2_causal_gqa",
      72021,
      FLAGDNN_DATA_FP8_E5M2,
      {1, 4, 48, 128},
      {1, 2, 48, 128},
      {1, 2, 48, 128});
  causal_gqa.options.diagonal_band_right_bound = 0;
  result.push_back(std::move(causal_gqa));

  SdpaFp8TestCase rectangular = make_fp8_forward_case(
      "sdpa_fp8_e4m3_rectangular",
      72041,
      FLAGDNN_DATA_FP8_E4M3,
      {1, 2, 32, 128},
      {1, 2, 40, 128},
      {1, 2, 40, 128});
  result.push_back(std::move(rectangular));

  SdpaFp8TestCase inference = make_fp8_forward_case(
      "sdpa_fp8_e4m3_inference_without_stats",
      72061,
      FLAGDNN_DATA_FP8_E4M3,
      {1, 2, 32, 128},
      {1, 2, 40, 128},
      {1, 2, 40, 128});
  inference.stats.reset();
  result.push_back(std::move(inference));

  for (const SdpaFp8TestCase& test_case : result) {
    validate_sdpa_fp8_case(test_case);
  }
  return result;
}

std::vector<SdpaFp8BackwardTestCase> make_sdpa_fp8_backward_cases() {
  std::vector<SdpaFp8BackwardTestCase> result;

  SdpaFp8BackwardTestCase dense = make_fp8_backward_case(
      "sdpa_fp8_backward_e4m3_dense_autotune",
      73001,
      FLAGDNN_DATA_FP8_E4M3,
      {1, 2, 64, 128},
      {1, 2, 64, 128},
      {1, 2, 64, 128});
  dense.autotune = true;
  result.push_back(std::move(dense));

  SdpaFp8BackwardTestCase causal_gqa = make_fp8_backward_case(
      "sdpa_fp8_backward_e5m2_causal_gqa",
      73031,
      FLAGDNN_DATA_FP8_E5M2,
      {1, 4, 48, 128},
      {1, 2, 48, 128},
      {1, 2, 48, 128});
  causal_gqa.options.diagonal_band_right_bound = 0;
  result.push_back(std::move(causal_gqa));

  for (const SdpaFp8BackwardTestCase& test_case : result) {
    validate_sdpa_fp8_backward_case(test_case);
  }
  return result;
}

void validate_sdpa_case(const SdpaTestCase& test_case) {
  if (test_case.name.empty() ||
      !std::isfinite(test_case.output_absolute_tolerance) ||
      !std::isfinite(test_case.output_relative_tolerance) ||
      !std::isfinite(test_case.stats_absolute_tolerance) ||
      !std::isfinite(test_case.stats_relative_tolerance) ||
      test_case.output_absolute_tolerance < 0.0 ||
      test_case.output_relative_tolerance < 0.0 ||
      test_case.stats_absolute_tolerance < 0.0 ||
      test_case.stats_relative_tolerance < 0.0) {
    throw std::invalid_argument("SDPA case metadata is invalid");
  }
  validate_options(test_case.options);
  const AttentionShape shape = validate_qkv(test_case.q,
                                             test_case.k,
                                             test_case.v);
  validate_tensor(test_case.output, "SDPA output", test_case.q.data_type);
  if (test_case.output.dimensions !=
      Shape{shape.batch,
            shape.query_heads,
            shape.sequence_q,
            shape.value_dimension}) {
    throw std::invalid_argument("SDPA output shape is invalid");
  }
  if (test_case.bias.has_value()) {
    validate_bias(*test_case.bias,
                  test_case.q.data_type,
                  shape,
                  "SDPA bias");
  }
  if (test_case.stats.has_value()) {
    validate_tensor(*test_case.stats,
                    "SDPA stats",
                    FLAGDNN_DATA_FLOAT32);
    if (test_case.stats->dimensions !=
        Shape{shape.batch, shape.query_heads, shape.sequence_q, 1}) {
      throw std::invalid_argument("SDPA stats shape is invalid");
    }
  }
  const TestTensor* bias = test_case.bias.has_value()
                               ? &*test_case.bias
                               : nullptr;
  const TestTensor* stats = test_case.stats.has_value()
                                ? &*test_case.stats
                                : nullptr;
  const std::vector<const TestTensor*> tensors{
      &test_case.q,
      &test_case.k,
      &test_case.v,
      bias,
      &test_case.output,
      stats};
  validate_unique_uids(tensors);
}

void validate_sdpa_backward_case(const SdpaBackwardTestCase& test_case) {
  if (test_case.name.empty() ||
      !std::isfinite(test_case.absolute_tolerance) ||
      !std::isfinite(test_case.relative_tolerance) ||
      test_case.absolute_tolerance < 0.0 ||
      test_case.relative_tolerance < 0.0) {
    throw std::invalid_argument("SDPA backward case metadata is invalid");
  }
  validate_options(test_case.options);
  const AttentionShape shape = validate_qkv(test_case.q,
                                             test_case.k,
                                             test_case.v);
  if (shape.key_heads != shape.value_heads) {
    throw std::invalid_argument(
        "SDPA backward currently requires matching K/V head counts");
  }
  const Shape output_shape{shape.batch,
                           shape.query_heads,
                           shape.sequence_q,
                           shape.value_dimension};
  for (const auto& [tensor_specification, expected_shape, name] :
       std::vector<std::tuple<const TestTensor*, Shape, const char*>>{
           {&test_case.output, output_shape, "SDPA backward O"},
           {&test_case.doutput, output_shape, "SDPA backward dO"},
           {&test_case.dq, test_case.q.dimensions, "SDPA backward dQ"},
           {&test_case.dk, test_case.k.dimensions, "SDPA backward dK"},
           {&test_case.dv, test_case.v.dimensions, "SDPA backward dV"}}) {
    validate_tensor(*tensor_specification, name, test_case.q.data_type);
    if (tensor_specification->dimensions != expected_shape) {
      throw std::invalid_argument(std::string(name) + " shape is invalid");
    }
  }
  validate_tensor(test_case.stats,
                  "SDPA backward stats",
                  FLAGDNN_DATA_FLOAT32);
  if (test_case.stats.dimensions !=
      Shape{shape.batch, shape.query_heads, shape.sequence_q, 1}) {
    throw std::invalid_argument("SDPA backward stats shape is invalid");
  }
  if (test_case.bias.has_value()) {
    validate_bias(*test_case.bias,
                  test_case.q.data_type,
                  shape,
                  "SDPA backward bias");
  }
  if (test_case.dbias.has_value()) {
    if (!test_case.bias.has_value()) {
      throw std::invalid_argument("SDPA dBias requires bias");
    }
    validate_bias(*test_case.dbias,
                  test_case.q.data_type,
                  shape,
                  "SDPA backward dBias");
    if (test_case.dbias->dimensions != test_case.bias->dimensions) {
      throw std::invalid_argument("SDPA bias/dBias shapes must match");
    }
  }
  const TestTensor* bias = test_case.bias.has_value()
                               ? &*test_case.bias
                               : nullptr;
  const TestTensor* dbias = test_case.dbias.has_value()
                                ? &*test_case.dbias
                                : nullptr;
  const std::vector<const TestTensor*> tensors{
      &test_case.q,
      &test_case.k,
      &test_case.v,
      bias,
      &test_case.output,
      &test_case.doutput,
      &test_case.stats,
      &test_case.dq,
      &test_case.dk,
      &test_case.dv,
      dbias};
  validate_unique_uids(tensors);
}

void validate_sdpa_fp8_case(const SdpaFp8TestCase& test_case) {
  if (test_case.name.empty() ||
      !valid_tolerance(test_case.output_absolute_tolerance) ||
      !valid_tolerance(test_case.output_relative_tolerance) ||
      !valid_tolerance(test_case.stats_absolute_tolerance) ||
      !valid_tolerance(test_case.stats_relative_tolerance) ||
      !valid_tolerance(test_case.amax_absolute_tolerance) ||
      !valid_tolerance(test_case.amax_relative_tolerance)) {
    throw std::invalid_argument("FP8 SDPA case metadata is invalid");
  }
  validate_options(test_case.options);
  const AttentionShape shape =
      validate_fp8_qkv(test_case.q, test_case.k, test_case.v);
  validate_tensor(
      test_case.output, "FP8 SDPA output", test_case.q.data_type);
  if (test_case.output.dimensions !=
      Shape{shape.batch,
            shape.query_heads,
            shape.sequence_q,
            shape.value_dimension}) {
    throw std::invalid_argument("FP8 SDPA output shape is invalid");
  }
  if (test_case.bias.has_value()) {
    if (test_case.bias->data_type != FLAGDNN_DATA_FLOAT16 &&
        test_case.bias->data_type != FLAGDNN_DATA_BFLOAT16 &&
        test_case.bias->data_type != FLAGDNN_DATA_FLOAT32) {
      throw std::invalid_argument("FP8 SDPA bias must be floating point");
    }
    validate_bias(*test_case.bias,
                  test_case.bias->data_type,
                  shape,
                  "FP8 SDPA bias");
  }
  if (test_case.stats.has_value()) {
    validate_tensor(
        *test_case.stats, "FP8 SDPA stats", FLAGDNN_DATA_FLOAT32);
    if (test_case.stats->dimensions !=
        Shape{shape.batch, shape.query_heads, shape.sequence_q, 1}) {
      throw std::invalid_argument("FP8 SDPA stats shape is invalid");
    }
  }
  for (const auto& [scalar_value, name] :
       std::vector<std::pair<const Fp8Scalar*, const char*>>{
           {&test_case.descale_q, "FP8 SDPA descale Q"},
           {&test_case.descale_k, "FP8 SDPA descale K"},
           {&test_case.descale_v, "FP8 SDPA descale V"},
           {&test_case.descale_s, "FP8 SDPA descale S"},
           {&test_case.scale_s, "FP8 SDPA scale S"},
           {&test_case.scale_o, "FP8 SDPA scale O"}}) {
    validate_fp8_scalar(*scalar_value, name);
  }
  for (const auto& [amax, name] :
       std::vector<std::pair<const TestTensor*, const char*>>{
           {&test_case.amax_s, "FP8 SDPA amax S"},
           {&test_case.amax_o, "FP8 SDPA amax O"}}) {
    validate_tensor(*amax, name, FLAGDNN_DATA_FLOAT32);
    if (amax->dimensions != Shape{1, 1, 1, 1}) {
      throw std::invalid_argument(std::string(name) + " shape is invalid");
    }
  }
  const TestTensor* bias =
      test_case.bias.has_value() ? &*test_case.bias : nullptr;
  const TestTensor* stats =
      test_case.stats.has_value() ? &*test_case.stats : nullptr;
  const std::array<const TestTensor*, 14> tensors{{
      &test_case.q,
      &test_case.k,
      &test_case.v,
      &test_case.descale_q.tensor,
      &test_case.descale_k.tensor,
      &test_case.descale_v.tensor,
      &test_case.descale_s.tensor,
      &test_case.scale_s.tensor,
      &test_case.scale_o.tensor,
      bias,
      &test_case.output,
      stats,
      &test_case.amax_s,
      &test_case.amax_o,
  }};
  validate_unique_uids(tensors);
}

void validate_sdpa_fp8_backward_case(
    const SdpaFp8BackwardTestCase& test_case) {
  if (test_case.name.empty() ||
      !valid_tolerance(test_case.gradient_absolute_tolerance) ||
      !valid_tolerance(test_case.gradient_relative_tolerance) ||
      !valid_tolerance(test_case.amax_absolute_tolerance) ||
      !valid_tolerance(test_case.amax_relative_tolerance)) {
    throw std::invalid_argument(
        "FP8 SDPA backward case metadata is invalid");
  }
  validate_options(test_case.options);
  const AttentionShape shape =
      validate_fp8_qkv(test_case.q, test_case.k, test_case.v);
  if (shape.key_heads != shape.value_heads ||
      shape.head_dimension != shape.value_dimension ||
      shape.head_dimension > 128) {
    throw std::invalid_argument(
        "FP8 SDPA backward requires matching K/V heads and D == V <= 128");
  }
  const Shape output_shape{shape.batch,
                           shape.query_heads,
                           shape.sequence_q,
                           shape.value_dimension};
  for (const auto& [tensor_specification, expected_shape, name] :
       std::vector<std::tuple<const TestTensor*, Shape, const char*>>{
           {&test_case.output, output_shape, "FP8 SDPA backward O"},
           {&test_case.doutput, output_shape, "FP8 SDPA backward dO"},
           {&test_case.dq, test_case.q.dimensions, "FP8 SDPA backward dQ"},
           {&test_case.dk, test_case.k.dimensions, "FP8 SDPA backward dK"},
           {&test_case.dv, test_case.v.dimensions, "FP8 SDPA backward dV"}}) {
    validate_tensor(*tensor_specification, name, test_case.q.data_type);
    if (tensor_specification->dimensions != expected_shape) {
      throw std::invalid_argument(std::string(name) + " shape is invalid");
    }
  }
  validate_tensor(test_case.stats,
                  "FP8 SDPA backward stats",
                  FLAGDNN_DATA_FLOAT32);
  if (test_case.stats.dimensions !=
      Shape{shape.batch, shape.query_heads, shape.sequence_q, 1}) {
    throw std::invalid_argument("FP8 SDPA backward stats shape is invalid");
  }
  const std::array<std::pair<const Fp8Scalar*, const char*>, 12> scales{{
      {&test_case.descale_q, "FP8 SDPA backward descale Q"},
      {&test_case.descale_k, "FP8 SDPA backward descale K"},
      {&test_case.descale_v, "FP8 SDPA backward descale V"},
      {&test_case.descale_o, "FP8 SDPA backward descale O"},
      {&test_case.descale_doutput, "FP8 SDPA backward descale dO"},
      {&test_case.descale_s, "FP8 SDPA backward descale S"},
      {&test_case.descale_dp, "FP8 SDPA backward descale dP"},
      {&test_case.scale_s, "FP8 SDPA backward scale S"},
      {&test_case.scale_dq, "FP8 SDPA backward scale dQ"},
      {&test_case.scale_dk, "FP8 SDPA backward scale dK"},
      {&test_case.scale_dv, "FP8 SDPA backward scale dV"},
      {&test_case.scale_dp, "FP8 SDPA backward scale dP"},
  }};
  for (const auto& [scalar_value, name] : scales) {
    validate_fp8_scalar(*scalar_value, name);
  }
  const std::array<std::pair<const TestTensor*, const char*>, 4> amaxes{{
      {&test_case.amax_dq, "FP8 SDPA backward amax dQ"},
      {&test_case.amax_dk, "FP8 SDPA backward amax dK"},
      {&test_case.amax_dv, "FP8 SDPA backward amax dV"},
      {&test_case.amax_dp, "FP8 SDPA backward amax dP"},
  }};
  for (const auto& [amax, name] : amaxes) {
    validate_tensor(*amax, name, FLAGDNN_DATA_FLOAT32);
    if (amax->dimensions != Shape{1, 1, 1, 1}) {
      throw std::invalid_argument(std::string(name) + " shape is invalid");
    }
  }
  std::vector<const TestTensor*> tensors{
      &test_case.q,
      &test_case.k,
      &test_case.v,
      &test_case.output,
      &test_case.doutput,
      &test_case.stats};
  for (const auto& [scalar_value, ignored_name] : scales) {
    static_cast<void>(ignored_name);
    tensors.push_back(&scalar_value->tensor);
  }
  tensors.insert(tensors.end(),
                 {&test_case.dq,
                  &test_case.dk,
                  &test_case.dv,
                  &test_case.amax_dq,
                  &test_case.amax_dk,
                  &test_case.amax_dv,
                  &test_case.amax_dp});
  validate_unique_uids(tensors);
}

std::unique_ptr<AttentionExecutable> build_flagdnn_sdpa(
    flagdnn::Handle& handle,
    const SdpaTestCase& test_case) {
  validate_sdpa_case(test_case);
  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_name(test_case.name)
      .set_io_data_type(frontend_data_type(test_case.q.data_type))
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT)
      .set_autotune(test_case.autotune);
  const auto q = make_tensor(graph, test_case.q, "q");
  const auto k = make_tensor(graph, test_case.k, "k");
  const auto v = make_tensor(graph, test_case.v, "v");
  fe::graph::SDPA_attributes attributes;
  attributes.set_name("sdpa")
      .set_compute_data_type(fe::DataType_t::FLOAT)
      .set_generate_stats(test_case.stats.has_value());
  apply_options(attributes, test_case.options);
  if (test_case.bias.has_value()) {
    attributes.set_bias(make_tensor(graph, *test_case.bias, "bias"));
  }
  auto result = graph->sdpa(q, k, v, attributes);
  set_output(result[0], test_case.output, "output");
  if (test_case.stats.has_value()) {
    if (result[1] == nullptr) {
      throw std::logic_error("FlagDNN SDPA did not return requested stats");
    }
    set_output(result[1], *test_case.stats, "stats");
  } else if (result[1] != nullptr) {
    throw std::logic_error("FlagDNN SDPA returned unrequested stats");
  }
  return std::make_unique<FlagdnnAttentionExecutable>(
      handle, std::move(graph), "SDPA");
}

std::unique_ptr<AttentionExecutable> build_flagdnn_sdpa_backward(
    flagdnn::Handle& handle,
    const SdpaBackwardTestCase& test_case) {
  validate_sdpa_backward_case(test_case);
  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_name(test_case.name)
      .set_io_data_type(frontend_data_type(test_case.q.data_type))
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT)
      .set_autotune(test_case.autotune);
  const auto q = make_tensor(graph, test_case.q, "q");
  const auto k = make_tensor(graph, test_case.k, "k");
  const auto v = make_tensor(graph, test_case.v, "v");
  const auto output = make_tensor(graph, test_case.output, "output");
  const auto doutput = make_tensor(graph, test_case.doutput, "doutput");
  const auto stats = make_tensor(graph, test_case.stats, "stats");
  fe::graph::SDPA_backward_attributes attributes;
  attributes.set_name("sdpa_backward")
      .set_compute_data_type(fe::DataType_t::FLOAT)
      .set_deterministic_algorithm(test_case.deterministic);
  apply_options(attributes, test_case.options);
  if (test_case.bias.has_value()) {
    attributes.set_bias(make_tensor(graph, *test_case.bias, "bias"));
  }
  if (test_case.dbias.has_value()) {
    auto dbias = make_tensor(graph, *test_case.dbias, "dbias");
    dbias->set_output(true);
    attributes.set_dbias(std::move(dbias));
  }
  auto gradients = graph->sdpa_backward(
      q, k, v, output, doutput, stats, attributes);
  set_output(gradients[0], test_case.dq, "dq");
  set_output(gradients[1], test_case.dk, "dk");
  set_output(gradients[2], test_case.dv, "dv");
  return std::make_unique<FlagdnnAttentionExecutable>(
      handle, std::move(graph), "SDPA backward");
}

std::unique_ptr<AttentionExecutable> build_flagdnn_sdpa_fp8(
    flagdnn::Handle& handle,
    const SdpaFp8TestCase& test_case) {
  validate_sdpa_fp8_case(test_case);
  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_name(test_case.name)
      .set_io_data_type(frontend_data_type(test_case.q.data_type))
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT)
      .set_autotune(test_case.autotune);
  const auto q = make_tensor(graph, test_case.q, "q");
  const auto k = make_tensor(graph, test_case.k, "k");
  const auto v = make_tensor(graph, test_case.v, "v");
  const auto descale_q =
      make_tensor(graph, test_case.descale_q.tensor, "descale_q");
  const auto descale_k =
      make_tensor(graph, test_case.descale_k.tensor, "descale_k");
  const auto descale_v =
      make_tensor(graph, test_case.descale_v.tensor, "descale_v");
  const auto descale_s =
      make_tensor(graph, test_case.descale_s.tensor, "descale_s");
  const auto scale_s =
      make_tensor(graph, test_case.scale_s.tensor, "scale_s");
  const auto scale_o =
      make_tensor(graph, test_case.scale_o.tensor, "scale_o");
  fe::graph::SDPA_fp8_attributes attributes;
  attributes.set_name("sdpa_fp8")
      .set_compute_data_type(fe::DataType_t::FLOAT)
      .set_generate_stats(test_case.stats.has_value());
  apply_options(attributes, test_case.options);
  if (test_case.bias.has_value()) {
    attributes.set_bias(
        make_tensor(graph, *test_case.bias, "bias"));
  }
  auto result = graph->sdpa_fp8(q,
                                 k,
                                 v,
                                 descale_q,
                                 descale_k,
                                 descale_v,
                                 descale_s,
                                 scale_s,
                                 scale_o,
                                 attributes);
  set_output(result[0], test_case.output, "output");
  if (test_case.stats.has_value()) {
    if (result[1] == nullptr) {
      throw std::logic_error(
          "FlagDNN FP8 SDPA did not return requested stats");
    }
    set_output(result[1], *test_case.stats, "stats");
  } else if (result[1] != nullptr) {
    throw std::logic_error("FlagDNN FP8 SDPA returned unrequested stats");
  }
  set_output(result[2], test_case.amax_s, "amax_s");
  set_output(result[3], test_case.amax_o, "amax_o");
  return std::make_unique<FlagdnnAttentionExecutable>(
      handle, std::move(graph), "FP8 SDPA");
}

std::unique_ptr<AttentionExecutable> build_flagdnn_sdpa_fp8_backward(
    flagdnn::Handle& handle,
    const SdpaFp8BackwardTestCase& test_case) {
  validate_sdpa_fp8_backward_case(test_case);
  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_name(test_case.name)
      .set_io_data_type(frontend_data_type(test_case.q.data_type))
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT)
      .set_autotune(test_case.autotune);
  const auto q = make_tensor(graph, test_case.q, "q");
  const auto k = make_tensor(graph, test_case.k, "k");
  const auto v = make_tensor(graph, test_case.v, "v");
  const auto output = make_tensor(graph, test_case.output, "output");
  const auto doutput = make_tensor(graph, test_case.doutput, "doutput");
  const auto stats = make_tensor(graph, test_case.stats, "stats");
  const auto scalar = [&](const Fp8Scalar& value, std::string_view name) {
    return make_tensor(graph, value.tensor, name);
  };
  fe::graph::SDPA_fp8_backward_attributes attributes;
  attributes.set_name("sdpa_fp8_backward")
      .set_compute_data_type(fe::DataType_t::FLOAT);
  apply_options(attributes, test_case.options);
  auto result = graph->sdpa_fp8_backward(
      q,
      k,
      v,
      output,
      doutput,
      stats,
      scalar(test_case.descale_q, "descale_q"),
      scalar(test_case.descale_k, "descale_k"),
      scalar(test_case.descale_v, "descale_v"),
      scalar(test_case.descale_o, "descale_o"),
      scalar(test_case.descale_doutput, "descale_doutput"),
      scalar(test_case.descale_s, "descale_s"),
      scalar(test_case.descale_dp, "descale_dp"),
      scalar(test_case.scale_s, "scale_s"),
      scalar(test_case.scale_dq, "scale_dq"),
      scalar(test_case.scale_dk, "scale_dk"),
      scalar(test_case.scale_dv, "scale_dv"),
      scalar(test_case.scale_dp, "scale_dp"),
      attributes);
  set_output(result[0], test_case.dq, "dq");
  set_output(result[1], test_case.dk, "dk");
  set_output(result[2], test_case.dv, "dv");
  set_output(result[3], test_case.amax_dq, "amax_dq");
  set_output(result[4], test_case.amax_dk, "amax_dk");
  set_output(result[5], test_case.amax_dv, "amax_dv");
  set_output(result[6], test_case.amax_dp, "amax_dp");
  return std::make_unique<FlagdnnAttentionExecutable>(
      handle, std::move(graph), "FP8 SDPA backward");
}

}  // namespace flagdnn::testing
