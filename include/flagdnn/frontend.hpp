/*
 * Copyright (c) 2025-2026 BAAI. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef FLAGDNN_FRONTEND_HPP_
#define FLAGDNN_FRONTEND_HPP_

#include <flagdnn/flagdnn.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

/*
 * This is the header-only, cuDNN-Frontend-style layer of FlagDNN.  It lowers
 * graph and attribute objects to the stable C ABI in flagdnn.h; it is not a
 * second binary ABI and it does not embed Python or initialize Torch.
 *
 * flagdnn_frontend is the canonical public namespace. The historical
 * flagnn_frontend misspelling remains as a source-compatibility alias below.
 */
namespace flagdnn_frontend {

using Handle = flagdnn::Handle;
using VariantPack = std::unordered_map<std::int64_t, void*>;

enum class DataType_t {
  NOT_SET,
  FLOAT,
  HALF,
  BFLOAT16,
  BOOLEAN,
  FP8_E4M3,
  FP8_E5M2,
};

enum class PointwiseMode_t {
  NOT_SET,
  RELU_FWD,
  ADD,
  SQRT,
  ERF,
  IDENTITY,
  EXP,
  LOG,
  NEG,
  ABS,
  CEIL,
  COS,
  FLOOR,
  RSQRT,
  SIN,
  TAN,
  RECIPROCAL,
  SUB,
  MUL,
  DIV,
  MIN,
  MAX,
  MOD,
  POW,
  LOGICAL_NOT,
  CMP_EQ,
  CMP_NEQ,
  CMP_GT,
  CMP_GE,
  CMP_LT,
  CMP_LE,
  LOGICAL_AND,
  LOGICAL_OR,
  SIGMOID_FWD,
  SIGMOID_BWD,
  BINARY_SELECT,
  TANH_FWD,
  ELU_FWD,
  GELU_FWD,
  SOFTPLUS_FWD,
  SWISH_FWD,
  GELU_APPROX_TANH_FWD,
};

enum class ReductionMode_t {
  NOT_SET,
  ADD,
  AVG,
  MUL,
};

enum class ReshapeMode_t {
  NOT_SET,
  VIEW_ONLY,
  LOGICAL,
};

enum class ConvolutionMode_t {
  CROSS_CORRELATION,
  CONVOLUTION,
};

enum class NormFwdPhase_t {
  NOT_SET,
  INFERENCE,
  TRAINING,
};

enum class DiagonalAlignment_t {
  TOP_LEFT,
  BOTTOM_RIGHT,
};

/*
 * Backend-neutral execution-plan candidate sources. These values are lowered
 * to versioned FlagDNN build-option bits and are never passed to a vendor API.
 */
enum class HeurMode_t {
  A,
  FALLBACK,
};

enum class BuildPlanPolicy_t {
  HEURISTICS_CHOICE,
};

class error_t {
 public:
  error_t() = default;

  error_t(flagdnnStatus_t status, std::string message)
      : status_(status), message_(std::move(message)) {}

  [[nodiscard]] bool is_bad() const noexcept {
    return status_ != FLAGDNN_STATUS_SUCCESS;
  }

  [[nodiscard]] bool is_good() const noexcept { return !is_bad(); }

  [[nodiscard]] flagdnnStatus_t get_status() const noexcept {
    return status_;
  }

  [[nodiscard]] const std::string& get_message() const noexcept {
    return message_;
  }

 private:
  flagdnnStatus_t status_ = FLAGDNN_STATUS_SUCCESS;
  std::string message_;
};

namespace detail {

inline error_t current_exception_as_error() noexcept {
  try {
    throw;
  } catch (const flagdnn::Error& error) {
    return {error.status(), error.what()};
  } catch (const std::invalid_argument& error) {
    return {FLAGDNN_STATUS_INVALID_VALUE, error.what()};
  } catch (const std::bad_alloc& error) {
    return {FLAGDNN_STATUS_ALLOC_FAILED, error.what()};
  } catch (const std::logic_error& error) {
    return {FLAGDNN_STATUS_NOT_INITIALIZED, error.what()};
  } catch (const std::exception& error) {
    return {FLAGDNN_STATUS_INTERNAL_ERROR, error.what()};
  } catch (...) {
    return {FLAGDNN_STATUS_INTERNAL_ERROR,
            "unknown FlagDNN frontend error"};
  }
}

}  // namespace detail

namespace graph {

enum class ScalarType {
  RUNTIME_PARAM,
  COMPILE_TIME_CONST,
};

class Tensor_attributes {
 public:
  Tensor_attributes& set_name(std::string name) {
    name_ = std::move(name);
    return *this;
  }

  Tensor_attributes& set_data_type(DataType_t data_type) noexcept {
    data_type_ = data_type;
    return *this;
  }

  Tensor_attributes& set_dim(std::vector<std::int64_t> dimensions) {
    dimensions_ = std::move(dimensions);
    return *this;
  }

  Tensor_attributes& set_dim(
      std::initializer_list<std::int64_t> dimensions) {
    dimensions_.assign(dimensions);
    return *this;
  }

  Tensor_attributes& set_stride(std::vector<std::int64_t> strides) {
    strides_ = std::move(strides);
    return *this;
  }

  Tensor_attributes& set_stride(
      std::initializer_list<std::int64_t> strides) {
    strides_.assign(strides);
    return *this;
  }

  Tensor_attributes& set_is_virtual(bool is_virtual) noexcept {
    is_virtual_ = is_virtual;
    return *this;
  }

  Tensor_attributes& set_output(bool output = true) noexcept {
    is_virtual_ = !output;
    return *this;
  }

  Tensor_attributes& set_uid(std::int64_t uid) noexcept {
    uid_ = uid;
    return *this;
  }

  Tensor_attributes& set_alignment(std::int64_t alignment) noexcept {
    alignment_ = alignment;
    return *this;
  }

  [[nodiscard]] const std::string& get_name() const noexcept {
    return name_;
  }

  [[nodiscard]] DataType_t get_data_type() const noexcept {
    return data_type_;
  }

  [[nodiscard]] const std::vector<std::int64_t>& get_dim() const noexcept {
    return dimensions_;
  }

  [[nodiscard]] const std::vector<std::int64_t>& get_stride() const noexcept {
    return strides_;
  }

  [[nodiscard]] bool get_is_virtual() const noexcept {
    return is_virtual_;
  }

  [[nodiscard]] std::int64_t get_uid() const noexcept { return uid_; }

  [[nodiscard]] std::int64_t get_alignment() const noexcept {
    return alignment_;
  }

  [[nodiscard]] bool is_scalar() const noexcept {
    return scalar_value_.has_value();
  }

  [[nodiscard]] std::optional<double> get_scalar_value() const noexcept {
    return scalar_value_;
  }

  [[nodiscard]] std::optional<ScalarType> get_scalar_type() const noexcept {
    return scalar_type_;
  }

 private:
  friend class Graph;

  std::string name_;
  DataType_t data_type_ = DataType_t::NOT_SET;
  std::vector<std::int64_t> dimensions_;
  std::vector<std::int64_t> strides_;
  bool is_virtual_ = false;
  std::int64_t uid_ = 0;
  std::int64_t alignment_ = 16;
  std::optional<double> scalar_value_;
  std::optional<ScalarType> scalar_type_;
};

class Pointwise_attributes {
 public:
  Pointwise_attributes& set_name(std::string name) {
    name_ = std::move(name);
    return *this;
  }

  Pointwise_attributes& set_mode(PointwiseMode_t mode) noexcept {
    mode_ = mode;
    return *this;
  }

  Pointwise_attributes& set_compute_data_type(
      DataType_t data_type) noexcept {
    compute_data_type_ = data_type;
    return *this;
  }

  /*
   * FlagDNN extension for ADD/SUB right-operand scaling.  The default is the
   * corresponding cuDNN pointwise behavior.
   */
  Pointwise_attributes& set_alpha(double alpha) noexcept {
    alpha_ = alpha;
    return *this;
  }

  Pointwise_attributes& set_relu_lower_clip_slope(
      float const negative_slope) noexcept {
    relu_lower_clip_slope_ = negative_slope;
    return *this;
  }

  Pointwise_attributes& set_relu_lower_clip(float const value) noexcept {
    relu_lower_clip_ = value;
    return *this;
  }

  Pointwise_attributes& set_relu_upper_clip(float const value) noexcept {
    relu_upper_clip_ = value;
    return *this;
  }

  Pointwise_attributes& set_swish_beta(float const value) noexcept {
    swish_beta_ = value;
    return *this;
  }

  [[nodiscard]] std::optional<float> get_swish_beta() const noexcept {
    return swish_beta_;
  }

  Pointwise_attributes& set_elu_alpha(float const value) noexcept {
    elu_alpha_ = value;
    return *this;
  }

  Pointwise_attributes& set_softplus_beta(float const value) noexcept {
    softplus_beta_ = value;
    return *this;
  }

  [[nodiscard]] const std::string& get_name() const noexcept {
    return name_;
  }

  [[nodiscard]] PointwiseMode_t get_mode() const noexcept { return mode_; }

  [[nodiscard]] DataType_t get_compute_data_type() const noexcept {
    return compute_data_type_;
  }

  [[nodiscard]] double get_alpha() const noexcept { return alpha_; }

 private:
  friend class Graph;

  std::string name_;
  PointwiseMode_t mode_ = PointwiseMode_t::NOT_SET;
  DataType_t compute_data_type_ = DataType_t::NOT_SET;
  double alpha_ = 1.0;
  std::optional<float> relu_lower_clip_;
  std::optional<float> relu_upper_clip_;
  std::optional<float> relu_lower_clip_slope_;
  std::optional<float> swish_beta_;
  std::optional<float> elu_alpha_;
  std::optional<float> softplus_beta_;
};

class Reduction_attributes {
 public:
  Reduction_attributes& set_name(std::string name) {
    name_ = std::move(name);
    return *this;
  }

  Reduction_attributes& set_mode(ReductionMode_t mode) noexcept {
    mode_ = mode;
    return *this;
  }

  Reduction_attributes& set_compute_data_type(
      DataType_t data_type) noexcept {
    compute_data_type_ = data_type;
    return *this;
  }

  /*
   * The current stable C ABI reduces one axis at a time.  These two setters
   * are FlagDNN extensions; other names intentionally match cuDNN Frontend.
   */
  Reduction_attributes& set_axis(std::int64_t axis) noexcept {
    axis_ = axis;
    return *this;
  }

  Reduction_attributes& set_keep_dimensions(
      bool keep_dimensions) noexcept {
    keep_dimensions_ = keep_dimensions;
    return *this;
  }

  [[nodiscard]] const std::string& get_name() const noexcept {
    return name_;
  }

  [[nodiscard]] ReductionMode_t get_mode() const noexcept { return mode_; }

  [[nodiscard]] DataType_t get_compute_data_type() const noexcept {
    return compute_data_type_;
  }

  [[nodiscard]] std::int64_t get_axis() const noexcept { return axis_; }

  [[nodiscard]] bool get_keep_dimensions() const noexcept {
    return keep_dimensions_;
  }

 private:
  std::string name_;
  ReductionMode_t mode_ = ReductionMode_t::NOT_SET;
  DataType_t compute_data_type_ = DataType_t::NOT_SET;
  std::int64_t axis_ = -1;
  bool keep_dimensions_ = false;
};

class Conv_fprop_attributes {
 public:
  Conv_fprop_attributes& set_name(std::string name) {
    name_ = std::move(name);
    return *this;
  }

  Conv_fprop_attributes& set_compute_data_type(
      DataType_t data_type) noexcept {
    compute_data_type_ = data_type;
    return *this;
  }

  Conv_fprop_attributes& set_padding(
      std::vector<std::int64_t> padding) {
    pre_padding_ = padding;
    post_padding_ = std::move(padding);
    return *this;
  }

  Conv_fprop_attributes& set_pre_padding(
      std::vector<std::int64_t> padding) {
    pre_padding_ = std::move(padding);
    return *this;
  }

  Conv_fprop_attributes& set_post_padding(
      std::vector<std::int64_t> padding) {
    post_padding_ = std::move(padding);
    return *this;
  }

  Conv_fprop_attributes& set_stride(std::vector<std::int64_t> stride) {
    stride_ = std::move(stride);
    return *this;
  }

  Conv_fprop_attributes& set_dilation(
      std::vector<std::int64_t> dilation) {
    dilation_ = std::move(dilation);
    return *this;
  }

  Conv_fprop_attributes& set_convolution_mode(
      ConvolutionMode_t mode) noexcept {
    convolution_mode_ = mode;
    return *this;
  }

  /*
   * FlagDNN exposes groups explicitly because its backend-neutral tensor
   * contract does not infer grouped convolution from a backend descriptor.
   */
  Conv_fprop_attributes& set_groups(std::int64_t groups) noexcept {
    groups_ = groups;
    return *this;
  }

  [[nodiscard]] const std::string& get_name() const noexcept {
    return name_;
  }

  [[nodiscard]] DataType_t get_compute_data_type() const noexcept {
    return compute_data_type_;
  }

  [[nodiscard]] const std::vector<std::int64_t>& get_pre_padding()
      const noexcept {
    return pre_padding_;
  }

  [[nodiscard]] const std::vector<std::int64_t>& get_post_padding()
      const noexcept {
    return post_padding_;
  }

  [[nodiscard]] const std::vector<std::int64_t>& get_stride()
      const noexcept {
    return stride_;
  }

  [[nodiscard]] const std::vector<std::int64_t>& get_dilation()
      const noexcept {
    return dilation_;
  }

  [[nodiscard]] ConvolutionMode_t get_convolution_mode() const noexcept {
    return convolution_mode_;
  }

  [[nodiscard]] std::int64_t get_groups() const noexcept {
    return groups_;
  }

 private:
  std::string name_;
  DataType_t compute_data_type_ = DataType_t::NOT_SET;
  std::vector<std::int64_t> pre_padding_;
  std::vector<std::int64_t> post_padding_;
  std::vector<std::int64_t> stride_;
  std::vector<std::int64_t> dilation_;
  ConvolutionMode_t convolution_mode_ =
      ConvolutionMode_t::CROSS_CORRELATION;
  std::int64_t groups_ = 1;
};

class Conv_dgrad_attributes {
 public:
  Conv_dgrad_attributes& set_name(std::string name) {
    name_ = std::move(name);
    return *this;
  }

  Conv_dgrad_attributes& set_compute_data_type(
      DataType_t data_type) noexcept {
    compute_data_type_ = data_type;
    return *this;
  }

  Conv_dgrad_attributes& set_padding(
      std::vector<std::int64_t> padding) {
    pre_padding_ = padding;
    post_padding_ = std::move(padding);
    return *this;
  }

  Conv_dgrad_attributes& set_pre_padding(
      std::vector<std::int64_t> padding) {
    pre_padding_ = std::move(padding);
    return *this;
  }

  Conv_dgrad_attributes& set_post_padding(
      std::vector<std::int64_t> padding) {
    post_padding_ = std::move(padding);
    return *this;
  }

  Conv_dgrad_attributes& set_stride(
      std::vector<std::int64_t> stride) {
    stride_ = std::move(stride);
    return *this;
  }

  Conv_dgrad_attributes& set_dilation(
      std::vector<std::int64_t> dilation) {
    dilation_ = std::move(dilation);
    return *this;
  }

  Conv_dgrad_attributes& set_convolution_mode(
      ConvolutionMode_t mode) noexcept {
    convolution_mode_ = mode;
    return *this;
  }

  Conv_dgrad_attributes& set_groups(std::int64_t groups) noexcept {
    groups_ = groups;
    return *this;
  }

  [[nodiscard]] const std::string& get_name() const noexcept {
    return name_;
  }

  [[nodiscard]] DataType_t get_compute_data_type() const noexcept {
    return compute_data_type_;
  }

  [[nodiscard]] const std::vector<std::int64_t>& get_pre_padding()
      const noexcept {
    return pre_padding_;
  }

  [[nodiscard]] const std::vector<std::int64_t>& get_post_padding()
      const noexcept {
    return post_padding_;
  }

  [[nodiscard]] const std::vector<std::int64_t>& get_stride()
      const noexcept {
    return stride_;
  }

  [[nodiscard]] const std::vector<std::int64_t>& get_dilation()
      const noexcept {
    return dilation_;
  }

  [[nodiscard]] ConvolutionMode_t get_convolution_mode() const noexcept {
    return convolution_mode_;
  }

  [[nodiscard]] std::int64_t get_groups() const noexcept {
    return groups_;
  }

 private:
  std::string name_;
  DataType_t compute_data_type_ = DataType_t::NOT_SET;
  std::vector<std::int64_t> pre_padding_;
  std::vector<std::int64_t> post_padding_;
  std::vector<std::int64_t> stride_;
  std::vector<std::int64_t> dilation_;
  ConvolutionMode_t convolution_mode_ =
      ConvolutionMode_t::CROSS_CORRELATION;
  std::int64_t groups_ = 1;
};

class Conv_wgrad_attributes {
 public:
  Conv_wgrad_attributes& set_name(std::string name) {
    name_ = std::move(name);
    return *this;
  }

  Conv_wgrad_attributes& set_compute_data_type(
      DataType_t data_type) noexcept {
    compute_data_type_ = data_type;
    return *this;
  }

  Conv_wgrad_attributes& set_padding(
      std::vector<std::int64_t> padding) {
    pre_padding_ = padding;
    post_padding_ = std::move(padding);
    return *this;
  }

  Conv_wgrad_attributes& set_pre_padding(
      std::vector<std::int64_t> padding) {
    pre_padding_ = std::move(padding);
    return *this;
  }

  Conv_wgrad_attributes& set_post_padding(
      std::vector<std::int64_t> padding) {
    post_padding_ = std::move(padding);
    return *this;
  }

  Conv_wgrad_attributes& set_stride(
      std::vector<std::int64_t> stride) {
    stride_ = std::move(stride);
    return *this;
  }

  Conv_wgrad_attributes& set_dilation(
      std::vector<std::int64_t> dilation) {
    dilation_ = std::move(dilation);
    return *this;
  }

  Conv_wgrad_attributes& set_convolution_mode(
      ConvolutionMode_t mode) noexcept {
    convolution_mode_ = mode;
    return *this;
  }

  Conv_wgrad_attributes& set_groups(std::int64_t groups) noexcept {
    groups_ = groups;
    return *this;
  }

  [[nodiscard]] const std::string& get_name() const noexcept {
    return name_;
  }

  [[nodiscard]] DataType_t get_compute_data_type() const noexcept {
    return compute_data_type_;
  }

  [[nodiscard]] const std::vector<std::int64_t>& get_pre_padding()
      const noexcept {
    return pre_padding_;
  }

  [[nodiscard]] const std::vector<std::int64_t>& get_post_padding()
      const noexcept {
    return post_padding_;
  }

  [[nodiscard]] const std::vector<std::int64_t>& get_stride()
      const noexcept {
    return stride_;
  }

  [[nodiscard]] const std::vector<std::int64_t>& get_dilation()
      const noexcept {
    return dilation_;
  }

  [[nodiscard]] ConvolutionMode_t get_convolution_mode() const noexcept {
    return convolution_mode_;
  }

  [[nodiscard]] std::int64_t get_groups() const noexcept {
    return groups_;
  }

 private:
  std::string name_;
  DataType_t compute_data_type_ = DataType_t::NOT_SET;
  std::vector<std::int64_t> pre_padding_;
  std::vector<std::int64_t> post_padding_;
  std::vector<std::int64_t> stride_;
  std::vector<std::int64_t> dilation_;
  ConvolutionMode_t convolution_mode_ =
      ConvolutionMode_t::CROSS_CORRELATION;
  std::int64_t groups_ = 1;
};

class Layernorm_attributes {
 public:
  Layernorm_attributes& set_name(std::string name) {
    name_ = std::move(name);
    return *this;
  }

  Layernorm_attributes& set_compute_data_type(
      DataType_t data_type) noexcept {
    compute_data_type_ = data_type;
    return *this;
  }

  Layernorm_attributes& set_forward_phase(
      NormFwdPhase_t phase) noexcept {
    forward_phase_ = phase;
    return *this;
  }

  Layernorm_attributes& set_epsilon(
      std::shared_ptr<Tensor_attributes>& epsilon) {
    epsilon_ = epsilon;
    epsilon_value_.reset();
    return *this;
  }

  Layernorm_attributes& set_epsilon(float epsilon) {
    epsilon_.reset();
    epsilon_value_ = static_cast<double>(epsilon);
    return *this;
  }

  [[nodiscard]] const std::string& get_name() const noexcept {
    return name_;
  }

  [[nodiscard]] DataType_t get_compute_data_type() const noexcept {
    return compute_data_type_;
  }

  [[nodiscard]] NormFwdPhase_t get_forward_phase() const noexcept {
    return forward_phase_;
  }

 private:
  friend class Graph;

  std::string name_;
  DataType_t compute_data_type_ = DataType_t::NOT_SET;
  NormFwdPhase_t forward_phase_ = NormFwdPhase_t::NOT_SET;
  std::shared_ptr<Tensor_attributes> epsilon_;
  std::optional<double> epsilon_value_;
};

class Rmsnorm_attributes {
 public:
  Rmsnorm_attributes& set_name(std::string name) {
    name_ = std::move(name);
    return *this;
  }

  Rmsnorm_attributes& set_compute_data_type(
      DataType_t data_type) noexcept {
    compute_data_type_ = data_type;
    return *this;
  }

  Rmsnorm_attributes& set_forward_phase(
      NormFwdPhase_t phase) noexcept {
    forward_phase_ = phase;
    return *this;
  }

  Rmsnorm_attributes& set_bias(
      std::shared_ptr<Tensor_attributes>& bias) {
    bias_ = bias;
    return *this;
  }

  Rmsnorm_attributes& set_epsilon(
      std::shared_ptr<Tensor_attributes>& epsilon) {
    epsilon_ = epsilon;
    return *this;
  }

  [[nodiscard]] const std::string& get_name() const noexcept {
    return name_;
  }

  [[nodiscard]] DataType_t get_compute_data_type() const noexcept {
    return compute_data_type_;
  }

  [[nodiscard]] NormFwdPhase_t get_forward_phase() const noexcept {
    return forward_phase_;
  }

 private:
  friend class Graph;

  std::string name_;
  DataType_t compute_data_type_ = DataType_t::NOT_SET;
  NormFwdPhase_t forward_phase_ = NormFwdPhase_t::NOT_SET;
  std::shared_ptr<Tensor_attributes> bias_;
  std::shared_ptr<Tensor_attributes> epsilon_;
};

class Batchnorm_attributes {
 public:
  Batchnorm_attributes& set_name(std::string name) {
    name_ = std::move(name);
    return *this;
  }

  Batchnorm_attributes& set_compute_data_type(
      DataType_t data_type) noexcept {
    compute_data_type_ = data_type;
    return *this;
  }

  Batchnorm_attributes& set_previous_running_stats(
      std::shared_ptr<Tensor_attributes>& mean,
      std::shared_ptr<Tensor_attributes>& variance,
      std::shared_ptr<Tensor_attributes>& momentum) {
    previous_running_mean_ = mean;
    previous_running_variance_ = variance;
    momentum_ = momentum;
    return *this;
  }

  Batchnorm_attributes& set_epsilon(
      std::shared_ptr<Tensor_attributes>& epsilon) {
    epsilon_ = epsilon;
    return *this;
  }

  Batchnorm_attributes& set_peer_stats(
      const std::vector<std::shared_ptr<Tensor_attributes>>& peer_stats) {
    peer_stats_ = peer_stats;
    return *this;
  }

  [[nodiscard]] const std::string& get_name() const noexcept {
    return name_;
  }

  [[nodiscard]] DataType_t get_compute_data_type() const noexcept {
    return compute_data_type_;
  }

 private:
  friend class Graph;

  std::string name_;
  DataType_t compute_data_type_ = DataType_t::NOT_SET;
  std::shared_ptr<Tensor_attributes> previous_running_mean_;
  std::shared_ptr<Tensor_attributes> previous_running_variance_;
  std::shared_ptr<Tensor_attributes> momentum_;
  std::shared_ptr<Tensor_attributes> epsilon_;
  std::vector<std::shared_ptr<Tensor_attributes>> peer_stats_;
};

class Batchnorm_inference_attributes {
 public:
  Batchnorm_inference_attributes& set_name(std::string name) {
    name_ = std::move(name);
    return *this;
  }

  Batchnorm_inference_attributes& set_compute_data_type(
      DataType_t data_type) noexcept {
    compute_data_type_ = data_type;
    return *this;
  }

  [[nodiscard]] const std::string& get_name() const noexcept {
    return name_;
  }

  [[nodiscard]] DataType_t get_compute_data_type() const noexcept {
    return compute_data_type_;
  }

 private:
  std::string name_;
  DataType_t compute_data_type_ = DataType_t::NOT_SET;
};

class Matmul_attributes {
 public:
  Matmul_attributes& set_name(std::string name) {
    name_ = std::move(name);
    return *this;
  }

  Matmul_attributes& set_compute_data_type(
      DataType_t data_type) noexcept {
    compute_data_type_ = data_type;
    return *this;
  }

  Matmul_attributes& set_padding(double value) noexcept {
    padding_ = value;
    return *this;
  }

  [[nodiscard]] const std::string& get_name() const noexcept {
    return name_;
  }

  [[nodiscard]] DataType_t get_compute_data_type() const noexcept {
    return compute_data_type_;
  }

  [[nodiscard]] double get_padding() const noexcept { return padding_; }

 private:
  std::string name_;
  DataType_t compute_data_type_ = DataType_t::NOT_SET;
  double padding_ = 0.0;
};

class Reshape_attributes {
 public:
  Reshape_attributes& set_name(std::string name) {
    name_ = std::move(name);
    return *this;
  }

  Reshape_attributes& set_compute_data_type(
      DataType_t data_type) noexcept {
    compute_data_type_ = data_type;
    return *this;
  }

  Reshape_attributes& set_dim(std::vector<std::int64_t> dimensions) {
    dimensions_ = std::move(dimensions);
    return *this;
  }

  Reshape_attributes& set_stride(std::vector<std::int64_t> strides) {
    strides_ = std::move(strides);
    return *this;
  }

  Reshape_attributes& set_reshape_mode(
      ReshapeMode_t mode) noexcept {
    reshape_mode_ = mode;
    return *this;
  }

  [[nodiscard]] const std::string& get_name() const noexcept {
    return name_;
  }

  [[nodiscard]] DataType_t get_compute_data_type() const noexcept {
    return compute_data_type_;
  }

  [[nodiscard]] const std::vector<std::int64_t>& get_dim()
      const noexcept {
    return dimensions_;
  }

  [[nodiscard]] const std::vector<std::int64_t>& get_stride()
      const noexcept {
    return strides_;
  }

  [[nodiscard]] ReshapeMode_t get_reshape_mode() const noexcept {
    return reshape_mode_;
  }

 private:
  std::string name_;
  DataType_t compute_data_type_ = DataType_t::NOT_SET;
  std::vector<std::int64_t> dimensions_;
  std::vector<std::int64_t> strides_;
  ReshapeMode_t reshape_mode_ = ReshapeMode_t::VIEW_ONLY;
};

class Transpose_attributes {
 public:
  Transpose_attributes& set_name(std::string name) {
    name_ = std::move(name);
    return *this;
  }

  Transpose_attributes& set_compute_data_type(
      DataType_t data_type) noexcept {
    compute_data_type_ = data_type;
    return *this;
  }

  Transpose_attributes& set_permutation(
      std::vector<std::int64_t> permutation) {
    permutation_ = std::move(permutation);
    return *this;
  }

  [[nodiscard]] const std::string& get_name() const noexcept {
    return name_;
  }

  [[nodiscard]] DataType_t get_compute_data_type() const noexcept {
    return compute_data_type_;
  }

  [[nodiscard]] const std::vector<std::int64_t>& get_permutation()
      const noexcept {
    return permutation_;
  }

 private:
  std::string name_;
  DataType_t compute_data_type_ = DataType_t::NOT_SET;
  std::vector<std::int64_t> permutation_;
};

class Slice_attributes {
 public:
  Slice_attributes& set_name(std::string name) {
    name_ = std::move(name);
    return *this;
  }

  Slice_attributes& set_compute_data_type(
      DataType_t data_type) noexcept {
    compute_data_type_ = data_type;
    return *this;
  }

  Slice_attributes& set_slices(
      std::vector<std::pair<std::int64_t, std::int64_t>> slices) {
    slices_ = std::move(slices);
    return *this;
  }

  Slice_attributes& set_strides(
      std::vector<std::int64_t> strides) {
    slice_strides_ = std::move(strides);
    return *this;
  }

  [[nodiscard]] const std::string& get_name() const noexcept {
    return name_;
  }

  [[nodiscard]] DataType_t get_compute_data_type() const noexcept {
    return compute_data_type_;
  }

  [[nodiscard]] const std::vector<
      std::pair<std::int64_t, std::int64_t>>& get_slices()
      const noexcept {
    return slices_;
  }

  [[nodiscard]] const std::vector<std::int64_t>& get_strides()
      const noexcept {
    return slice_strides_;
  }

 private:
  std::string name_;
  DataType_t compute_data_type_ = DataType_t::NOT_SET;
  std::vector<std::pair<std::int64_t, std::int64_t>> slices_;
  std::vector<std::int64_t> slice_strides_;
};

class SDPA_attributes {
 public:
  using Tensor = std::shared_ptr<Tensor_attributes>;

  SDPA_attributes& set_name(std::string name) {
    name_ = std::move(name);
    return *this;
  }

  SDPA_attributes& set_compute_data_type(DataType_t data_type) noexcept {
    compute_data_type_ = data_type;
    return *this;
  }

  SDPA_attributes& set_generate_stats(bool value) noexcept {
    generate_stats_ = value;
    return *this;
  }

  SDPA_attributes& set_is_inference(bool value) noexcept {
    generate_stats_ = !value;
    return *this;
  }

  SDPA_attributes& set_attn_scale(float value) {
    if (!std::isfinite(value) || value <= 0.0F) {
      throw std::invalid_argument(
          "SDPA attention scale must be positive and finite");
    }
    attn_scale_ = value;
    return *this;
  }

  SDPA_attributes& set_bias(Tensor value) {
    bias_ = std::move(value);
    return *this;
  }

  SDPA_attributes& set_diagonal_alignment(
      DiagonalAlignment_t value) noexcept {
    diagonal_alignment_ = value;
    return *this;
  }

  SDPA_attributes& set_diagonal_band_left_bound(std::int64_t value) {
    if (value < 1) {
      throw std::invalid_argument(
          "SDPA diagonal left bound must be at least one");
    }
    left_bound_ = value;
    return *this;
  }

  SDPA_attributes& set_diagonal_band_right_bound(std::int64_t value) {
    if (value < 0) {
      throw std::invalid_argument(
          "SDPA diagonal right bound must be nonnegative");
    }
    right_bound_ = value;
    return *this;
  }

  SDPA_attributes& set_causal_mask(bool value) {
    if (value) {
      diagonal_alignment_ = DiagonalAlignment_t::TOP_LEFT;
      right_bound_ = 0;
    }
    return *this;
  }

  SDPA_attributes& set_causal_mask_bottom_right(bool value) {
    if (value) {
      diagonal_alignment_ = DiagonalAlignment_t::BOTTOM_RIGHT;
      right_bound_ = 0;
    }
    return *this;
  }

  SDPA_attributes& set_sliding_window_length(std::int64_t value) {
    return set_diagonal_band_left_bound(value);
  }

  [[nodiscard]] const std::string& get_name() const noexcept {
    return name_;
  }
  [[nodiscard]] DataType_t get_compute_data_type() const noexcept {
    return compute_data_type_;
  }
  [[nodiscard]] bool get_generate_stats() const noexcept {
    return generate_stats_;
  }
  [[nodiscard]] const std::optional<float>& get_attn_scale() const noexcept {
    return attn_scale_;
  }
  [[nodiscard]] const Tensor& get_bias() const noexcept { return bias_; }
  [[nodiscard]] DiagonalAlignment_t get_diagonal_alignment() const noexcept {
    return diagonal_alignment_;
  }
  [[nodiscard]] const std::optional<std::int64_t>&
  get_diagonal_band_left_bound() const noexcept {
    return left_bound_;
  }
  [[nodiscard]] const std::optional<std::int64_t>&
  get_diagonal_band_right_bound() const noexcept {
    return right_bound_;
  }

 private:
  friend class Graph;
  std::string name_;
  DataType_t compute_data_type_ = DataType_t::NOT_SET;
  bool generate_stats_ = false;
  std::optional<float> attn_scale_;
  Tensor bias_;
  std::optional<std::int64_t> left_bound_;
  std::optional<std::int64_t> right_bound_;
  DiagonalAlignment_t diagonal_alignment_ =
      DiagonalAlignment_t::TOP_LEFT;
};

class SDPA_backward_attributes {
 public:
  using Tensor = std::shared_ptr<Tensor_attributes>;

  SDPA_backward_attributes& set_name(std::string name) {
    name_ = std::move(name);
    return *this;
  }
  SDPA_backward_attributes& set_compute_data_type(
      DataType_t data_type) noexcept {
    compute_data_type_ = data_type;
    return *this;
  }
  SDPA_backward_attributes& set_attn_scale(float value) {
    if (!std::isfinite(value) || value <= 0.0F) {
      throw std::invalid_argument(
          "SDPA backward attention scale must be positive and finite");
    }
    attn_scale_ = value;
    return *this;
  }
  SDPA_backward_attributes& set_bias(Tensor value) {
    bias_ = std::move(value);
    return *this;
  }
  SDPA_backward_attributes& set_dbias(Tensor value) {
    dbias_ = std::move(value);
    return *this;
  }
  SDPA_backward_attributes& set_diagonal_alignment(
      DiagonalAlignment_t value) noexcept {
    diagonal_alignment_ = value;
    return *this;
  }
  SDPA_backward_attributes& set_diagonal_band_left_bound(
      std::int64_t value) {
    if (value < 1) {
      throw std::invalid_argument(
          "SDPA backward diagonal left bound must be at least one");
    }
    left_bound_ = value;
    return *this;
  }
  SDPA_backward_attributes& set_diagonal_band_right_bound(
      std::int64_t value) {
    if (value < 0) {
      throw std::invalid_argument(
          "SDPA backward diagonal right bound must be nonnegative");
    }
    right_bound_ = value;
    return *this;
  }
  SDPA_backward_attributes& set_causal_mask(bool value) {
    if (value) {
      diagonal_alignment_ = DiagonalAlignment_t::TOP_LEFT;
      right_bound_ = 0;
    }
    return *this;
  }
  SDPA_backward_attributes& set_causal_mask_bottom_right(bool value) {
    if (value) {
      diagonal_alignment_ = DiagonalAlignment_t::BOTTOM_RIGHT;
      right_bound_ = 0;
    }
    return *this;
  }
  SDPA_backward_attributes& set_sliding_window_length(
      std::int64_t value) {
    return set_diagonal_band_left_bound(value);
  }
  SDPA_backward_attributes& set_deterministic_algorithm(
      bool value) noexcept {
    deterministic_ = value;
    return *this;
  }

  [[nodiscard]] const std::string& get_name() const noexcept {
    return name_;
  }
  [[nodiscard]] DataType_t get_compute_data_type() const noexcept {
    return compute_data_type_;
  }
  [[nodiscard]] const std::optional<float>& get_attn_scale() const noexcept {
    return attn_scale_;
  }
  [[nodiscard]] const Tensor& get_bias() const noexcept { return bias_; }
  [[nodiscard]] const Tensor& get_dbias() const noexcept { return dbias_; }
  [[nodiscard]] DiagonalAlignment_t get_diagonal_alignment() const noexcept {
    return diagonal_alignment_;
  }
  [[nodiscard]] const std::optional<std::int64_t>&
  get_diagonal_band_left_bound() const noexcept {
    return left_bound_;
  }
  [[nodiscard]] const std::optional<std::int64_t>&
  get_diagonal_band_right_bound() const noexcept {
    return right_bound_;
  }
  [[nodiscard]] bool get_deterministic_algorithm() const noexcept {
    return deterministic_;
  }

 private:
  friend class Graph;
  std::string name_;
  DataType_t compute_data_type_ = DataType_t::NOT_SET;
  std::optional<float> attn_scale_;
  Tensor bias_;
  Tensor dbias_;
  std::optional<std::int64_t> left_bound_;
  std::optional<std::int64_t> right_bound_;
  DiagonalAlignment_t diagonal_alignment_ =
      DiagonalAlignment_t::TOP_LEFT;
  bool deterministic_ = false;
};

/* cuDNN Frontend keeps the forward FP8 attributes API aligned with SDPA. */
using SDPA_fp8_attributes = SDPA_attributes;
using SDPA_fp8_backward_attributes = SDPA_backward_attributes;

class Graph {
 private:
  enum class LifecycleState {
    kUnvalidated,
    kValidated,
    kOperationGraph,
    kExecutionPlans,
    kSupported,
    kBuilt,
  };

 public:
  using Tensor = std::shared_ptr<Tensor_attributes>;

  Graph() = default;
  ~Graph() = default;

  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;
  Graph(Graph&&) noexcept = default;
  Graph& operator=(Graph&&) noexcept = default;

  Graph& set_name(std::string name) {
    name_ = std::move(name);
    invalidate();
    return *this;
  }

  Graph& set_io_data_type(DataType_t data_type) noexcept {
    io_data_type_ = data_type;
    invalidate();
    return *this;
  }

  Graph& set_intermediate_data_type(DataType_t data_type) noexcept {
    intermediate_data_type_ = data_type;
    invalidate();
    return *this;
  }

  Graph& set_compute_data_type(DataType_t data_type) noexcept {
    compute_data_type_ = data_type;
    invalidate();
    return *this;
  }

  Graph& set_autotune(bool enabled) noexcept {
    autotune_ = enabled;
    invalidate();
    return *this;
  }

  [[nodiscard]] Tensor tensor(Tensor_attributes attributes) {
    auto result =
        std::make_shared<Tensor_attributes>(std::move(attributes));
    tensors_.push_back(result);
    invalidate();
    return result;
  }

  [[nodiscard]] Tensor tensor(float const& scalar,
                              ScalarType scalar_type) {
    if (!std::isfinite(scalar)) {
      throw std::invalid_argument("frontend scalar must be finite");
    }
    auto result = std::make_shared<Tensor_attributes>();
    result->set_name("scalar")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 1, 1, 1})
        .set_stride({1, 1, 1, 1});
    result->scalar_value_ = static_cast<double>(scalar);
    result->scalar_type_ = scalar_type;
    tensors_.push_back(result);
    invalidate();
    return result;
  }

  [[nodiscard]] Tensor pointwise(
      const Tensor& input,
      const Pointwise_attributes& attributes) {
    require_tensor(input, "pointwise input");
    if (!is_unary_pointwise_mode(attributes.get_mode())) {
      throw std::invalid_argument(
          "pointwise mode is not a supported unary operation");
    }
    Tensor output = inferred_output(*input, "pointwise_output");
    if (attributes.get_mode() == PointwiseMode_t::LOGICAL_NOT) {
      output->set_data_type(DataType_t::BOOLEAN);
    }
    nodes_.push_back(Node::make_pointwise(input, output, attributes));
    tensors_.push_back(output);
    invalidate();
    return output;
  }

  [[nodiscard]] Tensor pointwise(
      const Tensor& left,
      const Tensor& right,
      const Pointwise_attributes& attributes) {
    require_tensor(left, "left pointwise input");
    require_tensor(right, "right pointwise input");
    if (!is_binary_pointwise_mode(attributes.get_mode())) {
      throw std::invalid_argument(
          "pointwise mode is not a supported binary operation");
    }
    if (!std::isfinite(attributes.get_alpha())) {
      throw std::invalid_argument("pointwise alpha must be finite");
    }
    if (attributes.get_mode() != PointwiseMode_t::ADD &&
        attributes.get_mode() != PointwiseMode_t::SUB &&
        attributes.get_alpha() != 1.0) {
      throw std::invalid_argument(
          "pointwise alpha is only supported by ADD and SUB modes");
    }
    if (attributes.get_mode() == PointwiseMode_t::SIGMOID_BWD &&
        left->get_dim() != right->get_dim()) {
      throw std::invalid_argument(
          "sigmoid backward inputs must have equal shapes");
    }
    Tensor output =
        inferred_binary_output(*left, *right, attributes.get_mode());
    nodes_.push_back(
        Node::make_binary_pointwise(left, right, output, attributes));
    tensors_.push_back(output);
    invalidate();
    return output;
  }

  [[nodiscard]] Tensor pointwise(
      const Tensor& a,
      const Tensor& b,
      const Tensor& t,
      const Pointwise_attributes& attributes) {
    require_tensor(a, "ternary pointwise A input");
    require_tensor(b, "ternary pointwise B input");
    require_tensor(t, "ternary pointwise T input");
    if (!is_ternary_pointwise_mode(attributes.get_mode())) {
      throw std::invalid_argument(
          "pointwise mode is not a supported ternary operation");
    }
    if (attributes.get_alpha() != 1.0) {
      throw std::invalid_argument(
          "ternary pointwise alpha must use its default value");
    }
    if (a->get_data_type() != b->get_data_type()) {
      throw std::invalid_argument(
          "ternary pointwise A/B data types must match");
    }
    if (t->get_data_type() != DataType_t::BOOLEAN) {
      throw std::invalid_argument(
          "ternary pointwise T predicate must use BOOLEAN data type");
    }
    Tensor output = inferred_ternary_output(*a, *b, *t);
    nodes_.push_back(
        Node::make_ternary_pointwise(a, b, t, output, attributes));
    tensors_.push_back(output);
    invalidate();
    return output;
  }

  [[nodiscard]] std::array<Tensor, 3> layernorm(
      const Tensor& x,
      const Tensor& scale,
      const Tensor& bias,
      Layernorm_attributes attributes) {
    require_tensor(x, "layernorm X");
    require_tensor(scale, "layernorm scale");
    require_tensor(bias, "layernorm bias");
    if (attributes.get_forward_phase() != NormFwdPhase_t::TRAINING) {
      throw flagdnn::Error(
          FLAGDNN_STATUS_NOT_SUPPORTED,
          "FlagDNN layernorm currently supports TRAINING phase only");
    }
    if (attributes.epsilon_ == nullptr &&
        attributes.epsilon_value_.has_value()) {
      attributes.epsilon_ = tensor(
          static_cast<float>(*attributes.epsilon_value_),
          ScalarType::COMPILE_TIME_CONST);
    }
    require_tensor(attributes.epsilon_, "layernorm epsilon");
    Tensor y = inferred_output(*x, "layernorm_output");
    Tensor mean = inferred_norm_stat_output(
        *x, *scale, "layernorm_mean");
    Tensor inv_variance = inferred_norm_stat_output(
        *x, *scale, "layernorm_inv_variance");
    nodes_.push_back(Node::make_layernorm(
        x, scale, bias, y, mean, inv_variance, attributes));
    tensors_.push_back(y);
    tensors_.push_back(mean);
    tensors_.push_back(inv_variance);
    invalidate();
    return {y, mean, inv_variance};
  }

  [[nodiscard]] std::array<Tensor, 2> rmsnorm(
      const Tensor& x,
      const Tensor& scale,
      Rmsnorm_attributes attributes) {
    require_tensor(x, "rmsnorm X");
    require_tensor(scale, "rmsnorm scale");
    if (attributes.get_forward_phase() != NormFwdPhase_t::TRAINING) {
      throw flagdnn::Error(
          FLAGDNN_STATUS_NOT_SUPPORTED,
          "FlagDNN rmsnorm currently supports TRAINING phase only");
    }
    if (attributes.bias_ == nullptr) {
      throw flagdnn::Error(
          FLAGDNN_STATUS_NOT_SUPPORTED,
          "FlagDNN rmsnorm currently requires bias");
    }
    require_tensor(attributes.bias_, "rmsnorm bias");
    require_tensor(attributes.epsilon_, "rmsnorm epsilon");
    Tensor y = inferred_output(*x, "rmsnorm_output");
    Tensor inv_variance = inferred_norm_stat_output(
        *x, *scale, "rmsnorm_inv_variance");
    nodes_.push_back(Node::make_rmsnorm(
        x, scale, attributes.bias_, y, inv_variance, attributes));
    tensors_.push_back(y);
    tensors_.push_back(inv_variance);
    invalidate();
    return {y, inv_variance};
  }

  [[nodiscard]] std::array<Tensor, 5> batchnorm(
      const Tensor& x,
      const Tensor& scale,
      const Tensor& bias,
      Batchnorm_attributes attributes) {
    require_tensor(x, "batchnorm X");
    require_tensor(scale, "batchnorm scale");
    require_tensor(bias, "batchnorm bias");
    require_tensor(attributes.epsilon_, "batchnorm epsilon");
    if (attributes.peer_stats_.size() > 0) {
      throw flagdnn::Error(
          FLAGDNN_STATUS_NOT_SUPPORTED,
          "FlagDNN batchnorm does not yet support peer stats");
    }
    if (attributes.previous_running_mean_ == nullptr ||
        attributes.previous_running_variance_ == nullptr ||
        attributes.momentum_ == nullptr) {
      throw flagdnn::Error(
          FLAGDNN_STATUS_NOT_SUPPORTED,
          "FlagDNN batchnorm currently requires previous running stats");
    }
    require_tensor(attributes.previous_running_mean_,
                   "batchnorm previous running mean");
    require_tensor(attributes.previous_running_variance_,
                   "batchnorm previous running variance");
    require_tensor(attributes.momentum_, "batchnorm momentum");

    Tensor y = inferred_output(*x, "batchnorm_output");
    Tensor mean = inferred_stat_output(*scale, "batchnorm_mean");
    Tensor inv_variance =
        inferred_stat_output(*scale, "batchnorm_inv_variance");
    Tensor next_running_mean = inferred_stat_output(
        *attributes.previous_running_mean_, "batchnorm_next_running_mean");
    Tensor next_running_variance = inferred_stat_output(
        *attributes.previous_running_variance_,
        "batchnorm_next_running_variance");
    nodes_.push_back(Node::make_batchnorm(
        x, scale, bias,
        attributes.previous_running_mean_,
        attributes.previous_running_variance_,
        y, mean, inv_variance,
        next_running_mean, next_running_variance, attributes));
    tensors_.push_back(y);
    tensors_.push_back(mean);
    tensors_.push_back(inv_variance);
    tensors_.push_back(next_running_mean);
    tensors_.push_back(next_running_variance);
    invalidate();
    return {y, mean, inv_variance,
            next_running_mean, next_running_variance};
  }

  [[nodiscard]] Tensor batchnorm_inference(
      const Tensor& x,
      const Tensor& mean,
      const Tensor& inv_variance,
      const Tensor& scale,
      const Tensor& bias,
      const Batchnorm_inference_attributes& attributes) {
    require_tensor(x, "batchnorm inference X");
    require_tensor(mean, "batchnorm inference mean");
    require_tensor(inv_variance, "batchnorm inference inverse variance");
    require_tensor(scale, "batchnorm inference scale");
    require_tensor(bias, "batchnorm inference bias");
    Tensor output = inferred_output(*x, "batchnorm_inference_output");
    nodes_.push_back(Node::make_batchnorm_inference(
        x, mean, inv_variance, scale, bias, output, attributes));
    tensors_.push_back(output);
    invalidate();
    return output;
  }

  [[nodiscard]] Tensor reshape(
      const Tensor& input,
      const Reshape_attributes& attributes) {
    require_tensor(input, "reshape input");
    if (attributes.get_reshape_mode() == ReshapeMode_t::NOT_SET) {
      throw std::invalid_argument("reshape mode is not set");
    }
    Tensor output = inferred_reshape_output(*input, attributes);
    nodes_.push_back(Node::make_reshape(input, output, attributes));
    tensors_.push_back(output);
    invalidate();
    return output;
  }

  [[nodiscard]] Tensor transpose(
      const Tensor& input,
      const Transpose_attributes& attributes) {
    require_tensor(input, "transpose input");
    Tensor output = inferred_transpose_output(*input, attributes);
    nodes_.push_back(Node::make_transpose(input, output, attributes));
    tensors_.push_back(output);
    invalidate();
    return output;
  }

  [[nodiscard]] Tensor slice(
      const Tensor& input,
      const Slice_attributes& attributes) {
    require_tensor(input, "slice input");
    Tensor output = inferred_slice_output(*input, attributes);
    nodes_.push_back(Node::make_slice(input, output, attributes));
    tensors_.push_back(output);
    invalidate();
    return output;
  }

  [[nodiscard]] Tensor matmul(
      const Tensor& a,
      const Tensor& b,
      const Matmul_attributes& attributes) {
    require_tensor(a, "MatMul A input");
    require_tensor(b, "MatMul B input");
    if (!std::isfinite(attributes.get_padding()) ||
        attributes.get_padding() != 0.0) {
      throw std::invalid_argument(
          "FlagDNN MatMul currently requires zero padding");
    }
    Tensor output = inferred_matmul_output(*a, *b);
    nodes_.push_back(Node::make_matmul(a, b, output, attributes));
    tensors_.push_back(output);
    invalidate();
    return output;
  }

  [[nodiscard]] std::array<Tensor, 2> sdpa(
      const Tensor& q,
      const Tensor& k,
      const Tensor& v,
      const SDPA_attributes& attributes) {
    require_tensor(q, "SDPA Q");
    require_tensor(k, "SDPA K");
    require_tensor(v, "SDPA V");
    if (attributes.get_bias() != nullptr) {
      require_tensor(attributes.get_bias(), "SDPA bias");
    }
    Tensor output = inferred_sdpa_output(*q, *v, "sdpa_output");
    Tensor stats = inferred_sdpa_stats(*q, "sdpa_stats");
    nodes_.push_back(
        Node::make_sdpa(q, k, v, output, stats, attributes));
    tensors_.push_back(output);
    tensors_.push_back(stats);
    if (attributes.get_bias() != nullptr &&
        std::find(tensors_.begin(),
                  tensors_.end(),
                  attributes.get_bias()) == tensors_.end()) {
      tensors_.push_back(attributes.get_bias());
    }
    invalidate();
    return {output, attributes.get_generate_stats() ? stats : nullptr};
  }

  [[nodiscard]] std::array<Tensor, 3> sdpa_backward(
      const Tensor& q,
      const Tensor& k,
      const Tensor& v,
      const Tensor& output,
      const Tensor& doutput,
      const Tensor& stats,
      const SDPA_backward_attributes& attributes) {
    require_tensor(q, "SDPA backward Q");
    require_tensor(k, "SDPA backward K");
    require_tensor(v, "SDPA backward V");
    require_tensor(output, "SDPA backward O");
    require_tensor(doutput, "SDPA backward dO");
    require_tensor(stats, "SDPA backward stats");
    if (attributes.get_bias() != nullptr) {
      require_tensor(attributes.get_bias(), "SDPA backward bias");
    }
    if (attributes.get_dbias() != nullptr) {
      require_tensor(attributes.get_dbias(), "SDPA backward dBias");
    }
    Tensor dq = inferred_output(*q, "sdpa_backward_dq");
    Tensor dk = inferred_output(*k, "sdpa_backward_dk");
    Tensor dv = inferred_output(*v, "sdpa_backward_dv");
    nodes_.push_back(Node::make_sdpa_backward(q,
                                              k,
                                              v,
                                              output,
                                              doutput,
                                              stats,
                                              dq,
                                              dk,
                                              dv,
                                              attributes));
    tensors_.push_back(dq);
    tensors_.push_back(dk);
    tensors_.push_back(dv);
    for (const Tensor& attribute_tensor :
         {attributes.get_bias(), attributes.get_dbias()}) {
      if (attribute_tensor != nullptr &&
          std::find(tensors_.begin(),
                    tensors_.end(),
                    attribute_tensor) == tensors_.end()) {
        tensors_.push_back(attribute_tensor);
      }
    }
    invalidate();
    return {dq, dk, dv};
  }

  [[nodiscard]] std::array<Tensor, 4> sdpa_fp8(
      const Tensor& q,
      const Tensor& k,
      const Tensor& v,
      const Tensor& descale_q,
      const Tensor& descale_k,
      const Tensor& descale_v,
      const Tensor& descale_s,
      const Tensor& scale_s,
      const Tensor& scale_o,
      const SDPA_fp8_attributes& attributes) {
    const std::array<std::pair<Tensor, const char*>, 9>
        required_inputs{{{q, "FP8 SDPA Q"},
                         {k, "FP8 SDPA K"},
                         {v, "FP8 SDPA V"},
                         {descale_q, "FP8 SDPA descale Q"},
                         {descale_k, "FP8 SDPA descale K"},
                         {descale_v, "FP8 SDPA descale V"},
                         {descale_s, "FP8 SDPA descale S"},
                         {scale_s, "FP8 SDPA scale S"},
                         {scale_o, "FP8 SDPA scale O"}}};
    for (const auto& [value, role] : required_inputs) {
      require_tensor(value, role);
    }
    if (attributes.get_bias() != nullptr) {
      require_tensor(attributes.get_bias(), "FP8 SDPA bias");
    }
    Tensor output =
        inferred_sdpa_output(*q, *v, "sdpa_fp8_output");
    Tensor stats = inferred_sdpa_stats(*q, "sdpa_fp8_stats");
    Tensor amax_s = inferred_amax_output("sdpa_fp8_amax_s");
    Tensor amax_o = inferred_amax_output("sdpa_fp8_amax_o");
    nodes_.push_back(Node::make_sdpa_fp8(q,
                                         k,
                                         v,
                                         descale_q,
                                         descale_k,
                                         descale_v,
                                         descale_s,
                                         scale_s,
                                         scale_o,
                                         output,
                                         stats,
                                         amax_s,
                                         amax_o,
                                         attributes));
    tensors_.insert(
        tensors_.end(), {output, stats, amax_s, amax_o});
    if (attributes.get_bias() != nullptr &&
        std::find(tensors_.begin(),
                  tensors_.end(),
                  attributes.get_bias()) == tensors_.end()) {
      tensors_.push_back(attributes.get_bias());
    }
    invalidate();
    return {output,
            attributes.get_generate_stats() ? stats : nullptr,
            amax_s,
            amax_o};
  }

  [[nodiscard]] std::array<Tensor, 7> sdpa_fp8_backward(
      const Tensor& q,
      const Tensor& k,
      const Tensor& v,
      const Tensor& output,
      const Tensor& doutput,
      const Tensor& stats,
      const Tensor& descale_q,
      const Tensor& descale_k,
      const Tensor& descale_v,
      const Tensor& descale_o,
      const Tensor& descale_doutput,
      const Tensor& descale_s,
      const Tensor& descale_dp,
      const Tensor& scale_s,
      const Tensor& scale_dq,
      const Tensor& scale_dk,
      const Tensor& scale_dv,
      const Tensor& scale_dp,
      const SDPA_fp8_backward_attributes& attributes) {
    const std::array<Tensor, 18> inputs{q,
                                        k,
                                        v,
                                        output,
                                        doutput,
                                        stats,
                                        descale_q,
                                        descale_k,
                                        descale_v,
                                        descale_o,
                                        descale_doutput,
                                        descale_s,
                                        descale_dp,
                                        scale_s,
                                        scale_dq,
                                        scale_dk,
                                        scale_dv,
                                        scale_dp};
    for (const Tensor& value : inputs) {
      require_tensor(value, "FP8 SDPA backward input");
    }
    if (attributes.get_bias() != nullptr ||
        attributes.get_dbias() != nullptr) {
      throw flagdnn::Error(
          FLAGDNN_STATUS_NOT_SUPPORTED,
          "FP8 SDPA backward bias gradients are not supported");
    }
    Tensor dq = inferred_output(*q, "sdpa_fp8_backward_dq");
    Tensor dk = inferred_output(*k, "sdpa_fp8_backward_dk");
    Tensor dv = inferred_output(*v, "sdpa_fp8_backward_dv");
    Tensor amax_dq = inferred_amax_output("sdpa_fp8_backward_amax_dq");
    Tensor amax_dk = inferred_amax_output("sdpa_fp8_backward_amax_dk");
    Tensor amax_dv = inferred_amax_output("sdpa_fp8_backward_amax_dv");
    Tensor amax_dp = inferred_amax_output("sdpa_fp8_backward_amax_dp");
    nodes_.push_back(Node::make_sdpa_fp8_backward(inputs,
                                                  dq,
                                                  dk,
                                                  dv,
                                                  amax_dq,
                                                  amax_dk,
                                                  amax_dv,
                                                  amax_dp,
                                                  attributes));
    tensors_.insert(tensors_.end(),
                    {dq, dk, dv, amax_dq, amax_dk, amax_dv, amax_dp});
    invalidate();
    return {dq, dk, dv, amax_dq, amax_dk, amax_dv, amax_dp};
  }

  [[nodiscard]] Tensor reduction(
      const Tensor& input,
      const Reduction_attributes& attributes) {
    require_tensor(input, "reduction input");
    if (attributes.get_mode() == ReductionMode_t::NOT_SET) {
      throw std::invalid_argument("reduction mode is not set");
    }
    Tensor output = inferred_reduction_output(*input, attributes);
    nodes_.push_back(Node::make_reduction(input, output, attributes));
    tensors_.push_back(output);
    invalidate();
    return output;
  }

  [[nodiscard]] Tensor conv_fprop(
      const Tensor& input,
      const Tensor& filter,
      const Conv_fprop_attributes& attributes) {
    require_tensor(input, "convolution input");
    require_tensor(filter, "convolution filter");
    if (attributes.get_convolution_mode() !=
        ConvolutionMode_t::CROSS_CORRELATION) {
      throw std::invalid_argument(
          "FlagDNN currently supports CROSS_CORRELATION only");
    }
    Tensor output = inferred_convolution_output(
        *input, *filter, attributes);
    nodes_.push_back(
        Node::make_convolution(input, filter, output, attributes));
    tensors_.push_back(output);
    invalidate();
    return output;
  }

  [[nodiscard]] Tensor conv_dgrad(
      const Tensor& loss,
      const Tensor& filter,
      const Conv_dgrad_attributes& attributes) {
    require_tensor(loss, "convolution loss");
    require_tensor(filter, "convolution filter");
    Tensor output = uninferred_convolution_output(
        *loss, "convolution_dgrad_output");
    nodes_.push_back(
        Node::make_convolution_dgrad(loss, filter, output, attributes));
    tensors_.push_back(output);
    invalidate();
    return output;
  }

  [[nodiscard]] Tensor conv_wgrad(
      const Tensor& loss,
      const Tensor& image,
      const Conv_wgrad_attributes& attributes) {
    require_tensor(loss, "convolution loss");
    require_tensor(image, "convolution image");
    Tensor output = uninferred_convolution_output(
        *loss, "convolution_wgrad_output");
    nodes_.push_back(
        Node::make_convolution_wgrad(loss, image, output, attributes));
    tensors_.push_back(output);
    invalidate();
    return output;
  }

  [[nodiscard]] error_t validate() noexcept {
    try {
      invalidate();
      (void)lower_to_native_graph(false);
      state_ = LifecycleState::kValidated;
      return {};
    } catch (...) {
      invalidate();
      return detail::current_exception_as_error();
    }
  }

  [[nodiscard]] error_t build_operation_graph(
      const Handle& handle) noexcept {
    try {
      require_state(LifecycleState::kValidated,
                    "validate() must succeed before build_operation_graph()");
      LoweredGraph lowered = lower_to_native_graph(true);
      const std::string backend(handle.backend_name());
      const std::string target(handle.target_fingerprint());
      native_graph_ = std::move(lowered.graph);
      required_uids_ = std::move(lowered.required_uids);
      backend_name_ = backend;
      target_fingerprint_ = target;
      state_ = LifecycleState::kOperationGraph;
      return {};
    } catch (...) {
      return detail::current_exception_as_error();
    }
  }

  [[nodiscard]] error_t create_execution_plans(
      std::initializer_list<HeurMode_t> modes = {
          HeurMode_t::A}) noexcept {
    try {
      require_state(
          LifecycleState::kOperationGraph,
          "build_operation_graph() must succeed before create_execution_plans()");
      if (modes.size() == 0) {
        throw std::invalid_argument(
            "at least one frontend heuristic mode is required");
      }
      std::vector<HeurMode_t> requested_modes;
      requested_modes.reserve(modes.size());
      for (const HeurMode_t mode : modes) {
        if (mode != HeurMode_t::A && mode != HeurMode_t::FALLBACK) {
          throw std::invalid_argument("unknown frontend heuristic mode");
        }
        if (std::find(requested_modes.begin(), requested_modes.end(), mode) ==
            requested_modes.end()) {
          requested_modes.push_back(mode);
        }
      }
      heuristic_modes_ = std::move(requested_modes);
      state_ = LifecycleState::kExecutionPlans;
      return {};
    } catch (...) {
      return detail::current_exception_as_error();
    }
  }

  [[nodiscard]] error_t check_support(const Handle& handle) noexcept {
    try {
      require_state(LifecycleState::kExecutionPlans,
                    "create_execution_plans() must succeed before check_support()");
      require_compatible_handle(handle);
      const flagdnnBuildOptions_t options = selected_build_options();
      auto candidate = std::make_unique<flagdnn::Executable>(
          handle, *native_graph_, &options);
      const std::size_t candidate_workspace = candidate->workspace_size();
      supported_candidate_ = std::move(candidate);
      supported_workspace_size_ = candidate_workspace;
      state_ = LifecycleState::kSupported;
      return {};
    } catch (...) {
      return detail::current_exception_as_error();
    }
  }

  [[nodiscard]] error_t build_plans(
      const Handle& handle,
      BuildPlanPolicy_t policy =
          BuildPlanPolicy_t::HEURISTICS_CHOICE) noexcept {
    try {
      require_state(LifecycleState::kSupported,
                    "check_support() must succeed before build_plans()");
      require_compatible_handle(handle);
      if (policy != BuildPlanPolicy_t::HEURISTICS_CHOICE) {
        throw std::invalid_argument("unknown frontend build-plan policy");
      }
      if (supported_candidate_ == nullptr) {
        throw std::logic_error(
            "frontend graph has no supported executable candidate");
      }
      executable_ = std::move(supported_candidate_);
      workspace_size_ = supported_workspace_size_;
      supported_workspace_size_ = 0;
      state_ = LifecycleState::kBuilt;
      return {};
    } catch (...) {
      return detail::current_exception_as_error();
    }
  }

  /* Convenience entry point equivalent to the full staged lifecycle. */
  [[nodiscard]] error_t build(
      const Handle& handle,
      std::initializer_list<HeurMode_t> modes = {HeurMode_t::A}) noexcept {
    error_t status = validate();
    if (status.is_bad()) {
      return status;
    }
    status = build_operation_graph(handle);
    if (status.is_bad()) {
      return status;
    }
    status = create_execution_plans(modes);
    if (status.is_bad()) {
      return status;
    }
    status = check_support(handle);
    if (status.is_bad()) {
      return status;
    }
    return build_plans(handle);
  }

  [[nodiscard]] error_t get_workspace_size(
      std::int64_t& workspace_size) const noexcept {
    try {
      require_built();
      if (workspace_size_ >
          static_cast<std::size_t>(
              std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error(
            "frontend workspace size does not fit int64_t");
      }
      workspace_size = static_cast<std::int64_t>(workspace_size_);
      return {};
    } catch (...) {
      return detail::current_exception_as_error();
    }
  }

  [[nodiscard]] std::size_t get_workspace_size() const {
    require_built();
    return workspace_size_;
  }

  /*
   * This span overload is the zero-allocation execution path used by native
   * tests and performance-sensitive callers.
   */
  [[nodiscard]] error_t execute(
      const Handle& handle,
      std::span<const flagdnnBinding_t> bindings,
      void* workspace,
      std::size_t workspace_size,
      flagdnnStream_t stream = nullptr) const noexcept {
    try {
      require_built();
      require_compatible_handle(handle);
      executable_->execute(bindings, workspace, workspace_size, stream);
      return {};
    } catch (...) {
      return detail::current_exception_as_error();
    }
  }

  [[nodiscard]] error_t execute(
      const Handle& handle,
      std::span<const flagdnnBinding_t> bindings,
      void* workspace,
      flagdnnStream_t stream = nullptr) const noexcept {
    return execute(handle, bindings, workspace, workspace_size_, stream);
  }

  /*
   * UID-to-pointer variant packs mirror cuDNN Frontend.  Callers that need
   * allocation-free launches should retain flagdnnBinding_t storage and use
   * the span overload above.
   */
  [[nodiscard]] error_t execute(
      const Handle& handle,
      const VariantPack& variant_pack,
      void* workspace,
      flagdnnStream_t stream = nullptr) const noexcept {
    try {
      require_built();
      std::vector<flagdnnBinding_t> bindings;
      bindings.reserve(required_uids_.size());
      for (const std::int64_t uid : required_uids_) {
        const auto found = variant_pack.find(uid);
        if (found == variant_pack.end()) {
          throw std::invalid_argument(
              "frontend variant pack is missing tensor UID " +
              std::to_string(uid));
        }
        bindings.push_back({uid, found->second});
      }
      return execute(
          handle, bindings, workspace, workspace_size_, stream);
    } catch (...) {
      return detail::current_exception_as_error();
    }
  }

  [[nodiscard]] bool is_built() const noexcept {
    return executable_ != nullptr;
  }

 private:
  enum class NodeKind {
    kPointwise,
    kBinaryPointwise,
    kTernaryPointwise,
    kReduction,
    kConvolution,
    kConvolutionDgrad,
    kConvolutionWgrad,
    kLayernorm,
    kRmsnorm,
    kBatchnorm,
    kBatchnormInference,
    kMatmul,
    kReshape,
    kTranspose,
    kSlice,
    kSdpa,
    kSdpaBackward,
    kSdpaFp8,
    kSdpaFp8Backward,
  };

  struct Node {
    static Node make_pointwise(
        const Tensor& input,
        const Tensor& output,
        const Pointwise_attributes& attributes) {
      Node result;
      result.kind = NodeKind::kPointwise;
      result.input = input;
      result.output = output;
      result.pointwise = attributes;
      return result;
    }

    static Node make_binary_pointwise(
        const Tensor& left,
        const Tensor& right,
        const Tensor& output,
        const Pointwise_attributes& attributes) {
      Node result;
      result.kind = NodeKind::kBinaryPointwise;
      result.input = left;
      result.second = right;
      result.output = output;
      result.pointwise = attributes;
      return result;
    }

    static Node make_ternary_pointwise(
        const Tensor& a,
        const Tensor& b,
        const Tensor& t,
        const Tensor& output,
        const Pointwise_attributes& attributes) {
      Node result;
      result.kind = NodeKind::kTernaryPointwise;
      result.input = a;
      result.second = b;
      result.third = t;
      result.output = output;
      result.pointwise = attributes;
      return result;
    }

    static Node make_reduction(
        const Tensor& input,
        const Tensor& output,
        const Reduction_attributes& attributes) {
      Node result;
      result.kind = NodeKind::kReduction;
      result.input = input;
      result.output = output;
      result.reduction_attributes = attributes;
      return result;
    }

    static Node make_matmul(
        const Tensor& a,
        const Tensor& b,
        const Tensor& output,
        const Matmul_attributes& attributes) {
      Node result;
      result.kind = NodeKind::kMatmul;
      result.input = a;
      result.second = b;
      result.output = output;
      result.matmul_attributes = attributes;
      return result;
    }

    static Node make_sdpa(
        const Tensor& q,
        const Tensor& k,
        const Tensor& v,
        const Tensor& output,
        const Tensor& stats,
        const SDPA_attributes& attributes) {
      Node result;
      result.kind = NodeKind::kSdpa;
      result.input = q;
      result.output = output;
      result.attention_inputs = {q, k, v};
      if (attributes.get_bias() != nullptr) {
        result.attention_inputs.push_back(attributes.get_bias());
      }
      result.attention_outputs = {output, stats};
      result.sdpa_attributes = attributes;
      return result;
    }

    static Node make_sdpa_backward(
        const Tensor& q,
        const Tensor& k,
        const Tensor& v,
        const Tensor& output,
        const Tensor& doutput,
        const Tensor& stats,
        const Tensor& dq,
        const Tensor& dk,
        const Tensor& dv,
        const SDPA_backward_attributes& attributes) {
      Node result;
      result.kind = NodeKind::kSdpaBackward;
      result.input = q;
      result.output = dq;
      result.attention_inputs = {q, k, v, output, doutput, stats};
      if (attributes.get_bias() != nullptr) {
        result.attention_inputs.push_back(attributes.get_bias());
      }
      result.attention_outputs = {dq, dk, dv};
      if (attributes.get_dbias() != nullptr) {
        result.attention_outputs.push_back(attributes.get_dbias());
      }
      result.sdpa_backward_attributes = attributes;
      return result;
    }

    static Node make_sdpa_fp8(
        const Tensor& q,
        const Tensor& k,
        const Tensor& v,
        const Tensor& descale_q,
        const Tensor& descale_k,
        const Tensor& descale_v,
        const Tensor& descale_s,
        const Tensor& scale_s,
        const Tensor& scale_o,
        const Tensor& output,
        const Tensor& stats,
        const Tensor& amax_s,
        const Tensor& amax_o,
        const SDPA_fp8_attributes& attributes) {
      Node result;
      result.kind = NodeKind::kSdpaFp8;
      result.input = q;
      result.output = output;
      result.attention_inputs = {q,
                                 k,
                                 v,
                                 descale_q,
                                 descale_k,
                                 descale_v,
                                 descale_s,
                                 scale_s,
                                 scale_o};
      if (attributes.get_bias() != nullptr) {
        result.attention_inputs.push_back(attributes.get_bias());
      }
      result.attention_outputs = {output, stats, amax_s, amax_o};
      result.sdpa_attributes = attributes;
      return result;
    }

    static Node make_sdpa_fp8_backward(
        const std::array<Tensor, 18>& inputs,
        const Tensor& dq,
        const Tensor& dk,
        const Tensor& dv,
        const Tensor& amax_dq,
        const Tensor& amax_dk,
        const Tensor& amax_dv,
        const Tensor& amax_dp,
        const SDPA_fp8_backward_attributes& attributes) {
      Node result;
      result.kind = NodeKind::kSdpaFp8Backward;
      result.input = inputs[0];
      result.output = dq;
      result.attention_inputs.assign(inputs.begin(), inputs.end());
      result.attention_outputs =
          {dq, dk, dv, amax_dq, amax_dk, amax_dv, amax_dp};
      result.sdpa_backward_attributes = attributes;
      return result;
    }

    static Node make_reshape(
        const Tensor& input,
        const Tensor& output,
        const Reshape_attributes& attributes) {
      Node result;
      result.kind = NodeKind::kReshape;
      result.input = input;
      result.output = output;
      result.reshape_attributes = attributes;
      return result;
    }

    static Node make_transpose(
        const Tensor& input,
        const Tensor& output,
        const Transpose_attributes& attributes) {
      Node result;
      result.kind = NodeKind::kTranspose;
      result.input = input;
      result.output = output;
      result.transpose_attributes = attributes;
      return result;
    }

    static Node make_slice(
        const Tensor& input,
        const Tensor& output,
        const Slice_attributes& attributes) {
      Node result;
      result.kind = NodeKind::kSlice;
      result.input = input;
      result.output = output;
      result.slice_attributes = attributes;
      return result;
    }

    static Node make_convolution_dgrad(
        const Tensor& loss,
        const Tensor& filter,
        const Tensor& output,
        const Conv_dgrad_attributes& attributes) {
      Node result;
      result.kind = NodeKind::kConvolutionDgrad;
      result.input = loss;
      result.second = filter;
      result.output = output;
      result.convolution_dgrad_attributes = attributes;
      return result;
    }

    static Node make_convolution_wgrad(
        const Tensor& loss,
        const Tensor& image,
        const Tensor& output,
        const Conv_wgrad_attributes& attributes) {
      Node result;
      result.kind = NodeKind::kConvolutionWgrad;
      result.input = loss;
      result.second = image;
      result.output = output;
      result.convolution_wgrad_attributes = attributes;
      return result;
    }

    static Node make_layernorm(
        const Tensor& x,
        const Tensor& scale,
        const Tensor& bias,
        const Tensor& y,
        const Tensor& mean,
        const Tensor& inv_variance,
        const Layernorm_attributes& attributes) {
      Node result;
      result.kind = NodeKind::kLayernorm;
      result.input = x;
      result.second = scale;
      result.third = bias;
      result.output = y;
      result.second_output = mean;
      result.third_output = inv_variance;
      result.layernorm_attributes = attributes;
      return result;
    }

    static Node make_rmsnorm(
        const Tensor& x,
        const Tensor& scale,
        const Tensor& bias,
        const Tensor& y,
        const Tensor& inv_variance,
        const Rmsnorm_attributes& attributes) {
      Node result;
      result.kind = NodeKind::kRmsnorm;
      result.input = x;
      result.second = scale;
      result.third = bias;
      result.output = y;
      result.second_output = inv_variance;
      result.rmsnorm_attributes = attributes;
      return result;
    }

    static Node make_batchnorm(
        const Tensor& x,
        const Tensor& scale,
        const Tensor& bias,
        const Tensor& previous_running_mean,
        const Tensor& previous_running_variance,
        const Tensor& y,
        const Tensor& mean,
        const Tensor& inv_variance,
        const Tensor& next_running_mean,
        const Tensor& next_running_variance,
        const Batchnorm_attributes& attributes) {
      Node result;
      result.kind = NodeKind::kBatchnorm;
      result.input = x;
      result.second = scale;
      result.third = bias;
      result.fourth = previous_running_mean;
      result.fifth = previous_running_variance;
      result.output = y;
      result.second_output = mean;
      result.third_output = inv_variance;
      result.fourth_output = next_running_mean;
      result.fifth_output = next_running_variance;
      result.batchnorm_attributes = attributes;
      return result;
    }

    static Node make_batchnorm_inference(
        const Tensor& x,
        const Tensor& mean,
        const Tensor& inv_variance,
        const Tensor& scale,
        const Tensor& bias,
        const Tensor& output,
        const Batchnorm_inference_attributes& attributes) {
      Node result;
      result.kind = NodeKind::kBatchnormInference;
      result.input = x;
      result.second = mean;
      result.third = inv_variance;
      result.fourth = scale;
      result.fifth = bias;
      result.output = output;
      result.batchnorm_inference_attributes = attributes;
      return result;
    }

    static Node make_convolution(
        const Tensor& input,
        const Tensor& filter,
        const Tensor& output,
        const Conv_fprop_attributes& attributes) {
      Node result;
      result.kind = NodeKind::kConvolution;
      result.input = input;
      result.second = filter;
      result.output = output;
      result.convolution_attributes = attributes;
      return result;
    }

    NodeKind kind = NodeKind::kPointwise;
    Tensor input;
    Tensor second;
    Tensor third;
    Tensor fourth;
    Tensor fifth;
    Tensor output;
    Tensor second_output;
    Tensor third_output;
    Tensor fourth_output;
    Tensor fifth_output;
    std::vector<Tensor> attention_inputs;
    std::vector<Tensor> attention_outputs;
    Pointwise_attributes pointwise;
    Reduction_attributes reduction_attributes;
    Conv_fprop_attributes convolution_attributes;
    Conv_dgrad_attributes convolution_dgrad_attributes;
    Conv_wgrad_attributes convolution_wgrad_attributes;
    Layernorm_attributes layernorm_attributes;
    Rmsnorm_attributes rmsnorm_attributes;
    Batchnorm_attributes batchnorm_attributes;
    Batchnorm_inference_attributes batchnorm_inference_attributes;
    Matmul_attributes matmul_attributes;
    Reshape_attributes reshape_attributes;
    Transpose_attributes transpose_attributes;
    Slice_attributes slice_attributes;
    SDPA_attributes sdpa_attributes;
    SDPA_backward_attributes sdpa_backward_attributes;
  };

  static bool is_unary_pointwise_mode(PointwiseMode_t mode) noexcept {
    switch (mode) {
      case PointwiseMode_t::RELU_FWD:
      case PointwiseMode_t::SQRT:
      case PointwiseMode_t::ERF:
      case PointwiseMode_t::IDENTITY:
      case PointwiseMode_t::EXP:
      case PointwiseMode_t::LOG:
      case PointwiseMode_t::NEG:
      case PointwiseMode_t::ABS:
      case PointwiseMode_t::CEIL:
      case PointwiseMode_t::COS:
      case PointwiseMode_t::FLOOR:
      case PointwiseMode_t::RSQRT:
      case PointwiseMode_t::SIN:
      case PointwiseMode_t::TAN:
      case PointwiseMode_t::RECIPROCAL:
      case PointwiseMode_t::LOGICAL_NOT:
      case PointwiseMode_t::SIGMOID_FWD:
      case PointwiseMode_t::TANH_FWD:
      case PointwiseMode_t::ELU_FWD:
      case PointwiseMode_t::GELU_FWD:
      case PointwiseMode_t::SOFTPLUS_FWD:
      case PointwiseMode_t::SWISH_FWD:
      case PointwiseMode_t::GELU_APPROX_TANH_FWD:
        return true;
      case PointwiseMode_t::NOT_SET:
      case PointwiseMode_t::SIGMOID_BWD:
      case PointwiseMode_t::BINARY_SELECT:
      case PointwiseMode_t::ADD:
      case PointwiseMode_t::SUB:
      case PointwiseMode_t::MUL:
      case PointwiseMode_t::DIV:
      case PointwiseMode_t::MIN:
      case PointwiseMode_t::MAX:
      case PointwiseMode_t::MOD:
      case PointwiseMode_t::POW:
      case PointwiseMode_t::CMP_EQ:
      case PointwiseMode_t::CMP_NEQ:
      case PointwiseMode_t::CMP_GT:
      case PointwiseMode_t::CMP_GE:
      case PointwiseMode_t::CMP_LT:
      case PointwiseMode_t::CMP_LE:
      case PointwiseMode_t::LOGICAL_AND:
      case PointwiseMode_t::LOGICAL_OR:
        return false;
    }
    return false;
  }

  static bool is_binary_pointwise_mode(PointwiseMode_t mode) noexcept {
    switch (mode) {
      case PointwiseMode_t::ADD:
      case PointwiseMode_t::SIGMOID_BWD:
      case PointwiseMode_t::SUB:
      case PointwiseMode_t::MUL:
      case PointwiseMode_t::DIV:
      case PointwiseMode_t::MIN:
      case PointwiseMode_t::MAX:
      case PointwiseMode_t::MOD:
      case PointwiseMode_t::POW:
      case PointwiseMode_t::CMP_EQ:
      case PointwiseMode_t::CMP_NEQ:
      case PointwiseMode_t::CMP_GT:
      case PointwiseMode_t::CMP_GE:
      case PointwiseMode_t::CMP_LT:
      case PointwiseMode_t::CMP_LE:
      case PointwiseMode_t::LOGICAL_AND:
      case PointwiseMode_t::LOGICAL_OR:
        return true;
      case PointwiseMode_t::NOT_SET:
      case PointwiseMode_t::BINARY_SELECT:
      case PointwiseMode_t::RELU_FWD:
      case PointwiseMode_t::SQRT:
      case PointwiseMode_t::ERF:
      case PointwiseMode_t::IDENTITY:
      case PointwiseMode_t::EXP:
      case PointwiseMode_t::LOG:
      case PointwiseMode_t::NEG:
      case PointwiseMode_t::ABS:
      case PointwiseMode_t::CEIL:
      case PointwiseMode_t::COS:
      case PointwiseMode_t::FLOOR:
      case PointwiseMode_t::RSQRT:
      case PointwiseMode_t::SIN:
      case PointwiseMode_t::TAN:
      case PointwiseMode_t::RECIPROCAL:
      case PointwiseMode_t::LOGICAL_NOT:
      case PointwiseMode_t::SIGMOID_FWD:
      case PointwiseMode_t::TANH_FWD:
      case PointwiseMode_t::ELU_FWD:
      case PointwiseMode_t::GELU_FWD:
      case PointwiseMode_t::SOFTPLUS_FWD:
      case PointwiseMode_t::SWISH_FWD:
      case PointwiseMode_t::GELU_APPROX_TANH_FWD:
        return false;
    }
    return false;
  }

  static bool is_ternary_pointwise_mode(
      PointwiseMode_t mode) noexcept {
    return mode == PointwiseMode_t::BINARY_SELECT;
  }

  static void require_tensor(const Tensor& tensor,
                             std::string_view role) {
    if (tensor == nullptr) {
      throw std::invalid_argument(
          std::string(role) + " tensor is null");
    }
  }

  static std::vector<std::int64_t> contiguous_strides(
      const std::vector<std::int64_t>& dimensions) {
    std::vector<std::int64_t> result(dimensions.size());
    std::int64_t stride = 1;
    for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
      result[axis - 1] = stride;
      if (dimensions[axis - 1] > 0 &&
          stride >
              std::numeric_limits<std::int64_t>::max() /
                  dimensions[axis - 1]) {
        throw std::overflow_error("tensor stride inference overflowed");
      }
      stride *= dimensions[axis - 1];
    }
    return result;
  }

  static Tensor inferred_output(const Tensor_attributes& input,
                                std::string name) {
    auto output = std::make_shared<Tensor_attributes>();
    output->set_name(std::move(name))
        .set_data_type(input.get_data_type())
        .set_dim(input.get_dim())
        .set_stride(input.get_stride())
        .set_is_virtual(true);
    return output;
  }

  static Tensor inferred_sdpa_output(const Tensor_attributes& q,
                                     const Tensor_attributes& v,
                                     std::string name) {
    if (q.get_dim().size() != 4 || v.get_dim().size() != 4) {
      throw std::invalid_argument(
          "SDPA Q and V must be rank-4 BHSD tensors");
    }
    std::vector<std::int64_t> dimensions{
        q.get_dim()[0], q.get_dim()[1], q.get_dim()[2], v.get_dim()[3]};
    auto output = std::make_shared<Tensor_attributes>();
    output->set_name(std::move(name))
        .set_data_type(q.get_data_type())
        .set_dim(dimensions)
        .set_stride(contiguous_strides(dimensions))
        .set_is_virtual(true);
    return output;
  }

  static Tensor inferred_sdpa_stats(const Tensor_attributes& q,
                                    std::string name) {
    if (q.get_dim().size() != 4) {
      throw std::invalid_argument("SDPA Q must be a rank-4 BHSD tensor");
    }
    std::vector<std::int64_t> dimensions{
        q.get_dim()[0], q.get_dim()[1], q.get_dim()[2], 1};
    auto output = std::make_shared<Tensor_attributes>();
    output->set_name(std::move(name))
        .set_data_type(DataType_t::FLOAT)
        .set_dim(dimensions)
        .set_stride(contiguous_strides(dimensions))
        .set_is_virtual(true);
    return output;
  }

  static Tensor inferred_amax_output(std::string name) {
    auto output = std::make_shared<Tensor_attributes>();
    output->set_name(std::move(name))
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_is_virtual(true);
    return output;
  }

  static Tensor inferred_norm_stat_output(
      const Tensor_attributes& input,
      const Tensor_attributes& scale,
      std::string name) {
    const auto& input_dimensions = input.get_dim();
    const auto& scale_dimensions = scale.get_dim();
    if (input_dimensions.empty() || scale_dimensions.empty() ||
        scale_dimensions.size() > input_dimensions.size()) {
      throw std::invalid_argument(
          "normalization scale rank must be in [1, input rank]");
    }
    const std::size_t leading =
        input_dimensions.size() - scale_dimensions.size();
    std::vector<std::int64_t> stat_dimensions = input_dimensions;
    bool normalized_suffix = false;
    for (std::size_t axis = 0; axis < input_dimensions.size(); ++axis) {
      const std::int64_t scale_dimension =
          axis < leading ? 1 : scale_dimensions[axis - leading];
      if (scale_dimension != 1) {
        if (scale_dimension != input_dimensions[axis]) {
          throw std::invalid_argument(
              "normalization scale shape does not match input suffix");
        }
        normalized_suffix = true;
      } else if (normalized_suffix && input_dimensions[axis] != 1) {
        throw std::invalid_argument(
            "normalization scale must describe a contiguous input suffix");
      }
      if (normalized_suffix) {
        stat_dimensions[axis] = 1;
      }
    }
    if (!normalized_suffix) {
      stat_dimensions.back() = 1;
    }
    auto output = std::make_shared<Tensor_attributes>();
    output->set_name(std::move(name))
        .set_data_type(DataType_t::FLOAT)
        .set_dim(stat_dimensions)
        .set_stride(contiguous_strides(stat_dimensions))
        .set_is_virtual(true);
    return output;
  }

  static Tensor inferred_stat_output(const Tensor_attributes& input,
                                     std::string name) {
    auto output = std::make_shared<Tensor_attributes>();
    output->set_name(std::move(name))
        .set_data_type(DataType_t::FLOAT)
        .set_dim(input.get_dim())
        .set_stride(input.get_stride())
        .set_is_virtual(true);
    return output;
  }

  static double compile_time_scalar(const Tensor& tensor,
                                    std::string_view role) {
    require_tensor(tensor, role);
    if (!tensor->scalar_value_.has_value() ||
        !tensor->scalar_type_.has_value()) {
      throw std::invalid_argument(
          std::string(role) + " must be a scalar tensor");
    }
    if (*tensor->scalar_type_ != ScalarType::COMPILE_TIME_CONST) {
      throw flagdnn::Error(
          FLAGDNN_STATUS_NOT_SUPPORTED,
          std::string(role) +
              " runtime parameters are not supported by this backend");
    }
    return *tensor->scalar_value_;
  }

  static std::vector<std::int64_t> broadcast_dimensions(
      const std::vector<std::int64_t>& left,
      const std::vector<std::int64_t>& right) {
    const std::size_t rank = std::max(left.size(), right.size());
    std::vector<std::int64_t> result(rank, 1);
    for (std::size_t trailing = 0; trailing < rank; ++trailing) {
      const std::int64_t left_dimension =
          trailing < left.size() ? left[left.size() - 1 - trailing] : 1;
      const std::int64_t right_dimension =
          trailing < right.size() ? right[right.size() - 1 - trailing] : 1;
      if (left_dimension != right_dimension && left_dimension != 1 &&
          right_dimension != 1) {
        throw std::invalid_argument(
            "binary pointwise input shapes are not broadcast-compatible");
      }
      result[rank - 1 - trailing] =
          std::max(left_dimension, right_dimension);
    }
    return result;
  }

  static bool has_boolean_output(PointwiseMode_t mode) noexcept {
    return mode == PointwiseMode_t::CMP_EQ ||
           mode == PointwiseMode_t::CMP_NEQ ||
           mode == PointwiseMode_t::CMP_GT ||
           mode == PointwiseMode_t::CMP_GE ||
           mode == PointwiseMode_t::CMP_LT ||
           mode == PointwiseMode_t::CMP_LE ||
           mode == PointwiseMode_t::LOGICAL_AND ||
           mode == PointwiseMode_t::LOGICAL_OR;
  }

  static Tensor inferred_binary_output(const Tensor_attributes& left,
                                       const Tensor_attributes& right,
                                       PointwiseMode_t mode) {
    auto output = std::make_shared<Tensor_attributes>();
    const std::vector<std::int64_t> dimensions =
        broadcast_dimensions(left.get_dim(), right.get_dim());
    output->set_name("pointwise_output")
        .set_data_type(has_boolean_output(mode) ? DataType_t::BOOLEAN
                                                : left.get_data_type())
        .set_dim(dimensions)
        .set_stride(contiguous_strides(dimensions))
        .set_is_virtual(true);
    return output;
  }

  static Tensor inferred_ternary_output(
      const Tensor_attributes& a,
      const Tensor_attributes& b,
      const Tensor_attributes& t) {
    auto output = std::make_shared<Tensor_attributes>();
    const std::vector<std::int64_t> dimensions = broadcast_dimensions(
        broadcast_dimensions(a.get_dim(), b.get_dim()), t.get_dim());
    output->set_name("pointwise_output")
        .set_data_type(a.get_data_type())
        .set_dim(dimensions)
        .set_stride(contiguous_strides(dimensions))
        .set_is_virtual(true);
    return output;
  }

  static std::int64_t logical_element_count(
      const std::vector<std::int64_t>& dimensions,
      std::string_view role) {
    std::int64_t result = 1;
    for (const std::int64_t dimension : dimensions) {
      if (dimension <= 0) {
        throw std::invalid_argument(
            std::string(role) + " dimensions must be positive");
      }
      if (result >
          std::numeric_limits<std::int64_t>::max() / dimension) {
        throw std::overflow_error(
            std::string(role) + " element count overflowed");
      }
      result *= dimension;
    }
    return result;
  }

  static Tensor inferred_reshape_output(
      const Tensor_attributes& input,
      const Reshape_attributes& attributes) {
    const auto& dimensions = attributes.get_dim();
    if (logical_element_count(input.get_dim(), "reshape input") !=
        logical_element_count(dimensions, "reshape output")) {
      throw std::invalid_argument(
          "reshape input/output element counts must match");
    }
    std::vector<std::int64_t> strides = attributes.get_stride();
    if (strides.empty()) {
      strides = contiguous_strides(dimensions);
    } else if (strides.size() != dimensions.size()) {
      throw std::invalid_argument(
          "reshape output dimensions and strides must have the same rank");
    }
    auto output = std::make_shared<Tensor_attributes>();
    output->set_name("reshape_output")
        .set_data_type(input.get_data_type())
        .set_dim(dimensions)
        .set_stride(std::move(strides))
        .set_is_virtual(true);
    return output;
  }

  static Tensor inferred_transpose_output(
      const Tensor_attributes& input,
      const Transpose_attributes& attributes) {
    const auto& input_dimensions = input.get_dim();
    const auto& input_strides = input.get_stride();
    const auto& permutation = attributes.get_permutation();
    const std::size_t rank = input_dimensions.size();
    if (rank == 0 || rank > 8 || input_strides.size() != rank) {
      throw std::invalid_argument(
          "transpose input rank must be in [1, 8] with matching strides");
    }
    if (permutation.size() != rank) {
      throw std::invalid_argument(
          "transpose permutation rank must match input rank");
    }
    std::vector<bool> seen(rank, false);
    std::vector<std::int64_t> dimensions(rank);
    std::vector<std::int64_t> strides(rank);
    for (std::size_t axis = 0; axis < rank; ++axis) {
      const std::int64_t source_axis = permutation[axis];
      if (source_axis < 0 ||
          source_axis >= static_cast<std::int64_t>(rank) ||
          seen[static_cast<std::size_t>(source_axis)]) {
        throw std::invalid_argument(
            "transpose permutation must contain each input axis once");
      }
      const std::size_t source = static_cast<std::size_t>(source_axis);
      seen[source] = true;
      dimensions[axis] = input_dimensions[source];
      strides[axis] = input_strides[source];
    }
    auto output = std::make_shared<Tensor_attributes>();
    output->set_name("transpose_output")
        .set_data_type(input.get_data_type())
        .set_dim(std::move(dimensions))
        .set_stride(std::move(strides))
        .set_is_virtual(true);
    return output;
  }

  static Tensor inferred_slice_output(
      const Tensor_attributes& input,
      const Slice_attributes& attributes) {
    const auto& input_dimensions = input.get_dim();
    const auto& input_strides = input.get_stride();
    const auto& slices = attributes.get_slices();
    const std::size_t rank = input_dimensions.size();
    if (rank == 0 || rank > 8 || input_strides.size() != rank) {
      throw std::invalid_argument(
          "slice input rank must be in [1, 8] with matching strides");
    }
    if (slices.size() != rank) {
      throw std::invalid_argument(
          "slice range count must match input rank");
    }
    std::vector<std::int64_t> slice_strides =
        attributes.get_strides();
    if (slice_strides.size() > rank) {
      throw std::invalid_argument(
          "slice stride count must not exceed input rank");
    }
    slice_strides.resize(rank, 1);
    std::vector<std::int64_t> dimensions(rank);
    std::vector<std::int64_t> strides(rank);
    for (std::size_t axis = 0; axis < rank; ++axis) {
      const std::int64_t start = slices[axis].first;
      const std::int64_t limit = slices[axis].second;
      const std::int64_t step = slice_strides[axis];
      if (start < 0 || limit <= start ||
          limit > input_dimensions[axis] || step <= 0) {
        throw std::invalid_argument("slice range or stride is invalid");
      }
      dimensions[axis] = (limit - start + step - 1) / step;
      if (input_strides[axis] >
          std::numeric_limits<std::int64_t>::max() / step) {
        throw std::overflow_error("slice output stride overflowed");
      }
      strides[axis] = input_strides[axis] * step;
    }
    auto output = std::make_shared<Tensor_attributes>();
    output->set_name("slice_output")
        .set_data_type(input.get_data_type())
        .set_dim(std::move(dimensions))
        .set_stride(std::move(strides))
        .set_is_virtual(true);
    return output;
  }

  static Tensor inferred_matmul_output(
      const Tensor_attributes& a,
      const Tensor_attributes& b) {
    const auto& a_dimensions = a.get_dim();
    const auto& b_dimensions = b.get_dim();
    if (a_dimensions.size() < 2 || b_dimensions.size() < 2) {
      throw std::invalid_argument("MatMul inputs must have rank at least two");
    }
    if (a_dimensions.back() !=
        b_dimensions[b_dimensions.size() - 2]) {
      throw std::invalid_argument(
          "MatMul contraction dimensions do not match");
    }
    const std::vector<std::int64_t> a_batch(
        a_dimensions.begin(), a_dimensions.end() - 2);
    const std::vector<std::int64_t> b_batch(
        b_dimensions.begin(), b_dimensions.end() - 2);
    const std::size_t batch_rank =
        std::max(a_batch.size(), b_batch.size());
    std::vector<std::int64_t> dimensions(batch_rank, 1);
    for (std::size_t trailing = 0; trailing < batch_rank; ++trailing) {
      const std::int64_t a_dimension =
          trailing < a_batch.size()
              ? a_batch[a_batch.size() - 1 - trailing]
              : 1;
      const std::int64_t b_dimension =
          trailing < b_batch.size()
              ? b_batch[b_batch.size() - 1 - trailing]
              : 1;
      if (a_dimension != b_dimension && a_dimension != 1 &&
          b_dimension != 1) {
        throw std::invalid_argument(
            "MatMul batch dimensions are not broadcast-compatible");
      }
      dimensions[batch_rank - 1 - trailing] =
          std::max(a_dimension, b_dimension);
    }
    dimensions.push_back(a_dimensions[a_dimensions.size() - 2]);
    dimensions.push_back(b_dimensions.back());
    auto output = std::make_shared<Tensor_attributes>();
    output->set_name("matmul_output")
        .set_data_type(a.get_data_type())
        .set_dim(dimensions)
        .set_stride(contiguous_strides(dimensions))
        .set_is_virtual(true);
    return output;
  }

  static Tensor inferred_reduction_output(
      const Tensor_attributes& input,
      const Reduction_attributes& attributes) {
    auto output = std::make_shared<Tensor_attributes>();
    std::vector<std::int64_t> dimensions = input.get_dim();
    if (!dimensions.empty()) {
      std::int64_t axis = attributes.get_axis();
      if (axis < 0) {
        axis += static_cast<std::int64_t>(dimensions.size());
      }
      if (axis < 0 ||
          axis >= static_cast<std::int64_t>(dimensions.size())) {
        throw std::invalid_argument("reduction axis is out of range");
      }
      if (attributes.get_keep_dimensions()) {
        dimensions[static_cast<std::size_t>(axis)] = 1;
      } else {
        dimensions.erase(dimensions.begin() + axis);
      }
    }
    output->set_name("reduction_output")
        .set_data_type(input.get_data_type())
        .set_dim(dimensions)
        .set_stride(contiguous_strides(dimensions))
        .set_is_virtual(true);
    return output;
  }

  static std::vector<std::int64_t> normalized_spatial_attribute(
      const std::vector<std::int64_t>& value,
      std::size_t spatial_rank,
      std::int64_t default_value,
      std::string_view name) {
    if (value.empty()) {
      return std::vector<std::int64_t>(spatial_rank, default_value);
    }
    if (value.size() != spatial_rank) {
      throw std::invalid_argument(
          "convolution " + std::string(name) +
          " rank does not match its tensors");
    }
    return value;
  }

  static Tensor uninferred_convolution_output(
      const Tensor_attributes& input,
      std::string name) {
    auto output = std::make_shared<Tensor_attributes>();
    output->set_name(std::move(name))
        .set_data_type(input.get_data_type())
        .set_is_virtual(true);
    return output;
  }

  static Tensor inferred_convolution_output(
      const Tensor_attributes& input,
      const Tensor_attributes& filter,
      const Conv_fprop_attributes& attributes) {
    auto output = std::make_shared<Tensor_attributes>();
    const auto& input_dimensions = input.get_dim();
    const auto& filter_dimensions = filter.get_dim();
    if (input_dimensions.size() < 3 ||
        input_dimensions.size() != filter_dimensions.size()) {
      output->set_name("convolution_output")
          .set_data_type(input.get_data_type())
          .set_is_virtual(true);
      return output;
    }

    const std::size_t spatial_rank = input_dimensions.size() - 2;
    const std::vector<std::int64_t> pre_padding =
        normalized_spatial_attribute(attributes.get_pre_padding(),
                                     spatial_rank,
                                     0,
                                     "pre-padding");
    const std::vector<std::int64_t> post_padding =
        normalized_spatial_attribute(attributes.get_post_padding(),
                                     spatial_rank,
                                     0,
                                     "post-padding");
    const std::vector<std::int64_t> stride =
        normalized_spatial_attribute(
            attributes.get_stride(), spatial_rank, 1, "stride");
    const std::vector<std::int64_t> dilation =
        normalized_spatial_attribute(
            attributes.get_dilation(), spatial_rank, 1, "dilation");

    std::vector<std::int64_t> dimensions(spatial_rank + 2);
    dimensions[0] = input_dimensions[0];
    dimensions[1] = filter_dimensions[0];
    for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
      if (stride[axis] <= 0 || dilation[axis] <= 0) {
        throw std::invalid_argument(
            "convolution stride and dilation must be positive");
      }
      const std::int64_t effective_filter =
          dilation[axis] * (filter_dimensions[axis + 2] - 1) + 1;
      const std::int64_t numerator =
          input_dimensions[axis + 2] + pre_padding[axis] +
          post_padding[axis] - effective_filter;
      if (numerator < 0) {
        throw std::invalid_argument(
            "convolution inferred a non-positive output dimension");
      }
      dimensions[axis + 2] = numerator / stride[axis] + 1;
    }
    output->set_name("convolution_output")
        .set_data_type(input.get_data_type())
        .set_dim(dimensions)
        .set_stride(contiguous_strides(dimensions))
        .set_is_virtual(true);
    return output;
  }

  [[nodiscard]] DataType_t resolved_data_type(
      const Tensor_attributes& tensor) const {
    if (tensor.get_data_type() != DataType_t::NOT_SET) {
      return tensor.get_data_type();
    }
    if (tensor.get_is_virtual() &&
        intermediate_data_type_ != DataType_t::NOT_SET) {
      return intermediate_data_type_;
    }
    return io_data_type_;
  }

  [[nodiscard]] DataType_t resolved_compute_data_type(
      DataType_t operation_data_type,
      const Tensor_attributes& input) const {
    if (operation_data_type != DataType_t::NOT_SET) {
      return operation_data_type;
    }
    if (compute_data_type_ != DataType_t::NOT_SET) {
      return compute_data_type_;
    }
    return resolved_data_type(input);
  }

  void set_operation_metadata(
      flagdnn::OperationDescriptor& operation,
      std::string_view name,
      DataType_t operation_data_type,
      const Tensor_attributes& input) const {
    operation.set_name(name);
    operation.set_compute_data_type(native_data_type(
        resolved_compute_data_type(operation_data_type, input)));
  }

  [[nodiscard]] flagdnn::TensorDescriptor make_descriptor(
      const Tensor_attributes& tensor,
      std::string_view role) const {
    if (tensor.get_uid() <= 0) {
      throw std::invalid_argument(
          std::string(role) + " tensor UID must be greater than zero");
    }
    if (tensor.get_dim().size() != tensor.get_stride().size()) {
      throw std::invalid_argument(
          std::string(role) +
          " tensor dimensions and strides must have the same rank");
    }
    flagdnn::TensorDescriptor result(tensor.get_uid(),
                                     native_data_type(
                                         resolved_data_type(tensor)),
                                     tensor.get_dim(),
                                     tensor.get_stride());
    result.set_alignment(tensor.get_alignment());
    result.set_virtual(tensor.get_is_virtual());
    return result;
  }

  static flagdnnDataType_t native_data_type(DataType_t data_type) {
    switch (data_type) {
      case DataType_t::FLOAT:
        return FLAGDNN_DATA_FLOAT32;
      case DataType_t::HALF:
        return FLAGDNN_DATA_FLOAT16;
      case DataType_t::BFLOAT16:
        return FLAGDNN_DATA_BFLOAT16;
      case DataType_t::BOOLEAN:
        return FLAGDNN_DATA_BOOLEAN;
      case DataType_t::FP8_E4M3:
        return FLAGDNN_DATA_FP8_E4M3;
      case DataType_t::FP8_E5M2:
        return FLAGDNN_DATA_FP8_E5M2;
      case DataType_t::NOT_SET:
        break;
    }
    throw std::invalid_argument("frontend tensor data type is not set");
  }

  static flagdnnPointwiseMode_t native_pointwise_mode(
      PointwiseMode_t mode) {
    switch (mode) {
      case PointwiseMode_t::ADD:
        return FLAGDNN_POINTWISE_ADD;
      case PointwiseMode_t::SUB:
        return FLAGDNN_POINTWISE_SUB;
      case PointwiseMode_t::MUL:
        return FLAGDNN_POINTWISE_MUL;
      case PointwiseMode_t::DIV:
        return FLAGDNN_POINTWISE_DIV;
      case PointwiseMode_t::MIN:
        return FLAGDNN_POINTWISE_MIN;
      case PointwiseMode_t::MAX:
        return FLAGDNN_POINTWISE_MAX;
      case PointwiseMode_t::MOD:
        return FLAGDNN_POINTWISE_MOD;
      case PointwiseMode_t::POW:
        return FLAGDNN_POINTWISE_POW;
      case PointwiseMode_t::LOGICAL_NOT:
        return FLAGDNN_POINTWISE_LOGICAL_NOT;
      case PointwiseMode_t::CMP_EQ:
        return FLAGDNN_POINTWISE_CMP_EQ;
      case PointwiseMode_t::CMP_NEQ:
        return FLAGDNN_POINTWISE_CMP_NEQ;
      case PointwiseMode_t::CMP_GT:
        return FLAGDNN_POINTWISE_CMP_GT;
      case PointwiseMode_t::CMP_GE:
        return FLAGDNN_POINTWISE_CMP_GE;
      case PointwiseMode_t::CMP_LT:
        return FLAGDNN_POINTWISE_CMP_LT;
      case PointwiseMode_t::CMP_LE:
        return FLAGDNN_POINTWISE_CMP_LE;
      case PointwiseMode_t::LOGICAL_AND:
        return FLAGDNN_POINTWISE_LOGICAL_AND;
      case PointwiseMode_t::LOGICAL_OR:
        return FLAGDNN_POINTWISE_LOGICAL_OR;
      case PointwiseMode_t::SIGMOID_BWD:
        return FLAGDNN_POINTWISE_SIGMOID_BWD;
      case PointwiseMode_t::BINARY_SELECT:
        return FLAGDNN_POINTWISE_BINARY_SELECT;
      case PointwiseMode_t::SIGMOID_FWD:
        return FLAGDNN_POINTWISE_SIGMOID_FWD;
      case PointwiseMode_t::TANH_FWD:
        return FLAGDNN_POINTWISE_TANH_FWD;
      case PointwiseMode_t::ELU_FWD:
        return FLAGDNN_POINTWISE_ELU_FWD;
      case PointwiseMode_t::GELU_FWD:
        return FLAGDNN_POINTWISE_GELU_FWD;
      case PointwiseMode_t::SOFTPLUS_FWD:
        return FLAGDNN_POINTWISE_SOFTPLUS_FWD;
      case PointwiseMode_t::SWISH_FWD:
        return FLAGDNN_POINTWISE_SWISH_FWD;
      case PointwiseMode_t::GELU_APPROX_TANH_FWD:
        return FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD;
      case PointwiseMode_t::RELU_FWD:
        return FLAGDNN_POINTWISE_RELU_FWD;
      case PointwiseMode_t::SQRT:
        return FLAGDNN_POINTWISE_SQRT;
      case PointwiseMode_t::ERF:
        return FLAGDNN_POINTWISE_ERF;
      case PointwiseMode_t::IDENTITY:
        return FLAGDNN_POINTWISE_IDENTITY;
      case PointwiseMode_t::EXP:
        return FLAGDNN_POINTWISE_EXP;
      case PointwiseMode_t::LOG:
        return FLAGDNN_POINTWISE_LOG;
      case PointwiseMode_t::NEG:
        return FLAGDNN_POINTWISE_NEG;
      case PointwiseMode_t::ABS:
        return FLAGDNN_POINTWISE_ABS;
      case PointwiseMode_t::CEIL:
        return FLAGDNN_POINTWISE_CEIL;
      case PointwiseMode_t::COS:
        return FLAGDNN_POINTWISE_COS;
      case PointwiseMode_t::FLOOR:
        return FLAGDNN_POINTWISE_FLOOR;
      case PointwiseMode_t::RSQRT:
        return FLAGDNN_POINTWISE_RSQRT;
      case PointwiseMode_t::SIN:
        return FLAGDNN_POINTWISE_SIN;
      case PointwiseMode_t::TAN:
        return FLAGDNN_POINTWISE_TAN;
      case PointwiseMode_t::RECIPROCAL:
        return FLAGDNN_POINTWISE_RECIPROCAL;
      case PointwiseMode_t::NOT_SET:
        break;
    }
    throw std::invalid_argument("frontend pointwise mode is not set");
  }

  static flagdnnPointwiseAttributes_t native_pointwise_attributes(
      const Pointwise_attributes& attributes) {
    flagdnnPointwiseAttributes_t result =
        FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
    if (attributes.relu_lower_clip_.has_value()) {
      result.flags |= FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP;
      result.relu_lower_clip = *attributes.relu_lower_clip_;
    }
    if (attributes.relu_upper_clip_.has_value()) {
      result.flags |= FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP;
      result.relu_upper_clip = *attributes.relu_upper_clip_;
    }
    if (attributes.relu_lower_clip_slope_.has_value()) {
      result.flags |=
          FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE;
      result.relu_lower_clip_slope =
          *attributes.relu_lower_clip_slope_;
    }
    if (attributes.swish_beta_.has_value()) {
      result.flags |= FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA;
      result.swish_beta = *attributes.swish_beta_;
    }
    if (attributes.elu_alpha_.has_value()) {
      result.flags |= FLAGDNN_POINTWISE_ATTRIBUTE_ELU_ALPHA;
      result.elu_alpha = *attributes.elu_alpha_;
    }
    if (attributes.softplus_beta_.has_value()) {
      result.flags |= FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA;
      result.softplus_beta = *attributes.softplus_beta_;
    }
    return result;
  }

  static flagdnnSdpaAttributes_t native_sdpa_attributes(
      const SDPA_attributes& attributes) {
    flagdnnSdpaAttributes_t result = FLAGDNN_SDPA_ATTRIBUTES_INITIALIZER;
    if (attributes.get_attn_scale().has_value()) {
      result.flags |= FLAGDNN_SDPA_ATTRIBUTE_ATTN_SCALE;
      result.attn_scale = *attributes.get_attn_scale();
    }
    if (attributes.get_diagonal_band_left_bound().has_value()) {
      result.flags |= FLAGDNN_SDPA_ATTRIBUTE_LEFT_BOUND;
      result.diagonal_band_left_bound =
          *attributes.get_diagonal_band_left_bound();
    }
    if (attributes.get_diagonal_band_right_bound().has_value()) {
      result.flags |= FLAGDNN_SDPA_ATTRIBUTE_RIGHT_BOUND;
      result.diagonal_band_right_bound =
          *attributes.get_diagonal_band_right_bound();
    }
    result.diagonal_alignment =
        attributes.get_diagonal_alignment() ==
                DiagonalAlignment_t::BOTTOM_RIGHT
            ? FLAGDNN_DIAGONAL_ALIGNMENT_BOTTOM_RIGHT
            : FLAGDNN_DIAGONAL_ALIGNMENT_TOP_LEFT;
    result.generate_stats = attributes.get_generate_stats() ? 1 : 0;
    return result;
  }

  static flagdnnSdpaAttributes_t native_sdpa_attributes(
      const SDPA_backward_attributes& attributes) {
    flagdnnSdpaAttributes_t result = FLAGDNN_SDPA_ATTRIBUTES_INITIALIZER;
    if (attributes.get_attn_scale().has_value()) {
      result.flags |= FLAGDNN_SDPA_ATTRIBUTE_ATTN_SCALE;
      result.attn_scale = *attributes.get_attn_scale();
    }
    if (attributes.get_diagonal_band_left_bound().has_value()) {
      result.flags |= FLAGDNN_SDPA_ATTRIBUTE_LEFT_BOUND;
      result.diagonal_band_left_bound =
          *attributes.get_diagonal_band_left_bound();
    }
    if (attributes.get_diagonal_band_right_bound().has_value()) {
      result.flags |= FLAGDNN_SDPA_ATTRIBUTE_RIGHT_BOUND;
      result.diagonal_band_right_bound =
          *attributes.get_diagonal_band_right_bound();
    }
    result.diagonal_alignment =
        attributes.get_diagonal_alignment() ==
                DiagonalAlignment_t::BOTTOM_RIGHT
            ? FLAGDNN_DIAGONAL_ALIGNMENT_BOTTOM_RIGHT
            : FLAGDNN_DIAGONAL_ALIGNMENT_TOP_LEFT;
    result.generate_stats = 1;
    return result;
  }

  static flagdnnReductionMode_t native_reduction_mode(
      ReductionMode_t mode) {
    switch (mode) {
      case ReductionMode_t::ADD:
        return FLAGDNN_REDUCTION_ADD;
      case ReductionMode_t::AVG:
        return FLAGDNN_REDUCTION_AVG;
      case ReductionMode_t::MUL:
        return FLAGDNN_REDUCTION_MUL;
      case ReductionMode_t::NOT_SET:
        break;
    }
    throw std::invalid_argument("frontend reduction mode is not set");
  }

  static std::int32_t checked_axis(std::int64_t axis) {
    if (axis < std::numeric_limits<std::int32_t>::min() ||
        axis > std::numeric_limits<std::int32_t>::max()) {
      throw std::invalid_argument("reduction axis is outside int32 range");
    }
    return static_cast<std::int32_t>(axis);
  }

  static void append_uid(std::vector<std::int64_t>& uids,
                         std::int64_t uid) {
    if (std::find(uids.begin(), uids.end(), uid) == uids.end()) {
      uids.push_back(uid);
    }
  }

  static void append_external_uid(
      std::vector<std::int64_t>& uids,
      const Tensor_attributes& tensor) {
    if (!tensor.get_is_virtual()) {
      append_uid(uids, tensor.get_uid());
    }
  }

  void assign_missing_virtual_uids() {
    std::int64_t candidate = 1;
    for (const Tensor& tensor : tensors_) {
      if (tensor == nullptr || !tensor->get_is_virtual() ||
          tensor->get_uid() > 0) {
        continue;
      }
      for (;;) {
        const bool used = std::any_of(
            tensors_.begin(), tensors_.end(), [&](const Tensor& existing) {
              return existing != nullptr &&
                     existing->get_uid() == candidate;
            });
        if (!used) {
          break;
        }
        if (candidate == std::numeric_limits<std::int64_t>::max()) {
          throw std::overflow_error("frontend tensor UID space exhausted");
        }
        ++candidate;
      }
      tensor->set_uid(candidate);
      if (candidate != std::numeric_limits<std::int64_t>::max()) {
        ++candidate;
      }
    }
  }

  struct LoweredGraph {
    std::unique_ptr<flagdnn::Graph> graph;
    std::vector<std::int64_t> required_uids;
  };

  [[nodiscard]] LoweredGraph lower_to_native_graph(bool finalize) {
    if (nodes_.empty()) {
      throw std::invalid_argument("cannot lower an empty frontend graph");
    }

    auto native_graph = std::make_unique<flagdnn::Graph>();
    native_graph->set_name(name_);
    assign_missing_virtual_uids();
    std::vector<std::int64_t> required_uids;
    for (const Node& node : nodes_) {
      flagdnn::TensorDescriptor input = make_descriptor(*node.input, "input");
      flagdnn::TensorDescriptor output = make_descriptor(*node.output, "output");
      append_external_uid(required_uids, *node.input);
      append_external_uid(required_uids, *node.output);

      switch (node.kind) {
        case NodeKind::kPointwise: {
          flagdnn::OperationDescriptor operation(FLAGDNN_OPERATION_POINTWISE);
          operation.set_pointwise(
              input,
              native_pointwise_mode(node.pointwise.get_mode()),
              output,
              native_pointwise_attributes(node.pointwise));
          set_operation_metadata(operation,
                                 node.pointwise.get_name(),
                                 node.pointwise.get_compute_data_type(),
                                 *node.input);
          native_graph->add(operation);
          break;
        }
        case NodeKind::kBinaryPointwise: {
          flagdnn::TensorDescriptor second =
              make_descriptor(*node.second, "second input");
          append_external_uid(required_uids, *node.second);
          flagdnn::OperationDescriptor operation(FLAGDNN_OPERATION_POINTWISE);
          operation.set_pointwise(
              input,
              second,
              native_pointwise_mode(node.pointwise.get_mode()),
              output,
              node.pointwise.get_alpha());
          set_operation_metadata(operation,
                                 node.pointwise.get_name(),
                                 node.pointwise.get_compute_data_type(),
                                 *node.input);
          native_graph->add(operation);
          break;
        }
        case NodeKind::kTernaryPointwise: {
          flagdnn::TensorDescriptor second =
              make_descriptor(*node.second, "B input");
          flagdnn::TensorDescriptor third =
              make_descriptor(*node.third, "T input");
          append_external_uid(required_uids, *node.second);
          append_external_uid(required_uids, *node.third);
          flagdnn::OperationDescriptor operation(FLAGDNN_OPERATION_POINTWISE);
          operation.set_pointwise(
              input,
              second,
              third,
              native_pointwise_mode(node.pointwise.get_mode()),
              output);
          set_operation_metadata(operation,
                                 node.pointwise.get_name(),
                                 node.pointwise.get_compute_data_type(),
                                 *node.input);
          native_graph->add(operation);
          break;
        }
        case NodeKind::kReduction: {
          flagdnn::OperationDescriptor operation(FLAGDNN_OPERATION_REDUCTION);
          operation.set_reduction(
              input,
              native_reduction_mode(node.reduction_attributes.get_mode()),
              checked_axis(node.reduction_attributes.get_axis()),
              node.reduction_attributes.get_keep_dimensions(),
              output);
          set_operation_metadata(
              operation,
              node.reduction_attributes.get_name(),
              node.reduction_attributes.get_compute_data_type(),
              *node.input);
          native_graph->add(operation);
          break;
        }
        case NodeKind::kMatmul: {
          flagdnn::TensorDescriptor second =
              make_descriptor(*node.second, "B input");
          append_external_uid(required_uids, *node.second);
          flagdnn::OperationDescriptor operation(FLAGDNN_OPERATION_MATMUL);
          operation.set_matmul(input, second, output);
          set_operation_metadata(
              operation,
              node.matmul_attributes.get_name(),
              node.matmul_attributes.get_compute_data_type(),
              *node.input);
          native_graph->add(operation);
          break;
        }
        case NodeKind::kSdpa: {
          if (node.attention_inputs.size() < 3 ||
              node.attention_inputs.size() > 4 ||
              node.attention_outputs.size() != 2) {
            throw std::logic_error("frontend SDPA node is malformed");
          }
          flagdnn::TensorDescriptor k =
              make_descriptor(*node.attention_inputs[1], "SDPA K");
          flagdnn::TensorDescriptor v =
              make_descriptor(*node.attention_inputs[2], "SDPA V");
          flagdnn::TensorDescriptor stats =
              make_descriptor(*node.attention_outputs[1], "SDPA stats");
          std::unique_ptr<flagdnn::TensorDescriptor> bias;
          if (node.attention_inputs.size() == 4) {
            bias = std::make_unique<flagdnn::TensorDescriptor>(
                make_descriptor(*node.attention_inputs[3], "SDPA bias"));
          }
          for (const Tensor& tensor : node.attention_inputs) {
            append_external_uid(required_uids, *tensor);
          }
          for (const Tensor& tensor : node.attention_outputs) {
            append_external_uid(required_uids, *tensor);
          }
          flagdnn::OperationDescriptor operation(FLAGDNN_OPERATION_SDPA);
          operation.set_sdpa(input,
                             k,
                             v,
                             bias.get(),
                             output,
                             stats,
                             native_sdpa_attributes(node.sdpa_attributes));
          set_operation_metadata(
              operation,
              node.sdpa_attributes.get_name(),
              node.sdpa_attributes.get_compute_data_type(),
              *node.input);
          native_graph->add(operation);
          break;
        }
        case NodeKind::kSdpaBackward: {
          if (node.attention_inputs.size() < 6 ||
              node.attention_inputs.size() > 7 ||
              node.attention_outputs.size() < 3 ||
              node.attention_outputs.size() > 4) {
            throw std::logic_error(
                "frontend SDPA backward node is malformed");
          }
          flagdnn::TensorDescriptor k =
              make_descriptor(*node.attention_inputs[1], "SDPA backward K");
          flagdnn::TensorDescriptor v =
              make_descriptor(*node.attention_inputs[2], "SDPA backward V");
          flagdnn::TensorDescriptor primal_output =
              make_descriptor(*node.attention_inputs[3], "SDPA backward O");
          flagdnn::TensorDescriptor doutput =
              make_descriptor(*node.attention_inputs[4], "SDPA backward dO");
          flagdnn::TensorDescriptor stats =
              make_descriptor(
                  *node.attention_inputs[5], "SDPA backward stats");
          flagdnn::TensorDescriptor dk =
              make_descriptor(
                  *node.attention_outputs[1], "SDPA backward dK");
          flagdnn::TensorDescriptor dv =
              make_descriptor(
                  *node.attention_outputs[2], "SDPA backward dV");
          std::unique_ptr<flagdnn::TensorDescriptor> bias;
          if (node.attention_inputs.size() == 7) {
            bias = std::make_unique<flagdnn::TensorDescriptor>(
                make_descriptor(
                    *node.attention_inputs[6], "SDPA backward bias"));
          }
          std::unique_ptr<flagdnn::TensorDescriptor> dbias;
          if (node.attention_outputs.size() == 4) {
            dbias = std::make_unique<flagdnn::TensorDescriptor>(
                make_descriptor(
                    *node.attention_outputs[3], "SDPA backward dBias"));
          }
          for (const Tensor& tensor : node.attention_inputs) {
            append_external_uid(required_uids, *tensor);
          }
          for (const Tensor& tensor : node.attention_outputs) {
            append_external_uid(required_uids, *tensor);
          }
          flagdnn::OperationDescriptor operation(
              FLAGDNN_OPERATION_SDPA_BACKWARD);
          operation.set_sdpa_backward(
              input,
              k,
              v,
              primal_output,
              doutput,
              stats,
              bias.get(),
              output,
              dk,
              dv,
              dbias.get(),
              native_sdpa_attributes(node.sdpa_backward_attributes));
          set_operation_metadata(
              operation,
              node.sdpa_backward_attributes.get_name(),
              node.sdpa_backward_attributes.get_compute_data_type(),
              *node.input);
          native_graph->add(operation);
          break;
        }
        case NodeKind::kSdpaFp8: {
          if (node.attention_inputs.size() < 9 ||
              node.attention_inputs.size() > 10 ||
              node.attention_outputs.size() != 4) {
            throw std::logic_error("frontend FP8 SDPA node is malformed");
          }
          auto descriptor = [&](std::size_t index,
                                std::string_view role) {
            return make_descriptor(*node.attention_inputs[index], role);
          };
          flagdnn::TensorDescriptor k = descriptor(1, "FP8 SDPA K");
          flagdnn::TensorDescriptor v = descriptor(2, "FP8 SDPA V");
          flagdnn::TensorDescriptor descale_q =
              descriptor(3, "FP8 SDPA descale Q");
          flagdnn::TensorDescriptor descale_k =
              descriptor(4, "FP8 SDPA descale K");
          flagdnn::TensorDescriptor descale_v =
              descriptor(5, "FP8 SDPA descale V");
          flagdnn::TensorDescriptor descale_s =
              descriptor(6, "FP8 SDPA descale S");
          flagdnn::TensorDescriptor scale_s =
              descriptor(7, "FP8 SDPA scale S");
          flagdnn::TensorDescriptor scale_o =
              descriptor(8, "FP8 SDPA scale O");
          std::unique_ptr<flagdnn::TensorDescriptor> bias;
          if (node.attention_inputs.size() == 10) {
            bias = std::make_unique<flagdnn::TensorDescriptor>(
                descriptor(9, "FP8 SDPA bias"));
          }
          flagdnn::TensorDescriptor stats = make_descriptor(
              *node.attention_outputs[1], "FP8 SDPA stats");
          flagdnn::TensorDescriptor amax_s = make_descriptor(
              *node.attention_outputs[2], "FP8 SDPA amax S");
          flagdnn::TensorDescriptor amax_o = make_descriptor(
              *node.attention_outputs[3], "FP8 SDPA amax O");
          for (const Tensor& tensor : node.attention_inputs) {
            append_external_uid(required_uids, *tensor);
          }
          for (const Tensor& tensor : node.attention_outputs) {
            append_external_uid(required_uids, *tensor);
          }
          flagdnn::OperationDescriptor operation(
              FLAGDNN_OPERATION_SDPA_FP8);
          operation.set_sdpa_fp8(
              input,
              k,
              v,
              descale_q,
              descale_k,
              descale_v,
              descale_s,
              scale_s,
              scale_o,
              bias.get(),
              output,
              stats,
              amax_s,
              amax_o,
              native_sdpa_attributes(node.sdpa_attributes));
          set_operation_metadata(
              operation,
              node.sdpa_attributes.get_name(),
              node.sdpa_attributes.get_compute_data_type(),
              *node.input);
          native_graph->add(operation);
          break;
        }
        case NodeKind::kSdpaFp8Backward: {
          if (node.attention_inputs.size() != 18 ||
              node.attention_outputs.size() != 7) {
            throw std::logic_error(
                "frontend FP8 SDPA backward node is malformed");
          }
          auto input_descriptor = [&](std::size_t index,
                                      std::string_view role) {
            return make_descriptor(*node.attention_inputs[index], role);
          };
          auto output_descriptor = [&](std::size_t index,
                                       std::string_view role) {
            return make_descriptor(*node.attention_outputs[index], role);
          };
          flagdnn::TensorDescriptor k =
              input_descriptor(1, "FP8 SDPA backward K");
          flagdnn::TensorDescriptor v =
              input_descriptor(2, "FP8 SDPA backward V");
          flagdnn::TensorDescriptor primal_output =
              input_descriptor(3, "FP8 SDPA backward O");
          flagdnn::TensorDescriptor doutput =
              input_descriptor(4, "FP8 SDPA backward dO");
          flagdnn::TensorDescriptor stats =
              input_descriptor(5, "FP8 SDPA backward stats");
          std::array<flagdnn::TensorDescriptor, 12> scales{
              input_descriptor(6, "FP8 SDPA backward descale Q"),
              input_descriptor(7, "FP8 SDPA backward descale K"),
              input_descriptor(8, "FP8 SDPA backward descale V"),
              input_descriptor(9, "FP8 SDPA backward descale O"),
              input_descriptor(10, "FP8 SDPA backward descale dO"),
              input_descriptor(11, "FP8 SDPA backward descale S"),
              input_descriptor(12, "FP8 SDPA backward descale dP"),
              input_descriptor(13, "FP8 SDPA backward scale S"),
              input_descriptor(14, "FP8 SDPA backward scale dQ"),
              input_descriptor(15, "FP8 SDPA backward scale dK"),
              input_descriptor(16, "FP8 SDPA backward scale dV"),
              input_descriptor(17, "FP8 SDPA backward scale dP")};
          flagdnn::TensorDescriptor dk =
              output_descriptor(1, "FP8 SDPA backward dK");
          flagdnn::TensorDescriptor dv =
              output_descriptor(2, "FP8 SDPA backward dV");
          std::array<flagdnn::TensorDescriptor, 4> amax{
              output_descriptor(3, "FP8 SDPA backward amax dQ"),
              output_descriptor(4, "FP8 SDPA backward amax dK"),
              output_descriptor(5, "FP8 SDPA backward amax dV"),
              output_descriptor(6, "FP8 SDPA backward amax dP")};
          for (const Tensor& tensor : node.attention_inputs) {
            append_external_uid(required_uids, *tensor);
          }
          for (const Tensor& tensor : node.attention_outputs) {
            append_external_uid(required_uids, *tensor);
          }
          flagdnn::OperationDescriptor operation(
              FLAGDNN_OPERATION_SDPA_FP8_BACKWARD);
          operation.set_sdpa_fp8_backward(
              input,
              k,
              v,
              primal_output,
              doutput,
              stats,
              scales[0],
              scales[1],
              scales[2],
              scales[3],
              scales[4],
              scales[5],
              scales[6],
              scales[7],
              scales[8],
              scales[9],
              scales[10],
              scales[11],
              output,
              dk,
              dv,
              amax[0],
              amax[1],
              amax[2],
              amax[3],
              native_sdpa_attributes(node.sdpa_backward_attributes));
          set_operation_metadata(
              operation,
              node.sdpa_backward_attributes.get_name(),
              node.sdpa_backward_attributes.get_compute_data_type(),
              *node.input);
          native_graph->add(operation);
          break;
        }
        case NodeKind::kReshape: {
          flagdnn::OperationDescriptor operation("reshape");
          operation.set_input("input", input);
          operation.set_output("output", output);
          operation.set_attribute(
              "reshape_mode",
              static_cast<std::int64_t>(
                  node.reshape_attributes.get_reshape_mode()));
          set_operation_metadata(
              operation,
              node.reshape_attributes.get_name(),
              node.reshape_attributes.get_compute_data_type(),
              *node.input);
          operation.finalize();
          native_graph->add(operation);
          break;
        }
        case NodeKind::kTranspose: {
          flagdnn::OperationDescriptor operation("transpose");
          operation.set_input("input", input);
          operation.set_output("output", output);
          operation.set_attribute(
              "permutation",
              node.transpose_attributes.get_permutation());
          set_operation_metadata(
              operation,
              node.transpose_attributes.get_name(),
              node.transpose_attributes.get_compute_data_type(),
              *node.input);
          operation.finalize();
          native_graph->add(operation);
          break;
        }
        case NodeKind::kSlice: {
          const auto& slices = node.slice_attributes.get_slices();
          std::vector<std::int64_t> starts;
          std::vector<std::int64_t> limits;
          starts.reserve(slices.size());
          limits.reserve(slices.size());
          for (const auto& range : slices) {
            starts.push_back(range.first);
            limits.push_back(range.second);
          }
          std::vector<std::int64_t> strides =
              node.slice_attributes.get_strides();
          strides.resize(slices.size(), 1);
          flagdnn::OperationDescriptor operation("slice");
          operation.set_input("input", input);
          operation.set_output("output", output);
          operation.set_attribute("starts", starts);
          operation.set_attribute("limits", limits);
          operation.set_attribute("slice_strides", strides);
          set_operation_metadata(
              operation,
              node.slice_attributes.get_name(),
              node.slice_attributes.get_compute_data_type(),
              *node.input);
          operation.finalize();
          native_graph->add(operation);
          break;
        }
        case NodeKind::kConvolutionDgrad: {
          flagdnn::TensorDescriptor filter =
              make_descriptor(*node.second, "filter");
          append_external_uid(required_uids, *node.second);
          if (node.input->get_dim().size() < 3) {
            throw std::invalid_argument(
                "convolution Dgrad tensors must have rank at least three");
          }
          const std::size_t spatial_rank =
              node.input->get_dim().size() - 2;
          flagdnn::OperationDescriptor operation("convolution_dgrad");
          operation.set_input("dy", input);
          operation.set_input("w", filter);
          operation.set_output("dx", output);
          operation.set_attribute(
              "spatial_rank", static_cast<std::int64_t>(spatial_rank));
          operation.set_attribute(
              "pre_padding",
              normalized_spatial_attribute(
                  node.convolution_dgrad_attributes.get_pre_padding(),
                  spatial_rank,
                  0,
                  "pre-padding"));
          operation.set_attribute(
              "post_padding",
              normalized_spatial_attribute(
                  node.convolution_dgrad_attributes.get_post_padding(),
                  spatial_rank,
                  0,
                  "post-padding"));
          operation.set_attribute(
              "stride",
              normalized_spatial_attribute(
                  node.convolution_dgrad_attributes.get_stride(),
                  spatial_rank,
                  1,
                  "stride"));
          operation.set_attribute(
              "dilation",
              normalized_spatial_attribute(
                  node.convolution_dgrad_attributes.get_dilation(),
                  spatial_rank,
                  1,
                  "dilation"));
          operation.set_attribute(
              "groups", node.convolution_dgrad_attributes.get_groups());
          operation.set_attribute(
              "convolution_mode",
              static_cast<std::int64_t>(
                  node.convolution_dgrad_attributes
                      .get_convolution_mode()));
          set_operation_metadata(
              operation,
              node.convolution_dgrad_attributes.get_name(),
              node.convolution_dgrad_attributes.get_compute_data_type(),
              *node.input);
          operation.finalize();
          native_graph->add(operation);
          break;
        }
        case NodeKind::kConvolutionWgrad: {
          flagdnn::TensorDescriptor image =
              make_descriptor(*node.second, "image");
          append_external_uid(required_uids, *node.second);
          if (node.input->get_dim().size() < 3) {
            throw std::invalid_argument(
                "convolution Wgrad tensors must have rank at least three");
          }
          const std::size_t spatial_rank =
              node.input->get_dim().size() - 2;
          flagdnn::OperationDescriptor operation("convolution_wgrad");
          operation.set_input("dy", input);
          operation.set_input("x", image);
          operation.set_output("dw", output);
          operation.set_attribute(
              "spatial_rank", static_cast<std::int64_t>(spatial_rank));
          operation.set_attribute(
              "pre_padding",
              normalized_spatial_attribute(
                  node.convolution_wgrad_attributes.get_pre_padding(),
                  spatial_rank,
                  0,
                  "pre-padding"));
          operation.set_attribute(
              "post_padding",
              normalized_spatial_attribute(
                  node.convolution_wgrad_attributes.get_post_padding(),
                  spatial_rank,
                  0,
                  "post-padding"));
          operation.set_attribute(
              "stride",
              normalized_spatial_attribute(
                  node.convolution_wgrad_attributes.get_stride(),
                  spatial_rank,
                  1,
                  "stride"));
          operation.set_attribute(
              "dilation",
              normalized_spatial_attribute(
                  node.convolution_wgrad_attributes.get_dilation(),
                  spatial_rank,
                  1,
                  "dilation"));
          operation.set_attribute(
              "groups", node.convolution_wgrad_attributes.get_groups());
          operation.set_attribute(
              "convolution_mode",
              static_cast<std::int64_t>(
                  node.convolution_wgrad_attributes
                      .get_convolution_mode()));
          set_operation_metadata(
              operation,
              node.convolution_wgrad_attributes.get_name(),
              node.convolution_wgrad_attributes.get_compute_data_type(),
              *node.input);
          operation.finalize();
          native_graph->add(operation);
          break;
        }
        case NodeKind::kLayernorm: {
          const double epsilon = compile_time_scalar(
              node.layernorm_attributes.epsilon_, "layernorm epsilon");
          flagdnn::TensorDescriptor scale =
              make_descriptor(*node.second, "scale");
          flagdnn::TensorDescriptor bias =
              make_descriptor(*node.third, "bias");
          flagdnn::TensorDescriptor mean =
              make_descriptor(*node.second_output, "mean");
          flagdnn::TensorDescriptor inv_variance =
              make_descriptor(*node.third_output, "inverse variance");
          append_external_uid(required_uids, *node.second);
          append_external_uid(required_uids, *node.third);
          append_external_uid(required_uids, *node.second_output);
          append_external_uid(required_uids, *node.third_output);
          flagdnn::OperationDescriptor operation("layernorm");
          operation.set_input("x", input);
          operation.set_input("scale", scale);
          operation.set_input("bias", bias);
          operation.set_output("y", output);
          operation.set_output("mean", mean);
          operation.set_output("inv_variance", inv_variance);
          operation.set_attribute("epsilon", epsilon);
          operation.set_attribute(
              "forward_phase",
              static_cast<std::int64_t>(
                  node.layernorm_attributes.get_forward_phase()));
          set_operation_metadata(
              operation,
              node.layernorm_attributes.get_name(),
              node.layernorm_attributes.get_compute_data_type(),
              *node.input);
          operation.finalize();
          native_graph->add(operation);
          break;
        }
        case NodeKind::kRmsnorm: {
          const double epsilon = compile_time_scalar(
              node.rmsnorm_attributes.epsilon_, "rmsnorm epsilon");
          flagdnn::TensorDescriptor scale =
              make_descriptor(*node.second, "scale");
          flagdnn::TensorDescriptor bias =
              make_descriptor(*node.third, "bias");
          flagdnn::TensorDescriptor inv_variance =
              make_descriptor(*node.second_output, "inverse variance");
          append_external_uid(required_uids, *node.second);
          append_external_uid(required_uids, *node.third);
          append_external_uid(required_uids, *node.second_output);
          flagdnn::OperationDescriptor operation("rmsnorm");
          operation.set_input("x", input);
          operation.set_input("scale", scale);
          operation.set_input("bias", bias);
          operation.set_output("y", output);
          operation.set_output("inv_variance", inv_variance);
          operation.set_attribute("epsilon", epsilon);
          operation.set_attribute(
              "forward_phase",
              static_cast<std::int64_t>(
                  node.rmsnorm_attributes.get_forward_phase()));
          set_operation_metadata(
              operation,
              node.rmsnorm_attributes.get_name(),
              node.rmsnorm_attributes.get_compute_data_type(),
              *node.input);
          operation.finalize();
          native_graph->add(operation);
          break;
        }

        case NodeKind::kBatchnorm: {
          const double epsilon = compile_time_scalar(
              node.batchnorm_attributes.epsilon_, "batchnorm epsilon");
          const double momentum = compile_time_scalar(
              node.batchnorm_attributes.momentum_, "batchnorm momentum");
          flagdnn::TensorDescriptor scale =
              make_descriptor(*node.second, "scale");
          flagdnn::TensorDescriptor bias =
              make_descriptor(*node.third, "bias");
          flagdnn::TensorDescriptor previous_running_mean =
              make_descriptor(*node.fourth, "previous running mean");
          flagdnn::TensorDescriptor previous_running_variance =
              make_descriptor(*node.fifth, "previous running variance");
          flagdnn::TensorDescriptor mean =
              make_descriptor(*node.second_output, "mean");
          flagdnn::TensorDescriptor inv_variance =
              make_descriptor(*node.third_output, "inverse variance");
          flagdnn::TensorDescriptor next_running_mean =
              make_descriptor(*node.fourth_output, "next running mean");
          flagdnn::TensorDescriptor next_running_variance =
              make_descriptor(*node.fifth_output, "next running variance");
          append_external_uid(required_uids, *node.second);
          append_external_uid(required_uids, *node.third);
          append_external_uid(required_uids, *node.fourth);
          append_external_uid(required_uids, *node.fifth);
          append_external_uid(required_uids, *node.second_output);
          append_external_uid(required_uids, *node.third_output);
          append_external_uid(required_uids, *node.fourth_output);
          append_external_uid(required_uids, *node.fifth_output);
          flagdnn::OperationDescriptor operation("batchnorm");
          operation.set_input("x", input);
          operation.set_input("scale", scale);
          operation.set_input("bias", bias);
          operation.set_input(
              "previous_running_mean", previous_running_mean);
          operation.set_input(
              "previous_running_variance", previous_running_variance);
          operation.set_output("y", output);
          operation.set_output("mean", mean);
          operation.set_output("inv_variance", inv_variance);
          operation.set_output("next_running_mean", next_running_mean);
          operation.set_output(
              "next_running_variance", next_running_variance);
          operation.set_attribute("epsilon", epsilon);
          operation.set_attribute("momentum", momentum);
          set_operation_metadata(
              operation,
              node.batchnorm_attributes.get_name(),
              node.batchnorm_attributes.get_compute_data_type(),
              *node.input);
          operation.finalize();
          native_graph->add(operation);
          break;
        }

        case NodeKind::kBatchnormInference: {
          flagdnn::TensorDescriptor mean =
              make_descriptor(*node.second, "mean");
          flagdnn::TensorDescriptor inv_variance =
              make_descriptor(*node.third, "inverse variance");
          flagdnn::TensorDescriptor scale =
              make_descriptor(*node.fourth, "scale");
          flagdnn::TensorDescriptor bias =
              make_descriptor(*node.fifth, "bias");
          append_external_uid(required_uids, *node.second);
          append_external_uid(required_uids, *node.third);
          append_external_uid(required_uids, *node.fourth);
          append_external_uid(required_uids, *node.fifth);
          flagdnn::OperationDescriptor operation("batchnorm_inference");
          operation.set_input("x", input);
          operation.set_input("mean", mean);
          operation.set_input("inv_variance", inv_variance);
          operation.set_input("scale", scale);
          operation.set_input("bias", bias);
          operation.set_output("y", output);
          set_operation_metadata(
              operation,
              node.batchnorm_inference_attributes.get_name(),
              node.batchnorm_inference_attributes.get_compute_data_type(),
              *node.input);
          operation.finalize();
          native_graph->add(operation);
          break;
        }
        case NodeKind::kConvolution: {
          flagdnn::TensorDescriptor filter =
              make_descriptor(*node.second, "filter");
          append_external_uid(required_uids, *node.second);
          flagdnn::OperationDescriptor operation(
              FLAGDNN_OPERATION_CONVOLUTION_FPROP);
          operation.set_convolution_fprop(
              input,
              filter,
              node.convolution_attributes.get_pre_padding(),
              node.convolution_attributes.get_post_padding(),
              node.convolution_attributes.get_stride(),
              node.convolution_attributes.get_dilation(),
              node.convolution_attributes.get_groups(),
              output);
          set_operation_metadata(
              operation,
              node.convolution_attributes.get_name(),
              node.convolution_attributes.get_compute_data_type(),
              *node.input);
          native_graph->add(operation);
          break;
        }
      }
    }
    if (finalize) {
      native_graph->finalize();
    } else {
      native_graph->validate();
    }
    return {std::move(native_graph), std::move(required_uids)};
  }

  [[nodiscard]] flagdnnBuildOptions_t selected_build_options() const {
    flagdnnBuildOptions_t options = FLAGDNN_BUILD_OPTIONS_INITIALIZER;
    for (const HeurMode_t mode : heuristic_modes_) {
      if (mode == HeurMode_t::A) {
        options.flags |= FLAGDNN_BUILD_OPTION_HEURISTIC_MODE_A;
      } else if (mode == HeurMode_t::FALLBACK) {
        options.flags |= FLAGDNN_BUILD_OPTION_HEURISTIC_MODE_FALLBACK;
      }
    }
    if (autotune_) {
      options.flags |= FLAGDNN_BUILD_OPTION_AUTOTUNE;
    }
    return options;
  }

  void require_state(LifecycleState expected, const char* message) const {
    if (state_ != expected) {
      throw std::logic_error(message);
    }
  }

  void require_compatible_handle(const Handle& handle) const {
    if (backend_name_.empty() || target_fingerprint_.empty()) {
      throw std::logic_error(
          "frontend operation graph has no backend affinity");
    }
    if (handle.backend_name() != backend_name_ ||
        handle.target_fingerprint() != target_fingerprint_) {
      throw std::invalid_argument(
          "frontend handle backend or target does not match operation graph");
    }
  }

  void require_built() const {
    if (state_ != LifecycleState::kBuilt || executable_ == nullptr) {
      throw std::logic_error("frontend graph has not been built");
    }
  }

  void invalidate() noexcept {
    executable_.reset();
    supported_candidate_.reset();
    native_graph_.reset();
    required_uids_.clear();
    heuristic_modes_.clear();
    backend_name_.clear();
    target_fingerprint_.clear();
    workspace_size_ = 0;
    supported_workspace_size_ = 0;
    state_ = LifecycleState::kUnvalidated;
  }

  std::string name_;
  DataType_t io_data_type_ = DataType_t::NOT_SET;
  DataType_t intermediate_data_type_ = DataType_t::NOT_SET;
  DataType_t compute_data_type_ = DataType_t::NOT_SET;
  std::vector<Tensor> tensors_;
  std::vector<Node> nodes_;
  std::unique_ptr<flagdnn::Graph> native_graph_;
  std::unique_ptr<flagdnn::Executable> supported_candidate_;
  std::unique_ptr<flagdnn::Executable> executable_;
  std::vector<std::int64_t> required_uids_;
  std::vector<HeurMode_t> heuristic_modes_;
  bool autotune_ = false;
  std::string backend_name_;
  std::string target_fingerprint_;
  std::size_t workspace_size_ = 0;
  std::size_t supported_workspace_size_ = 0;
  LifecycleState state_ = LifecycleState::kUnvalidated;
};

}  // namespace graph
}  // namespace flagdnn_frontend

namespace flagnn_frontend = flagdnn_frontend;

#endif  // FLAGDNN_FRONTEND_HPP_
