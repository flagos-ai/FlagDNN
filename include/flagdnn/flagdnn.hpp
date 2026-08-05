/*
 * Copyright (c) 2025-2026 BAAI. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef FLAGDNN_FLAGDNN_HPP_
#define FLAGDNN_FLAGDNN_HPP_

#include <flagdnn/flagdnn.h>

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace flagdnn {

class Error : public std::runtime_error {
 public:
  Error(flagdnnStatus_t status, std::string message)
      : std::runtime_error(std::move(message)), status_(status) {}

  [[nodiscard]] flagdnnStatus_t status() const noexcept { return status_; }

 private:
  flagdnnStatus_t status_;
};

inline void check(flagdnnStatus_t status) {
  if (status == FLAGDNN_STATUS_SUCCESS) {
    return;
  }
  const char* detail = flagdnnGetLastErrorString();
  if (detail == nullptr || detail[0] == '\0') {
    detail = flagdnnGetErrorString(status);
  }
  throw Error(status, detail == nullptr ? "FlagDNN error" : detail);
}

class Handle {
 public:
  Handle() { check(flagdnnCreate(&value_)); }

  explicit Handle(flagdnnBackend_t backend, std::int32_t device = 0) {
    check(flagdnnCreateWithBackend(backend, device, &value_));
  }

  explicit Handle(std::string_view backend_name, std::int32_t device = 0) {
    const std::string owned_name(backend_name);
    check(flagdnnCreateWithBackendName(owned_name.c_str(), device, &value_));
  }

  ~Handle() { reset(); }

  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;

  Handle(Handle&& other) noexcept
      : value_(std::exchange(other.value_, nullptr)) {}

  Handle& operator=(Handle&& other) noexcept {
    if (this != &other) {
      reset();
      value_ = std::exchange(other.value_, nullptr);
    }
    return *this;
  }


  void set_compiler(std::string const& executable,
                    std::string const& compiler,
                    std::string const& cache_directory) {
    check(flagdnnSetCompilerConfig(value_,
                                   executable.c_str(),
                                   compiler.c_str(),
                                   cache_directory.c_str()));
  }

  [[nodiscard]] std::string_view backend_name() const {
    const char* result = nullptr;
    check(flagdnnGetBackendName(value_, &result));
    return result;
  }

  [[nodiscard]] std::string_view target_fingerprint() const {
    const char* result = nullptr;
    check(flagdnnGetTargetFingerprint(value_, &result));
    return result;
  }

  [[nodiscard]] flagdnnHandle_t get() const noexcept { return value_; }

 private:
  void reset() noexcept {
    if (value_ != nullptr) {
      (void)flagdnnDestroy(value_);
      value_ = nullptr;
    }
  }

  flagdnnHandle_t value_ = nullptr;
};

class TensorDescriptor {
 public:
  TensorDescriptor() { check(flagdnnCreateTensorDescriptor(&value_)); }

  TensorDescriptor(std::int64_t uid,
                   flagdnnDataType_t data_type,
                   std::span<const std::int64_t> dimensions,
                   std::span<const std::int64_t> strides)
      : TensorDescriptor() {
    set(uid, data_type, dimensions, strides);
  }

  ~TensorDescriptor() { reset(); }

  TensorDescriptor(const TensorDescriptor&) = delete;
  TensorDescriptor& operator=(const TensorDescriptor&) = delete;

  TensorDescriptor(TensorDescriptor&& other) noexcept
      : value_(std::exchange(other.value_, nullptr)) {}

  TensorDescriptor& operator=(TensorDescriptor&& other) noexcept {
    if (this != &other) {
      reset();
      value_ = std::exchange(other.value_, nullptr);
    }
    return *this;
  }

  void set(std::int64_t uid,
           flagdnnDataType_t data_type,
           std::span<const std::int64_t> dimensions,
           std::span<const std::int64_t> strides) {
    if (dimensions.size() != strides.size()) {
      throw std::invalid_argument("dimensions and strides must have same rank");
    }
    check(flagdnnSetTensorNdDescriptor(
        value_,
        uid,
        data_type,
        static_cast<std::int32_t>(dimensions.size()),
        dimensions.data(),
        strides.data()));
  }

  [[nodiscard]] std::size_t size_in_bytes() const {
    std::size_t result = 0;
    check(flagdnnGetTensorSizeInBytes(value_, &result));
    return result;
  }

  void set_virtual(bool is_virtual = true) {
    check(flagdnnSetTensorDescriptorVirtual(
        value_, is_virtual ? 1 : 0));
  }

  [[nodiscard]] bool is_virtual() const {
    std::int32_t result = 0;
    check(flagdnnGetTensorDescriptorVirtual(value_, &result));
    return result != 0;
  }

  void set_alignment(std::int64_t alignment) {
    check(flagdnnSetTensorDescriptorAlignment(value_, alignment));
  }

  [[nodiscard]] std::int64_t alignment() const {
    std::int64_t result = 0;
    check(flagdnnGetTensorDescriptorAlignment(value_, &result));
    return result;
  }

  [[nodiscard]] flagdnnTensorDescriptor_t get() const noexcept {
    return value_;
  }

 private:
  void reset() noexcept {
    if (value_ != nullptr) {
      (void)flagdnnDestroyTensorDescriptor(value_);
      value_ = nullptr;
    }
  }

  flagdnnTensorDescriptor_t value_ = nullptr;
};

class OperationDescriptor {
 public:
  explicit OperationDescriptor(flagdnnOperation_t operation) {
    check(flagdnnCreateOperationDescriptor(operation, &value_));
  }

  explicit OperationDescriptor(std::string_view operation_kind) {
    const std::string owned_kind(operation_kind);
    check(flagdnnCreateOperationDescriptorByName(
        owned_kind.c_str(), &value_));
  }

  ~OperationDescriptor() { reset(); }

  OperationDescriptor(const OperationDescriptor&) = delete;
  OperationDescriptor& operator=(const OperationDescriptor&) = delete;

  OperationDescriptor(OperationDescriptor&& other) noexcept
      : value_(std::exchange(other.value_, nullptr)) {}

  OperationDescriptor& operator=(OperationDescriptor&& other) noexcept {
    if (this != &other) {
      reset();
      value_ = std::exchange(other.value_, nullptr);
    }
    return *this;
  }

  void set_input(std::string_view port_name,
                 TensorDescriptor const& tensor,
                 bool optional = false) {
    const std::string owned_name(port_name);
    check(flagdnnSetOperationDescriptorInput(
        value_, owned_name.c_str(), tensor.get(), optional ? 1 : 0));
  }

  void set_output(std::string_view port_name,
                  TensorDescriptor const& tensor,
                  bool optional = false) {
    const std::string owned_name(port_name);
    check(flagdnnSetOperationDescriptorOutput(
        value_, owned_name.c_str(), tensor.get(), optional ? 1 : 0));
  }

  void set_attribute(std::string_view attribute_name,
                     std::int64_t value) {
    const std::string owned_name(attribute_name);
    check(flagdnnSetOperationDescriptorAttributeInt64(
        value_, owned_name.c_str(), value));
  }

  void set_attribute(std::string_view attribute_name, double value) {
    const std::string owned_name(attribute_name);
    check(flagdnnSetOperationDescriptorAttributeDouble(
        value_, owned_name.c_str(), value));
  }

  void set_attribute(std::string_view attribute_name, bool value) {
    const std::string owned_name(attribute_name);
    check(flagdnnSetOperationDescriptorAttributeBoolean(
        value_, owned_name.c_str(), value ? 1 : 0));
  }

  void set_attribute(std::string_view attribute_name,
                     std::string_view value) {
    const std::string owned_name(attribute_name);
    const std::string owned_value(value);
    check(flagdnnSetOperationDescriptorAttributeString(
        value_, owned_name.c_str(), owned_value.c_str()));
  }

  void set_attribute(std::string_view attribute_name, const char* value) {
    const std::string owned_name(attribute_name);
    check(flagdnnSetOperationDescriptorAttributeString(
        value_, owned_name.c_str(), value));
  }

  void set_attribute(std::string_view attribute_name,
                     std::span<const std::int64_t> values) {
    const std::string owned_name(attribute_name);
    check(flagdnnSetOperationDescriptorAttributeInt64Array(
        value_, owned_name.c_str(), values.data(), values.size()));
  }

  void finalize() { check(flagdnnFinalizeOperationDescriptor(value_)); }

  void set_name(std::string_view name) {
    const std::string owned_name(name);
    check(flagdnnSetOperationDescriptorName(value_, owned_name.c_str()));
  }

  void set_compute_data_type(flagdnnDataType_t data_type) {
    check(flagdnnSetOperationDescriptorComputeDataType(value_, data_type));
  }

  void set_pointwise(TensorDescriptor const& input,
                     flagdnnPointwiseMode_t mode,
                     TensorDescriptor const& output) {
    check(flagdnnSetPointwiseUnaryOperationDescriptor(
        value_, input.get(), mode, output.get()));
  }

  void set_pointwise(TensorDescriptor const& input,
                     flagdnnPointwiseMode_t mode,
                     TensorDescriptor const& output,
                     const flagdnnPointwiseAttributes_t& attributes) {
    check(flagdnnSetPointwiseUnaryOperationDescriptorWithAttributes(
        value_, input.get(), mode, output.get(), &attributes));
  }

  void set_pointwise(TensorDescriptor const& left,
                     TensorDescriptor const& right,
                     flagdnnPointwiseMode_t mode,
                     TensorDescriptor const& output) {
    check(flagdnnSetPointwiseBinaryOperationDescriptor(
        value_, left.get(), right.get(), mode, output.get()));
  }

  void set_pointwise(TensorDescriptor const& left,
                     TensorDescriptor const& right,
                     flagdnnPointwiseMode_t mode,
                     TensorDescriptor const& output,
                     double alpha) {
    check(flagdnnSetPointwiseBinaryOperationDescriptorWithAlpha(
        value_, left.get(), right.get(), mode, output.get(), alpha));
  }

  void set_pointwise(TensorDescriptor const& a,
                     TensorDescriptor const& b,
                     TensorDescriptor const& t,
                     flagdnnPointwiseMode_t mode,
                     TensorDescriptor const& output) {
    check(flagdnnSetPointwiseTernaryOperationDescriptor(
        value_, a.get(), b.get(), t.get(), mode, output.get()));
  }

  void set_relu(TensorDescriptor const& input,
                TensorDescriptor const& output) {
    check(flagdnnSetReluOperationDescriptor(
        value_, input.get(), output.get()));
  }

  void set_add(TensorDescriptor const& left,
               TensorDescriptor const& right,
               TensorDescriptor const& output) {
    check(flagdnnSetAddOperationDescriptor(
        value_, left.get(), right.get(), output.get()));
  }

  void set_add(TensorDescriptor const& left,
               TensorDescriptor const& right,
               TensorDescriptor const& output,
               double alpha) {
    check(flagdnnSetAddOperationDescriptorWithAlpha(
        value_, left.get(), right.get(), output.get(), alpha));
  }

  void set_matmul(TensorDescriptor const& a,
                  TensorDescriptor const& b,
                  TensorDescriptor const& output) {
    check(flagdnnSetMatmulOperationDescriptor(
        value_, a.get(), b.get(), output.get()));
  }

  void set_sdpa(TensorDescriptor const& q,
                TensorDescriptor const& k,
                TensorDescriptor const& v,
                const TensorDescriptor* bias,
                TensorDescriptor const& output,
                TensorDescriptor const& stats,
                const flagdnnSdpaAttributes_t& attributes) {
    check(flagdnnSetSdpaOperationDescriptor(value_,
                                            q.get(),
                                            k.get(),
                                            v.get(),
                                            bias == nullptr
                                                ? nullptr
                                                : bias->get(),
                                            output.get(),
                                            stats.get(),
                                            &attributes));
  }

  void set_sdpa_backward(TensorDescriptor const& q,
                         TensorDescriptor const& k,
                         TensorDescriptor const& v,
                         TensorDescriptor const& output,
                         TensorDescriptor const& doutput,
                         TensorDescriptor const& stats,
                         const TensorDescriptor* bias,
                         TensorDescriptor const& dq,
                         TensorDescriptor const& dk,
                         TensorDescriptor const& dv,
                         const TensorDescriptor* dbias,
                         const flagdnnSdpaAttributes_t& attributes) {
    check(flagdnnSetSdpaBackwardOperationDescriptor(
        value_,
        q.get(),
        k.get(),
        v.get(),
        output.get(),
        doutput.get(),
        stats.get(),
        bias == nullptr ? nullptr : bias->get(),
        dq.get(),
        dk.get(),
        dv.get(),
        dbias == nullptr ? nullptr : dbias->get(),
        &attributes));
  }

  void set_sdpa_fp8(TensorDescriptor const& q,
                    TensorDescriptor const& k,
                    TensorDescriptor const& v,
                    TensorDescriptor const& descale_q,
                    TensorDescriptor const& descale_k,
                    TensorDescriptor const& descale_v,
                    TensorDescriptor const& descale_s,
                    TensorDescriptor const& scale_s,
                    TensorDescriptor const& scale_o,
                    const TensorDescriptor* bias,
                    TensorDescriptor const& output,
                    TensorDescriptor const& stats,
                    TensorDescriptor const& amax_s,
                    TensorDescriptor const& amax_o,
                    const flagdnnSdpaAttributes_t& attributes) {
    check(flagdnnSetSdpaFp8OperationDescriptor(
        value_,
        q.get(),
        k.get(),
        v.get(),
        descale_q.get(),
        descale_k.get(),
        descale_v.get(),
        descale_s.get(),
        scale_s.get(),
        scale_o.get(),
        bias == nullptr ? nullptr : bias->get(),
        output.get(),
        stats.get(),
        amax_s.get(),
        amax_o.get(),
        &attributes));
  }

  void set_sdpa_fp8_backward(
      TensorDescriptor const& q,
      TensorDescriptor const& k,
      TensorDescriptor const& v,
      TensorDescriptor const& output,
      TensorDescriptor const& doutput,
      TensorDescriptor const& stats,
      TensorDescriptor const& descale_q,
      TensorDescriptor const& descale_k,
      TensorDescriptor const& descale_v,
      TensorDescriptor const& descale_o,
      TensorDescriptor const& descale_doutput,
      TensorDescriptor const& descale_s,
      TensorDescriptor const& descale_dp,
      TensorDescriptor const& scale_s,
      TensorDescriptor const& scale_dq,
      TensorDescriptor const& scale_dk,
      TensorDescriptor const& scale_dv,
      TensorDescriptor const& scale_dp,
      TensorDescriptor const& dq,
      TensorDescriptor const& dk,
      TensorDescriptor const& dv,
      TensorDescriptor const& amax_dq,
      TensorDescriptor const& amax_dk,
      TensorDescriptor const& amax_dv,
      TensorDescriptor const& amax_dp,
      const flagdnnSdpaAttributes_t& attributes) {
    check(flagdnnSetSdpaFp8BackwardOperationDescriptor(
        value_,
        q.get(),
        k.get(),
        v.get(),
        output.get(),
        doutput.get(),
        stats.get(),
        descale_q.get(),
        descale_k.get(),
        descale_v.get(),
        descale_o.get(),
        descale_doutput.get(),
        descale_s.get(),
        descale_dp.get(),
        scale_s.get(),
        scale_dq.get(),
        scale_dk.get(),
        scale_dv.get(),
        scale_dp.get(),
        dq.get(),
        dk.get(),
        dv.get(),
        amax_dq.get(),
        amax_dk.get(),
        amax_dv.get(),
        amax_dp.get(),
        &attributes));
  }

  void set_reduction_sum(TensorDescriptor const& input,
                         std::int32_t axis,
                         bool keep_dimensions,
                         TensorDescriptor const& output) {
    check(flagdnnSetReductionSumOperationDescriptor(value_,
                                                     input.get(),
                                                     axis,
                                                     keep_dimensions ? 1 : 0,
                                                     output.get()));
  }

  void set_reduction(TensorDescriptor const& input,
                     flagdnnReductionMode_t mode,
                     std::int32_t axis,
                     bool keep_dimensions,
                     TensorDescriptor const& output) {
    check(flagdnnSetReductionOperationDescriptor(value_,
                                                  input.get(),
                                                  mode,
                                                  axis,
                                                  keep_dimensions ? 1 : 0,
                                                  output.get()));
  }

  void set_convolution_fprop(
      TensorDescriptor const& input,
      TensorDescriptor const& filter,
      std::span<const std::int64_t> pre_padding,
      std::span<const std::int64_t> post_padding,
      std::span<const std::int64_t> stride,
      std::span<const std::int64_t> dilation,
      std::int64_t groups,
      TensorDescriptor const& output) {
    const std::size_t spatial_rank = pre_padding.size();
    if (spatial_rank < 1 || spatial_rank > 3 ||
        post_padding.size() != spatial_rank ||
        stride.size() != spatial_rank || dilation.size() != spatial_rank) {
      throw std::invalid_argument(
          "convolution spatial arrays must have the same rank in [1, 3]");
    }
    check(flagdnnSetConvolutionFpropOperationDescriptor(
        value_,
        input.get(),
        filter.get(),
        static_cast<std::int32_t>(spatial_rank),
        pre_padding.data(),
        post_padding.data(),
        stride.data(),
        dilation.data(),
        groups,
        output.get()));
  }

  void set_conv2d_fprop(TensorDescriptor const& input,
                        TensorDescriptor const& filter,
                        std::span<const std::int64_t, 2> padding,
                        std::span<const std::int64_t, 2> stride,
                        std::span<const std::int64_t, 2> dilation,
                        std::int64_t groups,
                        TensorDescriptor const& output) {
    check(flagdnnSetConv2dFpropOperationDescriptor(value_,
                                                    input.get(),
                                                    filter.get(),
                                                    padding.data(),
                                                    stride.data(),
                                                    dilation.data(),
                                                    groups,
                                                    output.get()));
  }

  void set_conv2d_fprop(
      TensorDescriptor const& input,
      TensorDescriptor const& filter,
      std::span<const std::int64_t, 2> pre_padding,
      std::span<const std::int64_t, 2> post_padding,
      std::span<const std::int64_t, 2> stride,
      std::span<const std::int64_t, 2> dilation,
      std::int64_t groups,
      TensorDescriptor const& output) {
    check(
        flagdnnSetConv2dFpropOperationDescriptorWithAsymmetricPadding(
            value_,
            input.get(),
            filter.get(),
            pre_padding.data(),
            post_padding.data(),
            stride.data(),
            dilation.data(),
            groups,
            output.get()));
  }

  [[nodiscard]] flagdnnOperationDescriptor_t get() const noexcept {
    return value_;
  }

 private:
  void reset() noexcept {
    if (value_ != nullptr) {
      (void)flagdnnDestroyOperationDescriptor(value_);
      value_ = nullptr;
    }
  }

  flagdnnOperationDescriptor_t value_ = nullptr;
};

/**
 * Low-level descriptor graph used by the C ABI and Frontend lowering.
 *
 * Application code targeting the cuDNN-Frontend-style API should use
 * flagdnn_frontend::graph::Graph. In particular, pointwise Add is expressed
 * there as Graph::pointwise(..., PointwiseMode_t::ADD); add() below only
 * attaches an already finalized OperationDescriptor.
 */
class Graph {
 public:
  Graph() { check(flagdnnCreateGraph(&value_)); }

  ~Graph() { reset(); }

  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;

  Graph(Graph&& other) noexcept
      : value_(std::exchange(other.value_, nullptr)) {}

  Graph& operator=(Graph&& other) noexcept {
    if (this != &other) {
      reset();
      value_ = std::exchange(other.value_, nullptr);
    }
    return *this;
  }

  void set_name(std::string_view name) {
    const std::string owned_name(name);
    check(flagdnnSetGraphName(value_, owned_name.c_str()));
  }

  // Low-level descriptor attachment used by Frontend lowering. This is not
  // the pointwise ADD operator; public operator construction uses pointwise().
  void add(OperationDescriptor const& operation) {
    check(flagdnnGraphAddOperation(value_, operation.get()));
  }

  void pointwise(TensorDescriptor const& input,
                 flagdnnPointwiseMode_t mode,
                 TensorDescriptor const& output) {
    OperationDescriptor operation(FLAGDNN_OPERATION_POINTWISE);
    operation.set_pointwise(input, mode, output);
    add(operation);
  }

  void pointwise(TensorDescriptor const& input,
                 flagdnnPointwiseMode_t mode,
                 TensorDescriptor const& output,
                 const flagdnnPointwiseAttributes_t& attributes) {
    OperationDescriptor operation(FLAGDNN_OPERATION_POINTWISE);
    operation.set_pointwise(input, mode, output, attributes);
    add(operation);
  }

  void pointwise(TensorDescriptor const& left,
                 TensorDescriptor const& right,
                 flagdnnPointwiseMode_t mode,
                 TensorDescriptor const& output,
                 double alpha = 1.0) {
    OperationDescriptor operation(FLAGDNN_OPERATION_POINTWISE);
    operation.set_pointwise(left, right, mode, output, alpha);
    add(operation);
  }

  void pointwise(TensorDescriptor const& a,
                 TensorDescriptor const& b,
                 TensorDescriptor const& t,
                 flagdnnPointwiseMode_t mode,
                 TensorDescriptor const& output) {
    OperationDescriptor operation(FLAGDNN_OPERATION_POINTWISE);
    operation.set_pointwise(a, b, t, mode, output);
    add(operation);
  }

  void relu(TensorDescriptor const& input,
            TensorDescriptor const& output) {
    OperationDescriptor operation(FLAGDNN_OPERATION_RELU);
    operation.set_relu(input, output);
    add(operation);
  }

  void matmul(TensorDescriptor const& a,
              TensorDescriptor const& b,
              TensorDescriptor const& output) {
    OperationDescriptor operation(FLAGDNN_OPERATION_MATMUL);
    operation.set_matmul(a, b, output);
    add(operation);
  }

  void reduction_sum(TensorDescriptor const& input,
                     std::int32_t axis,
                     bool keep_dimensions,
                     TensorDescriptor const& output) {
    OperationDescriptor operation(FLAGDNN_OPERATION_REDUCTION_SUM);
    operation.set_reduction_sum(input, axis, keep_dimensions, output);
    add(operation);
  }

  void reduction(TensorDescriptor const& input,
                 flagdnnReductionMode_t mode,
                 std::int32_t axis,
                 bool keep_dimensions,
                 TensorDescriptor const& output) {
    OperationDescriptor operation(FLAGDNN_OPERATION_REDUCTION);
    operation.set_reduction(input, mode, axis, keep_dimensions, output);
    add(operation);
  }

  void reduction_avg(TensorDescriptor const& input,
                     std::int32_t axis,
                     bool keep_dimensions,
                     TensorDescriptor const& output) {
    reduction(
        input, FLAGDNN_REDUCTION_AVG, axis, keep_dimensions, output);
  }

  void reduction_mul(TensorDescriptor const& input,
                     std::int32_t axis,
                     bool keep_dimensions,
                     TensorDescriptor const& output) {
    reduction(
        input, FLAGDNN_REDUCTION_MUL, axis, keep_dimensions, output);
  }

  void convolution_fprop(
      TensorDescriptor const& input,
      TensorDescriptor const& filter,
      std::span<const std::int64_t> padding,
      std::span<const std::int64_t> stride,
      std::span<const std::int64_t> dilation,
      std::int64_t groups,
      TensorDescriptor const& output) {
    convolution_fprop(input,
                      filter,
                      padding,
                      padding,
                      stride,
                      dilation,
                      groups,
                      output);
  }

  void convolution_fprop(
      TensorDescriptor const& input,
      TensorDescriptor const& filter,
      std::span<const std::int64_t> pre_padding,
      std::span<const std::int64_t> post_padding,
      std::span<const std::int64_t> stride,
      std::span<const std::int64_t> dilation,
      std::int64_t groups,
      TensorDescriptor const& output) {
    OperationDescriptor operation(FLAGDNN_OPERATION_CONVOLUTION_FPROP);
    operation.set_convolution_fprop(input,
                                    filter,
                                    pre_padding,
                                    post_padding,
                                    stride,
                                    dilation,
                                    groups,
                                    output);
    add(operation);
  }

  void conv1d_fprop(TensorDescriptor const& input,
                    TensorDescriptor const& filter,
                    std::span<const std::int64_t, 1> padding,
                    std::span<const std::int64_t, 1> stride,
                    std::span<const std::int64_t, 1> dilation,
                    std::int64_t groups,
                    TensorDescriptor const& output) {
    convolution_fprop(
        input, filter, padding, stride, dilation, groups, output);
  }

  void conv1d_fprop(
      TensorDescriptor const& input,
      TensorDescriptor const& filter,
      std::span<const std::int64_t, 1> pre_padding,
      std::span<const std::int64_t, 1> post_padding,
      std::span<const std::int64_t, 1> stride,
      std::span<const std::int64_t, 1> dilation,
      std::int64_t groups,
      TensorDescriptor const& output) {
    convolution_fprop(input,
                      filter,
                      pre_padding,
                      post_padding,
                      stride,
                      dilation,
                      groups,
                      output);
  }

  void conv2d_fprop(TensorDescriptor const& input,
                    TensorDescriptor const& filter,
                    std::span<const std::int64_t, 2> padding,
                    std::span<const std::int64_t, 2> stride,
                    std::span<const std::int64_t, 2> dilation,
                    std::int64_t groups,
                    TensorDescriptor const& output) {
    OperationDescriptor operation(FLAGDNN_OPERATION_CONV2D_FPROP);
    operation.set_conv2d_fprop(
        input, filter, padding, stride, dilation, groups, output);
    add(operation);
  }

  void conv2d_fprop(
      TensorDescriptor const& input,
      TensorDescriptor const& filter,
      std::span<const std::int64_t, 2> pre_padding,
      std::span<const std::int64_t, 2> post_padding,
      std::span<const std::int64_t, 2> stride,
      std::span<const std::int64_t, 2> dilation,
      std::int64_t groups,
      TensorDescriptor const& output) {
    OperationDescriptor operation(FLAGDNN_OPERATION_CONV2D_FPROP);
    operation.set_conv2d_fprop(input,
                               filter,
                               pre_padding,
                               post_padding,
                               stride,
                               dilation,
                               groups,
                               output);
    add(operation);
  }

  void conv3d_fprop(TensorDescriptor const& input,
                    TensorDescriptor const& filter,
                    std::span<const std::int64_t, 3> padding,
                    std::span<const std::int64_t, 3> stride,
                    std::span<const std::int64_t, 3> dilation,
                    std::int64_t groups,
                    TensorDescriptor const& output) {
    convolution_fprop(
        input, filter, padding, stride, dilation, groups, output);
  }

  void conv3d_fprop(
      TensorDescriptor const& input,
      TensorDescriptor const& filter,
      std::span<const std::int64_t, 3> pre_padding,
      std::span<const std::int64_t, 3> post_padding,
      std::span<const std::int64_t, 3> stride,
      std::span<const std::int64_t, 3> dilation,
      std::int64_t groups,
      TensorDescriptor const& output) {
    convolution_fprop(input,
                      filter,
                      pre_padding,
                      post_padding,
                      stride,
                      dilation,
                      groups,
                      output);
  }

  void validate() const { check(flagdnnValidateGraph(value_)); }

  void finalize() { check(flagdnnFinalizeGraph(value_)); }

  [[nodiscard]] std::size_t operation_count() const {
    std::size_t result = 0;
    check(flagdnnGetGraphOperationCount(value_, &result));
    return result;
  }

  [[nodiscard]] flagdnnGraph_t get() const noexcept { return value_; }

 private:
  void reset() noexcept {
    if (value_ != nullptr) {
      (void)flagdnnDestroyGraph(value_);
      value_ = nullptr;
    }
  }

  flagdnnGraph_t value_ = nullptr;
};

class Executable {
 public:
  Executable() = default;

  Executable(Handle const& handle,
             Graph const& graph,
             const flagdnnBuildOptions_t* options = nullptr) {
    check(flagdnnBuildExecutable(
        handle.get(), graph.get(), options, &value_));
  }

  ~Executable() { reset(); }

  Executable(const Executable&) = delete;
  Executable& operator=(const Executable&) = delete;

  Executable(Executable&& other) noexcept
      : value_(std::exchange(other.value_, nullptr)) {}

  Executable& operator=(Executable&& other) noexcept {
    if (this != &other) {
      reset();
      value_ = std::exchange(other.value_, nullptr);
    }
    return *this;
  }

  [[nodiscard]] std::size_t operation_count() const {
    std::size_t result = 0;
    check(flagdnnGetExecutableOperationCount(value_, &result));
    return result;
  }

  [[nodiscard]] std::size_t workspace_size() const {
    std::size_t result = 0;
    check(flagdnnGetExecutableWorkspaceSize(value_, &result));
    return result;
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) const {
    check(flagdnnExecuteAsync(value_,
                              bindings.data(),
                              bindings.size(),
                              workspace,
                              workspace_size,
                              stream));
  }

  [[nodiscard]] flagdnnExecutable_t get() const noexcept { return value_; }

 private:
  void reset() noexcept {
    if (value_ != nullptr) {
      (void)flagdnnDestroyExecutable(value_);
      value_ = nullptr;
    }
  }

  flagdnnExecutable_t value_ = nullptr;
};

}  // namespace flagdnn

#endif  // FLAGDNN_FLAGDNN_HPP_
