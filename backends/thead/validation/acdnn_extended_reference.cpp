// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "acdnn_extended_reference.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <map>

#include "acdnn_graph.hpp"
#include "backend_pointwise_reference.hpp"
namespace flagdnn::validation::thead {
namespace {
using namespace flagdnn::testing;
class StatisticsReference final : public AcdnnGraphReference {
 public:
  explicit StatisticsReference(const StatisticsTestCase& t) {
    const bool gen = t.operation == "genstats";
    auto& op = descriptor(
        gen ? ACDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR
            : ACDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR);
    const auto bind = [&](acdnnBackendAttributeName_t attr,
                          const TestTensor& value) {
      op.set(attr, ACDNN_TYPE_BACKEND_DESCRIPTOR,
             tensor(value, acdnn_float_type(value.data_type)));
    };
    if (gen) {
      op.set(ACDNN_ATTR_OPERATION_GENSTATS_MODE, ACDNN_TYPE_GENSTATS_MODE,
             ACDNN_GENSTATS_SUM_SQSUM);
      op.set(ACDNN_ATTR_OPERATION_GENSTATS_MATH_PREC, ACDNN_TYPE_DATA_TYPE,
             ACDNN_DATA_FLOAT);
      bind(ACDNN_ATTR_OPERATION_GENSTATS_XDESC, t.inputs.at(0));
      bind(ACDNN_ATTR_OPERATION_GENSTATS_SUMDESC, t.outputs.at(0));
      bind(ACDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC, t.outputs.at(1));
    } else {
      op.set(ACDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE,
             ACDNN_TYPE_BN_FINALIZE_STATS_MODE,
             ACDNN_BN_FINALIZE_STATISTICS_TRAINING);
      op.set(ACDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC, ACDNN_TYPE_DATA_TYPE,
             ACDNN_DATA_FLOAT);
      const std::array input_attrs{
          ACDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC,
          ACDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC,
          ACDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC,
          ACDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC,
          ACDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC,
          ACDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC};
      const std::array output_attrs{
          ACDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC,
          ACDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC,
          ACDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC,
          ACDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC,
          ACDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC,
          ACDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC};
      for (std::size_t i = 0; i < t.inputs.size(); ++i)
        bind(input_attrs.at(i), t.inputs[i]);
      for (std::size_t i = 0; i < t.outputs.size(); ++i)
        bind(output_attrs.at(i), t.outputs[i]);
      op.set(ACDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC,
             ACDNN_TYPE_BACKEND_DESCRIPTOR,
             constant(10001, static_cast<std::int64_t>(t.accum_count),
                      ACDNN_DATA_INT64));
      op.set(ACDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC,
             ACDNN_TYPE_BACKEND_DESCRIPTOR,
             constant(10002, static_cast<float>(t.epsilon), ACDNN_DATA_FLOAT));
      op.set(ACDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC,
             ACDNN_TYPE_BACKEND_DESCRIPTOR,
             constant(10003, static_cast<float>(t.momentum), ACDNN_DATA_FLOAT));
    }
    op.finalize();
    build(op.get());
  }
};
class ResampleReference final : public TestExecutable {
 public:
  explicit ResampleReference(const ResampleTestCase& t) : test_(t) {
    if (t.mode == 3 || t.mode == 4) {
      throw std::invalid_argument(
          "acDNN 1400 resampling descriptor is not implemented");
    }
    const auto& input = t.inputs.at(0);
    const auto& output = t.outputs.at(0);
    for (std::size_t axis = 0; axis < t.window.size(); ++axis) {
      if ((input.dimensions[axis + 2] + 2 * t.pre[axis] - t.window[axis]) /
                  t.stride[axis] +
              1 !=
          output.dimensions[axis + 2]) {
        throw std::invalid_argument(
            "acDNN stable pooling requires symmetric output padding");
      }
    }
    set(x_, input);
    set(y_, output);
    check_acdnn(acdnnCreatePoolingDescriptor(&pool_),
                "acdnnCreatePoolingDescriptor");
    const auto mode = t.mode == 1 ? ACDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
                      : t.mode == 2
                          ? ACDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
                          : ACDNN_POOLING_MAX;
    const std::vector<int> window(t.window.begin(), t.window.end()),
        pad(t.pre.begin(), t.pre.end()),
        stride(t.stride.begin(), t.stride.end());
    try {
      check_acdnn(
          acdnnSetPoolingNdDescriptor(pool_, mode, ACDNN_NOT_PROPAGATE_NAN,
                                      static_cast<int>(window.size()),
                                      window.data(), pad.data(), stride.data()),
          "acdnnSetPoolingNdDescriptor");
    } catch (...) {
      acdnnDestroyPoolingDescriptor(pool_);
      throw;
    }
  }
  ~ResampleReference() override {
    if (pool_) (void)acdnnDestroyPoolingDescriptor(pool_);
  }
  std::size_t workspace_size() const noexcept override { return 0; }
  void execute(std::span<const flagdnnBinding_t> bindings, void*, std::size_t,
               flagdnnStream_t stream) override {
    std::map<std::int64_t, void*> p;
    for (const auto& b : bindings) p.emplace(b.uid, b.device_pointer);
    handle_.set_stream(reinterpret_cast<hggcStream_t>(stream));
    const float one = 1, zero = 0;
    check_acdnn(acdnnPoolingForward(handle_.get(), pool_, &one, x_.get(),
                                    p.at(test_.inputs[0].uid), &zero, y_.get(),
                                    p.at(test_.outputs[0].uid)),
                "acdnnPoolingForward");
  }

 private:
  static void set(AcdnnTensorDescriptor& d, const TestTensor& t) {
    const std::vector<int> dims(t.dimensions.begin(), t.dimensions.end()),
        strides(t.strides.begin(), t.strides.end());
    d.set(acdnn_float_type(t.data_type), dims, strides);
  }
  ResampleTestCase test_;
  AcdnnHandle handle_;
  AcdnnTensorDescriptor x_, y_;
  acdnnPoolingDescriptor_t pool_ = nullptr;
};
class ConvertedResampleReference final : public TestExecutable {
 public:
  explicit ConvertedResampleReference(const ResampleTestCase& test_case)
      : input_(test_case.inputs.at(0)), output_(test_case.outputs.at(0)) {
    if (test_case.inputs.size() != 1 || test_case.outputs.size() != 1 ||
        input_.data_type != FLAGDNN_DATA_BFLOAT16 ||
        output_.data_type != FLAGDNN_DATA_BFLOAT16) {
      throw std::invalid_argument(
          "converted pooling requires BF16 input/output");
    }
    auto converted = test_case;
    auto& x = converted.inputs[0];
    auto& y = converted.outputs[0];
    const auto last_uid = std::max(input_.uid, output_.uid);
    if (last_uid > std::numeric_limits<std::int64_t>::max() - 2) {
      throw std::overflow_error("converted pooling UID overflows");
    }
    x.uid = last_uid + 1;
    y.uid = last_uid + 2;
    x.data_type = y.data_type = FLAGDNN_DATA_FLOAT32;
    x.binding_byte_offset = y.binding_byte_offset = 0;
    floating_input_ = x;
    floating_output_ = y;
    input_buffer_ = DeviceBuffer(storage_bytes(x));
    output_buffer_ = DeviceBuffer(storage_bytes(y));
    input_conversion_ = make_acdnn_backend_pointwise_reference(
        {.mode = ACDNN_POINTWISE_IDENTITY_FWD,
         .inputs = {input_},
         .output = x,
         .primitive =
             "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,pooling-input-fp32)"});
    pooling_ = std::make_unique<ResampleReference>(converted);
    output_conversion_ = make_acdnn_backend_pointwise_reference(
        {.mode = ACDNN_POINTWISE_IDENTITY_FWD,
         .inputs = {y},
         .output = output_,
         .primitive = "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,pooling-"
                      "output-bfloat16)"});
    workspace_ = std::max(input_conversion_->workspace_size(),
                          output_conversion_->workspace_size());
  }

  std::size_t workspace_size() const noexcept override { return workspace_; }

  void execute(std::span<const flagdnnBinding_t> bindings, void* workspace,
               std::size_t size, flagdnnStream_t stream) override {
    std::map<std::int64_t, void*> pointers;
    for (const auto& binding : bindings) {
      if (!binding.device_pointer ||
          !pointers.emplace(binding.uid, binding.device_pointer).second) {
        throw std::invalid_argument("invalid converted pooling binding");
      }
    }
    if (pointers.size() != 2 || !pointers.contains(input_.uid) ||
        !pointers.contains(output_.uid) || size < workspace_ ||
        (workspace_ != 0 && !workspace)) {
      throw std::invalid_argument(
          "converted pooling bindings/workspace mismatch");
    }
    const std::array<flagdnnBinding_t, 2> input_bindings{
        {{input_.uid, pointers.at(input_.uid)},
         {floating_input_.uid, input_buffer_.data()}}};
    const std::array<flagdnnBinding_t, 2> pooling_bindings{
        {{floating_input_.uid, input_buffer_.data()},
         {floating_output_.uid, output_buffer_.data()}}};
    const std::array<flagdnnBinding_t, 2> output_bindings{
        {{floating_output_.uid, output_buffer_.data()},
         {output_.uid, pointers.at(output_.uid)}}};
    input_conversion_->execute(input_bindings, workspace,
                               input_conversion_->workspace_size(), stream);
    pooling_->execute(pooling_bindings, nullptr, 0, stream);
    output_conversion_->execute(output_bindings, workspace,
                                output_conversion_->workspace_size(), stream);
  }

 private:
  static std::size_t storage_bytes(const TestTensor& tensor) {
    std::size_t elements = 1;
    for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
      if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0) {
        throw std::invalid_argument("invalid converted pooling geometry");
      }
      const auto extent = static_cast<std::size_t>(tensor.dimensions[axis] - 1);
      const auto stride = static_cast<std::size_t>(tensor.strides[axis]);
      if (extent >
          (std::numeric_limits<std::size_t>::max() - elements) / stride) {
        throw std::overflow_error("converted pooling storage overflows");
      }
      elements += extent * stride;
    }
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
      throw std::overflow_error("converted pooling allocation overflows");
    }
    return elements * sizeof(float);
  }
  TestTensor input_, output_, floating_input_, floating_output_;
  DeviceBuffer input_buffer_, output_buffer_;
  std::unique_ptr<TestExecutable> input_conversion_, pooling_,
      output_conversion_;
  std::size_t workspace_ = 0;
};

class BatchnormBackwardReference final : public TestExecutable {
 public:
  explicit BatchnormBackwardReference(const ExtendedNormalizationTestCase& t)
      : test_(t) {
    if (t.operation != "batchnorm_backward" || t.inputs.size() != 5 ||
        t.outputs.size() != 3)
      throw std::invalid_argument("invalid acDNN batchnorm backward case");
    for (std::size_t i = 0; i < 3; ++i)
      set(data_[i], i == 0 ? t.inputs[1] : i == 1 ? t.inputs[0] : t.outputs[0]);
    set(parameter_, t.inputs[2]);
  }
  std::size_t workspace_size() const noexcept override { return 0; }
  void execute(std::span<const flagdnnBinding_t> bindings, void*, std::size_t,
               flagdnnStream_t stream) override {
    std::map<std::int64_t, void*> p;
    for (const auto& b : bindings) p.emplace(b.uid, b.device_pointer);
    handle_.set_stream(reinterpret_cast<hggcStream_t>(stream));
    const float one = 1, zero = 0;
    check_acdnn(acdnnBatchNormalizationBackward(
                    handle_.get(), ACDNN_BATCHNORM_SPATIAL, &one, &zero, &one,
                    &zero, data_[0].get(), p.at(test_.inputs[1].uid),
                    data_[1].get(), p.at(test_.inputs[0].uid), data_[2].get(),
                    p.at(test_.outputs[0].uid), parameter_.get(),
                    p.at(test_.inputs[2].uid), p.at(test_.outputs[1].uid),
                    p.at(test_.outputs[2].uid), test_.epsilon,
                    p.at(test_.inputs[3].uid), p.at(test_.inputs[4].uid)),
                "acdnnBatchNormalizationBackward");
  }

 private:
  static void set(AcdnnTensorDescriptor& d, const TestTensor& t) {
    std::vector<int> dims(t.dimensions.begin(), t.dimensions.end()),
        strides(t.strides.begin(), t.strides.end());
    while (dims.size() < 4) {
      dims.push_back(1);
      strides.push_back(1);
    }
    d.set(acdnn_float_type(t.data_type), dims, strides);
  }
  ExtendedNormalizationTestCase test_;
  AcdnnHandle handle_;
  std::array<AcdnnTensorDescriptor, 3> data_;
  AcdnnTensorDescriptor parameter_;
};
}  // namespace
std::unique_ptr<TestExecutable> make_acdnn_extended_reference(
    const StatisticsTestCase& t) {
  return std::make_unique<StatisticsReference>(t);
}
std::unique_ptr<TestExecutable> make_acdnn_extended_reference(
    const ResampleTestCase& t) {
  if (t.inputs.at(0).data_type == FLAGDNN_DATA_BFLOAT16) {
    return std::make_unique<ConvertedResampleReference>(t);
  }
  return std::make_unique<ResampleReference>(t);
}
std::unique_ptr<TestExecutable> make_acdnn_extended_reference(
    const ExtendedNormalizationTestCase& t) {
  return std::make_unique<BatchnormBackwardReference>(t);
}
}  // namespace flagdnn::validation::thead
