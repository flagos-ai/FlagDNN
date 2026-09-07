// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "acdnn_normalization_reference.hpp"

#include "acdnn_reference.hpp"
#include "backend_pointwise_reference.hpp"
#include "numeric_types.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#ifndef FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG
#define FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG "capability.json"
#endif

namespace flagdnn::validation::thead {
namespace {

constexpr std::string_view kInferencePlan =
    "acdnnOpTensor(ADD[-mean])->MUL(inv_variance)->MUL(scale)->ADD(bias)";

const std::vector<std::string> kTrainingPlan = {
    "acdnnTransformTensor(previous_running_mean)",
    "acdnnTransformTensor(previous_running_variance)",
    "acdnnBatchNormalizationForwardTraining(SPATIAL)",
};

const std::vector<std::string> kLayernormPlan = {
    "acdnnReduceTensor(AVG,mean)",
    "acdnnOpTensor(ADD[-mean])",
    "acdnnOpTensor(MUL,square)",
    "acdnnReduceTensor(AVG,variance)",
    "acdnnOpTensor(ADD,epsilon)",
    "acdnnBackendExecute(POINTWISE_RSQRT)",
    "acdnnOpTensor(MUL,normalize)",
    "acdnnOpTensor(MUL,scale)",
    "acdnnOpTensor(ADD,bias)",
};

const std::vector<std::string> kRmsnormPlan = {
    "acdnnOpTensor(MUL,square)",
    "acdnnReduceTensor(AVG,mean_square)",
    "acdnnOpTensor(ADD,epsilon)",
    "acdnnBackendExecute(POINTWISE_RSQRT)",
    "acdnnOpTensor(MUL,normalize)",
    "acdnnOpTensor(MUL,scale)",
    "acdnnOpTensor(ADD,bias)",
};

constexpr std::string_view kConvertXPrimitive =
    "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-x-fp32)";
constexpr std::string_view kConvertScalePrimitive =
    "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-scale-fp32)";
constexpr std::string_view kConvertBiasPrimitive =
    "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-bias-fp32)";
constexpr std::string_view kConvertYPrimitive =
    "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-y-data-type)";

std::vector<std::string> typed_inference_plan() {
  return {std::string(kConvertXPrimitive), std::string(kInferencePlan),
          std::string(kConvertYPrimitive)};
}

std::vector<std::string> typed_training_plan() {
  return {std::string(kConvertXPrimitive),
          std::string(kConvertScalePrimitive),
          std::string(kConvertBiasPrimitive),
          kTrainingPlan[0], kTrainingPlan[1], kTrainingPlan[2],
          std::string(kConvertYPrimitive)};
}

std::vector<std::string> typed_suffix_plan(bool layernorm) {
  std::vector<std::string> result = {
      std::string(kConvertXPrimitive), std::string(kConvertScalePrimitive),
      std::string(kConvertBiasPrimitive)};
  const std::vector<std::string> &base =
      layernorm ? kLayernormPlan : kRmsnormPlan;
  result.insert(result.end(), base.begin(), base.end());
  result.push_back(std::string(kConvertYPrimitive));
  return result;
}

std::vector<int> checked_ints(std::span<const std::int64_t> values,
                              std::string_view description) {
  if (values.empty() || values.size() > 8) {
    throw std::invalid_argument(std::string(description) + " rank is invalid");
  }
  std::vector<int> result;
  result.reserve(values.size());
  for (const std::int64_t value : values) {
    if (value <= 0 || value > std::numeric_limits<int>::max()) {
      throw std::invalid_argument(std::string(description) +
                                  " is outside positive int32");
    }
    result.push_back(static_cast<int>(value));
  }
  return result;
}

acdnnDataType_t acdnn_data_type(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_FLOAT32:
      return ACDNN_DATA_FLOAT;
    case FLAGDNN_DATA_FLOAT16:
      return ACDNN_DATA_HALF;
    case FLAGDNN_DATA_BFLOAT16:
      return ACDNN_DATA_BF16;
    default:
      throw std::invalid_argument(
          "qualified acDNN normalization data type is unsupported");
  }
}

void set_descriptor(AcdnnTensorDescriptor &descriptor,
                    const flagdnn::testing::TestTensor &tensor) {
  descriptor.set(acdnn_data_type(tensor.data_type),
                 checked_ints(tensor.dimensions, "tensor dimension"),
                 checked_ints(tensor.strides, "tensor stride"));
}

std::size_t element_count(const flagdnn::testing::TestTensor &tensor) {
  std::size_t result = 1;
  for (const std::int64_t dimension : tensor.dimensions) {
    if (dimension <= 0 ||
        static_cast<std::uint64_t>(dimension) >
            std::numeric_limits<std::size_t>::max() / result) {
      throw std::invalid_argument("BatchNorm tensor size is invalid");
    }
    result *= static_cast<std::size_t>(dimension);
  }
  return result;
}

std::map<std::int64_t, void *> binding_map(
    std::span<const flagdnnBinding_t> bindings, std::size_t expected) {
  std::map<std::int64_t, void *> result;
  for (const flagdnnBinding_t &binding : bindings) {
    if (binding.device_pointer == nullptr ||
        !result.emplace(binding.uid, binding.device_pointer).second) {
      throw std::invalid_argument("acDNN BatchNorm binding is null or duplicate");
    }
  }
  if (result.size() != expected) {
    throw std::invalid_argument("acDNN BatchNorm binding count is invalid");
  }
  return result;
}

void require_plan(const CapabilityRecord &capability,
                  std::span<const std::string> expected,
                  ReferencePath expected_path =
                      ReferencePath::kStablePrimitive) {
  const ReferenceSelection selection = select_reference(capability);
  if (!std::holds_alternative<ReferencePlan>(selection)) {
    throw std::invalid_argument("unsupported BatchNorm reached acDNN reference");
  }
  const ReferencePlan &plan = std::get<ReferencePlan>(selection);
  if (plan.path != expected_path ||
      plan.primitives != std::vector<std::string>(expected.begin(),
                                                  expected.end())) {
    throw std::invalid_argument("THead acDNN BatchNorm plan mismatch");
  }
}

class ReduceDescriptor final {
 public:
  ReduceDescriptor() {
    check_acdnn(acdnnCreateReduceTensorDescriptor(&descriptor_),
                "acdnnCreateReduceTensorDescriptor(normalization)");
    check_acdnn(
        acdnnSetReduceTensorDescriptor(
            descriptor_, ACDNN_REDUCE_TENSOR_AVG, ACDNN_DATA_FLOAT,
            ACDNN_NOT_PROPAGATE_NAN, ACDNN_REDUCE_TENSOR_NO_INDICES,
            ACDNN_32BIT_INDICES),
        "acdnnSetReduceTensorDescriptor(normalization AVG)");
  }

  ~ReduceDescriptor() {
    if (descriptor_ != nullptr) {
      (void)acdnnDestroyReduceTensorDescriptor(descriptor_);
    }
  }

  ReduceDescriptor(const ReduceDescriptor &) = delete;
  ReduceDescriptor &operator=(const ReduceDescriptor &) = delete;

  [[nodiscard]] acdnnReduceTensorDescriptor_t get() const noexcept {
    return descriptor_;
  }

 private:
  acdnnReduceTensorDescriptor_t descriptor_ = nullptr;
};

template <typename Case>
bool dense_tensor(const Case &tensor) {
  if (tensor.dimensions.empty() ||
      tensor.dimensions.size() != tensor.strides.size()) {
    return false;
  }
  std::int64_t expected = 1;
  for (std::size_t axis = tensor.dimensions.size(); axis != 0; --axis) {
    const std::int64_t dimension = tensor.dimensions[axis - 1];
    if (dimension <= 0 || tensor.strides[axis - 1] != expected ||
        dimension > std::numeric_limits<std::int64_t>::max() / expected) {
      return false;
    }
    expected *= dimension;
  }
  return true;
}

template <typename Case>
class AcdnnSuffixNormalization final
    : public flagdnn::testing::NormalizationExecutable {
 public:
  AcdnnSuffixNormalization(Case test_case, const CapabilityRecord &capability,
                           bool layernorm)
      : test_case_(std::move(test_case)),
        layernorm_(layernorm),
        first_data_(element_count(test_case_.x) * sizeof(float)),
        second_data_(element_count(test_case_.x) * sizeof(float)),
        first_statistic_(element_count(test_case_.inv_variance) *
                         sizeof(float)),
        second_statistic_(element_count(test_case_.inv_variance) *
                          sizeof(float)),
        epsilon_(element_count(test_case_.inv_variance) * sizeof(float)) {
    const ReferenceSelection selection = select_reference(capability);
    if (!std::holds_alternative<ReferencePlan>(selection)) {
      throw std::invalid_argument(
          "unsupported normalization reached acDNN reference");
    }
    const ReferencePlan &plan = std::get<ReferencePlan>(selection);
    convert_typed_ = test_case_.x.data_type != FLAGDNN_DATA_FLOAT32;
    const std::vector<std::string> expected =
        convert_typed_ ? typed_suffix_plan(layernorm_)
                       : (layernorm_ ? kLayernormPlan : kRmsnormPlan);
    if (plan.path != ReferencePath::kBackendDescriptor ||
        plan.primitives != expected) {
      throw std::invalid_argument("THead acDNN normalization plan mismatch");
    }
    if (test_case_.x.data_type != test_case_.scale.data_type ||
        test_case_.x.data_type != test_case_.bias.data_type ||
        test_case_.x.data_type != test_case_.y.data_type ||
        test_case_.inv_variance.data_type != FLAGDNN_DATA_FLOAT32 ||
        !dense_tensor(test_case_.x) || !dense_tensor(test_case_.scale) ||
        !dense_tensor(test_case_.bias) || !dense_tensor(test_case_.y) ||
        !dense_tensor(test_case_.inv_variance) ||
        test_case_.x.dimensions != test_case_.y.dimensions ||
        test_case_.scale.dimensions != test_case_.bias.dimensions ||
        !std::isfinite(test_case_.epsilon) || test_case_.epsilon <= 0.0) {
      throw std::invalid_argument(
          "qualified acDNN normalization geometry is invalid");
    }
    if constexpr (std::is_same_v<Case,
                                 flagdnn::testing::LayernormTestCase>) {
      if (!layernorm_ ||
          test_case_.mean.data_type != FLAGDNN_DATA_FLOAT32 ||
          !dense_tensor(test_case_.mean) ||
          test_case_.mean.dimensions != test_case_.inv_variance.dimensions) {
        throw std::invalid_argument(
            "qualified acDNN LayerNorm statistic geometry is invalid");
      }
    } else if (layernorm_) {
      throw std::invalid_argument("RMSNorm cannot use the LayerNorm plan");
    }

    const std::size_t normalized = element_count(test_case_.scale);
    if (normalized == 0 || normalized > 65536 ||
        element_count(test_case_.x) % normalized != 0) {
      throw std::invalid_argument(
          "qualified acDNN normalization suffix is invalid");
    }
    flagdnn::testing::TestTensor data_specification = test_case_.x;
    flagdnn::testing::TestTensor parameter_specification = test_case_.scale;
    if (convert_typed_) {
      constexpr std::array<std::int64_t, 4> kConversionUids = {
          std::numeric_limits<std::int64_t>::max() - 31,
          std::numeric_limits<std::int64_t>::max() - 32,
          std::numeric_limits<std::int64_t>::max() - 33,
          std::numeric_limits<std::int64_t>::max() - 34,
      };
      const auto converted = [](const flagdnn::testing::TestTensor &source,
                                std::int64_t uid) {
        flagdnn::testing::TestTensor result = source;
        result.uid = uid;
        result.data_type = FLAGDNN_DATA_FLOAT32;
        result.binding_byte_offset = 0;
        return result;
      };
      converted_x_specification_ = converted(test_case_.x, kConversionUids[0]);
      converted_scale_specification_ =
          converted(test_case_.scale, kConversionUids[1]);
      converted_bias_specification_ =
          converted(test_case_.bias, kConversionUids[2]);
      converted_y_specification_ = converted(test_case_.y, kConversionUids[3]);
      converted_x_ = DeviceBuffer(element_count(converted_x_specification_) *
                                  sizeof(float));
      converted_scale_ = DeviceBuffer(
          element_count(converted_scale_specification_) * sizeof(float));
      converted_bias_ = DeviceBuffer(
          element_count(converted_bias_specification_) * sizeof(float));
      converted_y_ = DeviceBuffer(element_count(converted_y_specification_) *
                                  sizeof(float));
      x_conversion_ = make_acdnn_backend_pointwise_reference(
          {.mode = ACDNN_POINTWISE_IDENTITY_FWD,
           .inputs = {test_case_.x},
           .output = converted_x_specification_,
           .primitive = std::string(kConvertXPrimitive)});
      scale_conversion_ = make_acdnn_backend_pointwise_reference(
          {.mode = ACDNN_POINTWISE_IDENTITY_FWD,
           .inputs = {test_case_.scale},
           .output = converted_scale_specification_,
           .primitive = std::string(kConvertScalePrimitive)});
      bias_conversion_ = make_acdnn_backend_pointwise_reference(
          {.mode = ACDNN_POINTWISE_IDENTITY_FWD,
           .inputs = {test_case_.bias},
           .output = converted_bias_specification_,
           .primitive = std::string(kConvertBiasPrimitive)});
      y_conversion_ = make_acdnn_backend_pointwise_reference(
          {.mode = ACDNN_POINTWISE_IDENTITY_FWD,
           .inputs = {converted_y_specification_},
           .output = test_case_.y,
           .primitive = std::string(kConvertYPrimitive)});
      data_specification = converted_x_specification_;
      parameter_specification = converted_scale_specification_;
    }
    set_descriptor(data_, data_specification);
    set_descriptor(parameter_, parameter_specification);
    set_descriptor(statistic_, test_case_.inv_variance);
    multiply_.set(ACDNN_OP_TENSOR_MUL, ACDNN_DATA_FLOAT,
                  ACDNN_NOT_PROPAGATE_NAN);
    add_.set(ACDNN_OP_TENSOR_ADD, ACDNN_DATA_FLOAT,
             ACDNN_NOT_PROPAGATE_NAN);
    check_acdnn(
        acdnnGetReductionWorkspaceSize(
            handle_.get(), reduction_.get(), data_.get(), statistic_.get(),
            &reduction_workspace_size_),
        "acdnnGetReductionWorkspaceSize(normalization)");

    constexpr std::int64_t kRsqrtInputUid =
        std::numeric_limits<std::int64_t>::max() - 17;
    if (test_case_.inv_variance.uid == kRsqrtInputUid) {
      throw std::invalid_argument("normalization internal UID collides");
    }
    flagdnn::testing::TestTensor rsqrt_input = test_case_.inv_variance;
    rsqrt_input.uid = kRsqrtInputUid;
    rsqrt_input_uid_ = kRsqrtInputUid;
    rsqrt_ = make_acdnn_backend_pointwise_reference(
        {.mode = ACDNN_POINTWISE_RSQRT,
         .inputs = {rsqrt_input},
         .output = test_case_.inv_variance,
         .primitive = "acdnnBackendExecute(POINTWISE_RSQRT)"});

    const float epsilon = static_cast<float>(test_case_.epsilon);
    check_driver(
        cuMemsetD32(epsilon_.address(), std::bit_cast<std::uint32_t>(epsilon),
                    element_count(test_case_.inv_variance)),
        "cuMemsetD32(acDNN normalization epsilon)");
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return std::max(
        {std::size_t{1}, reduction_workspace_size_, rsqrt_->workspace_size(),
         x_conversion_ == nullptr ? std::size_t{0}
                                  : x_conversion_->workspace_size(),
         scale_conversion_ == nullptr ? std::size_t{0}
                                      : scale_conversion_->workspace_size(),
         bias_conversion_ == nullptr ? std::size_t{0}
                                     : bias_conversion_->workspace_size(),
         y_conversion_ == nullptr ? std::size_t{0}
                                  : y_conversion_->workspace_size()});
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    const std::size_t expected_bindings = layernorm_ ? 6 : 5;
    if (stream == nullptr || workspace == nullptr ||
        workspace_size < this->workspace_size()) {
      throw std::invalid_argument("acDNN normalization launch is invalid");
    }
    const auto pointers = binding_map(bindings, expected_bindings);
    void *x = pointers.at(test_case_.x.uid);
    void *scale = pointers.at(test_case_.scale.uid);
    void *bias = pointers.at(test_case_.bias.uid);
    void *y = pointers.at(test_case_.y.uid);
    void *const inv_variance =
        pointers.at(test_case_.inv_variance.uid);
    handle_.set_stream(reinterpret_cast<hggcStream_t>(stream));

    const auto convert = [workspace, stream](
                             flagdnn::testing::TestExecutable &executable,
                             std::int64_t input_uid, void *input,
                             std::int64_t output_uid, void *output) {
      const std::array<flagdnnBinding_t, 2> conversion_bindings = {{
          {input_uid, input}, {output_uid, output}}};
      executable.prepare(conversion_bindings, stream);
      executable.execute(conversion_bindings, workspace,
                         executable.workspace_size(), stream);
    };
    if (convert_typed_) {
      convert(*x_conversion_, test_case_.x.uid, x,
              converted_x_specification_.uid, converted_x_.data());
      convert(*scale_conversion_, test_case_.scale.uid, scale,
              converted_scale_specification_.uid, converted_scale_.data());
      convert(*bias_conversion_, test_case_.bias.uid, bias,
              converted_bias_specification_.uid, converted_bias_.data());
      x = converted_x_.data();
      scale = converted_scale_.data();
      bias = converted_bias_.data();
      y = converted_y_.data();
    }

    constexpr float one = 1.0F;
    constexpr float minus_one = -1.0F;
    if (layernorm_) {
      if constexpr (std::is_same_v<Case,
                                   flagdnn::testing::LayernormTestCase>) {
        void *const mean = pointers.at(test_case_.mean.uid);
        reduce(x, mean, workspace);
        op(add_, data_, x, minus_one, statistic_, mean, data_,
           first_data_.data(), kLayernormPlan[1]);
        op(multiply_, data_, first_data_.data(), one, data_,
           first_data_.data(), data_, second_data_.data(),
           kLayernormPlan[2]);
      }
    } else {
      op(multiply_, data_, x, one, data_, x, data_, first_data_.data(),
         kRmsnormPlan[0]);
    }

    const std::size_t reduction_plan_index = layernorm_ ? 3 : 1;
    reduce(layernorm_ ? second_data_.data() : first_data_.data(),
           first_statistic_.data(), workspace,
           layernorm_ ? kLayernormPlan[reduction_plan_index]
                      : kRmsnormPlan[reduction_plan_index]);
    op(add_, statistic_, first_statistic_.data(), one, statistic_,
       epsilon_.data(), statistic_, second_statistic_.data(),
       layernorm_ ? kLayernormPlan[4] : kRmsnormPlan[2]);

    const std::array<flagdnnBinding_t, 2> rsqrt_bindings = {{
        {rsqrt_input_uid_, second_statistic_.data()},
        {test_case_.inv_variance.uid, inv_variance},
    }};
    rsqrt_->execute(rsqrt_bindings, workspace, rsqrt_->workspace_size(),
                    stream);

    void *const normalization_input = layernorm_ ? first_data_.data() : x;
    op(multiply_, data_, normalization_input, one, statistic_, inv_variance,
       data_, first_data_.data(),
       layernorm_ ? kLayernormPlan[6] : kRmsnormPlan[4]);
    op(multiply_, data_, first_data_.data(), one, parameter_, scale, data_,
       second_data_.data(),
       layernorm_ ? kLayernormPlan[7] : kRmsnormPlan[5]);
    op(add_, data_, second_data_.data(), one, parameter_, bias, data_, y,
       layernorm_ ? kLayernormPlan[8] : kRmsnormPlan[6]);
    if (convert_typed_) {
      convert(*y_conversion_, converted_y_specification_.uid,
              converted_y_.data(), test_case_.y.uid,
              pointers.at(test_case_.y.uid));
    }
  }

 private:
  void op(AcdnnOpTensorDescriptor &descriptor,
          AcdnnTensorDescriptor &left_descriptor, void *left,
          float right_coefficient,
          AcdnnTensorDescriptor &right_descriptor, void *right,
          AcdnnTensorDescriptor &output_descriptor, void *output,
          std::string_view description) {
    constexpr float one = 1.0F;
    constexpr float zero = 0.0F;
    check_acdnn(
        acdnnOpTensor(handle_.get(), descriptor.get(), &one,
                      left_descriptor.get(), left, &right_coefficient,
                      right_descriptor.get(), right, &zero,
                      output_descriptor.get(), output),
        description);
  }

  void reduce(
      void *input, void *output, void *workspace,
      std::string_view description = "acdnnReduceTensor(AVG,mean)") {
    constexpr float one = 1.0F;
    constexpr float zero = 0.0F;
    check_acdnn(
        acdnnReduceTensor(
            handle_.get(), reduction_.get(), workspace, 0, workspace,
            reduction_workspace_size_, &one, data_.get(),
            input, &zero, statistic_.get(), output),
        description);
  }

  Case test_case_;
  bool layernorm_ = false;
  bool convert_typed_ = false;
  AcdnnHandle handle_;
  AcdnnTensorDescriptor data_;
  AcdnnTensorDescriptor parameter_;
  AcdnnTensorDescriptor statistic_;
  AcdnnOpTensorDescriptor multiply_;
  AcdnnOpTensorDescriptor add_;
  ReduceDescriptor reduction_;
  DeviceBuffer first_data_;
  DeviceBuffer second_data_;
  DeviceBuffer first_statistic_;
  DeviceBuffer second_statistic_;
  DeviceBuffer epsilon_;
  flagdnn::testing::TestTensor converted_x_specification_;
  flagdnn::testing::TestTensor converted_scale_specification_;
  flagdnn::testing::TestTensor converted_bias_specification_;
  flagdnn::testing::TestTensor converted_y_specification_;
  DeviceBuffer converted_x_;
  DeviceBuffer converted_scale_;
  DeviceBuffer converted_bias_;
  DeviceBuffer converted_y_;
  std::unique_ptr<flagdnn::testing::TestExecutable> x_conversion_;
  std::unique_ptr<flagdnn::testing::TestExecutable> scale_conversion_;
  std::unique_ptr<flagdnn::testing::TestExecutable> bias_conversion_;
  std::unique_ptr<flagdnn::testing::TestExecutable> y_conversion_;
  std::unique_ptr<flagdnn::testing::TestExecutable> rsqrt_;
  std::int64_t rsqrt_input_uid_ = 0;
  std::size_t reduction_workspace_size_ = 0;
};

class AcdnnBatchnormInference final
    : public flagdnn::testing::NormalizationExecutable {
 public:
  AcdnnBatchnormInference(
      flagdnn::testing::BatchnormInferenceTestCase test_case,
      const CapabilityRecord &capability)
      : test_case_(std::move(test_case)),
        first_(element_count(test_case_.x) * sizeof(float)),
        second_(element_count(test_case_.x) * sizeof(float)) {
    convert_typed_ = test_case_.x.data_type != FLAGDNN_DATA_FLOAT32;
    if (convert_typed_) {
      const std::vector<std::string> plan = typed_inference_plan();
      require_plan(capability, plan, ReferencePath::kBackendDescriptor);
    } else {
      const std::string plan(kInferencePlan);
      require_plan(capability, std::span<const std::string>(&plan, 1));
    }
    if (test_case_.x.data_type != test_case_.y.data_type ||
        test_case_.mean.data_type != FLAGDNN_DATA_FLOAT32 ||
        test_case_.inv_variance.data_type != FLAGDNN_DATA_FLOAT32 ||
        test_case_.scale.data_type != FLAGDNN_DATA_FLOAT32 ||
        test_case_.bias.data_type != FLAGDNN_DATA_FLOAT32) {
      throw std::invalid_argument(
          "qualified acDNN BatchNorm inference types are invalid");
    }
    flagdnn::testing::TestTensor data_specification = test_case_.x;
    if (convert_typed_) {
      constexpr std::int64_t kConvertedXUid =
          std::numeric_limits<std::int64_t>::max() - 41;
      constexpr std::int64_t kConvertedYUid =
          std::numeric_limits<std::int64_t>::max() - 42;
      converted_x_specification_ = test_case_.x;
      converted_x_specification_.uid = kConvertedXUid;
      converted_x_specification_.data_type = FLAGDNN_DATA_FLOAT32;
      converted_y_specification_ = test_case_.y;
      converted_y_specification_.uid = kConvertedYUid;
      converted_y_specification_.data_type = FLAGDNN_DATA_FLOAT32;
      converted_x_ = DeviceBuffer(element_count(converted_x_specification_) *
                                  sizeof(float));
      converted_y_ = DeviceBuffer(element_count(converted_y_specification_) *
                                  sizeof(float));
      x_conversion_ = make_acdnn_backend_pointwise_reference(
          {.mode = ACDNN_POINTWISE_IDENTITY_FWD,
           .inputs = {test_case_.x},
           .output = converted_x_specification_,
           .primitive = std::string(kConvertXPrimitive)});
      y_conversion_ = make_acdnn_backend_pointwise_reference(
          {.mode = ACDNN_POINTWISE_IDENTITY_FWD,
           .inputs = {converted_y_specification_},
           .output = test_case_.y,
           .primitive = std::string(kConvertYPrimitive)});
      data_specification = converted_x_specification_;
    }
    set_descriptor(data_, data_specification);
    set_descriptor(parameter_, test_case_.mean);
    add_.set(ACDNN_OP_TENSOR_ADD, ACDNN_DATA_FLOAT,
             ACDNN_NOT_PROPAGATE_NAN);
    multiply_.set(ACDNN_OP_TENSOR_MUL, ACDNN_DATA_FLOAT,
                  ACDNN_NOT_PROPAGATE_NAN);
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return std::max(
        x_conversion_ == nullptr ? std::size_t{0}
                                 : x_conversion_->workspace_size(),
        y_conversion_ == nullptr ? std::size_t{0}
                                 : y_conversion_->workspace_size());
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    if (stream == nullptr || workspace_size != this->workspace_size() ||
        (workspace_size != 0 && workspace == nullptr)) {
      throw std::invalid_argument("acDNN BatchNorm inference launch is invalid");
    }
    const auto pointers = binding_map(bindings, 6);
    handle_.set_stream(reinterpret_cast<hggcStream_t>(stream));
    void *x = pointers.at(test_case_.x.uid);
    void *y = pointers.at(test_case_.y.uid);
    if (convert_typed_) {
      const std::array<flagdnnBinding_t, 2> conversion_bindings = {{
          {test_case_.x.uid, x},
          {converted_x_specification_.uid, converted_x_.data()},
      }};
      x_conversion_->prepare(conversion_bindings, stream);
      x_conversion_->execute(conversion_bindings, workspace,
                             x_conversion_->workspace_size(), stream);
      x = converted_x_.data();
      y = converted_y_.data();
    }
    constexpr float one = 1.0F;
    constexpr float minus_one = -1.0F;
    constexpr float zero = 0.0F;
    check_acdnn(
        acdnnOpTensor(handle_.get(), add_.get(), &one, data_.get(),
                      x, &minus_one,
                      parameter_.get(), pointers.at(test_case_.mean.uid),
                      &zero, data_.get(), first_.data()),
        "acdnnOpTensor(BatchNorm inference subtract mean)");
    check_acdnn(
        acdnnOpTensor(handle_.get(), multiply_.get(), &one, data_.get(),
                      first_.data(), &one, parameter_.get(),
                      pointers.at(test_case_.inv_variance.uid), &zero,
                      data_.get(), second_.data()),
        "acdnnOpTensor(BatchNorm inference inv variance)");
    check_acdnn(
        acdnnOpTensor(handle_.get(), multiply_.get(), &one, data_.get(),
                      second_.data(), &one, parameter_.get(),
                      pointers.at(test_case_.scale.uid), &zero, data_.get(),
                      first_.data()),
        "acdnnOpTensor(BatchNorm inference scale)");
    check_acdnn(
        acdnnOpTensor(handle_.get(), add_.get(), &one, data_.get(),
                      first_.data(), &one, parameter_.get(),
                      pointers.at(test_case_.bias.uid), &zero, data_.get(),
                      y),
        kInferencePlan);
    if (convert_typed_) {
      const std::array<flagdnnBinding_t, 2> conversion_bindings = {{
          {converted_y_specification_.uid, converted_y_.data()},
          {test_case_.y.uid, pointers.at(test_case_.y.uid)},
      }};
      y_conversion_->prepare(conversion_bindings, stream);
      y_conversion_->execute(conversion_bindings, workspace,
                             y_conversion_->workspace_size(), stream);
    }
  }

 private:
  flagdnn::testing::BatchnormInferenceTestCase test_case_;
  AcdnnHandle handle_;
  AcdnnTensorDescriptor data_;
  AcdnnTensorDescriptor parameter_;
  AcdnnOpTensorDescriptor add_;
  AcdnnOpTensorDescriptor multiply_;
  DeviceBuffer first_;
  DeviceBuffer second_;
  bool convert_typed_ = false;
  flagdnn::testing::TestTensor converted_x_specification_;
  flagdnn::testing::TestTensor converted_y_specification_;
  DeviceBuffer converted_x_;
  DeviceBuffer converted_y_;
  std::unique_ptr<flagdnn::testing::TestExecutable> x_conversion_;
  std::unique_ptr<flagdnn::testing::TestExecutable> y_conversion_;
};

class AcdnnBatchnormTraining final
    : public flagdnn::testing::NormalizationExecutable {
 public:
  AcdnnBatchnormTraining(flagdnn::testing::BatchnormTestCase test_case,
                         const CapabilityRecord &capability)
      : test_case_(std::move(test_case)) {
    convert_typed_ = test_case_.x.data_type != FLAGDNN_DATA_FLOAT32;
    if (convert_typed_) {
      const std::vector<std::string> plan = typed_training_plan();
      require_plan(capability, plan, ReferencePath::kBackendDescriptor);
    } else {
      require_plan(capability, kTrainingPlan);
    }
    if (test_case_.scale.data_type != test_case_.x.data_type ||
        test_case_.bias.data_type != test_case_.x.data_type ||
        test_case_.y.data_type != test_case_.x.data_type ||
        test_case_.previous_running_mean.data_type != FLAGDNN_DATA_FLOAT32 ||
        test_case_.previous_running_variance.data_type != FLAGDNN_DATA_FLOAT32 ||
        test_case_.mean.data_type != FLAGDNN_DATA_FLOAT32 ||
        test_case_.inv_variance.data_type != FLAGDNN_DATA_FLOAT32 ||
        test_case_.next_running_mean.data_type != FLAGDNN_DATA_FLOAT32 ||
        test_case_.next_running_variance.data_type != FLAGDNN_DATA_FLOAT32) {
      throw std::invalid_argument(
          "acDNN BatchNorm training data/statistic types are invalid");
    }
    flagdnn::testing::TestTensor data_specification = test_case_.x;
    flagdnn::testing::TestTensor parameter_specification = test_case_.scale;
    if (convert_typed_) {
      constexpr std::array<std::int64_t, 4> kConversionUids = {
          std::numeric_limits<std::int64_t>::max() - 51,
          std::numeric_limits<std::int64_t>::max() - 52,
          std::numeric_limits<std::int64_t>::max() - 53,
          std::numeric_limits<std::int64_t>::max() - 54,
      };
      const auto converted = [](const flagdnn::testing::TestTensor &source,
                                std::int64_t uid) {
        flagdnn::testing::TestTensor result = source;
        result.uid = uid;
        result.data_type = FLAGDNN_DATA_FLOAT32;
        result.binding_byte_offset = 0;
        return result;
      };
      converted_x_specification_ = converted(test_case_.x, kConversionUids[0]);
      converted_scale_specification_ =
          converted(test_case_.scale, kConversionUids[1]);
      converted_bias_specification_ =
          converted(test_case_.bias, kConversionUids[2]);
      converted_y_specification_ = converted(test_case_.y, kConversionUids[3]);
      converted_x_ = DeviceBuffer(element_count(converted_x_specification_) *
                                  sizeof(float));
      converted_scale_ = DeviceBuffer(
          element_count(converted_scale_specification_) * sizeof(float));
      converted_bias_ = DeviceBuffer(
          element_count(converted_bias_specification_) * sizeof(float));
      converted_y_ = DeviceBuffer(element_count(converted_y_specification_) *
                                  sizeof(float));
      x_conversion_ = make_acdnn_backend_pointwise_reference(
          {.mode = ACDNN_POINTWISE_IDENTITY_FWD,
           .inputs = {test_case_.x},
           .output = converted_x_specification_,
           .primitive = std::string(kConvertXPrimitive)});
      scale_conversion_ = make_acdnn_backend_pointwise_reference(
          {.mode = ACDNN_POINTWISE_IDENTITY_FWD,
           .inputs = {test_case_.scale},
           .output = converted_scale_specification_,
           .primitive = std::string(kConvertScalePrimitive)});
      bias_conversion_ = make_acdnn_backend_pointwise_reference(
          {.mode = ACDNN_POINTWISE_IDENTITY_FWD,
           .inputs = {test_case_.bias},
           .output = converted_bias_specification_,
           .primitive = std::string(kConvertBiasPrimitive)});
      y_conversion_ = make_acdnn_backend_pointwise_reference(
          {.mode = ACDNN_POINTWISE_IDENTITY_FWD,
           .inputs = {converted_y_specification_},
           .output = test_case_.y,
           .primitive = std::string(kConvertYPrimitive)});
      data_specification = converted_x_specification_;
      parameter_specification = converted_scale_specification_;
    }
    set_descriptor(data_, data_specification);
    set_descriptor(parameter_, parameter_specification);
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return std::max(
        {x_conversion_ == nullptr ? std::size_t{0}
                                  : x_conversion_->workspace_size(),
         scale_conversion_ == nullptr ? std::size_t{0}
                                      : scale_conversion_->workspace_size(),
         bias_conversion_ == nullptr ? std::size_t{0}
                                     : bias_conversion_->workspace_size(),
         y_conversion_ == nullptr ? std::size_t{0}
                                  : y_conversion_->workspace_size()});
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    if (stream == nullptr || workspace_size != this->workspace_size() ||
        (workspace_size != 0 && workspace == nullptr)) {
      throw std::invalid_argument("acDNN BatchNorm training launch is invalid");
    }
    const auto pointers = binding_map(bindings, 10);
    handle_.set_stream(reinterpret_cast<hggcStream_t>(stream));
    void *x = pointers.at(test_case_.x.uid);
    void *scale = pointers.at(test_case_.scale.uid);
    void *bias = pointers.at(test_case_.bias.uid);
    void *y = pointers.at(test_case_.y.uid);
    const auto convert = [workspace, stream](
                             flagdnn::testing::TestExecutable &executable,
                             std::int64_t input_uid, void *input,
                             std::int64_t output_uid, void *output) {
      const std::array<flagdnnBinding_t, 2> conversion_bindings = {{
          {input_uid, input}, {output_uid, output}}};
      executable.prepare(conversion_bindings, stream);
      executable.execute(conversion_bindings, workspace,
                         executable.workspace_size(), stream);
    };
    if (convert_typed_) {
      convert(*x_conversion_, test_case_.x.uid, x,
              converted_x_specification_.uid, converted_x_.data());
      convert(*scale_conversion_, test_case_.scale.uid, scale,
              converted_scale_specification_.uid, converted_scale_.data());
      convert(*bias_conversion_, test_case_.bias.uid, bias,
              converted_bias_specification_.uid, converted_bias_.data());
      x = converted_x_.data();
      scale = converted_scale_.data();
      bias = converted_bias_.data();
      y = converted_y_.data();
    }
    constexpr float one = 1.0F;
    constexpr float zero = 0.0F;
    check_acdnn(
        acdnnTransformTensor(
            handle_.get(), &one, parameter_.get(),
            pointers.at(test_case_.previous_running_mean.uid), &zero,
            parameter_.get(), pointers.at(test_case_.next_running_mean.uid)),
        kTrainingPlan[0]);
    check_acdnn(
        acdnnTransformTensor(
            handle_.get(), &one, parameter_.get(),
            pointers.at(test_case_.previous_running_variance.uid), &zero,
            parameter_.get(),
            pointers.at(test_case_.next_running_variance.uid)),
        kTrainingPlan[1]);
    check_acdnn(
        acdnnBatchNormalizationForwardTraining(
            handle_.get(), ACDNN_BATCHNORM_SPATIAL, &one, &zero,
            data_.get(), x, data_.get(), y, parameter_.get(), scale,
            bias, test_case_.momentum,
            pointers.at(test_case_.next_running_mean.uid),
            pointers.at(test_case_.next_running_variance.uid),
            test_case_.epsilon, pointers.at(test_case_.mean.uid),
            pointers.at(test_case_.inv_variance.uid)),
        kTrainingPlan[2]);
    if (convert_typed_) {
      convert(*y_conversion_, converted_y_specification_.uid,
              converted_y_.data(), test_case_.y.uid,
              pointers.at(test_case_.y.uid));
    }
  }

 private:
  flagdnn::testing::BatchnormTestCase test_case_;
  AcdnnHandle handle_;
  AcdnnTensorDescriptor data_;
  AcdnnTensorDescriptor parameter_;
  bool convert_typed_ = false;
  flagdnn::testing::TestTensor converted_x_specification_;
  flagdnn::testing::TestTensor converted_scale_specification_;
  flagdnn::testing::TestTensor converted_bias_specification_;
  flagdnn::testing::TestTensor converted_y_specification_;
  DeviceBuffer converted_x_;
  DeviceBuffer converted_scale_;
  DeviceBuffer converted_bias_;
  DeviceBuffer converted_y_;
  std::unique_ptr<flagdnn::testing::TestExecutable> x_conversion_;
  std::unique_ptr<flagdnn::testing::TestExecutable> scale_conversion_;
  std::unique_ptr<flagdnn::testing::TestExecutable> bias_conversion_;
  std::unique_ptr<flagdnn::testing::TestExecutable> y_conversion_;
};

}  // namespace

std::unique_ptr<flagdnn::testing::NormalizationExecutable>
make_acdnn_layernorm_reference(
    const flagdnn::testing::LayernormTestCase &test_case,
    const CapabilityRecord &capability) {
  return std::make_unique<
      AcdnnSuffixNormalization<flagdnn::testing::LayernormTestCase>>(
      test_case, capability, true);
}

std::unique_ptr<flagdnn::testing::NormalizationExecutable>
make_acdnn_rmsnorm_reference(
    const flagdnn::testing::RmsnormTestCase &test_case,
    const CapabilityRecord &capability) {
  return std::make_unique<
      AcdnnSuffixNormalization<flagdnn::testing::RmsnormTestCase>>(
      test_case, capability, false);
}

std::unique_ptr<flagdnn::testing::NormalizationExecutable>
make_acdnn_batchnorm_reference(
    const flagdnn::testing::BatchnormTestCase &test_case,
    const CapabilityRecord &capability) {
  return std::make_unique<AcdnnBatchnormTraining>(test_case, capability);
}

std::unique_ptr<flagdnn::testing::NormalizationExecutable>
make_acdnn_batchnorm_inference_reference(
    const flagdnn::testing::BatchnormInferenceTestCase &test_case,
    const CapabilityRecord &capability) {
  return std::make_unique<AcdnnBatchnormInference>(test_case, capability);
}

}  // namespace flagdnn::validation::thead

namespace flagdnn::testing {

std::unique_ptr<NormalizationExecutable> build_layernorm_reference(
    const LayernormTestCase &test_case) {
  using namespace flagdnn::validation::thead;
  static const CapabilityCatalog catalog = CapabilityCatalog::load(
      FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
  return make_acdnn_layernorm_reference(
      test_case, catalog.lookup("layernorm", test_case.name));
}

std::unique_ptr<NormalizationExecutable> build_rmsnorm_reference(
    const RmsnormTestCase &test_case) {
  using namespace flagdnn::validation::thead;
  static const CapabilityCatalog catalog = CapabilityCatalog::load(
      FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
  return make_acdnn_rmsnorm_reference(
      test_case, catalog.lookup("rmsnorm", test_case.name));
}

TestTensor batchnorm_reference_data_tensor(const TestTensor &tensor) {
  return tensor;
}

std::unique_ptr<NormalizationExecutable> build_batchnorm_reference(
    const BatchnormTestCase &test_case) {
  using namespace flagdnn::validation::thead;
  static const CapabilityCatalog catalog = CapabilityCatalog::load(
      FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
  return make_acdnn_batchnorm_reference(
      test_case, catalog.lookup("batchnorm", test_case.name));
}

std::unique_ptr<NormalizationExecutable>
build_batchnorm_inference_reference(
    const BatchnormInferenceTestCase &test_case) {
  using namespace flagdnn::validation::thead;
  static const CapabilityCatalog catalog = CapabilityCatalog::load(
      FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
  return make_acdnn_batchnorm_inference_reference(
      test_case, catalog.lookup("batchnorm_inference", test_case.name));
}

}  // namespace flagdnn::testing
