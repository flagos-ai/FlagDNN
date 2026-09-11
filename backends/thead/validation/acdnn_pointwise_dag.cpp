// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "acdnn_pointwise_dag.hpp"
#include "acdnn_reference.hpp"
#include "backend_pointwise_reference.hpp"
#include "ppu_driver.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace flagdnn::validation::thead {
namespace {
using flagdnn::testing::TestExecutable;
using flagdnn::testing::TestTensor;

std::size_t aligned(std::size_t bytes) {
  if (bytes > std::numeric_limits<std::size_t>::max() - 255) {
    throw std::overflow_error("acDNN pointwise DAG workspace overflows");
  }
  return (bytes + 255) / 256 * 256;
}

class GeluDerivative final : public TestExecutable {
 public:
  GeluDerivative(std::size_t elements, std::array<std::int64_t, 4> uids)
      : uids_(uids) {
    if (elements > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::overflow_error("acDNN GELU derivative extent exceeds int32");
    }
    const int count = static_cast<int>(elements);
    tensor_.set(ACDNN_DATA_FLOAT, std::array<int, 4>{1, 1, 1, count},
                std::array<int, 4>{count, count, count, 1});
    activation_.set(ACDNN_ACTIVATION_GELU, ACDNN_NOT_PROPAGATE_NAN, 0.0);
  }
  std::size_t workspace_size() const noexcept override { return 0; }
  void execute(std::span<const flagdnnBinding_t> bindings, void *,
               std::size_t workspace_size, flagdnnStream_t stream) override {
    if (workspace_size != 0 || bindings.size() != 4) {
      throw std::invalid_argument("invalid acDNN GELU derivative bindings");
    }
    std::map<std::int64_t, void *> pointers;
    for (const auto &binding : bindings) {
      if (!binding.device_pointer ||
          !pointers.emplace(binding.uid, binding.device_pointer).second) {
        throw std::invalid_argument("invalid acDNN GELU derivative pointer");
      }
    }
    handle_.set_stream(reinterpret_cast<hggcStream_t>(stream));
    const float one = 1.0F, zero = 0.0F;
    // The forward result and unit upstream gradient are acDNN inputs to the
    // exact GELU backward primitive. No host input arithmetic is performed.
    check_acdnn(acdnnActivationForward(
        handle_.get(), activation_.get(), &one, tensor_.get(),
        pointers.at(uids_[0]), &zero, tensor_.get(), pointers.at(uids_[1])),
        "acdnnActivationForward(GELU,Erf-DAG)");
    check_acdnn(acdnnActivationBackward(
        handle_.get(), activation_.get(), &one, tensor_.get(),
        pointers.at(uids_[1]), tensor_.get(), pointers.at(uids_[2]),
        tensor_.get(), pointers.at(uids_[0]), &zero, tensor_.get(),
        pointers.at(uids_[3])), "acdnnActivationBackward(GELU,Erf-DAG)");
  }
 private:
  std::array<std::int64_t, 4> uids_;
  AcdnnHandle handle_;
  AcdnnTensorDescriptor tensor_;
  AcdnnActivationDescriptor activation_;
};

class TensorTransform final : public TestExecutable {
 public:
  TensorTransform(const TestTensor &input, const TestTensor &output)
      : input_uid_(input.uid), output_uid_(output.uid) {
    if (input.data_type != output.data_type || input.dimensions != output.dimensions) {
      throw std::invalid_argument("acDNN DAG layout transform type/shape mismatch");
    }
    const acdnnDataType_t type = input.data_type == FLAGDNN_DATA_FLOAT32 ? ACDNN_DATA_FLOAT :
        input.data_type == FLAGDNN_DATA_FLOAT16 ? ACDNN_DATA_HALF : ACDNN_DATA_BF16;
    const auto integers = [](const std::vector<std::int64_t> &values) {
      std::vector<int> result;
      for (auto value : values) {
        if (value <= 0 || value > std::numeric_limits<int>::max()) {
          throw std::overflow_error("acDNN DAG transform metadata exceeds int32");
        }
        result.push_back(static_cast<int>(value));
      }
      return result;
    };
    const std::int64_t length = input.strides.back() == 1 && output.strides.back() == 1
                                    ? input.dimensions.back() : 1;
    std::size_t elements = 1;
    for (auto extent : input.dimensions) {
      if (extent <= 0 || static_cast<std::uint64_t>(extent) >
          std::numeric_limits<std::size_t>::max() / elements) {
        throw std::overflow_error("acDNN DAG transform size overflows");
      }
      elements *= static_cast<std::size_t>(extent);
    }
    scalar_bytes_ = input.data_type == FLAGDNN_DATA_FLOAT32 ? 4 : 2;
    const auto offset = [&](std::size_t logical, const TestTensor &tensor) {
      std::size_t physical = 0;
      for (std::size_t axis = tensor.dimensions.size(); axis != 0; --axis) {
        const auto dimension = static_cast<std::size_t>(tensor.dimensions[axis - 1]);
        const auto coordinate = logical % dimension;
        logical /= dimension;
        const auto stride = static_cast<std::size_t>(tensor.strides[axis - 1]);
        if (coordinate && stride > (std::numeric_limits<std::size_t>::max() - physical) / coordinate) {
          throw std::overflow_error("acDNN DAG transform offset overflows");
        }
        physical += coordinate * stride;
      }
      if (physical > std::numeric_limits<std::size_t>::max() / scalar_bytes_) {
        throw std::overflow_error("acDNN DAG transform byte offset overflows");
      }
      return physical * scalar_bytes_;
    };
    for (std::size_t logical = 0; logical < elements; logical += static_cast<std::size_t>(length)) {
      segments_.emplace_back(offset(logical, input), offset(logical, output));
    }
    // Legacy TransformTensor ignores padding for some strided descriptors.
    // Copy only contiguous logical segments, as in the qualified Slice reference.
    input_.set(type, integers({1, 1, length}), integers({length, length, 1}));
    output_.set(type, integers({1, 1, length}), integers({length, length, 1}));
  }
  std::size_t workspace_size() const noexcept override { return 0; }
  void execute(std::span<const flagdnnBinding_t> bindings, void *,
               std::size_t size, flagdnnStream_t stream) override {
    std::map<std::int64_t, void *> pointers;
    for (const auto &binding : bindings) pointers.emplace(binding.uid, binding.device_pointer);
    if (size || bindings.size() != 2 || !pointers.at(input_uid_) || !pointers.at(output_uid_)) {
      throw std::invalid_argument("invalid acDNN DAG transform bindings");
    }
    handle_.set_stream(reinterpret_cast<hggcStream_t>(stream));
    const float one = 1, zero = 0;
    for (const auto &[input_offset, output_offset] : segments_) {
      check_acdnn(acdnnTransformTensor(handle_.get(), &one, input_.get(),
          static_cast<std::byte *>(pointers.at(input_uid_)) + input_offset,
          &zero, output_.get(), static_cast<std::byte *>(pointers.at(output_uid_)) + output_offset),
          "acdnnTransformTensor(pointwise DAG layout segment)");
    }
  }
 private:
  std::int64_t input_uid_, output_uid_;
  std::size_t scalar_bytes_ = 0;
  std::vector<std::pair<std::size_t, std::size_t>> segments_;
  AcdnnHandle handle_;
  AcdnnTensorDescriptor input_, output_;
};

class PointwiseDag final : public TestExecutable {
 public:
  PointwiseDag(const PointwiseReferenceSpecification &specification,
               const CapabilityRecord &capability) : output_(specification.output), capability_(capability) {
    const auto plan = acdnn_pointwise_dag_plan(specification.mode);
    const auto selection = select_reference(capability);
    if (plan.empty() || !std::holds_alternative<ReferencePlan>(selection) ||
        capability.path != ReferencePath::kBackendDescriptor ||
        capability.reference_plan != plan || specification.alpha != 1.0 ||
        specification.attributes.flags != 0) {
      throw std::invalid_argument("acDNN pointwise DAG reference plan mismatch");
    }
    const bool logical = specification.mode == FLAGDNN_POINTWISE_LOGICAL_NOT ||
                         specification.mode == FLAGDNN_POINTWISE_LOGICAL_AND ||
                         specification.mode == FLAGDNN_POINTWISE_LOGICAL_OR;
    const bool select = specification.mode == FLAGDNN_POINTWISE_BINARY_SELECT;
    const std::size_t arity = select ? 3 :
        (specification.mode == FLAGDNN_POINTWISE_MOD ||
         specification.mode == FLAGDNN_POINTWISE_LOGICAL_AND ||
         specification.mode == FLAGDNN_POINTWISE_LOGICAL_OR) ? 2 : 1;
    if (specification.inputs.size() != arity || output_.dimensions.empty() ||
        output_.dimensions.size() > 8 ||
        (logical ? output_.data_type != FLAGDNN_DATA_BOOLEAN :
         !floating(output_.data_type))) {
      throw std::invalid_argument("acDNN pointwise DAG tensor contract mismatch");
    }
    register_external(output_);
    for (std::size_t index = 0; index < arity; ++index) {
      const auto &input = specification.inputs[index];
      const auto expected = (logical || (select && index == 2))
                                ? FLAGDNN_DATA_BOOLEAN : output_.data_type;
      if (input.dimensions != output_.dimensions || input.data_type != expected) {
        throw std::invalid_argument("acDNN pointwise DAG input contract mismatch");
      }
      register_external(input);
    }
    dense_ = output_;
    dense_.data_type = FLAGDNN_DATA_FLOAT32;
    elements_ = 1;
    for (std::size_t axis = dense_.dimensions.size(); axis != 0; --axis) {
      const auto extent = dense_.dimensions[axis - 1];
      if (extent <= 0 || elements_ >
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) /
              static_cast<std::size_t>(extent)) {
        throw std::overflow_error("acDNN pointwise DAG tensor size overflows");
      }
      dense_.strides[axis - 1] = static_cast<std::int64_t>(elements_);
      elements_ *= static_cast<std::size_t>(extent);
    }
    if (elements_ > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
      throw std::overflow_error("acDNN pointwise DAG allocation overflows");
    }
    std::uint64_t last = 0;
    for (std::size_t axis = 0; axis < output_.dimensions.size(); ++axis) {
      const auto extent = static_cast<std::uint64_t>(output_.dimensions[axis] - 1);
      const auto stride = static_cast<std::uint64_t>(output_.strides[axis]);
      if (extent && stride > (std::numeric_limits<std::uint64_t>::max() - last) / extent) {
        throw std::overflow_error("acDNN pointwise DAG output storage overflows");
      }
      last += extent * stride;
    }
    // Preserve a shared dense permutation (for example NHWC). AcDNN identity
    // conversion requires matching layouts; elementwise math is permutation invariant.
    if (last == elements_ - 1) dense_.strides = output_.strides;
    std::vector<TestTensor> inputs;
    for (const auto &input : specification.inputs) {
      inputs.push_back(node(ACDNN_POINTWISE_IDENTITY_FWD, {input}));
    }
    TestTensor result;
    if (logical) {
      if (arity == 1) result = node(ACDNN_POINTWISE_SUB, {constant(1), inputs[0]});
      else result = node(specification.mode == FLAGDNN_POINTWISE_LOGICAL_AND
                             ? ACDNN_POINTWISE_MUL : ACDNN_POINTWISE_MAX,
                         {inputs[0], inputs[1]});
    } else if (select) {
      // The qualified reference domain contains finite values. Masking with
      // 0/1 preserves those values exactly in FP32 (including FP16/BF16 casts).
      auto selected = node(ACDNN_POINTWISE_MUL, {inputs[2], inputs[0]});
      auto inverse = node(ACDNN_POINTWISE_SUB, {constant(1), inputs[2]});
      auto other = node(ACDNN_POINTWISE_MUL, {inverse, inputs[1]});
      result = node(ACDNN_POINTWISE_ADD, {selected, other});
    } else if (specification.mode == FLAGDNN_POINTWISE_MOD) {
      auto quotient = node(ACDNN_POINTWISE_DIV, {inputs[0], inputs[1]});
      auto floor = node(ACDNN_POINTWISE_FLOOR, {quotient});
      auto ceil = node(ACDNN_POINTWISE_CEIL, {quotient});
      auto positive = node(ACDNN_POINTWISE_MAX, {floor, constant(0)});
      auto negative = node(ACDNN_POINTWISE_MIN, {ceil, constant(0)});
      auto truncated = node(ACDNN_POINTWISE_ADD, {positive, negative});
      auto product = node(ACDNN_POINTWISE_MUL, {inputs[1], truncated});
      result = node(ACDNN_POINTWISE_SUB, {inputs[0], product});
    } else {
      // erf(x) = 2*GELU'(sqrt(2)*x) - 2/sqrt(pi)*x*exp(-x*x) - 1.
      // This identity is regular at x=0, unlike division by GELU's input.
      auto scaled = node(ACDNN_POINTWISE_MUL, {inputs[0], constant(1.4142135623730951F)});
      auto forward = temporary();
      auto derivative = temporary();
      auto ones = constant(1);
      Step step;
      for (const auto &tensor : {scaled, forward, ones, derivative}) {
        step.bindings.emplace_back(tensor.uid, tensor.uid);
      }
      step.executable = std::make_unique<GeluDerivative>(elements_,
          std::array<std::int64_t, 4>{scaled.uid, forward.uid, ones.uid, derivative.uid});
      steps_.push_back(std::move(step));
      auto twice = node(ACDNN_POINTWISE_MUL, {derivative, constant(2)});
      auto square = node(ACDNN_POINTWISE_MUL, {inputs[0], inputs[0]});
      auto negative_square = node(ACDNN_POINTWISE_SUB, {constant(0), square});
      auto exponential = node(ACDNN_POINTWISE_EXP, {negative_square});
      auto weighted = node(ACDNN_POINTWISE_MUL, {inputs[0], constant(1.1283791670955126F)});
      auto correction = node(ACDNN_POINTWISE_MUL, {weighted, exponential});
      result = node(ACDNN_POINTWISE_SUB, {twice, correction});
      result = node(ACDNN_POINTWISE_SUB, {result, ones});
    }
    add_node(ACDNN_POINTWISE_IDENTITY_FWD, {result}, output_);
    scratch_offset_ = workspace_bytes_;
    std::size_t scratch = 0;
    for (const auto &step : steps_) scratch = std::max(scratch, step.executable->workspace_size());
    if (scratch > std::numeric_limits<std::size_t>::max() - workspace_bytes_) {
      throw std::overflow_error("acDNN pointwise DAG scratch overflows");
    }
    workspace_bytes_ += scratch;
  }

  std::size_t workspace_size() const noexcept override { return workspace_bytes_; }
  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t size, flagdnnStream_t stream) override {
    if (!workspace || size < workspace_bytes_ || bindings.size() != external_.size()) {
      throw std::invalid_argument("acDNN pointwise DAG workspace/bindings mismatch");
    }
    std::map<std::int64_t, void *> pointers;
    for (const auto &binding : bindings) {
      if (!binding.device_pointer || !external_.contains(binding.uid) ||
          !pointers.emplace(binding.uid, binding.device_pointer).second) {
        throw std::invalid_argument("acDNN pointwise DAG invalid external binding");
      }
    }
    auto *base = static_cast<std::byte *>(workspace);
    for (const auto &[uid, offset] : offsets_) pointers.emplace(uid, base + offset);
    for (const auto &[uid, value] : constants_) {
      check_driver(cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(pointers.at(uid)),
          std::bit_cast<unsigned int>(value), elements_, reinterpret_cast<CUstream>(stream)),
          "cuMemsetD32Async(acDNN DAG constant)");
    }
    for (auto &step : steps_) {
      std::vector<flagdnnBinding_t> stage_bindings;
      for (const auto &[alias, uid] : step.bindings) {
        stage_bindings.push_back({alias, pointers.at(uid)});
      }
      step.executable->prepare(stage_bindings, stream);
      const auto scratch = step.executable->workspace_size();
      step.executable->execute(stage_bindings, scratch ? base + scratch_offset_ : nullptr,
                                scratch, stream);
    }
  }
 private:
  static bool floating(flagdnnDataType_t type) {
    return type == FLAGDNN_DATA_FLOAT32 || type == FLAGDNN_DATA_FLOAT16 ||
           type == FLAGDNN_DATA_BFLOAT16;
  }
  void register_external(const TestTensor &tensor) {
    if (tensor.dimensions.size() != tensor.strides.size() ||
        std::any_of(tensor.strides.begin(), tensor.strides.end(),
                    [](auto stride) { return stride <= 0; }) ||
        !external_.emplace(tensor.uid, tensor).second) {
      throw std::invalid_argument("acDNN pointwise DAG invalid tensor metadata");
    }
    next_uid_ = std::max(next_uid_, tensor.uid);
  }
  std::int64_t uid() {
    if (next_uid_ == std::numeric_limits<std::int64_t>::max()) {
      throw std::overflow_error("acDNN pointwise DAG UID overflows");
    }
    return ++next_uid_;
  }
  TestTensor temporary() {
    auto tensor = dense_;
    tensor.uid = uid();
    tensor.binding_byte_offset = 0;
    const auto bytes = aligned(elements_ * sizeof(float));
    if (bytes > std::numeric_limits<std::size_t>::max() - workspace_bytes_) {
      throw std::overflow_error("acDNN pointwise DAG workspace overflows");
    }
    offsets_.emplace(tensor.uid, workspace_bytes_);
    workspace_bytes_ += bytes;
    return tensor;
  }
  TestTensor constant(float value) {
    for (const auto &[id, current] : constants_) {
      if (current == value) { auto tensor = dense_; tensor.uid = id; return tensor; }
    }
    auto tensor = temporary();
    constants_.emplace(tensor.uid, value);
    return tensor;
  }
  TestTensor node(acdnnPointwiseMode_t mode, std::vector<TestTensor> inputs) {
    auto output = temporary();
    add_node(mode, std::move(inputs), output);
    return output;
  }
  void add_node(acdnnPointwiseMode_t mode, std::vector<TestTensor> inputs,
                const TestTensor &output) {
    Step step;
    if (mode == ACDNN_POINTWISE_IDENTITY_FWD && inputs.size() == 1 &&
        inputs[0].strides != output.strides) {
      if (!floating(inputs[0].data_type) || inputs[0].data_type != output.data_type) {
        throw std::invalid_argument("acDNN DAG layout conversion requires matching floating types");
      }
      step.bindings = {{inputs[0].uid, inputs[0].uid}, {output.uid, output.uid}};
      step.executable = std::make_unique<TensorTransform>(inputs[0], output);
      steps_.push_back(std::move(step));
      return;
    }
    for (auto &input : inputs) {
      auto original = input.uid;
      input.uid = uid();
      step.bindings.emplace_back(input.uid, original);
    }
    step.bindings.emplace_back(output.uid, output.uid);
    if (mode == ACDNN_POINTWISE_MIN || mode == ACDNN_POINTWISE_MAX) {
      const bool minimum = mode == ACDNN_POINTWISE_MIN;
      CapabilityRecord capability = capability_;
      capability.path = ReferencePath::kStablePrimitive;
      capability.reference_plan = {minimum ? "acdnnOpTensor(MIN)" : "acdnnOpTensor(MAX)"};
      capability.constraints->dtypes = {"fp32"};
      capability.constraints->compute_type = "fp32";
      step.executable = make_acdnn_pointwise_reference(
          {.mode = minimum ? FLAGDNN_POINTWISE_MIN : FLAGDNN_POINTWISE_MAX,
           .inputs = std::move(inputs), .output = output}, capability);
      steps_.push_back(std::move(step));
      return;
    }
    step.executable = make_acdnn_backend_pointwise_reference(
        {.mode = mode, .inputs = std::move(inputs), .output = output,
         .primitive = "acdnnBackendExecute(pointwise DAG node)"});
    steps_.push_back(std::move(step));
  }
  struct Step {
    std::unique_ptr<TestExecutable> executable;
    std::vector<std::pair<std::int64_t, std::int64_t>> bindings;
  };
  TestTensor output_, dense_;
  CapabilityRecord capability_;
  std::int64_t next_uid_ = 0;
  std::size_t elements_ = 0, workspace_bytes_ = 0, scratch_offset_ = 0;
  std::map<std::int64_t, TestTensor> external_;
  std::map<std::int64_t, std::size_t> offsets_;
  std::map<std::int64_t, float> constants_;
  std::vector<Step> steps_;
};
}  // namespace

std::vector<std::string> acdnn_pointwise_dag_plan(flagdnnPointwiseMode_t mode) {
  const std::string cast = "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-fp32)";
  const std::string output = "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-output)";
  switch (mode) {
    case FLAGDNN_POINTWISE_LOGICAL_NOT:
      return {cast, "acdnnBackendExecute(POINTWISE_SUB,1-x)", output};
    case FLAGDNN_POINTWISE_LOGICAL_AND:
      return {cast, "acdnnBackendExecute(POINTWISE_MUL,x*y)", output};
    case FLAGDNN_POINTWISE_LOGICAL_OR:
      return {cast, "acdnnOpTensor(MAX,x,y)", output};
    case FLAGDNN_POINTWISE_BINARY_SELECT:
      return {cast, "acdnnTransformTensor(layout-if-needed)",
              "acdnnBackendExecute(POINTWISE_MUL,SUB,ADD,select-DAG)", output};
    case FLAGDNN_POINTWISE_MOD:
      return {cast, "acdnnBackendExecute(POINTWISE_DIV,FLOOR,CEIL,ADD,MUL,SUB,trunc-remainder-DAG)", "acdnnOpTensor(MAX,MIN,trunc-remainder-DAG)", output};
    case FLAGDNN_POINTWISE_ERF:
      return {cast, "acdnnActivationForward(GELU,Erf-DAG)",
              "acdnnActivationBackward(GELU,Erf-DAG)",
              "acdnnBackendExecute(POINTWISE_MUL,EXP,SUB,Erf-DAG)", output};
    default: return {};
  }
}

std::unique_ptr<TestExecutable> make_acdnn_pointwise_dag(
    const PointwiseReferenceSpecification &specification,
    const CapabilityRecord &capability) {
  return std::make_unique<PointwiseDag>(specification, capability);
}
}  // namespace flagdnn::validation::thead
