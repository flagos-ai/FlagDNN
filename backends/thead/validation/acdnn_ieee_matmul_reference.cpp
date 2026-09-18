// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <bit>
#include <limits>
#include <map>

#include "acdnn_matmul_reference.hpp"
#include "acdnn_reference.hpp"
#include "backend_pointwise_reference.hpp"

namespace flagdnn::validation::thead {
namespace {
using flagdnn::testing::MatmulTestCase;
using flagdnn::testing::TestExecutable;
using flagdnn::testing::TestTensor;

std::size_t elements(const TestTensor &tensor) {
  std::size_t result = 1;
  for (auto dimension : tensor.dimensions) {
    if (dimension <= 0 ||
        static_cast<std::uint64_t>(dimension) >
            std::numeric_limits<std::size_t>::max() / result) {
      throw std::overflow_error("IEEE MatMul tensor size overflows");
    }
    result *= static_cast<std::size_t>(dimension);
  }
  return result;
}

std::vector<std::int64_t> dense_strides(const TestTensor &tensor) {
  std::vector<std::int64_t> result(tensor.dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = result.size(); axis != 0; --axis) {
    result[axis - 1] = stride;
    if (tensor.dimensions[axis - 1] >
        std::numeric_limits<std::int64_t>::max() / stride)
      throw std::overflow_error("IEEE MatMul strides overflow");
    stride *= tensor.dimensions[axis - 1];
  }
  return result;
}

class IeeeMatmul final : public TestExecutable {
 public:
  IeeeMatmul(const MatmulTestCase &test_case,
             const CapabilityRecord &capability)
      : test_(test_case), capability_(capability) {
    if (!std::holds_alternative<ReferencePlan>(select_reference(capability)) ||
        capability.path != ReferencePath::kBackendDescriptor ||
        capability.reference_plan != acdnn_ieee_matmul_plan() ||
        test_case.input_precision != 1 ||
        test_case.output.strides != dense_strides(test_case.output)) {
      throw std::invalid_argument(
          "IEEE acDNN MatMul reference contract mismatch");
    }
    for (const auto &tensor : {test_.a, test_.b, test_.output}) {
      if (tensor.uid <= 0 || tensor.dimensions.size() < 2 ||
          tensor.dimensions.size() > 8 ||
          tensor.dimensions.size() != tensor.strides.size() ||
          std::ranges::any_of(tensor.dimensions,
                              [](auto d) { return d <= 0; }) ||
          std::ranges::any_of(tensor.strides, [](auto d) { return d <= 0; }))
        throw std::invalid_argument("invalid IEEE MatMul tensor geometry");
      if (tensor.data_type != FLAGDNN_DATA_FLOAT32)
        throw std::invalid_argument("IEEE MatMul reference requires FP32");
      std::size_t storage = 1;
      for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
        const auto extent =
            static_cast<std::size_t>(tensor.dimensions[axis] - 1);
        const auto stride = static_cast<std::size_t>(tensor.strides[axis]);
        if (extent >
            (std::numeric_limits<std::size_t>::max() - storage) / stride)
          throw std::overflow_error("IEEE MatMul storage overflows");
        storage += extent * stride;
      }
      if (storage != elements(tensor))
        throw std::invalid_argument(
            "IEEE MatMul qualification requires dense matrix storage");
      next_uid_ = std::max(next_uid_, tensor.uid);
    }
    const auto [ah, al] = split(test_.a);
    const auto [bh, bl] = split(test_.b);
    auto hh = multiply(ah, bh);
    auto cross =
        pointwise(ACDNN_POINTWISE_ADD, {multiply(ah, bl), multiply(al, bh)});
    auto low = scale(multiply(al, bl), 1.0F / 2048.0F);
    auto correction =
        scale(pointwise(ACDNN_POINTWISE_ADD, {cross, low}), 1.0F / 2048.0F);
    add_pointwise(ACDNN_POINTWISE_ADD, {hh, correction}, test_.output);
  }

  std::size_t workspace_size() const noexcept override { return workspace_; }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t size, flagdnnStream_t stream) override {
    if (bindings.size() != 3 || size < workspace_ || (workspace_ && !workspace))
      throw std::invalid_argument("IEEE MatMul bindings/workspace mismatch");
    std::map<std::int64_t, void *> pointers;
    for (const auto &binding : bindings) {
      if (!binding.device_pointer ||
          !pointers.emplace(binding.uid, binding.device_pointer).second)
        throw std::invalid_argument("invalid IEEE MatMul binding");
    }
    for (auto &[uid, buffer] : buffers_) pointers.emplace(uid, buffer.data());
    for (auto &step : steps_) {
      std::vector<flagdnnBinding_t> arguments;
      for (auto uid : step.uids) arguments.push_back({uid, pointers.at(uid)});
      step.executable->prepare(arguments, stream);
      const auto bytes = step.executable->workspace_size();
      step.executable->execute(arguments, bytes ? workspace : nullptr, bytes,
                               stream);
    }
  }

 private:
  TestTensor temporary(const TestTensor &source,
                       flagdnnDataType_t type = FLAGDNN_DATA_FLOAT32) {
    if (next_uid_ == std::numeric_limits<std::int64_t>::max())
      throw std::overflow_error("IEEE MatMul UID overflows");
    auto result = source;
    result.uid = ++next_uid_;
    result.data_type = type;
    result.binding_byte_offset = 0;
    const auto count = elements(result);
    const std::size_t width = type == FLAGDNN_DATA_FLOAT16 ? 2 : 4;
    if (count > std::numeric_limits<std::size_t>::max() / width)
      throw std::overflow_error("IEEE MatMul allocation overflows");
    buffers_.emplace(result.uid, DeviceBuffer(count * width));
    return result;
  }
  static TestTensor flat(TestTensor tensor) {
    const auto count = elements(tensor);
    if (count >
        static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
      throw std::overflow_error("IEEE MatMul flat extent overflows");
    tensor.dimensions = {1, 1, static_cast<std::int64_t>(count)};
    tensor.strides = {static_cast<std::int64_t>(count),
                      static_cast<std::int64_t>(count), 1};
    return tensor;
  }
  void append(std::unique_ptr<TestExecutable> executable,
              std::vector<std::int64_t> uids) {
    workspace_ = std::max(workspace_, executable->workspace_size());
    steps_.push_back({std::move(executable), std::move(uids)});
  }
  void add_pointwise(acdnnPointwiseMode_t mode, std::vector<TestTensor> inputs,
                     const TestTensor &output) {
    std::vector<std::int64_t> uids;
    for (auto &input : inputs) {
      uids.push_back(input.uid);
      input = flat(input);
    }
    uids.push_back(output.uid);
    append(make_acdnn_backend_pointwise_reference(
               {.mode = mode,
                .inputs = std::move(inputs),
                .output = flat(output),
                .primitive = "acdnnBackendExecute(IEEE MatMul residual node)"}),
           std::move(uids));
  }
  TestTensor pointwise(acdnnPointwiseMode_t mode,
                       std::vector<TestTensor> inputs) {
    auto output = temporary(inputs.front());
    add_pointwise(mode, std::move(inputs), output);
    return output;
  }
  TestTensor scale(const TestTensor &input, float factor) {
    auto constant = temporary(input);
    check_driver(
        cuMemsetD32(buffers_.at(constant.uid).address(),
                    std::bit_cast<unsigned int>(factor), elements(constant)),
        "IEEE MatMul scale constant");
    check_driver(cuStreamSynchronize(nullptr), "IEEE MatMul constant ready");
    return pointwise(ACDNN_POINTWISE_MUL, {input, constant});
  }
  std::pair<TestTensor, TestTensor> split(const TestTensor &input) {
    // All public precision inputs lie in the finite FP16 range. Scaling the
    // residual protects its mantissa and subnormal values in SDK 1400 MatMul.
    auto half = temporary(input, FLAGDNN_DATA_FLOAT16);
    add_pointwise(ACDNN_POINTWISE_IDENTITY_FWD, {input}, half);
    auto high = pointwise(ACDNN_POINTWISE_IDENTITY_FWD, {half});
    auto low = scale(pointwise(ACDNN_POINTWISE_SUB, {input, high}), 2048.0F);
    return {high, low};
  }
  TestTensor multiply(const TestTensor &a, const TestTensor &b) {
    auto output = temporary(test_.output);
    auto product = test_;
    product.a = a;
    product.b = b;
    product.output = output;
    product.input_precision = 2;
    auto capability = capability_;
    capability.reference_plan = {"acdnnBackendExecute(MATMUL)"};
    append(make_acdnn_matmul_reference(product, capability),
           {a.uid, b.uid, output.uid});
    return output;
  }
  struct Step {
    std::unique_ptr<TestExecutable> executable;
    std::vector<std::int64_t> uids;
  };
  MatmulTestCase test_;
  CapabilityRecord capability_;
  std::int64_t next_uid_ = 0;
  std::size_t workspace_ = 0;
  std::map<std::int64_t, DeviceBuffer> buffers_;
  std::vector<Step> steps_;
};
}  // namespace

std::vector<std::string> acdnn_ieee_matmul_plan() {
  return {"acdnnBackendExecute(IDENTITY,SUB,MUL,FP32-high-low-split)",
          "acdnnBackendExecute(MATMUL,FP32-residual-products)",
          "acdnnBackendExecute(ADD,MUL,FP32-residual-combine)"};
}

std::unique_ptr<flagdnn::testing::MatmulExecutable>
make_acdnn_ieee_matmul_reference(
    const flagdnn::testing::MatmulTestCase &test_case,
    const CapabilityRecord &capability) {
  return std::make_unique<IeeeMatmul>(test_case, capability);
}
}  // namespace flagdnn::validation::thead
