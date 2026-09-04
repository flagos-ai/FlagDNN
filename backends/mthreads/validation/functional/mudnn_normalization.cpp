/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/normalization.hpp"

#include "backends/mthreads/validation/functional/tensor_io_adapter.hpp"
#include "backends/mthreads/validation/mudnn_normalization.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>

namespace flagdnn::testing {
namespace {

namespace mv = validation::mthreads;

template <typename Operation>
class MudnnNormalizationExecutable final
    : public NormalizationExecutable {
 public:
  template <typename Descriptor>
  explicit MudnnNormalizationExecutable(Descriptor descriptor)
      : operation_(std::move(descriptor)) {}

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return operation_.workspace_size();
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    operation_.execute(bindings, workspace, workspace_size, stream);
  }

 private:
  Operation operation_;
};

}  // namespace

TestTensor batchnorm_reference_data_tensor(const TestTensor& tensor) {
  // muDNN 3.1.5 BatchNorm rejects explicit channels-last strides.  The
  // functional runner scatters the same logical values into this dense
  // reference tensor and gathers the result before comparing it with the
  // caller-layout FlagDNN output.
  TestTensor result = tensor;
  result.strides.assign(result.dimensions.size(), 1);
  std::int64_t running = 1;
  for (std::size_t remaining = result.dimensions.size(); remaining != 0;
       --remaining) {
    const std::size_t axis = remaining - 1;
    result.strides[axis] = running;
    if (result.dimensions[axis] >
        std::numeric_limits<std::int64_t>::max() / running) {
      throw std::overflow_error(
          "muDNN BatchNorm reference layout extent overflows int64");
    }
    running *= result.dimensions[axis];
  }
  return result;
}

std::unique_ptr<NormalizationExecutable> build_layernorm_reference(
    const LayernormTestCase& test_case) {
  validate_normalization_case(test_case);
  return std::make_unique<
      MudnnNormalizationExecutable<mv::MudnnLayernormOperation>>(
      mv::MudnnLayernormDescriptor{
          mv::describe_tensor(test_case.x),
          mv::describe_tensor(test_case.scale),
          mv::describe_tensor(test_case.bias),
          mv::describe_tensor(test_case.y),
          mv::describe_tensor(test_case.mean),
          mv::describe_tensor(test_case.inv_variance),
          test_case.epsilon,
      });
}

std::unique_ptr<NormalizationExecutable> build_rmsnorm_reference(
    const RmsnormTestCase& test_case) {
  validate_normalization_case(test_case);
  return std::make_unique<
      MudnnNormalizationExecutable<mv::MudnnRmsnormOperation>>(
      mv::MudnnRmsnormDescriptor{
          mv::describe_tensor(test_case.x),
          mv::describe_tensor(test_case.scale),
          mv::describe_tensor(test_case.bias),
          mv::describe_tensor(test_case.y),
          mv::describe_tensor(test_case.inv_variance),
          test_case.epsilon,
      });
}

std::unique_ptr<NormalizationExecutable> build_batchnorm_reference(
    const BatchnormTestCase& test_case) {
  validate_normalization_case(test_case);
  const TestTensor x = batchnorm_reference_data_tensor(test_case.x);
  const TestTensor y = batchnorm_reference_data_tensor(test_case.y);
  TestTensor scale = test_case.scale;
  TestTensor bias = test_case.bias;
  // muDNN BatchNorm requires FP32 affine parameters for FP16/BF16 data.
  // The runner first quantizes their logical values to the public Graph dtype,
  // then encodes those same values into these promoted reference tensors.
  scale.data_type = FLAGDNN_DATA_FLOAT32;
  bias.data_type = FLAGDNN_DATA_FLOAT32;
  return std::make_unique<
      MudnnNormalizationExecutable<mv::MudnnBatchnormOperation>>(
      mv::MudnnBatchnormDescriptor{
          mv::describe_tensor(x),
          mv::describe_tensor(scale),
          mv::describe_tensor(bias),
          mv::describe_tensor(test_case.previous_running_mean),
          mv::describe_tensor(test_case.previous_running_variance),
          mv::describe_tensor(y),
          mv::describe_tensor(test_case.mean),
          mv::describe_tensor(test_case.inv_variance),
          mv::describe_tensor(test_case.next_running_mean),
          mv::describe_tensor(test_case.next_running_variance),
          test_case.epsilon,
          test_case.momentum,
      });
}

std::unique_ptr<NormalizationExecutable>
build_batchnorm_inference_reference(
    const BatchnormInferenceTestCase& test_case) {
  validate_normalization_case(test_case);
  const TestTensor x = batchnorm_reference_data_tensor(test_case.x);
  const TestTensor y = batchnorm_reference_data_tensor(test_case.y);
  return std::make_unique<MudnnNormalizationExecutable<
      mv::MudnnBatchnormInferenceOperation>>(
      mv::MudnnBatchnormInferenceDescriptor{
          mv::describe_tensor(x),
          mv::describe_tensor(test_case.mean),
          mv::describe_tensor(test_case.inv_variance),
          mv::describe_tensor(test_case.scale),
          mv::describe_tensor(test_case.bias),
          mv::describe_tensor(y),
      });
}

}  // namespace flagdnn::testing
