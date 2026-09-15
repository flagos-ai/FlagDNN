/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include <cudnn_subquadratic_ops.h>

#include <limits>

#include "common/causal_convolution.hpp"
#include "validation/functional/cudnn_graph.hpp"
namespace flagdnn::testing {
namespace {
class CudnnCausalConvolution final : public TestExecutable {
 public:
  explicit CudnnCausalConvolution(const CausalConvolutionTestCase& test_case)
      : test_case_(test_case) {
    if (test_case.dilation != 1 || test_case.precision == 2 ||
        test_case.inputs.size() != 3)
      throw std::invalid_argument(
          "cuDNN causal conv1d requires dilation one, bias and FP32/FP16/BF16");
    for (const auto& tensor : test_case.inputs) validate_tensor(tensor);
    for (const auto& tensor : test_case.outputs) validate_tensor(tensor);
  }
  std::size_t workspace_size() const noexcept override { return 0; }
  void execute(std::span<const flagdnnBinding_t> bindings, void*, std::size_t,
               flagdnnStream_t stream) override {
    const auto pointers = cuda::make_cudnn_binding_map(bindings);
    const auto& x = test_case_.inputs[0];
    const auto& weight = test_case_.inputs[1];
    const auto dtype = x.data_type == FLAGDNN_DATA_FLOAT32 ? CUDNN_DATA_FLOAT
                       : x.data_type == FLAGDNN_DATA_FLOAT16
                           ? CUDNN_DATA_HALF
                           : CUDNN_DATA_BFLOAT16;
    cuda::check_cudnn(
        cudnnCausalConv1dForward(
            reinterpret_cast<cudaStream_t>(stream), pointers.at(x.uid),
            pointers.at(weight.uid), pointers.at(test_case_.inputs[2].uid),
            pointers.at(test_case_.outputs[0].uid),
            static_cast<int>(x.dimensions.at(0)),
            static_cast<int>(x.dimensions.at(1)),
            static_cast<int>(x.dimensions.at(2)),
            static_cast<int>(weight.dimensions.at(1)), dtype,
            test_case_.silu ? CUDNN_CAUSAL_CONV1D_ACTIVATION_SILU
                            : CUDNN_CAUSAL_CONV1D_ACTIVATION_IDENTITY),
        "cudnnCausalConv1dForward");
  }

 private:
  static void validate_tensor(const TestTensor& tensor) {
    if (tensor.data_type != FLAGDNN_DATA_FLOAT32 &&
        tensor.data_type != FLAGDNN_DATA_FLOAT16 &&
        tensor.data_type != FLAGDNN_DATA_BFLOAT16)
      throw std::invalid_argument("unsupported cuDNN causal conv1d datatype");
    std::int64_t stride = 1;
    for (std::size_t axis = tensor.dimensions.size(); axis > 0; --axis) {
      const auto dim = tensor.dimensions[axis - 1];
      if (dim <= 0 || dim > std::numeric_limits<int>::max() ||
          tensor.strides.at(axis - 1) != stride)
        throw std::invalid_argument(
            "cuDNN causal conv1d requires compact tensors");
      stride *= dim;
    }
  }
  CausalConvolutionTestCase test_case_;
};
}  // namespace
std::unique_ptr<TestExecutable> build_cudnn_causal_convolution(
    const CausalConvolutionTestCase& test_case) {
  return std::make_unique<CudnnCausalConvolution>(test_case);
}
}  // namespace flagdnn::testing
