/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/causal_convolution.hpp"

#include <flagdnn_frontend.h>

#include <array>
#include <bit>
#include <cmath>
#include <numeric>
#include <stdexcept>
namespace flagdnn::testing {
namespace {
namespace fe = flagdnn_frontend;
fe::DataType_t frontend_type(flagdnnDataType_t type) {
  if (type == FLAGDNN_DATA_FLOAT32) return fe::DataType_t::FLOAT;
  if (type == FLAGDNN_DATA_FLOAT16) return fe::DataType_t::HALF;
  if (type == FLAGDNN_DATA_BFLOAT16) return fe::DataType_t::BFLOAT16;
  throw std::invalid_argument("causal_conv1d requires floating input");
}
void check(fe::error_t status) {
  if (status.is_bad()) throw std::runtime_error(status.get_message());
}
class Executable final : public TestExecutable {
 public:
  Executable(flagdnn::Handle& handle,
             const CausalConvolutionTestCase& test_case)
      : handle_(handle) {
    std::vector<fe::graph::Graph::Tensor> inputs;
    for (const auto& input : test_case.inputs)
      inputs.push_back(
          graph_.tensor(fe::graph::Tensor_attributes()
                            .set_uid(input.uid)
                            .set_data_type(frontend_type(input.data_type))
                            .set_dim(input.dimensions)
                            .set_stride(input.strides)));
    auto attributes =
        fe::graph::Causal_conv1d_attributes()
            .set_name(test_case.name)
            .set_input_precision(
                static_cast<fe::InputPrecision_t>(test_case.precision))
            .set_dilation(test_case.dilation)
            .set_activation(test_case.silu ? fe::PointwiseMode_t::SWISH_FWD
                                           : fe::PointwiseMode_t::IDENTITY);
    if (inputs.size() == 3) attributes.set_bias(inputs[2]);
    const auto output = graph_.causal_conv1d(inputs[0], inputs[1], attributes);
    const auto& out = test_case.outputs[0];
    output->set_uid(out.uid)
        .set_data_type(frontend_type(out.data_type))
        .set_dim(out.dimensions)
        .set_stride(out.strides)
        .set_output(true);
    check(graph_.build(handle_, {fe::HeurMode_t::A}));
    std::int64_t size = 0;
    check(graph_.get_workspace_size(size));
    workspace_size_ = static_cast<std::size_t>(size);
  }
  std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }
  void execute(std::span<const flagdnnBinding_t> bindings, void* workspace,
               std::size_t size, flagdnnStream_t stream) override {
    check(graph_.execute(handle_, bindings, workspace, size, stream));
  }

 private:
  flagdnn::Handle& handle_;
  fe::graph::Graph graph_;
  std::size_t workspace_size_ = 0;
};
}  // namespace
std::vector<CausalConvolutionTestCase> make_causal_convolution_cases() {
  const std::vector<std::array<std::int64_t, 4>> shapes = {
      {1, 4, 8, 2},      {2, 8, 17, 3},    {1, 16, 32, 4},   {2, 16, 65, 7},
      {3, 8, 127, 8},    {1, 32, 129, 16}, {2, 32, 257, 32}, {1, 64, 513, 64},
      {2, 8, 1025, 3},   {4, 16, 33, 4},   {2, 64, 2048, 7}, {1, 128, 4096, 16},
      {2, 16, 257, 128}, {1, 32, 513, 256}};
  std::vector<CausalConvolutionTestCase> result;
  for (std::size_t index = 0; index < shapes.size(); ++index) {
    const auto [batch, channels, length, width] = shapes[index];
    for (auto type :
         {FLAGDNN_DATA_FLOAT32, FLAGDNN_DATA_FLOAT16, FLAGDNN_DATA_BFLOAT16}) {
      CausalConvolutionTestCase test_case;
      test_case.name = std::string("causal_conv1d_") +
                       (type == FLAGDNN_DATA_FLOAT32   ? "fp32"
                        : type == FLAGDNN_DATA_FLOAT16 ? "fp16"
                                                       : "bf16");
      for (auto dim : shapes[index])
        test_case.name += "_" + std::to_string(dim);
      test_case.precision = type == FLAGDNN_DATA_FLOAT32 ? 1 : 0;
      test_case.dilation = 1;
      test_case.silu = index % 2;
      test_case.inputs = {
          {1, type, {batch, channels, length}, {channels * length, length, 1}},
          {2, type, {channels, width}, {width, 1}},
          {3, type, {channels}, {1}}};
      test_case.outputs = {
          {4, type, {batch, channels, length}, {channels * length, length, 1}}};
      result.push_back(std::move(test_case));
    }
  }
  return result;
}
std::vector<std::vector<float>> causal_convolution_inputs(
    const CausalConvolutionTestCase& test_case) {
  std::vector<std::vector<float>> result;
  for (std::size_t input = 0; input < test_case.inputs.size(); ++input) {
    const auto& shape = test_case.inputs[input].dimensions;
    const auto count = std::accumulate(shape.begin(), shape.end(),
                                       std::size_t{1}, std::multiplies<>());
    std::vector<float> values(count);
    for (std::size_t index = 0; index < count; ++index)
      values[index] =
          static_cast<float>(static_cast<int>((index * 17 + input * 7) % 61) -
                             30) /
          19.0F;
    result.push_back(std::move(values));
  }
  return result;
}
std::unique_ptr<TestExecutable> build_flagdnn_causal_convolution(
    flagdnn::Handle& handle, const CausalConvolutionTestCase& test_case) {
  return std::make_unique<Executable>(handle, test_case);
}
}  // namespace flagdnn::testing
