/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/position_embedding.hpp"

#include <flagdnn_frontend.h>

#include <array>
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
  throw std::invalid_argument("RoPE requires floating input");
}
void check(fe::error_t status) {
  if (status.is_bad()) throw std::runtime_error(status.get_message());
}
class Executable final : public TestExecutable {
 public:
  Executable(flagdnn::Handle& handle, const RoPETestCase& test_case)
      : handle_(handle) {
    std::vector<fe::graph::Graph::Tensor> inputs;
    for (const auto& input : test_case.inputs)
      inputs.push_back(
          graph_.tensor(fe::graph::Tensor_attributes()
                            .set_uid(input.uid)
                            .set_data_type(frontend_type(input.data_type))
                            .set_dim(input.dimensions)
                            .set_stride(input.strides)));
    const auto output =
        test_case.operation == "rope"
            ? graph_.rope(inputs[0], inputs[1],
                          fe::graph::RoPE_attributes()
                              .set_name(test_case.name)
                              .set_rope_dim(test_case.rope_dim)
                              .set_output_scale(test_case.output_scale))
            : graph_.rope_backward(
                  inputs[0], inputs[1],
                  fe::graph::RoPE_backward_attributes()
                      .set_name(test_case.name)
                      .set_rope_dim(test_case.rope_dim)
                      .set_output_scale(test_case.output_scale));
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
std::vector<RoPETestCase> make_rope_cases(bool backward) {
  const std::vector<std::vector<std::int64_t>> shapes = {
      {1, 1, 3, 2},    {2, 3, 7, 8},    {1, 4, 17, 16},  {2, 3, 31, 32},
      {3, 2, 33, 64},  {1, 8, 65, 80},  {2, 4, 127, 96}, {1, 3, 129, 128},
      {2, 8, 17, 192}, {4, 2, 33, 256}, {2, 4, 257, 64}, {1, 16, 513, 128},
      {1, 2, 7, 34},   {2, 3, 5, 130}};
  std::vector<RoPETestCase> result;
  for (std::size_t index = 0; index < shapes.size(); ++index) {
    const auto& shape = shapes[index];
    const auto b = shape[0], h = shape[1], s = shape[2], d = shape[3];
    for (auto type :
         {FLAGDNN_DATA_FLOAT32, FLAGDNN_DATA_FLOAT16, FLAGDNN_DATA_BFLOAT16}) {
      RoPETestCase test_case;
      test_case.operation = backward ? "rope_backward" : "rope";
      test_case.name = test_case.operation + "_" +
                       (type == FLAGDNN_DATA_FLOAT32   ? "fp32"
                        : type == FLAGDNN_DATA_FLOAT16 ? "fp16"
                                                       : "bf16");
      for (auto dim : shape) test_case.name += "_" + std::to_string(dim);
      test_case.rope_dim = index % 2 ? d / 4 * 2 : 0;
      const auto width = test_case.rope_dim ? test_case.rope_dim : d;
      test_case.output_scale = index % 3 == 0   ? 1.0F
                               : index % 3 == 1 ? 0.125F
                                                : -0.75F;
      const std::vector<std::int64_t> x_stride =
          index % 2
              ? std::vector<std::int64_t>{s * h * d, d, h * d, 1}
              : std::vector<std::int64_t>{2 * h * s * d, 2 * s * d, 2 * d, 2};
      const std::vector<std::int64_t> output_stride =
          index % 2 ? std::vector<std::int64_t>{h * s * (d + 3), s * (d + 3),
                                                d + 3, 1}
                    : std::vector<std::int64_t>{h * d, d, b * h * d,
                                                1};  // SBHD physical layout.
      const std::vector<std::int64_t> frequency_shape = {s, 1, 1, width},
                                      frequency_stride = {2 * width, 2 * width,
                                                          2 * width, 2};
      test_case.inputs = {
          {1, type, shape, x_stride},
          {2, FLAGDNN_DATA_FLOAT32, frequency_shape, frequency_stride}};
      test_case.outputs = {{3, type, shape, output_stride}};
      result.push_back(std::move(test_case));
    }
  }
  return result;
}
std::vector<std::vector<float>> rope_inputs(const RoPETestCase& test_case) {
  std::vector<std::vector<float>> result;
  for (std::size_t input = 0; input < test_case.inputs.size(); ++input) {
    const auto& shape = test_case.inputs[input].dimensions;
    const auto count = std::accumulate(shape.begin(), shape.end(),
                                       std::size_t{1}, std::multiplies<>());
    std::vector<float> values(count);
    for (std::size_t index = 0; index < count; ++index) {
      if (input == 0)
        values[index] =
            static_cast<float>(static_cast<int>((index * 17) % 61) - 30) /
            19.0F;
      else {
        const auto d = static_cast<std::size_t>(shape[3]), position = index / d,
                   frequency = index % (d / 2);
        values[index] = static_cast<float>(
            position * std::pow(10000.0, -2.0 * frequency / d));
      }
    }
    result.push_back(std::move(values));
  }
  return result;
}
std::unique_ptr<TestExecutable> build_flagdnn_rope(
    flagdnn::Handle& handle, const RoPETestCase& test_case) {
  return std::make_unique<Executable>(handle, test_case);
}
}  // namespace flagdnn::testing
