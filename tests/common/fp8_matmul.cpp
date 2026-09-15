/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/fp8_matmul.hpp"

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
  if (type == FLAGDNN_DATA_FP8_E4M3) return fe::DataType_t::FP8_E4M3;
  if (type == FLAGDNN_DATA_FP8_E5M2) return fe::DataType_t::FP8_E5M2;
  if (type == FLAGDNN_DATA_FP8_E8M0) return fe::DataType_t::FP8_E8M0;
  throw std::invalid_argument("invalid FP8 matmul test tensor type");
}
void check(fe::error_t status) {
  if (status.is_bad()) throw std::runtime_error(status.get_message());
}
class Executable final : public TestExecutable {
 public:
  Executable(flagdnn::Handle& handle, const Fp8MatmulTestCase& test_case)
      : handle_(handle) {
    std::vector<fe::graph::Graph::Tensor> inputs;
    for (const auto& input : test_case.inputs)
      inputs.push_back(
          graph_.tensor(fe::graph::Tensor_attributes()
                            .set_uid(input.uid)
                            .set_data_type(frontend_type(input.data_type))
                            .set_dim(input.dimensions)
                            .set_stride(input.strides)));
    fe::graph::Graph::Tensor output;
    if (test_case.plain_matmul)
      output = graph_.matmul(
          inputs[0], inputs[1],
          fe::graph::Matmul_attributes().set_name(test_case.name));
    else {
      auto attributes = fe::graph::Matmul_fp8_attributes()
                            .set_name(test_case.name)
                            .set_mxfp8(test_case.scale_mode == 2);
      if (test_case.scale_mode)
        attributes.set_descale_a(inputs[2]).set_descale_b(inputs[3]);
      output = graph_.matmul_fp8(inputs[0], inputs[1], attributes);
    }
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
using Shape = std::vector<std::int64_t>;
std::int64_t elements(const Shape& shape) {
  return std::accumulate(shape.begin(), shape.end(), std::int64_t{1},
                         std::multiplies<>());
}
TestTensor tensor(std::int64_t uid, flagdnnDataType_t type, Shape shape,
                  int layout) {
  Shape strides(shape.size());
  std::vector<std::size_t> order(shape.size());
  std::iota(order.begin(), order.end(), 0);
  if (layout % 2 && shape.size() >= 2)
    std::swap(order[shape.size() - 1], order[shape.size() - 2]);
  std::int64_t stride = 1;
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    strides[*it] = stride;
    stride *= shape[*it];
  }
  return {uid, type, std::move(shape), std::move(strides), 16};
}
}  // namespace
std::vector<Fp8MatmulTestCase> make_fp8_matmul_cases() {
  const std::vector<std::array<std::int64_t, 3>> shapes = {
      {16, 16, 32},    {32, 32, 64},   {16, 64, 64},   {64, 16, 64},
      {32, 64, 128},   {64, 32, 128},  {64, 64, 64},   {128, 32, 64},
      {32, 128, 64},   {64, 128, 128}, {128, 64, 128}, {128, 128, 256},
      {256, 128, 256}, {128, 256, 512}};  // M,N,K
  std::vector<Fp8MatmulTestCase> result;
  for (std::size_t index = 0; index < shapes.size(); ++index) {
    const auto [m, n, k] = shapes[index];
    for (int format = 0; format < 4; ++format) {
      for (int mode : {0, 1, 2, 3}) {
        Fp8MatmulTestCase test_case;
        test_case.scale_mode = mode == 3 ? 0 : mode;
        test_case.plain_matmul = mode == 3;
        test_case.name = std::string(mode == 3   ? "matmul"
                                     : mode == 2 ? "matmul_mxfp8"
                                                 : "matmul_fp8") +
                         "_a" + (format & 1 ? "e5m2" : "e4m3") + "_b" +
                         (format & 2 ? "e5m2" : "e4m3") + "_mode" +
                         std::to_string(mode);
        for (auto dim : shapes[index])
          test_case.name += "_" + std::to_string(dim);
        const std::int64_t batch = index % 3 == 0 ? 2 : 1;
        Shape a{batch}, b{index % 2 ? 1 : batch}, c{batch};
        a.insert(a.end(), {m, k});
        b.insert(b.end(), {k, n});
        c.insert(c.end(), {m, n});
        test_case.inputs = {
            tensor(1,
                   format & 1 ? FLAGDNN_DATA_FP8_E5M2 : FLAGDNN_DATA_FP8_E4M3,
                   a, 0),
            tensor(2,
                   format & 2 ? FLAGDNN_DATA_FP8_E5M2 : FLAGDNN_DATA_FP8_E4M3,
                   b, index % 2)};
        if (mode == 1) {
          test_case.inputs.push_back(tensor(3, FLAGDNN_DATA_FLOAT32, {1}, 0));
          test_case.inputs.push_back(tensor(4, FLAGDNN_DATA_FLOAT32, {1}, 0));
        } else if (mode == 2) {
          a.back() = (k + 31) / 32;
          b[b.size() - 2] = (k + 31) / 32;
          test_case.inputs.push_back(
              tensor(3, FLAGDNN_DATA_FP8_E8M0, a, index + 1));
          test_case.inputs.push_back(
              tensor(4, FLAGDNN_DATA_FP8_E8M0, b, index + 2));
        }
        test_case.outputs = {tensor(5, FLAGDNN_DATA_FLOAT32, c, 0)};
        result.push_back(test_case);
        if (index == 0 || index == 13) {
          for (const auto output_type :
               {FLAGDNN_DATA_FLOAT16, FLAGDNN_DATA_BFLOAT16}) {
            auto converted = test_case;
            converted.name +=
                output_type == FLAGDNN_DATA_FLOAT16 ? "_out_fp16" : "_out_bf16";
            converted.outputs[0].data_type = output_type;
            result.push_back(std::move(converted));
          }
        }
      }
    }
  }
  return result;
}
std::vector<std::vector<float>> fp8_matmul_inputs(
    const Fp8MatmulTestCase& test_case) {
  std::vector<std::vector<float>> result;
  for (std::size_t input = 0; input < test_case.inputs.size(); ++input) {
    std::vector<float> values(elements(test_case.inputs[input].dimensions));
    for (std::size_t index = 0; index < values.size(); ++index)
      values[index] =
          input < 2
              ? static_cast<float>(
                    static_cast<int>((index * 17 + input * 7) % 61) - 30) /
                    19.0F
          : test_case.scale_mode == 1
              ? (input == 2 ? 0.75F : 1.25F)
              : std::ldexp(1.0F, static_cast<int>((index + input) % 7) - 3);
    if (test_case.scale_variant) {
      float value = 0;
      if (input < 2) {
        value = test_case.scale_variant == 1
                    ? 2.0F
                    : std::ldexp(1.0F, test_case.inputs[input].data_type ==
                                               FLAGDNN_DATA_FP8_E5M2
                                           ? -16
                                           : -9);
      } else {
        const int exponent =
            test_case.scale_variant == 3
                ? 70
                : (input == 2 ? 1 : -1) *
                      (test_case.scale_variant == 1 ? 127 : -127);
        value = std::ldexp(1.0F, exponent);
      }
      std::fill(values.begin(), values.end(), value);
    }
    result.push_back(std::move(values));
  }
  return result;
}
std::unique_ptr<TestExecutable> build_flagdnn_fp8_matmul(
    flagdnn::Handle& handle, const Fp8MatmulTestCase& test_case) {
  return std::make_unique<Executable>(handle, test_case);
}
}  // namespace flagdnn::testing
