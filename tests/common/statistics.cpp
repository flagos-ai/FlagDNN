/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/statistics.hpp"

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
  throw std::invalid_argument("statistics require floating input");
}
void check(fe::error_t status) {
  if (status.is_bad()) throw std::runtime_error(status.get_message());
}
class Executable final : public TestExecutable {
 public:
  Executable(flagdnn::Handle& handle, const StatisticsTestCase& test_case)
      : handle_(handle) {
    std::vector<fe::graph::Graph::Tensor> inputs, outputs;
    for (const auto& input : test_case.inputs)
      inputs.push_back(
          graph_.tensor(fe::graph::Tensor_attributes()
                            .set_uid(input.uid)
                            .set_data_type(frontend_type(input.data_type))
                            .set_dim(input.dimensions)
                            .set_stride(input.strides)));
    if (test_case.operation == "genstats") {
      const auto result = graph_.genstats(
          inputs[0], fe::graph::Genstats_attributes().set_name(test_case.name));
      outputs.assign(result.begin(), result.end());
    } else if (test_case.operation == "bn_finalize") {
      auto attributes = fe::graph::BN_finalize_attributes()
                            .set_name(test_case.name)
                            .set_compute_data_type(fe::DataType_t::FLOAT);
      if (inputs.size() == 6)
        attributes.set_previous_running_stats(
            inputs[4], inputs[5],
            graph_.tensor(static_cast<float>(test_case.momentum),
                          fe::graph::ScalarType::COMPILE_TIME_CONST));
      const auto result = graph_.bn_finalize(
          inputs[0], inputs[1], inputs[2], inputs[3],
          graph_.tensor(static_cast<float>(test_case.epsilon),
                        fe::graph::ScalarType::COMPILE_TIME_CONST),
          graph_.tensor(static_cast<float>(test_case.accum_count),
                        fe::graph::ScalarType::COMPILE_TIME_CONST),
          attributes);
      for (const auto& output : result)
        if (output) outputs.push_back(output);
    } else
      throw std::invalid_argument("unknown statistics operation");
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      const auto& out = test_case.outputs.at(index);
      outputs[index]
          ->set_uid(out.uid)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim(out.dimensions)
          .set_stride(out.strides)
          .set_output(true);
    }
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
std::vector<StatisticsTestCase> make_genstats_cases() {
  const std::vector<std::vector<std::int64_t>> shapes = {
      {1, 3},          {3, 7},
      {2, 8, 17},      {4, 3, 33},
      {2, 7, 65},      {3, 8, 127},
      {2, 16, 257},    {2, 3, 5, 7},
      {1, 8, 17, 19},  {3, 16, 7, 9},
      {2, 32, 16, 17}, {4, 8, 31, 33},
      {2, 3, 5, 7, 9}, {1, 2, 3, 2, 3, 2, 3, 2}};
  std::vector<StatisticsTestCase> result;
  for (std::size_t index = 0; index < shapes.size(); ++index) {
    for (auto type :
         {FLAGDNN_DATA_FLOAT32, FLAGDNN_DATA_FLOAT16, FLAGDNN_DATA_BFLOAT16}) {
      StatisticsTestCase test_case;
      test_case.name =
          std::string("genstats_") + (type == FLAGDNN_DATA_FLOAT32   ? "fp32"
                                      : type == FLAGDNN_DATA_FLOAT16 ? "fp16"
                                                                     : "bf16");
      const auto& shape = shapes[index];
      for (auto dimension : shape)
        test_case.name += "_" + std::to_string(dimension);
      std::vector<std::int64_t> strides(shape.size());
      std::int64_t stride = index % 3 == 0 ? 2 : 1;
      if (index % 2) {
        strides[1] = stride;
        stride *= shape[1];
        for (std::size_t axis = shape.size(); axis > 2; --axis) {
          strides[axis - 1] = stride;
          stride *= shape[axis - 1];
        }
        strides[0] = stride;
      } else {
        for (std::size_t axis = shape.size(); axis != 0; --axis) {
          strides[axis - 1] = stride;
          stride *= shape[axis - 1];
        }
      }
      test_case.inputs.push_back({1, type, shape, strides});
      std::vector<std::int64_t> output_shape(shape.size(), 1),
          output_stride(shape.size(), 1);
      output_shape[1] = shape[1];
      output_stride[1] = index % 2 + 1;
      test_case.outputs = {
          {2, FLAGDNN_DATA_FLOAT32, output_shape, output_stride},
          {3, FLAGDNN_DATA_FLOAT32, output_shape, output_stride}};
      result.push_back(std::move(test_case));
    }
  }
  return result;
}
std::vector<StatisticsTestCase> make_bn_finalize_cases() {
  std::vector<StatisticsTestCase> result;
  const std::array<std::int64_t, 14> channels = {
      1, 2, 3, 7, 16, 31, 33, 65, 127, 257, 511, 1024, 2049, 4096};
  for (std::size_t index = 0; index < channels.size(); ++index) {
    StatisticsTestCase test_case;
    test_case.operation = "bn_finalize";
    test_case.name = "bn_finalize_fp32_" + std::to_string(channels[index]);
    const std::vector<std::int64_t> shape{1, channels[index], 1, 1},
        strides{channels[index], 1, 1, 1};
    for (std::int64_t input = 0; input < 6; ++input)
      test_case.inputs.push_back(
          {input + 1, FLAGDNN_DATA_FLOAT32, shape, strides});
    for (std::int64_t output = 0; output < 6; ++output)
      test_case.outputs.push_back(
          {output + 20, FLAGDNN_DATA_FLOAT32, shape, strides});
    test_case.accum_count = index % 3 == 0 ? 2.0 : 17.0 + index;
    test_case.momentum = index % 3 == 0 ? 0.0 : index % 3 == 1 ? 1.0 : 0.125;
    result.push_back(std::move(test_case));
  }
  return result;
}
float statistics_input_value(const StatisticsTestCase& test_case,
                             std::size_t input, std::size_t index) {
  const float value =
      static_cast<float>(static_cast<int>((index * 17) % 61) - 30) / 19.0F;
  if (test_case.operation != "bn_finalize") return value;
  // Binary fractions keep the zero-variance inputs exactly representable
  // through the FP32 sum/squared-sum interface. Zero variance uses a
  // power-of-two count, so the cuDNN FP32 reciprocal introduces no error;
  // non-power-of-two counts still cover nonzero moments.
  const double mean = static_cast<double>(static_cast<int>((index * 17) % 61) -
                                          30) /
                      64.0,
               variance = index % 5 == 0 && test_case.accum_count == 2.0
                              ? 0.0
                              : 0.25 + static_cast<double>(index % 13) / 8.0;
  if (input == 0) return static_cast<float>(mean * test_case.accum_count);
  if (input == 1)
    return static_cast<float>((mean * mean + variance) * test_case.accum_count);
  if (input == 2) return value;
  if (input == 3 || input == 4) return value * 0.125F;
  return 1.0F + std::abs(value);
}
std::unique_ptr<TestExecutable> build_flagdnn_statistics(
    flagdnn::Handle& handle, const StatisticsTestCase& test_case) {
  return std::make_unique<Executable>(handle, test_case);
}
}  // namespace flagdnn::testing
