/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/normalization_extended.hpp"

#include <flagdnn_frontend.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
namespace flagdnn::testing {
namespace {
namespace fe = flagdnn_frontend;
using Shape = std::vector<std::int64_t>;
fe::DataType_t frontend_type(flagdnnDataType_t type) {
  if (type == FLAGDNN_DATA_FLOAT32) return fe::DataType_t::FLOAT;
  if (type == FLAGDNN_DATA_FLOAT16) return fe::DataType_t::HALF;
  if (type == FLAGDNN_DATA_BFLOAT16) return fe::DataType_t::BFLOAT16;
  throw std::invalid_argument("normalization requires floating input");
}
void check(fe::error_t status) {
  if (status.is_bad()) throw std::runtime_error(status.get_message());
}
Shape strides(const Shape& shape, std::size_t) {
  Shape result(shape.size());
  std::int64_t stride = 1;
  for (std::size_t axis = shape.size(); axis > 0; --axis) {
    result[axis - 1] = stride;
    stride *= shape[axis - 1];
  }
  return result;
}
class Executable final : public TestExecutable {
 public:
  Executable(flagdnn::Handle& handle,
             const ExtendedNormalizationTestCase& test_case)
      : handle_(handle) {
    std::vector<fe::graph::Graph::Tensor> inputs;
    for (const auto& input : test_case.inputs)
      inputs.push_back(
          graph_.tensor(fe::graph::Tensor_attributes()
                            .set_uid(input.uid)
                            .set_data_type(frontend_type(input.data_type))
                            .set_dim(input.dimensions)
                            .set_stride(input.strides)));
    std::array<fe::graph::Graph::Tensor, 3> outputs;
    if (test_case.operation == "instancenorm")
      outputs = graph_.instancenorm(
          inputs[0], inputs[1], inputs[2],
          fe::graph::Instancenorm_attributes()
              .set_name(test_case.name)
              .set_compute_data_type(fe::DataType_t::FLOAT)
              .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
              .set_epsilon(static_cast<float>(test_case.epsilon)));
    else if (test_case.operation == "adalayernorm")
      outputs = graph_.adalayernorm(
          inputs[0], inputs[1], inputs[2],
          fe::graph::AdaLayernorm_attributes()
              .set_name(test_case.name)
              .set_compute_data_type(fe::DataType_t::FLOAT)
              .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
              .set_epsilon(static_cast<float>(test_case.epsilon)));
    else if (test_case.operation == "layernorm_backward")
      outputs = graph_.layernorm_backward(
          inputs[0], inputs[1], inputs[2],
          fe::graph::Layernorm_backward_attributes()
              .set_name(test_case.name)
              .set_compute_data_type(fe::DataType_t::FLOAT)
              .set_saved_mean_and_inv_variance(inputs[3], inputs[4]));
    else if (test_case.operation == "batchnorm_backward")
      outputs = graph_.batchnorm_backward(
          inputs[0], inputs[1], inputs[2],
          fe::graph::Batchnorm_backward_attributes()
              .set_name(test_case.name)
              .set_compute_data_type(fe::DataType_t::FLOAT)
              .set_saved_mean_and_inv_variance(inputs[3], inputs[4]));
    else if (test_case.operation == "instancenorm_backward")
      outputs = graph_.instancenorm_backward(
          inputs[0], inputs[1], inputs[2],
          fe::graph::Instancenorm_backward_attributes()
              .set_name(test_case.name)
              .set_compute_data_type(fe::DataType_t::FLOAT)
              .set_saved_mean_and_inv_variance(inputs[3], inputs[4]));
    else if (test_case.operation == "adalayernorm_backward")
      outputs = graph_.adalayernorm_backward(
          inputs[0], inputs[1], inputs[2],
          fe::graph::AdaLayernorm_backward_attributes()
              .set_name(test_case.name)
              .set_compute_data_type(fe::DataType_t::FLOAT)
              .set_saved_mean_and_inv_variance(inputs[3], inputs[4]));
    else if (test_case.operation == "rmsnorm_backward")
      outputs = graph_.rmsnorm_backward(
          inputs[0], inputs[1], inputs[2], inputs[3],
          fe::graph::Rmsnorm_backward_attributes()
              .set_name(test_case.name)
              .set_compute_data_type(fe::DataType_t::FLOAT));
    else
      throw std::invalid_argument("unknown normalization test operation");
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      const auto& out = test_case.outputs.at(index);
      outputs[index]
          ->set_uid(out.uid)
          .set_data_type(frontend_type(out.data_type))
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
std::vector<ExtendedNormalizationTestCase> make_extended_normalization_cases(
    const std::string& operation) {
  const std::vector<Shape> shapes = {
      {2, 4, 4, 8},     {2, 8, 8, 16},   {3, 16, 4, 32},   {4, 8, 16, 64},
      {2, 16, 32, 128}, {1, 32, 4, 256}, {2, 8, 8, 512},   {3, 4, 16, 1024},
      {1, 16, 8, 33},   {2, 32, 4, 65},  {4, 16, 16, 129}, {2, 64, 8, 257},
      {1, 8, 32, 1025}, {2, 16, 4, 4097}};
  const bool backward = operation.ends_with("_backward"),
             rms = operation == "rmsnorm_backward";
  std::vector<ExtendedNormalizationTestCase> result;
  for (std::size_t index = 0; index < shapes.size(); ++index) {
    for (auto type :
         {FLAGDNN_DATA_FLOAT32, FLAGDNN_DATA_FLOAT16, FLAGDNN_DATA_BFLOAT16}) {
      ExtendedNormalizationTestCase test_case;
      test_case.operation = operation;
      test_case.name = operation + "_" +
                       (type == FLAGDNN_DATA_FLOAT32   ? "fp32"
                        : type == FLAGDNN_DATA_FLOAT16 ? "fp16"
                                                       : "bf16");
      auto shape = shapes[index];
      // cuDNN batchnorm backward vectorizes low precision channels in groups
      // of eight. Replace the unsupported C=4 configuration for those dtypes.
      if (operation == "batchnorm_backward" && type != FLAGDNN_DATA_FLOAT32 &&
          shape[1] < 8)
        shape[1] = 8;
      for (auto dimension : shape)
        test_case.name += "_" + std::to_string(dimension);
      Shape parameter(shape.size(), 1), statistic = shape;
      if (operation.starts_with("batchnorm") ||
          operation.starts_with("instancenorm")) {
        parameter[1] = shape[1];
        for (std::size_t axis = 0; axis < shape.size(); ++axis)
          if (operation.starts_with("batchnorm") ? axis != 1 : axis >= 2)
            test_case.axes.push_back(axis);
      } else {
        parameter.back() = shape.back();
        test_case.axes.push_back(shape.size() - 1);
        if (operation.starts_with("adalayernorm")) parameter[0] = shape[0];
      }
      for (auto axis : test_case.axes) statistic[axis] = 1;
      const auto add_input = [&](const Shape& dimensions,
                                 flagdnnDataType_t dtype, std::size_t layout) {
        test_case.inputs.push_back(
            {static_cast<std::int64_t>(test_case.inputs.size() + 1), dtype,
             dimensions, strides(dimensions, layout)});
      };
      if (backward) add_input(shape, type, index + 1);  // dy
      add_input(shape, type, index);
      // cuDNN normalization backward requires affine parameter and gradient
      // dtypes to agree. Keep these in FP32 for all activation dtypes.
      add_input(parameter, FLAGDNN_DATA_FLOAT32, index + 2);
      if (!backward)
        add_input(parameter, FLAGDNN_DATA_FLOAT32, index + 3);
      else {
        if (!rms) add_input(statistic, FLAGDNN_DATA_FLOAT32, index + 4);
        add_input(statistic, FLAGDNN_DATA_FLOAT32, index + 5);
      }
      test_case.outputs = {
          {20, type, shape, strides(shape, index + 4)},
          {21, FLAGDNN_DATA_FLOAT32, backward ? parameter : statistic,
           strides(backward ? parameter : statistic, index + 5)},
          {22, FLAGDNN_DATA_FLOAT32, backward ? parameter : statistic,
           strides(backward ? parameter : statistic, index + 6)}};
      test_case.epsilon = index % 2 ? 1.0e-4 : 1.0e-5;
      result.push_back(std::move(test_case));
    }
  }
  return result;
}
std::unique_ptr<TestExecutable> build_flagdnn_extended_normalization(
    flagdnn::Handle& handle, const ExtendedNormalizationTestCase& test_case) {
  return std::make_unique<Executable>(handle, test_case);
}
}  // namespace flagdnn::testing
