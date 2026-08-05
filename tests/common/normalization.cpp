/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/normalization.hpp"

#include <flagdnn/flagdnn.hpp>
#include <flagdnn_frontend.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::testing {
namespace {

namespace fe = ::flagdnn_frontend;
using Shape = std::vector<std::int64_t>;

constexpr std::array<flagdnnDataType_t, 3> kDataTypes = {
    FLAGDNN_DATA_FLOAT32,
    FLAGDNN_DATA_FLOAT16,
    FLAGDNN_DATA_BFLOAT16,
};

Shape contiguous_strides(const Shape& dimensions) {
  Shape result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    const std::int64_t dimension = dimensions[axis - 1];
    if (dimension <= 0 ||
        stride > std::numeric_limits<std::int64_t>::max() / dimension) {
      throw std::invalid_argument("normalization tensor shape is invalid");
    }
    result[axis - 1] = stride;
    stride *= dimension;
  }
  return result;
}

Shape channels_last_strides(const Shape& dimensions) {
  if (dimensions.size() != 4) {
    throw std::invalid_argument("BatchNorm channels-last cases must be rank 4");
  }
  return {dimensions[1] * dimensions[2] * dimensions[3],
          1,
          dimensions[3] * dimensions[1],
          dimensions[1]};
}

TestTensor tensor(std::int64_t uid,
                  Shape dimensions,
                  flagdnnDataType_t data_type,
                  Shape strides = {}) {
  if (strides.empty()) {
    strides = contiguous_strides(dimensions);
  }
  return {uid,
          data_type,
          std::move(dimensions),
          std::move(strides)};
}

std::string data_type_name(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_FLOAT32:
      return "fp32";
    case FLAGDNN_DATA_FLOAT16:
      return "fp16";
    case FLAGDNN_DATA_BFLOAT16:
      return "bfloat16";
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
    case FLAGDNN_DATA_BOOLEAN:
      break;
  }
  throw std::invalid_argument("unsupported normalization data type");
}

std::string shape_name(const Shape& shape) {
  std::string result;
  for (const std::int64_t dimension : shape) {
    if (!result.empty()) {
      result += 'x';
    }
    result += std::to_string(dimension);
  }
  return result;
}

fe::DataType_t frontend_data_type(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_FLOAT32:
      return fe::DataType_t::FLOAT;
    case FLAGDNN_DATA_FLOAT16:
      return fe::DataType_t::HALF;
    case FLAGDNN_DATA_BFLOAT16:
      return fe::DataType_t::BFLOAT16;
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
    case FLAGDNN_DATA_BOOLEAN:
      break;
  }
  throw std::invalid_argument("unsupported FlagDNN normalization data type");
}

void check_frontend(fe::error_t status, std::string_view operation) {
  if (status.is_bad()) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + status.get_message());
  }
}

std::shared_ptr<fe::graph::Graph> make_graph(std::string_view name,
                                             flagdnnDataType_t data_type,
                                             bool autotune) {
  auto graph = std::make_shared<fe::graph::Graph>();
  graph->set_name(std::string(name))
      .set_io_data_type(frontend_data_type(data_type))
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT)
      .set_autotune(autotune);
  return graph;
}

std::shared_ptr<fe::graph::Tensor_attributes> make_tensor(
    const std::shared_ptr<fe::graph::Graph>& graph,
    const TestTensor& tensor_specification,
    std::string_view name) {
  return graph->tensor(
      fe::graph::Tensor_attributes()
          .set_name(std::string(name))
          .set_uid(tensor_specification.uid)
          .set_data_type(frontend_data_type(tensor_specification.data_type))
          .set_dim(tensor_specification.dimensions)
          .set_stride(tensor_specification.strides));
}

void mark_output(const std::shared_ptr<fe::graph::Tensor_attributes>& output,
                 const TestTensor& expected,
                 std::string_view name) {
  output->set_name(std::string(name))
      .set_uid(expected.uid)
      .set_data_type(frontend_data_type(expected.data_type))
      .set_dim(expected.dimensions)
      .set_stride(expected.strides)
      .set_output(true);
}

class FlagdnnNormalizationExecutable final
    : public NormalizationExecutable {
 public:
  FlagdnnNormalizationExecutable(
      flagdnn::Handle& handle,
      std::shared_ptr<fe::graph::Graph> graph,
      std::string_view operation)
      : handle_(handle), graph_(std::move(graph)) {
    check_frontend(
        graph_->build(handle_, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}),
        std::string("FlagDNN ") + std::string(operation) + " graph build");
    std::int64_t workspace_size = 0;
    check_frontend(graph_->get_workspace_size(workspace_size),
                   "FlagDNN normalization workspace query");
    if (workspace_size < 0) {
      throw std::runtime_error(
          "FlagDNN returned a negative normalization workspace size");
    }
    workspace_size_ = static_cast<std::size_t>(workspace_size);
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }

  void execute(std::span<const flagdnnBinding_t> bindings,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) override {
    if (workspace_size < workspace_size_ ||
        (workspace_size_ != 0 && workspace == nullptr)) {
      throw std::invalid_argument(
          "FlagDNN normalization workspace is too small");
    }
    check_frontend(
        graph_->execute(handle_, bindings, workspace, workspace_size, stream),
        "FlagDNN normalization graph execute");
  }

 private:
  flagdnn::Handle& handle_;
  std::shared_ptr<fe::graph::Graph> graph_;
  std::size_t workspace_size_ = 0;
};

Shape parameter_shape(const Shape& x, std::size_t normalized_rank) {
  Shape result(x.size(), 1);
  for (std::size_t axis = x.size() - normalized_rank; axis < x.size(); ++axis) {
    result[axis] = x[axis];
  }
  return result;
}

Shape statistic_shape(const Shape& x, std::size_t normalized_rank) {
  Shape result = x;
  for (std::size_t axis = x.size() - normalized_rank; axis < x.size(); ++axis) {
    result[axis] = 1;
  }
  return result;
}

void set_norm_tolerance(flagdnnDataType_t data_type,
                        double& absolute,
                        double& relative) {
  absolute = data_type == FLAGDNN_DATA_FLOAT32 ? 2.0e-4 : 2.0e-2;
  relative = absolute;
}

LayernormTestCase make_layernorm_case(const Shape& shape,
                                      std::size_t normalized_rank,
                                      flagdnnDataType_t data_type,
                                      std::int64_t uid) {
  LayernormTestCase result;
  result.name = "layernorm_" + data_type_name(data_type) + "_" +
                shape_name(shape) + "_suffix" +
                std::to_string(normalized_rank);
  const Shape parameters = parameter_shape(shape, normalized_rank);
  const Shape statistics = statistic_shape(shape, normalized_rank);
  result.x = tensor(uid, shape, data_type);
  result.scale = tensor(uid + 1, parameters, data_type);
  result.bias = tensor(uid + 2, parameters, data_type);
  result.y = tensor(uid + 3, shape, data_type);
  result.mean = tensor(uid + 4, statistics, FLAGDNN_DATA_FLOAT32);
  result.inv_variance =
      tensor(uid + 5, statistics, FLAGDNN_DATA_FLOAT32);
  set_norm_tolerance(
      data_type, result.absolute_tolerance, result.relative_tolerance);
  return result;
}

RmsnormTestCase make_rmsnorm_case(const Shape& shape,
                                  std::size_t normalized_rank,
                                  flagdnnDataType_t data_type,
                                  std::int64_t uid) {
  RmsnormTestCase result;
  result.name = "rmsnorm_" + data_type_name(data_type) + "_" +
                shape_name(shape) + "_suffix" +
                std::to_string(normalized_rank);
  const Shape parameters = parameter_shape(shape, normalized_rank);
  const Shape statistics = statistic_shape(shape, normalized_rank);
  result.x = tensor(uid, shape, data_type);
  result.scale = tensor(uid + 1, parameters, data_type);
  result.bias = tensor(uid + 2, parameters, data_type);
  result.y = tensor(uid + 3, shape, data_type);
  result.inv_variance =
      tensor(uid + 4, statistics, FLAGDNN_DATA_FLOAT32);
  set_norm_tolerance(
      data_type, result.absolute_tolerance, result.relative_tolerance);
  return result;
}

void validate_tensor(const TestTensor& specification,
                     std::string_view name,
                     bool allow_float32_statistics = true) {
  if (specification.uid <= 0 || specification.dimensions.empty() ||
      specification.dimensions.size() > 8 ||
      specification.dimensions.size() != specification.strides.size()) {
    throw std::invalid_argument(std::string(name) + " metadata is invalid");
  }
  for (std::size_t axis = 0; axis < specification.dimensions.size(); ++axis) {
    if (specification.dimensions[axis] <= 0 || specification.strides[axis] <= 0) {
      throw std::invalid_argument(
          std::string(name) + " dimensions and strides must be positive");
    }
  }
  if (specification.data_type != FLAGDNN_DATA_FLOAT32 &&
      specification.data_type != FLAGDNN_DATA_FLOAT16 &&
      specification.data_type != FLAGDNN_DATA_BFLOAT16) {
    throw std::invalid_argument(
        std::string(name) + " has an unsupported data type");
  }
  (void)allow_float32_statistics;
}

template <typename Case>
void validate_common(const Case& test_case, std::string_view operation) {
  if (test_case.name.empty() || !std::isfinite(test_case.absolute_tolerance) ||
      !std::isfinite(test_case.relative_tolerance) ||
      test_case.absolute_tolerance < 0.0 ||
      test_case.relative_tolerance < 0.0) {
    throw std::invalid_argument(
        std::string(operation) + " case metadata is invalid");
  }
  validate_tensor(test_case.x, std::string(operation) + " X");
  validate_tensor(test_case.y, std::string(operation) + " Y");
  if (test_case.x.data_type != test_case.y.data_type ||
      test_case.x.dimensions != test_case.y.dimensions) {
    throw std::invalid_argument(
        std::string(operation) + " X/Y metadata does not match");
  }
}

}  // namespace

std::vector<LayernormTestCase> make_layernorm_cases() {
  const std::array<std::pair<Shape, std::size_t>, 3> definitions = {
      std::pair<Shape, std::size_t>{{2, 5, 17}, 1},
      std::pair<Shape, std::size_t>{{2, 4, 4096}, 1},
      std::pair<Shape, std::size_t>{{2, 3, 4, 5}, 2},
  };
  std::vector<LayernormTestCase> result;
  std::int64_t uid = 71000;
  for (const auto& [shape, normalized_rank] : definitions) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(
          make_layernorm_case(shape, normalized_rank, data_type, uid));
      uid += 6;
    }
  }
  result.front().autotune = true;
  for (const LayernormTestCase& test_case : result) {
    validate_normalization_case(test_case);
  }
  return result;
}

std::vector<RmsnormTestCase> make_rmsnorm_cases() {
  const std::array<std::pair<Shape, std::size_t>, 3> definitions = {
      std::pair<Shape, std::size_t>{{2, 5, 17}, 1},
      std::pair<Shape, std::size_t>{{2, 4, 4096}, 1},
      std::pair<Shape, std::size_t>{{2, 3, 4, 5}, 2},
  };
  std::vector<RmsnormTestCase> result;
  std::int64_t uid = 72000;
  for (const auto& [shape, normalized_rank] : definitions) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(
          make_rmsnorm_case(shape, normalized_rank, data_type, uid));
      uid += 5;
    }
  }
  result.front().autotune = true;
  for (const RmsnormTestCase& test_case : result) {
    validate_normalization_case(test_case);
  }
  return result;
}

std::vector<BatchnormTestCase> make_batchnorm_cases() {
  const Shape shape = {2, 8, 8, 8};
  const Shape parameters = {1, 8, 1, 1};
  std::vector<BatchnormTestCase> result;
  std::int64_t uid = 73000;
  for (const bool channels_last : {false, true}) {
    const Shape data_strides = channels_last
                                   ? channels_last_strides(shape)
                                   : contiguous_strides(shape);
    for (const flagdnnDataType_t data_type : kDataTypes) {
      BatchnormTestCase test_case;
      test_case.name = "batchnorm_" + data_type_name(data_type) + "_" +
                       shape_name(shape) +
                       (channels_last ? "_channels_last" : "_contiguous");
      test_case.x = tensor(uid, shape, data_type, data_strides);
      test_case.scale = tensor(uid + 1, parameters, data_type);
      test_case.bias = tensor(uid + 2, parameters, data_type);
      test_case.previous_running_mean =
          tensor(uid + 3, parameters, FLAGDNN_DATA_FLOAT32);
      test_case.previous_running_variance =
          tensor(uid + 4, parameters, FLAGDNN_DATA_FLOAT32);
      test_case.y = tensor(uid + 5, shape, data_type, data_strides);
      test_case.mean = tensor(uid + 6, parameters, FLAGDNN_DATA_FLOAT32);
      test_case.inv_variance =
          tensor(uid + 7, parameters, FLAGDNN_DATA_FLOAT32);
      test_case.next_running_mean =
          tensor(uid + 8, parameters, FLAGDNN_DATA_FLOAT32);
      test_case.next_running_variance =
          tensor(uid + 9, parameters, FLAGDNN_DATA_FLOAT32);
      test_case.absolute_tolerance =
          data_type == FLAGDNN_DATA_FLOAT32
              ? 2.0e-4
              : (data_type == FLAGDNN_DATA_FLOAT16 ? 3.0e-2 : 7.0e-2);
      test_case.relative_tolerance = test_case.absolute_tolerance;
      result.push_back(std::move(test_case));
      uid += 10;
    }
  }
  result.front().autotune = true;
  for (const BatchnormTestCase& test_case : result) {
    validate_normalization_case(test_case);
  }
  return result;
}

std::vector<BatchnormInferenceTestCase>
make_batchnorm_inference_cases() {
  const std::array<Shape, 3> shapes = {
      Shape{2, 8, 16, 16},
      Shape{4, 16, 8, 8},
      Shape{2, 32, 7, 9},
  };
  std::vector<BatchnormInferenceTestCase> result;
  std::int64_t uid = 74000;
  for (std::size_t shape_index = 0; shape_index < shapes.size(); ++shape_index) {
    const auto append_layout = [&](bool channels_last) {
      const Shape& shape = shapes[shape_index];
      const Shape parameters = {1, shape[1], 1, 1};
      const Shape data_strides = channels_last
                                     ? channels_last_strides(shape)
                                     : contiguous_strides(shape);
      for (const flagdnnDataType_t data_type : kDataTypes) {
        BatchnormInferenceTestCase test_case;
        test_case.name = "batchnorm_inference_" +
                         data_type_name(data_type) + "_" + shape_name(shape) +
                         (channels_last ? "_channels_last" : "_contiguous");
        test_case.x = tensor(uid, shape, data_type, data_strides);
        test_case.mean =
            tensor(uid + 1, parameters, FLAGDNN_DATA_FLOAT32);
        test_case.inv_variance =
            tensor(uid + 2, parameters, FLAGDNN_DATA_FLOAT32);
        test_case.scale =
            tensor(uid + 3, parameters, FLAGDNN_DATA_FLOAT32);
        test_case.bias =
            tensor(uid + 4, parameters, FLAGDNN_DATA_FLOAT32);
        test_case.y = tensor(uid + 5, shape, data_type, data_strides);
        test_case.absolute_tolerance =
            data_type == FLAGDNN_DATA_FLOAT32
                ? 1.0e-5
                : (data_type == FLAGDNN_DATA_FLOAT16 ? 2.0e-2 : 5.0e-2);
        test_case.relative_tolerance = test_case.absolute_tolerance;
        result.push_back(std::move(test_case));
        uid += 6;
      }
    };
    append_layout(false);
    if (shape_index == 0) {
      append_layout(true);
    }
  }
  result.front().autotune = true;
  for (const BatchnormInferenceTestCase& test_case : result) {
    validate_normalization_case(test_case);
  }
  return result;
}

void validate_normalization_case(const LayernormTestCase& test_case) {
  validate_common(test_case, "LayerNorm");
  validate_tensor(test_case.scale, "LayerNorm scale");
  validate_tensor(test_case.bias, "LayerNorm bias");
  validate_tensor(test_case.mean, "LayerNorm mean");
  validate_tensor(test_case.inv_variance, "LayerNorm inverse variance");
  if (test_case.scale.data_type != test_case.x.data_type ||
      test_case.bias.data_type != test_case.x.data_type ||
      test_case.scale.dimensions != test_case.bias.dimensions ||
      test_case.mean.data_type != FLAGDNN_DATA_FLOAT32 ||
      test_case.inv_variance.data_type != FLAGDNN_DATA_FLOAT32 ||
      test_case.mean.dimensions != test_case.inv_variance.dimensions ||
      !std::isfinite(test_case.epsilon) || test_case.epsilon <= 0.0) {
    throw std::invalid_argument("LayerNorm case semantics are invalid");
  }
}

void validate_normalization_case(const RmsnormTestCase& test_case) {
  validate_common(test_case, "RMSNorm");
  validate_tensor(test_case.scale, "RMSNorm scale");
  validate_tensor(test_case.bias, "RMSNorm bias");
  validate_tensor(test_case.inv_variance, "RMSNorm inverse variance");
  if (test_case.scale.data_type != test_case.x.data_type ||
      test_case.bias.data_type != test_case.x.data_type ||
      test_case.scale.dimensions != test_case.bias.dimensions ||
      test_case.inv_variance.data_type != FLAGDNN_DATA_FLOAT32 ||
      !std::isfinite(test_case.epsilon) || test_case.epsilon <= 0.0) {
    throw std::invalid_argument("RMSNorm case semantics are invalid");
  }
}

void validate_normalization_case(const BatchnormTestCase& test_case) {
  validate_common(test_case, "BatchNorm");
  const std::array<const TestTensor*, 8> parameters = {
      &test_case.scale,
      &test_case.bias,
      &test_case.previous_running_mean,
      &test_case.previous_running_variance,
      &test_case.mean,
      &test_case.inv_variance,
      &test_case.next_running_mean,
      &test_case.next_running_variance,
  };
  for (const TestTensor* parameter : parameters) {
    validate_tensor(*parameter, "BatchNorm parameter/statistic");
  }
  if (test_case.scale.data_type != test_case.x.data_type ||
      test_case.bias.data_type != test_case.x.data_type ||
      !std::isfinite(test_case.epsilon) || test_case.epsilon <= 0.0 ||
      !std::isfinite(test_case.momentum) || test_case.momentum < 0.0 ||
      test_case.momentum > 1.0) {
    throw std::invalid_argument("BatchNorm case semantics are invalid");
  }
}

void validate_normalization_case(
    const BatchnormInferenceTestCase& test_case) {
  validate_common(test_case, "BatchNorm Inference");
  for (const TestTensor* parameter :
       {&test_case.mean,
        &test_case.inv_variance,
        &test_case.scale,
        &test_case.bias}) {
    validate_tensor(*parameter, "BatchNorm Inference parameter");
    if (parameter->data_type != FLAGDNN_DATA_FLOAT32) {
      throw std::invalid_argument(
          "BatchNorm Inference parameters must use FP32");
    }
  }
}

std::unique_ptr<NormalizationExecutable> build_flagdnn_layernorm(
    flagdnn::Handle& handle,
    const LayernormTestCase& test_case) {
  validate_normalization_case(test_case);
  auto graph = make_graph(
      test_case.name, test_case.x.data_type, test_case.autotune);
  const auto x = make_tensor(graph, test_case.x, "x");
  const auto scale = make_tensor(graph, test_case.scale, "scale");
  const auto bias = make_tensor(graph, test_case.bias, "bias");
  auto outputs = graph->layernorm(
      x,
      scale,
      bias,
      fe::graph::Layernorm_attributes()
          .set_name("layernorm")
          .set_compute_data_type(fe::DataType_t::FLOAT)
          .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
          .set_epsilon(static_cast<float>(test_case.epsilon)));
  mark_output(outputs[0], test_case.y, "y");
  mark_output(outputs[1], test_case.mean, "mean");
  mark_output(outputs[2], test_case.inv_variance, "inv_variance");
  return std::make_unique<FlagdnnNormalizationExecutable>(
      handle, std::move(graph), "LayerNorm");
}

std::unique_ptr<NormalizationExecutable> build_flagdnn_rmsnorm(
    flagdnn::Handle& handle,
    const RmsnormTestCase& test_case) {
  validate_normalization_case(test_case);
  auto graph = make_graph(
      test_case.name, test_case.x.data_type, test_case.autotune);
  const auto x = make_tensor(graph, test_case.x, "x");
  const auto scale = make_tensor(graph, test_case.scale, "scale");
  auto bias = make_tensor(graph, test_case.bias, "bias");
  auto epsilon = graph->tensor(
      static_cast<float>(test_case.epsilon),
      fe::graph::ScalarType::COMPILE_TIME_CONST);
  auto attributes = fe::graph::Rmsnorm_attributes()
                        .set_name("rmsnorm")
                        .set_compute_data_type(fe::DataType_t::FLOAT)
                        .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
                        .set_bias(bias)
                        .set_epsilon(epsilon);
  auto outputs = graph->rmsnorm(x, scale, std::move(attributes));
  mark_output(outputs[0], test_case.y, "y");
  mark_output(outputs[1], test_case.inv_variance, "inv_variance");
  return std::make_unique<FlagdnnNormalizationExecutable>(
      handle, std::move(graph), "RMSNorm");
}

std::unique_ptr<NormalizationExecutable> build_flagdnn_batchnorm(
    flagdnn::Handle& handle,
    const BatchnormTestCase& test_case) {
  validate_normalization_case(test_case);
  auto graph = make_graph(
      test_case.name, test_case.x.data_type, test_case.autotune);
  const auto x = make_tensor(graph, test_case.x, "x");
  const auto scale = make_tensor(graph, test_case.scale, "scale");
  const auto bias = make_tensor(graph, test_case.bias, "bias");
  auto previous_mean = make_tensor(
      graph, test_case.previous_running_mean, "previous_running_mean");
  auto previous_variance = make_tensor(
      graph,
      test_case.previous_running_variance,
      "previous_running_variance");
  auto epsilon = graph->tensor(
      static_cast<float>(test_case.epsilon),
      fe::graph::ScalarType::COMPILE_TIME_CONST);
  auto momentum = graph->tensor(
      static_cast<float>(test_case.momentum),
      fe::graph::ScalarType::COMPILE_TIME_CONST);
  auto attributes = fe::graph::Batchnorm_attributes()
                        .set_name("batchnorm")
                        .set_compute_data_type(fe::DataType_t::FLOAT)
                        .set_previous_running_stats(
                            previous_mean, previous_variance, momentum)
                        .set_epsilon(epsilon);
  auto outputs = graph->batchnorm(x, scale, bias, std::move(attributes));
  mark_output(outputs[0], test_case.y, "y");
  mark_output(outputs[1], test_case.mean, "mean");
  mark_output(outputs[2], test_case.inv_variance, "inv_variance");
  mark_output(outputs[3], test_case.next_running_mean, "next_running_mean");
  mark_output(outputs[4],
              test_case.next_running_variance,
              "next_running_variance");
  return std::make_unique<FlagdnnNormalizationExecutable>(
      handle, std::move(graph), "BatchNorm");
}

std::unique_ptr<NormalizationExecutable>
build_flagdnn_batchnorm_inference(
    flagdnn::Handle& handle,
    const BatchnormInferenceTestCase& test_case) {
  validate_normalization_case(test_case);
  auto graph = make_graph(
      test_case.name, test_case.x.data_type, test_case.autotune);
  const auto x = make_tensor(graph, test_case.x, "x");
  const auto mean = make_tensor(graph, test_case.mean, "mean");
  const auto inv_variance =
      make_tensor(graph, test_case.inv_variance, "inv_variance");
  const auto scale = make_tensor(graph, test_case.scale, "scale");
  const auto bias = make_tensor(graph, test_case.bias, "bias");
  auto output = graph->batchnorm_inference(
      x,
      mean,
      inv_variance,
      scale,
      bias,
      fe::graph::Batchnorm_inference_attributes()
          .set_name("batchnorm_inference")
          .set_compute_data_type(fe::DataType_t::FLOAT));
  mark_output(output, test_case.y, "y");
  return std::make_unique<FlagdnnNormalizationExecutable>(
      handle, std::move(graph), "BatchNorm Inference");
}

}  // namespace flagdnn::testing
