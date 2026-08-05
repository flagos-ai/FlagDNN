/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/composite.hpp"

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

struct ConvBiasReluDefinition {
  Shape x;
  Shape w;
  Shape stride;
  Shape padding;
  Shape dilation;
};

Shape contiguous_strides(const Shape& dimensions) {
  Shape result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    const std::int64_t dimension = dimensions[axis - 1];
    if (dimension <= 0 ||
        stride > std::numeric_limits<std::int64_t>::max() / dimension) {
      throw std::invalid_argument("composite tensor shape is invalid");
    }
    result[axis - 1] = stride;
    stride *= dimension;
  }
  return result;
}

Shape channels_last_strides(const Shape& dimensions) {
  if (dimensions.size() != 4) {
    throw std::invalid_argument("composite channels-last tensor must be rank 4");
  }
  return {dimensions[1] * dimensions[2] * dimensions[3],
          1,
          dimensions[3] * dimensions[1],
          dimensions[1]};
}

Shape pointwise_strides(const Shape& dimensions) {
  return dimensions.size() == 4 ? channels_last_strides(dimensions)
                                : contiguous_strides(dimensions);
}

TestTensor tensor(std::int64_t uid,
                  Shape dimensions,
                  flagdnnDataType_t data_type,
                  bool channels_last) {
  Shape strides = channels_last ? channels_last_strides(dimensions)
                                : contiguous_strides(dimensions);
  return {uid, data_type, std::move(dimensions), std::move(strides)};
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
  throw std::invalid_argument("unsupported composite data type");
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

std::int64_t output_dimension(std::int64_t input,
                              std::int64_t filter,
                              std::int64_t padding,
                              std::int64_t stride,
                              std::int64_t dilation) {
  return 1 + (input + 2 * padding - dilation * (filter - 1) - 1) /
                 stride;
}

Shape convolution_output_shape(const ConvBiasReluDefinition& definition) {
  return {definition.x[0],
          definition.w[0],
          output_dimension(definition.x[2],
                           definition.w[2],
                           definition.padding[0],
                           definition.stride[0],
                           definition.dilation[0]),
          output_dimension(definition.x[3],
                           definition.w[3],
                           definition.padding[1],
                           definition.stride[1],
                           definition.dilation[1])};
}

void set_tolerance(AddSquareTestCase& test_case) {
  test_case.absolute_tolerance =
      test_case.output.data_type == FLAGDNN_DATA_BFLOAT16 ? 5.0e-2 : 2.0e-2;
  test_case.relative_tolerance = 1.0e-2;
}

void set_tolerance(ConvBiasReluTestCase& test_case) {
  test_case.absolute_tolerance =
      test_case.output.data_type == FLAGDNN_DATA_FLOAT16 ? 2.0e-2 : 5.0e-2;
  test_case.relative_tolerance = 5.0e-2;
}

AddSquareTestCase make_add_square_case(const Shape& shape,
                                       flagdnnDataType_t data_type,
                                       std::int64_t uid) {
  AddSquareTestCase result;
  result.name = "add_square_" + data_type_name(data_type) + "_" +
                shape_name(shape);
  const Shape strides = pointwise_strides(shape);
  result.left = {uid, data_type, shape, strides};
  result.right = {uid + 1, data_type, shape, strides};
  result.output = {uid + 2, data_type, shape, strides};
  set_tolerance(result);
  return result;
}

ConvBiasReluTestCase make_conv_bias_relu_case(
    const ConvBiasReluDefinition& definition,
    flagdnnDataType_t data_type,
    std::int64_t uid) {
  ConvBiasReluTestCase result;
  result.name = "conv_bias_relu_" + data_type_name(data_type) + "_x" +
                shape_name(definition.x) + "_w" + shape_name(definition.w) +
                "_s" + shape_name(definition.stride) + "_p" +
                shape_name(definition.padding) + "_d" +
                shape_name(definition.dilation);
  const Shape output_shape = convolution_output_shape(definition);
  result.x = tensor(uid, definition.x, data_type, true);
  result.w = tensor(uid + 1, definition.w, data_type, true);
  result.bias = tensor(uid + 2, {1, definition.w[0], 1, 1}, data_type, true);
  result.output = tensor(uid + 3, output_shape, data_type, true);
  result.padding = definition.padding;
  result.stride = definition.stride;
  result.dilation = definition.dilation;
  set_tolerance(result);
  return result;
}

const std::vector<Shape>& add_square_shapes() {
  static const std::vector<Shape> shapes = {
      {1, 1, 16},
      {2, 4, 8},
      {1, 4, 8, 16},
      {2, 4, 8, 16},
      {1, 3, 17},
      {3, 5, 7},
      {1, 3, 5, 7},
      {2, 3, 5, 7},
  };
  return shapes;
}

const std::vector<ConvBiasReluDefinition>& conv_bias_relu_definitions() {
  static const std::vector<ConvBiasReluDefinition> definitions = {
      {{2, 8, 16, 16}, {16, 8, 3, 3}, {1, 1}, {1, 1}, {1, 1}},
      {{1, 4, 15, 17}, {6, 4, 3, 5}, {2, 1}, {1, 2}, {1, 1}},
      {{2, 3, 8, 8}, {5, 3, 1, 1}, {1, 1}, {0, 0}, {1, 1}},
      {{1, 6, 20, 18}, {8, 6, 5, 3}, {1, 2}, {2, 1}, {1, 1}},
      {{2, 4, 12, 10}, {7, 4, 3, 3}, {2, 2}, {1, 1}, {1, 1}},
      {{1, 5, 19, 21}, {9, 5, 3, 3}, {1, 1}, {2, 2}, {2, 2}},
      {{1, 3, 32, 32}, {8, 3, 3, 3}, {1, 1}, {1, 1}, {1, 1}},
      {{2, 8, 9, 11}, {4, 8, 1, 1}, {1, 1}, {0, 0}, {1, 1}},
      {{1, 4, 18, 18}, {4, 4, 3, 3}, {1, 1}, {0, 0}, {1, 1}},
      {{2, 12, 13, 15}, {10, 12, 3, 3}, {1, 1}, {1, 1}, {1, 1}},
  };
  return definitions;
}

void validate_tensor(const TestTensor& tensor_specification,
                     std::string_view role) {
  if (tensor_specification.uid <= 0 ||
      tensor_specification.dimensions.empty() ||
      tensor_specification.dimensions.size() !=
          tensor_specification.strides.size()) {
    throw std::invalid_argument(std::string(role) + " metadata is invalid");
  }
  for (std::size_t axis = 0;
       axis < tensor_specification.dimensions.size();
       ++axis) {
    if (tensor_specification.dimensions[axis] <= 0 ||
        tensor_specification.strides[axis] <= 0) {
      throw std::invalid_argument(
          std::string(role) + " dimensions and strides must be positive");
    }
  }
  if (tensor_specification.data_type != FLAGDNN_DATA_FLOAT32 &&
      tensor_specification.data_type != FLAGDNN_DATA_FLOAT16 &&
      tensor_specification.data_type != FLAGDNN_DATA_BFLOAT16) {
    throw std::invalid_argument(std::string(role) + " data type is invalid");
  }
}

void validate_tolerance(double absolute, double relative) {
  if (!std::isfinite(absolute) || !std::isfinite(relative) || absolute < 0.0 ||
      relative < 0.0) {
    throw std::invalid_argument("composite tolerance is invalid");
  }
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
  throw std::invalid_argument("unsupported FlagDNN composite data type");
}

void check_frontend(fe::error_t status, std::string_view operation) {
  if (status.is_bad()) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + status.get_message());
  }
}

std::shared_ptr<fe::graph::Tensor_attributes> make_tensor(
    const std::shared_ptr<fe::graph::Graph>& graph,
    const TestTensor& specification,
    std::string_view name) {
  return graph->tensor(
      fe::graph::Tensor_attributes()
          .set_name(std::string(name))
          .set_uid(specification.uid)
          .set_data_type(frontend_data_type(specification.data_type))
          .set_dim(specification.dimensions)
          .set_stride(specification.strides));
}

void describe_tensor(
    const std::shared_ptr<fe::graph::Tensor_attributes>& tensor_value,
    const TestTensor& specification,
    std::string_view name,
    std::int64_t uid,
    bool output) {
  tensor_value->set_name(std::string(name))
      .set_uid(uid)
      .set_data_type(frontend_data_type(specification.data_type))
      .set_dim(specification.dimensions)
      .set_stride(specification.strides)
      .set_is_virtual(!output)
      .set_output(output);
}

class FlagdnnCompositeExecutable final : public CompositeExecutable {
 public:
  FlagdnnCompositeExecutable(flagdnn::Handle& handle,
                             const AddSquareTestCase& test_case)
      : handle_(handle), graph_(std::make_shared<fe::graph::Graph>()) {
    validate_composite_case(test_case);
    initialize_graph(test_case.name, test_case.output.data_type, test_case.autotune);
    const auto left = make_tensor(graph_, test_case.left, "left");
    const auto right = make_tensor(graph_, test_case.right, "right");
    const auto square = graph_->pointwise(
        right,
        right,
        fe::graph::Pointwise_attributes()
            .set_name("square")
            .set_mode(fe::PointwiseMode_t::MUL)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_tensor(
        square, test_case.output, "square", test_case.output.uid + 1, false);
    const auto output = graph_->pointwise(
        left,
        square,
        fe::graph::Pointwise_attributes()
            .set_name("add_square")
            .set_mode(fe::PointwiseMode_t::ADD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_tensor(
        output, test_case.output, "output", test_case.output.uid, true);
    build("AddSquare");
  }

  FlagdnnCompositeExecutable(flagdnn::Handle& handle,
                             const ConvBiasReluTestCase& test_case)
      : handle_(handle), graph_(std::make_shared<fe::graph::Graph>()) {
    validate_composite_case(test_case);
    initialize_graph(test_case.name, test_case.output.data_type, test_case.autotune);
    const auto x = make_tensor(graph_, test_case.x, "x");
    const auto w = make_tensor(graph_, test_case.w, "w");
    const auto bias = make_tensor(graph_, test_case.bias, "bias");
    const auto convolution = graph_->conv_fprop(
        x,
        w,
        fe::graph::Conv_fprop_attributes()
            .set_name("convolution")
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_pre_padding(test_case.padding)
            .set_post_padding(test_case.padding)
            .set_stride(test_case.stride)
            .set_dilation(test_case.dilation)
            .set_convolution_mode(fe::ConvolutionMode_t::CROSS_CORRELATION)
            .set_groups(1));
    describe_tensor(convolution,
                    test_case.output,
                    "convolution",
                    test_case.output.uid + 1,
                    false);
    const auto biased = graph_->pointwise(
        convolution,
        bias,
        fe::graph::Pointwise_attributes()
            .set_name("bias_add")
            .set_mode(fe::PointwiseMode_t::ADD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_tensor(
        biased, test_case.output, "biased", test_case.output.uid + 2, false);
    const auto output = graph_->pointwise(
        biased,
        fe::graph::Pointwise_attributes()
            .set_name("relu")
            .set_mode(fe::PointwiseMode_t::RELU_FWD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    describe_tensor(
        output, test_case.output, "output", test_case.output.uid, true);
    build("ConvBiasRelu");
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
      throw std::invalid_argument("FlagDNN composite workspace is too small");
    }
    check_frontend(
        graph_->execute(handle_, bindings, workspace, workspace_size, stream),
        "FlagDNN composite graph execute");
  }

 private:
  void initialize_graph(std::string_view name,
                        flagdnnDataType_t data_type,
                        bool autotune) {
    graph_->set_name(std::string(name))
        .set_io_data_type(frontend_data_type(data_type))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_autotune(autotune);
  }

  void build(std::string_view operation) {
    check_frontend(
        graph_->build(handle_, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}),
        std::string("FlagDNN ") + std::string(operation) + " graph build");
    std::int64_t workspace_size = 0;
    check_frontend(graph_->get_workspace_size(workspace_size),
                   "FlagDNN composite workspace query");
    if (workspace_size < 0) {
      throw std::runtime_error(
          "FlagDNN returned a negative composite workspace size");
    }
    workspace_size_ = static_cast<std::size_t>(workspace_size);
  }

  flagdnn::Handle& handle_;
  std::shared_ptr<fe::graph::Graph> graph_;
  std::size_t workspace_size_ = 0;
};

}  // namespace

std::vector<AddSquareTestCase> make_add_square_cases() {
  std::vector<AddSquareTestCase> result;
  result.reserve(add_square_shapes().size() * kDataTypes.size());
  std::int64_t uid = 75000;
  for (const Shape& shape : add_square_shapes()) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_add_square_case(shape, data_type, uid));
      uid += 3;
    }
  }
  result.front().autotune = true;
  for (const AddSquareTestCase& test_case : result) {
    validate_composite_case(test_case);
  }
  return result;
}

std::vector<ConvBiasReluTestCase> make_conv_bias_relu_cases() {
  std::vector<ConvBiasReluTestCase> result;
  result.reserve(conv_bias_relu_definitions().size() * kDataTypes.size());
  std::int64_t uid = 76000;
  for (const ConvBiasReluDefinition& definition :
       conv_bias_relu_definitions()) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_conv_bias_relu_case(definition, data_type, uid));
      uid += 4;
    }
  }
  result.front().autotune = true;
  for (const ConvBiasReluTestCase& test_case : result) {
    validate_composite_case(test_case);
  }
  return result;
}

void validate_composite_case(const AddSquareTestCase& test_case) {
  if (test_case.name.empty() || test_case.left.uid == test_case.right.uid ||
      test_case.left.uid == test_case.output.uid ||
      test_case.right.uid == test_case.output.uid) {
    throw std::invalid_argument("AddSquare case metadata is invalid");
  }
  validate_tensor(test_case.left, "AddSquare left");
  validate_tensor(test_case.right, "AddSquare right");
  validate_tensor(test_case.output, "AddSquare output");
  if (test_case.left.data_type != test_case.right.data_type ||
      test_case.left.data_type != test_case.output.data_type ||
      test_case.left.dimensions != test_case.right.dimensions ||
      test_case.left.dimensions != test_case.output.dimensions ||
      test_case.left.strides != test_case.right.strides ||
      test_case.left.strides != test_case.output.strides) {
    throw std::invalid_argument(
        "AddSquare tensors must have the same type, shape, and layout");
  }
  validate_tolerance(
      test_case.absolute_tolerance, test_case.relative_tolerance);
}

void validate_composite_case(const ConvBiasReluTestCase& test_case) {
  if (test_case.name.empty() || test_case.x.uid == test_case.w.uid ||
      test_case.x.uid == test_case.bias.uid ||
      test_case.x.uid == test_case.output.uid ||
      test_case.w.uid == test_case.bias.uid ||
      test_case.w.uid == test_case.output.uid ||
      test_case.bias.uid == test_case.output.uid) {
    throw std::invalid_argument("ConvBiasRelu case metadata is invalid");
  }
  validate_tensor(test_case.x, "ConvBiasRelu x");
  validate_tensor(test_case.w, "ConvBiasRelu w");
  validate_tensor(test_case.bias, "ConvBiasRelu bias");
  validate_tensor(test_case.output, "ConvBiasRelu output");
  if (test_case.x.dimensions.size() != 4 ||
      test_case.w.dimensions.size() != 4 ||
      test_case.bias.dimensions.size() != 4 ||
      test_case.output.dimensions.size() != 4 ||
      test_case.x.data_type != test_case.w.data_type ||
      test_case.x.data_type != test_case.bias.data_type ||
      test_case.x.data_type != test_case.output.data_type ||
      test_case.x.dimensions[1] != test_case.w.dimensions[1] ||
      test_case.bias.dimensions != Shape({1, test_case.w.dimensions[0], 1, 1}) ||
      test_case.output.dimensions[0] != test_case.x.dimensions[0] ||
      test_case.output.dimensions[1] != test_case.w.dimensions[0]) {
    throw std::invalid_argument("ConvBiasRelu tensor metadata is inconsistent");
  }
  if (test_case.padding.size() != 2 || test_case.stride.size() != 2 ||
      test_case.dilation.size() != 2) {
    throw std::invalid_argument("ConvBiasRelu spatial attributes are invalid");
  }
  for (std::size_t axis = 0; axis < 2; ++axis) {
    if (test_case.padding[axis] < 0 || test_case.stride[axis] <= 0 ||
        test_case.dilation[axis] <= 0) {
      throw std::invalid_argument(
          "ConvBiasRelu spatial attributes must be nonnegative/positive");
    }
    const std::int64_t expected = output_dimension(
        test_case.x.dimensions[axis + 2],
        test_case.w.dimensions[axis + 2],
        test_case.padding[axis],
        test_case.stride[axis],
        test_case.dilation[axis]);
    if (expected <= 0 || test_case.output.dimensions[axis + 2] != expected) {
      throw std::invalid_argument("ConvBiasRelu output shape is invalid");
    }
  }
  validate_tolerance(
      test_case.absolute_tolerance, test_case.relative_tolerance);
}

std::unique_ptr<CompositeExecutable> build_flagdnn_add_square(
    flagdnn::Handle& handle,
    const AddSquareTestCase& test_case) {
  return std::make_unique<FlagdnnCompositeExecutable>(handle, test_case);
}

std::unique_ptr<CompositeExecutable> build_flagdnn_conv_bias_relu(
    flagdnn::Handle& handle,
    const ConvBiasReluTestCase& test_case) {
  return std::make_unique<FlagdnnCompositeExecutable>(handle, test_case);
}

}  // namespace flagdnn::testing
