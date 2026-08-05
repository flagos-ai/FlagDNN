/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/convolution.hpp"

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

struct CaseDefinition {
  Shape x;
  Shape w;
  Shape stride;
  Shape pre_padding;
  Shape post_padding;
  Shape dilation;
  std::int64_t groups = 1;
  bool x_channels_last = false;
  bool w_channels_last = false;
  bool y_channels_last = false;
  std::string label;
  ConvolutionMode mode = ConvolutionMode::kCrossCorrelation;
};

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
      throw std::invalid_argument("convolution tensor shape is invalid");
    }
    result[axis - 1] = stride;
    stride *= dimension;
  }
  return result;
}

Shape channels_last_strides(const Shape& dimensions) {
  if (dimensions.size() < 3 || dimensions.size() > 5) {
    throw std::invalid_argument(
        "channels-last convolution tensor rank must be 3, 4, or 5");
  }
  Shape result(dimensions.size());
  result[1] = 1;
  std::int64_t stride = dimensions[1];
  for (std::size_t axis = dimensions.size(); axis != 2; --axis) {
    const std::size_t current = axis - 1;
    result[current] = stride;
    stride *= dimensions[current];
  }
  result[0] = stride;
  return result;
}

TestTensor tensor(std::int64_t uid,
                  Shape dimensions,
                  flagdnnDataType_t data_type,
                  bool channels_last) {
  Shape strides = channels_last ? channels_last_strides(dimensions)
                                : contiguous_strides(dimensions);
  return {uid,
          data_type,
          std::move(dimensions),
          std::move(strides)};
}

std::int64_t output_dimension(std::int64_t input,
                              std::int64_t filter,
                              std::int64_t pre_padding,
                              std::int64_t post_padding,
                              std::int64_t stride,
                              std::int64_t dilation) {
  return (input + pre_padding + post_padding -
          dilation * (filter - 1) - 1) /
             stride +
         1;
}

Shape output_shape(const CaseDefinition& definition) {
  Shape result = {definition.x[0], definition.w[0]};
  for (std::size_t axis = 0; axis < definition.stride.size(); ++axis) {
    result.push_back(output_dimension(definition.x[axis + 2],
                                      definition.w[axis + 2],
                                      definition.pre_padding[axis],
                                      definition.post_padding[axis],
                                      definition.stride[axis],
                                      definition.dilation[axis]));
  }
  return result;
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
  throw std::invalid_argument("unsupported convolution data type");
}

std::string direction_name(ConvolutionDirection direction) {
  switch (direction) {
    case ConvolutionDirection::kFprop:
      return "fprop";
    case ConvolutionDirection::kDgrad:
      return "dgrad";
    case ConvolutionDirection::kWgrad:
      return "wgrad";
  }
  throw std::invalid_argument("unsupported convolution direction");
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

void set_tolerance(ConvolutionTestCase& test_case) {
  const flagdnnDataType_t data_type = test_case.x.data_type;
  switch (test_case.direction) {
    case ConvolutionDirection::kFprop:
      test_case.absolute_tolerance =
          data_type == FLAGDNN_DATA_FLOAT16 ? 2.0e-2 : 5.0e-2;
      test_case.relative_tolerance = 5.0e-2;
      return;
    case ConvolutionDirection::kDgrad:
      test_case.absolute_tolerance =
          data_type == FLAGDNN_DATA_FLOAT16 ? 2.0e-2 : 7.0e-2;
      test_case.relative_tolerance = 7.0e-2;
      return;
    case ConvolutionDirection::kWgrad:
      test_case.absolute_tolerance =
          data_type == FLAGDNN_DATA_FLOAT16 ? 8.0e-2 : 2.0e-1;
      test_case.relative_tolerance = 1.0e-1;
      return;
  }
  throw std::invalid_argument("unsupported convolution direction");
}

ConvolutionTestCase make_case(const CaseDefinition& definition,
                              ConvolutionDirection direction,
                              flagdnnDataType_t data_type,
                              std::int64_t uid) {
  ConvolutionTestCase result;
  const std::size_t spatial_rank = definition.x.size() - 2;
  result.name = "conv" + std::to_string(spatial_rank) + "d_" +
                direction_name(direction) + "_" + data_type_name(data_type) +
                "_" + definition.label + "_" + shape_name(definition.x) +
                "_by_" + shape_name(definition.w);
  if (definition.mode == ConvolutionMode::kConvolution) {
    result.name += "_convolution_mode";
  }
  result.direction = direction;
  result.x = tensor(
      uid, definition.x, data_type, definition.x_channels_last);
  result.w = tensor(
      uid + 1, definition.w, data_type, definition.w_channels_last);
  result.y = tensor(uid + 2,
                    output_shape(definition),
                    data_type,
                    definition.y_channels_last);
  result.pre_padding = definition.pre_padding;
  result.post_padding = definition.post_padding;
  result.stride = definition.stride;
  result.dilation = definition.dilation;
  result.groups = definition.groups;
  result.mode = definition.mode;
  set_tolerance(result);
  return result;
}

const std::vector<CaseDefinition>& fprop_definitions() {
  static const std::vector<CaseDefinition> definitions = {
      {{1, 2, 5, 5},
       {2, 2, 3, 3},
       {1, 1}, {1, 1}, {1, 1}, {1, 1},
       1, false, false, false, "nchw_smoke"},
      {{2, 8, 16, 16},
       {16, 8, 3, 3},
       {1, 1}, {1, 1}, {1, 1}, {1, 1},
       1, true, true, true, "nhwc_symmetric"},
      {{1, 4, 15, 17},
       {6, 4, 3, 5},
       {2, 1}, {1, 2}, {1, 2}, {1, 1},
       1, true, true, true, "nhwc_nonuniform_stride"},
      {{1, 5, 19, 21},
       {9, 5, 3, 3},
       {1, 1}, {2, 1}, {0, 3}, {2, 1},
       1, true, true, true, "nhwc_asymmetric_dilation"},
      {{1, 4, 7, 7},
       {6, 2, 3, 3},
       {1, 1}, {1, 1}, {1, 1}, {1, 1},
       2, false, false, false, "nchw_groups2"},
      {{2, 4, 16},
       {6, 4, 3},
       {1}, {1}, {1}, {1},
       1, false, false, true, "ncw_to_nwc"},
      {{1, 2, 5, 6, 7},
       {4, 2, 3, 3, 3},
       {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1},
       1, true, true, true, "ndhwc_symmetric"},
      {{1, 2, 6, 7, 8},
       {3, 2, 2, 3, 3},
       {1, 1, 1}, {1, 0, 1}, {0, 1, 2}, {1, 1, 1},
       1, true, true, true, "ndhwc_asymmetric"},
  };
  return definitions;
}

std::vector<CaseDefinition> backward_definitions() {
  std::vector<CaseDefinition> result = {
      {{2, 8, 16, 16},
       {16, 8, 3, 3},
       {1, 1}, {1, 1}, {1, 1}, {1, 1},
       1, false, false, false, "nchw_symmetric"},
      {{1, 4, 15, 17},
       {6, 4, 3, 5},
       {2, 1}, {1, 2}, {1, 2}, {1, 1},
       1, false, false, false, "nchw_nonuniform_stride"},
      {{2, 3, 8, 8},
       {5, 3, 1, 1},
       {1, 1}, {0, 0}, {0, 0}, {1, 1},
       1, false, false, false, "nchw_1x1"},
      {{2, 4, 12, 13},
       {7, 4, 3, 3},
       {1, 2}, {1, 0}, {2, 3}, {1, 1},
       1, false, false, false, "nchw_asymmetric_padding"},
      {{1, 4, 7, 7},
       {6, 2, 3, 3},
       {1, 1}, {1, 1}, {1, 1}, {1, 1},
       2, false, false, false, "nchw_groups2"},
      {{2, 4, 16},
       {6, 4, 3},
       {1}, {1}, {1}, {1},
       1, false, false, false, "ncw_symmetric"},
      {{1, 2, 5, 6, 7},
       {4, 2, 3, 3, 3},
       {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1},
       1, false, false, false, "ncdhw_symmetric"},
      {{1, 2, 6, 7, 8},
       {3, 2, 2, 3, 3},
       {1, 1, 1}, {1, 0, 1}, {0, 1, 2}, {1, 1, 1},
       1, false, false, false, "ncdhw_asymmetric"},
  };
  result.push_back({{2, 4, 8, 9},
                    {6, 4, 3, 3},
                    {1, 1}, {1, 1}, {1, 1}, {1, 1},
                    1, false, false, false, "explicit_filter_flip",
                    ConvolutionMode::kConvolution});
  return result;
}

void validate_tensor(const TestTensor& tensor_specification,
                     std::string_view name) {
  if (tensor_specification.uid <= 0 ||
      tensor_specification.dimensions.size() < 3 ||
      tensor_specification.dimensions.size() > 5 ||
      tensor_specification.dimensions.size() !=
          tensor_specification.strides.size()) {
    throw std::invalid_argument(std::string(name) + " metadata is invalid");
  }
  for (std::size_t axis = 0;
       axis < tensor_specification.dimensions.size();
       ++axis) {
    if (tensor_specification.dimensions[axis] <= 0 ||
        tensor_specification.strides[axis] <= 0) {
      throw std::invalid_argument(
          std::string(name) + " dimensions and strides must be positive");
    }
  }
  if (tensor_specification.data_type != FLAGDNN_DATA_FLOAT32 &&
      tensor_specification.data_type != FLAGDNN_DATA_FLOAT16 &&
      tensor_specification.data_type != FLAGDNN_DATA_BFLOAT16) {
    throw std::invalid_argument(
        std::string(name) + " data type is not supported by convolution");
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
  throw std::invalid_argument("unsupported FlagDNN convolution data type");
}

fe::ConvolutionMode_t frontend_convolution_mode(ConvolutionMode mode) {
  switch (mode) {
    case ConvolutionMode::kCrossCorrelation:
      return fe::ConvolutionMode_t::CROSS_CORRELATION;
    case ConvolutionMode::kConvolution:
      return fe::ConvolutionMode_t::CONVOLUTION;
  }
  throw std::invalid_argument("unsupported convolution mode");
}

void check_frontend(fe::error_t status, std::string_view operation) {
  if (status.is_bad()) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + status.get_message());
  }
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

template <typename Attributes>
Attributes apply_attributes(Attributes attributes,
                            const ConvolutionTestCase& test_case) {
  return attributes
      .set_name(direction_name(test_case.direction))
      .set_compute_data_type(fe::DataType_t::FLOAT)
      .set_pre_padding(test_case.pre_padding)
      .set_post_padding(test_case.post_padding)
      .set_stride(test_case.stride)
      .set_dilation(test_case.dilation)
      .set_convolution_mode(frontend_convolution_mode(test_case.mode))
      .set_groups(test_case.groups);
}

class FlagdnnConvolutionExecutable final : public ConvolutionExecutable {
 public:
  FlagdnnConvolutionExecutable(flagdnn::Handle& handle,
                               const ConvolutionTestCase& test_case)
      : handle_(handle), graph_(std::make_shared<fe::graph::Graph>()) {
    validate_convolution_case(test_case);
    const fe::DataType_t io_type = frontend_data_type(test_case.x.data_type);
    graph_->set_name(test_case.name)
        .set_io_data_type(io_type)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_autotune(test_case.autotune);
    const auto x = test_case.direction == ConvolutionDirection::kDgrad
                       ? nullptr
                       : make_tensor(graph_, test_case.x, "x");
    const auto w = test_case.direction == ConvolutionDirection::kWgrad
                       ? nullptr
                       : make_tensor(graph_, test_case.w, "w");
    const auto y = test_case.direction == ConvolutionDirection::kFprop
                       ? nullptr
                       : make_tensor(graph_, test_case.y, "y");
    std::shared_ptr<fe::graph::Tensor_attributes> output;
    switch (test_case.direction) {
      case ConvolutionDirection::kFprop:
        output = graph_->conv_fprop(
            x,
            w,
            apply_attributes(fe::graph::Conv_fprop_attributes(), test_case));
        break;
      case ConvolutionDirection::kDgrad:
        output = graph_->conv_dgrad(
            y,
            w,
            apply_attributes(fe::graph::Conv_dgrad_attributes(), test_case));
        break;
      case ConvolutionDirection::kWgrad:
        output = graph_->conv_wgrad(
            y,
            x,
            apply_attributes(fe::graph::Conv_wgrad_attributes(), test_case));
        break;
    }
    const TestTensor& expected = convolution_output_tensor(test_case);
    output->set_name("output")
        .set_uid(expected.uid)
        .set_data_type(frontend_data_type(expected.data_type))
        .set_dim(expected.dimensions)
        .set_stride(expected.strides)
        .set_output(true);

    check_frontend(
        graph_->build(handle_, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}),
        "FlagDNN convolution graph build");
    std::int64_t workspace_size = 0;
    check_frontend(graph_->get_workspace_size(workspace_size),
                   "FlagDNN convolution workspace query");
    if (workspace_size < 0) {
      throw std::runtime_error(
          "FlagDNN returned a negative convolution workspace size");
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
          "FlagDNN convolution workspace is too small");
    }
    check_frontend(
        graph_->execute(handle_, bindings, workspace, workspace_size, stream),
        "FlagDNN convolution graph execute");
  }

 private:
  flagdnn::Handle& handle_;
  std::shared_ptr<fe::graph::Graph> graph_;
  std::size_t workspace_size_ = 0;
};

}  // namespace

std::vector<ConvolutionTestCase> make_convolution_cases(
    ConvolutionDirection direction) {
  const std::vector<CaseDefinition> definitions =
      direction == ConvolutionDirection::kFprop
          ? fprop_definitions()
          : backward_definitions();
  std::vector<ConvolutionTestCase> result;
  result.reserve(definitions.size() * kDataTypes.size());
  std::int64_t uid =
      direction == ConvolutionDirection::kFprop
          ? 61000
          : (direction == ConvolutionDirection::kDgrad ? 62000 : 63000);
  for (const CaseDefinition& definition : definitions) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_case(definition, direction, data_type, uid));
      uid += 3;
    }
  }
  if (!result.empty()) {
    result.front().autotune = true;
    if (direction == ConvolutionDirection::kFprop) {
      result.front().absolute_tolerance = 2.0e-5;
      result.front().relative_tolerance = 1.0e-5;
    }
  }
  for (const ConvolutionTestCase& test_case : result) {
    validate_convolution_case(test_case);
  }
  return result;
}

void validate_convolution_case(const ConvolutionTestCase& test_case) {
  if (test_case.name.empty() || test_case.groups <= 0 ||
      test_case.x.uid == test_case.w.uid ||
      test_case.x.uid == test_case.y.uid ||
      test_case.w.uid == test_case.y.uid ||
      !std::isfinite(test_case.absolute_tolerance) ||
      !std::isfinite(test_case.relative_tolerance) ||
      test_case.absolute_tolerance < 0.0 ||
      test_case.relative_tolerance < 0.0) {
    throw std::invalid_argument("convolution case metadata is invalid");
  }
  validate_tensor(test_case.x, "convolution X");
  validate_tensor(test_case.w, "convolution W");
  validate_tensor(test_case.y, "convolution Y");
  if (test_case.x.data_type != test_case.w.data_type ||
      test_case.x.data_type != test_case.y.data_type ||
      test_case.x.dimensions.size() != test_case.w.dimensions.size() ||
      test_case.x.dimensions.size() != test_case.y.dimensions.size()) {
    throw std::invalid_argument(
        "convolution tensor ranks and data types must match");
  }
  const std::size_t spatial_rank = test_case.x.dimensions.size() - 2;
  if (test_case.pre_padding.size() != spatial_rank ||
      test_case.post_padding.size() != spatial_rank ||
      test_case.stride.size() != spatial_rank ||
      test_case.dilation.size() != spatial_rank) {
    throw std::invalid_argument(
        "convolution spatial attributes have the wrong rank");
  }
  if (test_case.direction == ConvolutionDirection::kFprop &&
      test_case.mode != ConvolutionMode::kCrossCorrelation) {
    throw std::invalid_argument(
        "FlagDNN FProp supports cross-correlation mode only");
  }
  const std::int64_t channels = test_case.x.dimensions[1];
  const std::int64_t filters = test_case.w.dimensions[0];
  if (channels % test_case.groups != 0 ||
      filters % test_case.groups != 0 ||
      test_case.w.dimensions[1] != channels / test_case.groups ||
      test_case.y.dimensions[0] != test_case.x.dimensions[0] ||
      test_case.y.dimensions[1] != filters) {
    throw std::invalid_argument("convolution channel metadata is invalid");
  }
  for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
    if (test_case.pre_padding[axis] < 0 ||
        test_case.post_padding[axis] < 0 ||
        test_case.stride[axis] <= 0 || test_case.dilation[axis] <= 0 ||
        test_case.y.dimensions[axis + 2] !=
            output_dimension(test_case.x.dimensions[axis + 2],
                             test_case.w.dimensions[axis + 2],
                             test_case.pre_padding[axis],
                             test_case.post_padding[axis],
                             test_case.stride[axis],
                             test_case.dilation[axis])) {
      throw std::invalid_argument(
          "convolution spatial metadata or output shape is invalid");
    }
  }
}

const TestTensor& convolution_output_tensor(
    const ConvolutionTestCase& test_case) {
  switch (test_case.direction) {
    case ConvolutionDirection::kFprop:
      return test_case.y;
    case ConvolutionDirection::kDgrad:
      return test_case.x;
    case ConvolutionDirection::kWgrad:
      return test_case.w;
  }
  throw std::invalid_argument("unsupported convolution direction");
}

std::unique_ptr<ConvolutionExecutable> build_flagdnn_convolution(
    flagdnn::Handle& handle,
    const ConvolutionTestCase& test_case) {
  return std::make_unique<FlagdnnConvolutionExecutable>(handle, test_case);
}

}  // namespace flagdnn::testing
