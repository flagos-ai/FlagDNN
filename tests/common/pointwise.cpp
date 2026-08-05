/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/pointwise.hpp"

#include <flagdnn/flagdnn.hpp>
#include <flagdnn_frontend.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace flagdnn::testing {
namespace {

namespace fe = ::flagdnn_frontend;
using Shape = std::vector<std::int64_t>;

constexpr std::array<flagdnnDataType_t, 3> kFloatingDataTypes = {
    FLAGDNN_DATA_FLOAT32,
    FLAGDNN_DATA_FLOAT16,
    FLAGDNN_DATA_BFLOAT16,
};

const std::vector<Shape>& numeric_shapes() {
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

const std::vector<Shape>& identity_shapes() {
  static const std::vector<Shape> shapes = {
      {2, 3, 4},
      {4, 5, 6},
      {1, 8, 16},
      {3, 1, 17},
      {2, 4, 8},
      {5, 7, 11},
      {1, 33, 65},
      {2, 16, 257},
      {4, 32, 128},
  };
  return shapes;
}

// Some platform Graph references store BOOLEAN tensors bit-packed.  Every shape below has a
// contiguous logical extent divisible by eight, so both public Graph APIs can
// be compared without a host fallback or an ABI-specific padding exception.
const std::vector<Shape>& packed_boolean_shapes() {
  static const std::vector<Shape> shapes = {
      {1, 1, 16},
      {2, 8, 8},
      {1, 8, 4, 8},
      {2, 8, 4, 8},
  };
  return shapes;
}

std::vector<std::int64_t> contiguous_strides(const Shape& dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    result[axis - 1] = stride;
    if (dimensions[axis - 1] <= 0 ||
        stride > std::numeric_limits<std::int64_t>::max() /
                     dimensions[axis - 1]) {
      throw std::invalid_argument("pointwise shape is invalid or too large");
    }
    stride *= dimensions[axis - 1];
  }
  return result;
}

std::vector<std::int64_t> pointwise_strides(const Shape& dimensions) {
  if (dimensions.size() != 4) {
    return contiguous_strides(dimensions);
  }
  const std::int64_t channels = dimensions[1];
  const std::int64_t height = dimensions[2];
  const std::int64_t width = dimensions[3];
  return {channels * height * width, 1, width * channels, channels};
}

TestTensor make_tensor(std::int64_t uid,
                       const Shape& dimensions,
                       flagdnnDataType_t data_type) {
  return TestTensor{uid, data_type, dimensions, pointwise_strides(dimensions)};
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
      return "bool";
  }
  throw std::invalid_argument("unsupported pointwise data type");
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

bool is_comparison_mode(flagdnnPointwiseMode_t mode) {
  return mode == FLAGDNN_POINTWISE_CMP_EQ ||
         mode == FLAGDNN_POINTWISE_CMP_NEQ ||
         mode == FLAGDNN_POINTWISE_CMP_GT ||
         mode == FLAGDNN_POINTWISE_CMP_GE ||
         mode == FLAGDNN_POINTWISE_CMP_LT ||
         mode == FLAGDNN_POINTWISE_CMP_LE;
}

bool is_logical_binary_mode(flagdnnPointwiseMode_t mode) {
  return mode == FLAGDNN_POINTWISE_LOGICAL_AND ||
         mode == FLAGDNN_POINTWISE_LOGICAL_OR;
}

bool is_unary_mode(flagdnnPointwiseMode_t mode) {
  switch (mode) {
    case FLAGDNN_POINTWISE_RELU_FWD:
    case FLAGDNN_POINTWISE_SQRT:
    case FLAGDNN_POINTWISE_ERF:
    case FLAGDNN_POINTWISE_IDENTITY:
    case FLAGDNN_POINTWISE_EXP:
    case FLAGDNN_POINTWISE_LOG:
    case FLAGDNN_POINTWISE_NEG:
    case FLAGDNN_POINTWISE_ABS:
    case FLAGDNN_POINTWISE_CEIL:
    case FLAGDNN_POINTWISE_COS:
    case FLAGDNN_POINTWISE_FLOOR:
    case FLAGDNN_POINTWISE_RSQRT:
    case FLAGDNN_POINTWISE_SIN:
    case FLAGDNN_POINTWISE_TAN:
    case FLAGDNN_POINTWISE_RECIPROCAL:
    case FLAGDNN_POINTWISE_LOGICAL_NOT:
    case FLAGDNN_POINTWISE_SIGMOID_FWD:
    case FLAGDNN_POINTWISE_TANH_FWD:
    case FLAGDNN_POINTWISE_ELU_FWD:
    case FLAGDNN_POINTWISE_GELU_FWD:
    case FLAGDNN_POINTWISE_SOFTPLUS_FWD:
    case FLAGDNN_POINTWISE_SWISH_FWD:
    case FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD:
      return true;
    default:
      return false;
  }
}

bool is_binary_mode(flagdnnPointwiseMode_t mode) {
  switch (mode) {
    case FLAGDNN_POINTWISE_ADD:
    case FLAGDNN_POINTWISE_SUB:
    case FLAGDNN_POINTWISE_MUL:
    case FLAGDNN_POINTWISE_DIV:
    case FLAGDNN_POINTWISE_MIN:
    case FLAGDNN_POINTWISE_MAX:
    case FLAGDNN_POINTWISE_MOD:
    case FLAGDNN_POINTWISE_POW:
    case FLAGDNN_POINTWISE_CMP_EQ:
    case FLAGDNN_POINTWISE_CMP_NEQ:
    case FLAGDNN_POINTWISE_CMP_GT:
    case FLAGDNN_POINTWISE_CMP_GE:
    case FLAGDNN_POINTWISE_CMP_LT:
    case FLAGDNN_POINTWISE_CMP_LE:
    case FLAGDNN_POINTWISE_LOGICAL_AND:
    case FLAGDNN_POINTWISE_LOGICAL_OR:
    case FLAGDNN_POINTWISE_SIGMOID_BWD:
      return true;
    default:
      return false;
  }
}

bool uses_boolean_compute(flagdnnPointwiseMode_t mode) {
  return mode == FLAGDNN_POINTWISE_LOGICAL_NOT ||
         is_comparison_mode(mode) || is_logical_binary_mode(mode);
}

flagdnnDataType_t output_data_type(flagdnnPointwiseMode_t mode,
                                   flagdnnDataType_t input_data_type) {
  if (mode == FLAGDNN_POINTWISE_LOGICAL_NOT || is_comparison_mode(mode) ||
      is_logical_binary_mode(mode)) {
    return FLAGDNN_DATA_BOOLEAN;
  }
  return input_data_type;
}

void set_tolerance(PointwiseTestCase& test_case) {
  if (test_case.output.data_type == FLAGDNN_DATA_BOOLEAN ||
      test_case.mode == FLAGDNN_POINTWISE_CEIL ||
      test_case.mode == FLAGDNN_POINTWISE_FLOOR ||
      test_case.mode == FLAGDNN_POINTWISE_IDENTITY) {
    test_case.absolute_tolerance = 0.0;
    test_case.relative_tolerance = 0.0;
    return;
  }
  test_case.absolute_tolerance =
      test_case.output.data_type == FLAGDNN_DATA_BFLOAT16 ? 5.0e-2 : 2.0e-2;
  test_case.relative_tolerance = 1.0e-2;
}

std::vector<flagdnnDataType_t> input_data_types(flagdnnPointwiseMode_t mode) {
  if (mode == FLAGDNN_POINTWISE_LOGICAL_NOT ||
      is_logical_binary_mode(mode)) {
    return {FLAGDNN_DATA_BOOLEAN};
  }
  return {kFloatingDataTypes.begin(), kFloatingDataTypes.end()};
}

bool broadcasts_to(const TestTensor& input, const TestTensor& output) {
  if (input.dimensions.size() > output.dimensions.size()) {
    return false;
  }
  const std::size_t leading =
      output.dimensions.size() - input.dimensions.size();
  for (std::size_t axis = 0; axis < input.dimensions.size(); ++axis) {
    const std::int64_t input_dimension = input.dimensions[axis];
    const std::int64_t output_dimension = output.dimensions[leading + axis];
    if (input_dimension != 1 && input_dimension != output_dimension) {
      return false;
    }
  }
  return true;
}

void validate_tensor(const TestTensor& tensor, std::string_view name) {
  if (tensor.uid <= 0) {
    throw std::invalid_argument(std::string(name) + " UID must be positive");
  }
  if (tensor.dimensions.empty() || tensor.dimensions.size() > 4 ||
      tensor.dimensions.size() != tensor.strides.size()) {
    throw std::invalid_argument(std::string(name) + " metadata is invalid");
  }
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0) {
      throw std::invalid_argument(
          std::string(name) + " dimensions and strides must be positive");
    }
  }
  (void)data_type_name(tensor.data_type);
}

void validate_attributes(const flagdnnPointwiseAttributes_t& attributes) {
  if (attributes.struct_size != sizeof(flagdnnPointwiseAttributes_t) ||
      attributes.version != FLAGDNN_POINTWISE_ATTRIBUTES_VERSION ||
      (attributes.flags & ~FLAGDNN_POINTWISE_ATTRIBUTE_FLAGS_ALL) != 0U) {
    throw std::invalid_argument("pointwise attributes ABI is invalid");
  }
  const std::array<double, 6> values = {
      attributes.relu_lower_clip,
      attributes.relu_upper_clip,
      attributes.relu_lower_clip_slope,
      attributes.swish_beta,
      attributes.elu_alpha,
      attributes.softplus_beta,
  };
  if (!std::all_of(values.begin(), values.end(), [](double value) {
        return std::isfinite(value);
      })) {
    throw std::invalid_argument("pointwise attributes must be finite");
  }
}

void check_frontend(fe::error_t status, std::string_view operation) {
  if (status.is_bad()) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + status.get_message());
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
      return fe::DataType_t::BOOLEAN;
  }
  throw std::invalid_argument("unsupported FlagDNN pointwise data type");
}

fe::PointwiseMode_t frontend_pointwise_mode(flagdnnPointwiseMode_t mode) {
  switch (mode) {
    case FLAGDNN_POINTWISE_RELU_FWD:
      return fe::PointwiseMode_t::RELU_FWD;
    case FLAGDNN_POINTWISE_ADD:
      return fe::PointwiseMode_t::ADD;
    case FLAGDNN_POINTWISE_SQRT:
      return fe::PointwiseMode_t::SQRT;
    case FLAGDNN_POINTWISE_ERF:
      return fe::PointwiseMode_t::ERF;
    case FLAGDNN_POINTWISE_IDENTITY:
      return fe::PointwiseMode_t::IDENTITY;
    case FLAGDNN_POINTWISE_EXP:
      return fe::PointwiseMode_t::EXP;
    case FLAGDNN_POINTWISE_LOG:
      return fe::PointwiseMode_t::LOG;
    case FLAGDNN_POINTWISE_NEG:
      return fe::PointwiseMode_t::NEG;
    case FLAGDNN_POINTWISE_ABS:
      return fe::PointwiseMode_t::ABS;
    case FLAGDNN_POINTWISE_CEIL:
      return fe::PointwiseMode_t::CEIL;
    case FLAGDNN_POINTWISE_COS:
      return fe::PointwiseMode_t::COS;
    case FLAGDNN_POINTWISE_FLOOR:
      return fe::PointwiseMode_t::FLOOR;
    case FLAGDNN_POINTWISE_RSQRT:
      return fe::PointwiseMode_t::RSQRT;
    case FLAGDNN_POINTWISE_SIN:
      return fe::PointwiseMode_t::SIN;
    case FLAGDNN_POINTWISE_TAN:
      return fe::PointwiseMode_t::TAN;
    case FLAGDNN_POINTWISE_RECIPROCAL:
      return fe::PointwiseMode_t::RECIPROCAL;
    case FLAGDNN_POINTWISE_SUB:
      return fe::PointwiseMode_t::SUB;
    case FLAGDNN_POINTWISE_MUL:
      return fe::PointwiseMode_t::MUL;
    case FLAGDNN_POINTWISE_DIV:
      return fe::PointwiseMode_t::DIV;
    case FLAGDNN_POINTWISE_MIN:
      return fe::PointwiseMode_t::MIN;
    case FLAGDNN_POINTWISE_MAX:
      return fe::PointwiseMode_t::MAX;
    case FLAGDNN_POINTWISE_MOD:
      return fe::PointwiseMode_t::MOD;
    case FLAGDNN_POINTWISE_POW:
      return fe::PointwiseMode_t::POW;
    case FLAGDNN_POINTWISE_LOGICAL_NOT:
      return fe::PointwiseMode_t::LOGICAL_NOT;
    case FLAGDNN_POINTWISE_CMP_EQ:
      return fe::PointwiseMode_t::CMP_EQ;
    case FLAGDNN_POINTWISE_CMP_NEQ:
      return fe::PointwiseMode_t::CMP_NEQ;
    case FLAGDNN_POINTWISE_CMP_GT:
      return fe::PointwiseMode_t::CMP_GT;
    case FLAGDNN_POINTWISE_CMP_GE:
      return fe::PointwiseMode_t::CMP_GE;
    case FLAGDNN_POINTWISE_CMP_LT:
      return fe::PointwiseMode_t::CMP_LT;
    case FLAGDNN_POINTWISE_CMP_LE:
      return fe::PointwiseMode_t::CMP_LE;
    case FLAGDNN_POINTWISE_LOGICAL_AND:
      return fe::PointwiseMode_t::LOGICAL_AND;
    case FLAGDNN_POINTWISE_LOGICAL_OR:
      return fe::PointwiseMode_t::LOGICAL_OR;
    case FLAGDNN_POINTWISE_SIGMOID_BWD:
      return fe::PointwiseMode_t::SIGMOID_BWD;
    case FLAGDNN_POINTWISE_BINARY_SELECT:
      return fe::PointwiseMode_t::BINARY_SELECT;
    case FLAGDNN_POINTWISE_SIGMOID_FWD:
      return fe::PointwiseMode_t::SIGMOID_FWD;
    case FLAGDNN_POINTWISE_TANH_FWD:
      return fe::PointwiseMode_t::TANH_FWD;
    case FLAGDNN_POINTWISE_ELU_FWD:
      return fe::PointwiseMode_t::ELU_FWD;
    case FLAGDNN_POINTWISE_GELU_FWD:
      return fe::PointwiseMode_t::GELU_FWD;
    case FLAGDNN_POINTWISE_SOFTPLUS_FWD:
      return fe::PointwiseMode_t::SOFTPLUS_FWD;
    case FLAGDNN_POINTWISE_SWISH_FWD:
      return fe::PointwiseMode_t::SWISH_FWD;
    case FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD:
      return fe::PointwiseMode_t::GELU_APPROX_TANH_FWD;
    case FLAGDNN_POINTWISE_NOT_SET:
      break;
  }
  throw std::invalid_argument("unsupported FlagDNN pointwise mode");
}

void apply_pointwise_attributes(
    fe::graph::Pointwise_attributes& output,
    const flagdnnPointwiseAttributes_t& input) {
  if ((input.flags & FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP) != 0U) {
    output.set_relu_lower_clip(static_cast<float>(input.relu_lower_clip));
  }
  if ((input.flags & FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP) != 0U) {
    output.set_relu_upper_clip(static_cast<float>(input.relu_upper_clip));
  }
  if ((input.flags &
       FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE) != 0U) {
    output.set_relu_lower_clip_slope(
        static_cast<float>(input.relu_lower_clip_slope));
  }
  if ((input.flags & FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA) != 0U) {
    output.set_swish_beta(static_cast<float>(input.swish_beta));
  }
  if ((input.flags & FLAGDNN_POINTWISE_ATTRIBUTE_ELU_ALPHA) != 0U) {
    output.set_elu_alpha(static_cast<float>(input.elu_alpha));
  }
  if ((input.flags & FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA) != 0U) {
    output.set_softplus_beta(static_cast<float>(input.softplus_beta));
  }
}

std::shared_ptr<fe::graph::Tensor_attributes> make_frontend_tensor(
    const std::shared_ptr<fe::graph::Graph>& graph,
    const TestTensor& tensor,
    std::string name) {
  return graph->tensor(
      fe::graph::Tensor_attributes()
          .set_name(std::move(name))
          .set_uid(tensor.uid)
          .set_data_type(frontend_data_type(tensor.data_type))
          .set_dim(tensor.dimensions)
          .set_stride(tensor.strides));
}

class FlagdnnPointwiseExecutable final : public PointwiseExecutable {
 public:
  FlagdnnPointwiseExecutable(flagdnn::Handle& handle,
                             const PointwiseTestCase& test_case)
      : handle_(handle), graph_(std::make_shared<fe::graph::Graph>()) {
    validate_pointwise_case(test_case);

    graph_->set_name(test_case.name)
        .set_io_data_type(frontend_data_type(test_case.inputs.front().data_type))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_autotune(test_case.autotune);

    fe::graph::Pointwise_attributes attributes;
    attributes.set_name(test_case.name)
        .set_mode(frontend_pointwise_mode(test_case.mode))
        .set_compute_data_type(uses_boolean_compute(test_case.mode)
                                   ? fe::DataType_t::BOOLEAN
                                   : fe::DataType_t::FLOAT)
        .set_alpha(test_case.alpha);
    apply_pointwise_attributes(attributes, test_case.attributes);

    std::vector<std::shared_ptr<fe::graph::Tensor_attributes>> inputs;
    inputs.reserve(test_case.inputs.size());
    for (std::size_t index = 0; index < test_case.inputs.size(); ++index) {
      inputs.push_back(make_frontend_tensor(
          graph_, test_case.inputs[index], "input_" + std::to_string(index)));
    }

    std::shared_ptr<fe::graph::Tensor_attributes> output;
    if (inputs.size() == 1) {
      output = graph_->pointwise(inputs[0], attributes);
    } else if (inputs.size() == 2) {
      output = graph_->pointwise(inputs[0], inputs[1], attributes);
    } else {
      output = graph_->pointwise(inputs[0], inputs[1], inputs[2], attributes);
    }
    output->set_name("output")
        .set_uid(test_case.output.uid)
        .set_data_type(frontend_data_type(test_case.output.data_type))
        .set_dim(test_case.output.dimensions)
        .set_stride(test_case.output.strides)
        .set_output(true);

    check_frontend(graph_->build(handle_, {fe::HeurMode_t::A}),
                   "FlagDNN pointwise graph build");
    std::int64_t workspace_size = 0;
    check_frontend(graph_->get_workspace_size(workspace_size),
                   "FlagDNN pointwise workspace query");
    if (workspace_size < 0) {
      throw std::runtime_error("FlagDNN returned a negative workspace size");
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
      throw std::invalid_argument("FlagDNN pointwise workspace is too small");
    }
    check_frontend(
        graph_->execute(handle_, bindings, workspace, workspace_size, stream),
        "FlagDNN pointwise graph execute");
  }

 private:
  flagdnn::Handle& handle_;
  std::shared_ptr<fe::graph::Graph> graph_;
  std::size_t workspace_size_ = 0;
};

}  // namespace

std::vector<PointwiseTestCase> make_unary_pointwise_cases(
    const PointwiseCaseDefinition& definition) {
  if (!is_unary_mode(definition.mode) || definition.operation_name.empty()) {
    throw std::invalid_argument("unary pointwise definition is invalid");
  }
  const std::vector<Shape>& shapes =
      definition.mode == FLAGDNN_POINTWISE_IDENTITY
          ? identity_shapes()
          : (definition.mode == FLAGDNN_POINTWISE_LOGICAL_NOT
                 ? packed_boolean_shapes()
                 : numeric_shapes());
  const std::vector<flagdnnDataType_t> data_types =
      input_data_types(definition.mode);
  std::vector<PointwiseTestCase> result;
  result.reserve(shapes.size() * data_types.size());
  std::int64_t uid = 1000;
  for (const Shape& shape : shapes) {
    for (const flagdnnDataType_t data_type : data_types) {
      PointwiseTestCase test_case;
      test_case.name = definition.operation_name + "_" +
                       data_type_name(data_type) + "_" + shape_name(shape);
      test_case.mode = definition.mode;
      test_case.inputs = {make_tensor(uid, shape, data_type)};
      test_case.output = make_tensor(
          uid + 1, shape, output_data_type(definition.mode, data_type));
      test_case.input_domains = {definition.input_domain};
      test_case.attributes = definition.attributes;
      test_case.autotune = result.empty() || definition.autotune;
      set_tolerance(test_case);
      validate_pointwise_case(test_case);
      result.push_back(std::move(test_case));
      uid += 2;
    }
  }

  if (definition.mode == FLAGDNN_POINTWISE_NEG) {
    const Shape shape = {2, 3, 4};
    const flagdnnDataType_t data_type = data_types.front();
    PointwiseTestCase test_case;
    test_case.name = definition.operation_name + "_strided_" +
                     data_type_name(data_type) + "_2x3x4";
    test_case.mode = definition.mode;
    test_case.inputs = {
        TestTensor{uid, data_type, shape, {31, 9, 1}},
    };
    test_case.output = TestTensor{
        uid + 1,
        output_data_type(definition.mode, data_type),
        shape,
        {37, 11, 1},
    };
    test_case.input_domains = {definition.input_domain};
    test_case.attributes = definition.attributes;
    test_case.autotune = true;
    test_case.use_host_reference = true;
    set_tolerance(test_case);
    validate_pointwise_case(test_case);
    result.push_back(std::move(test_case));
  }
  return result;
}

std::vector<PointwiseTestCase> make_binary_pointwise_cases(
    const PointwiseCaseDefinition& definition) {
  if (!is_binary_mode(definition.mode) || definition.operation_name.empty()) {
    throw std::invalid_argument("binary pointwise definition is invalid");
  }
  const bool packed_boolean = is_comparison_mode(definition.mode) ||
                              is_logical_binary_mode(definition.mode);
  const auto& shapes = packed_boolean ? packed_boolean_shapes()
                                      : numeric_shapes();
  const auto data_types = input_data_types(definition.mode);
  std::vector<PointwiseTestCase> result;
  std::int64_t uid = 3000;
  const PointwiseInputDomain domain =
      is_comparison_mode(definition.mode)
          ? PointwiseInputDomain::kComparison
          : definition.input_domain;
  for (const Shape& shape : shapes) {
    for (const flagdnnDataType_t data_type : data_types) {
      PointwiseTestCase test_case;
      test_case.name = definition.operation_name + "_" +
                       data_type_name(data_type) + "_" + shape_name(shape);
      test_case.mode = definition.mode;
      test_case.inputs = {make_tensor(uid, shape, data_type),
                          make_tensor(uid + 1, shape, data_type)};
      test_case.output = make_tensor(
          uid + 2, shape, output_data_type(definition.mode, data_type));
      test_case.input_domains = {domain, domain};
      test_case.attributes = definition.attributes;
      test_case.autotune = result.empty() || definition.autotune;
      set_tolerance(test_case);
      validate_pointwise_case(test_case);
      result.push_back(std::move(test_case));
      uid += 3;
    }
  }

  if (definition.mode == FLAGDNN_POINTWISE_SUB) {
    for (const double alpha : {0.5, -2.0}) {
      for (const flagdnnDataType_t data_type : data_types) {
        PointwiseTestCase test_case;
        test_case.name = definition.operation_name + "_" +
                         data_type_name(data_type) + "_2x4x8" +
                         (alpha > 0.0 ? "_alpha_0p5" : "_alpha_neg2");
        test_case.mode = definition.mode;
        test_case.inputs = {make_tensor(uid, {2, 4, 8}, data_type),
                            make_tensor(uid + 1, {2, 4, 8}, data_type)};
        test_case.output = make_tensor(uid + 2, {2, 4, 8}, data_type);
        test_case.input_domains = {domain, domain};
        test_case.attributes = definition.attributes;
        test_case.alpha = alpha;
        test_case.autotune = result.empty() || definition.autotune;
        set_tolerance(test_case);
        validate_pointwise_case(test_case);
        result.push_back(std::move(test_case));
        uid += 3;
      }
    }
  }

  if (definition.mode == FLAGDNN_POINTWISE_MOD) {
    for (const flagdnnDataType_t data_type : data_types) {
      PointwiseTestCase test_case;
      test_case.name = definition.operation_name + "_" +
                       data_type_name(data_type) + "_1x1x6_signed";
      test_case.mode = definition.mode;
      test_case.inputs = {make_tensor(uid, {1, 1, 6}, data_type),
                          make_tensor(uid + 1, {1, 1, 6}, data_type)};
      test_case.output = make_tensor(uid + 2, {1, 1, 6}, data_type);
      test_case.input_domains = {PointwiseInputDomain::kModuloSigned,
                                 PointwiseInputDomain::kModuloSigned};
      test_case.attributes = definition.attributes;
      test_case.autotune = result.empty() || definition.autotune;
      set_tolerance(test_case);
      validate_pointwise_case(test_case);
      result.push_back(std::move(test_case));
      uid += 3;
    }
  }
  return result;
}

std::vector<PointwiseTestCase> make_binary_select_cases(
    const PointwiseCaseDefinition& definition) {
  if (definition.mode != FLAGDNN_POINTWISE_BINARY_SELECT ||
      definition.operation_name.empty()) {
    throw std::invalid_argument("binary-select definition is invalid");
  }
  std::vector<PointwiseTestCase> result;
  std::int64_t uid = 5000;
  for (const Shape& shape : packed_boolean_shapes()) {
    for (const flagdnnDataType_t data_type : kFloatingDataTypes) {
      PointwiseTestCase test_case;
      test_case.name = definition.operation_name + "_" +
                       data_type_name(data_type) + "_" + shape_name(shape);
      test_case.mode = definition.mode;
      test_case.inputs = {make_tensor(uid, shape, data_type),
                          make_tensor(uid + 1, shape, data_type),
                          make_tensor(uid + 2, shape, FLAGDNN_DATA_BOOLEAN)};
      test_case.output = make_tensor(uid + 3, shape, data_type);
      test_case.input_domains = {PointwiseInputDomain::kReal,
                                 PointwiseInputDomain::kReal,
                                 PointwiseInputDomain::kLogical};
      test_case.attributes = definition.attributes;
      test_case.autotune = result.empty() || definition.autotune;
      set_tolerance(test_case);
      validate_pointwise_case(test_case);
      result.push_back(std::move(test_case));
      uid += 4;
    }
  }

  {
    const Shape shape = {2, 3, 4};
    constexpr flagdnnDataType_t data_type = FLAGDNN_DATA_FLOAT32;
    PointwiseTestCase test_case;
    test_case.name = definition.operation_name + "_strided_fp32_2x3x4";
    test_case.mode = definition.mode;
    test_case.inputs = {
        TestTensor{uid, data_type, shape, {31, 9, 1}},
        TestTensor{uid + 1, data_type, shape, {37, 11, 1}},
        make_tensor(uid + 2, shape, FLAGDNN_DATA_BOOLEAN),
    };
    test_case.output =
        TestTensor{uid + 3, data_type, shape, {43, 14, 1}};
    test_case.input_domains = {PointwiseInputDomain::kReal,
                               PointwiseInputDomain::kReal,
                               PointwiseInputDomain::kLogical};
    test_case.attributes = definition.attributes;
    test_case.autotune = true;
    test_case.use_host_reference = true;
    set_tolerance(test_case);
    validate_pointwise_case(test_case);
    result.push_back(std::move(test_case));
  }
  return result;
}

void validate_pointwise_case(const PointwiseTestCase& test_case) {
  if (test_case.name.empty() || test_case.inputs.empty() ||
      test_case.inputs.size() > 3) {
    throw std::invalid_argument("pointwise case name or arity is invalid");
  }
  validate_attributes(test_case.attributes);
  if (!std::isfinite(test_case.alpha) ||
      test_case.absolute_tolerance < 0.0 ||
      test_case.relative_tolerance < 0.0) {
    throw std::invalid_argument("pointwise alpha or tolerance is invalid");
  }
  const std::size_t expected_inputs =
      is_unary_mode(test_case.mode)
          ? 1
          : (is_binary_mode(test_case.mode)
                 ? 2
                 : (test_case.mode == FLAGDNN_POINTWISE_BINARY_SELECT ? 3
                                                                      : 0));
  if (expected_inputs == 0 || test_case.inputs.size() != expected_inputs) {
    throw std::invalid_argument("pointwise case mode and arity disagree");
  }
  if (test_case.input_domains.size() != test_case.inputs.size()) {
    throw std::invalid_argument("pointwise input domains do not match arity");
  }
  if (test_case.alpha != 1.0 && test_case.mode != FLAGDNN_POINTWISE_ADD &&
      test_case.mode != FLAGDNN_POINTWISE_SUB) {
    throw std::invalid_argument("pointwise alpha only applies to ADD or SUB");
  }

  std::unordered_set<std::int64_t> uids;
  for (std::size_t index = 0; index < test_case.inputs.size(); ++index) {
    validate_tensor(test_case.inputs[index],
                    "input " + std::to_string(index));
    if (!uids.emplace(test_case.inputs[index].uid).second) {
      throw std::invalid_argument("pointwise tensor UIDs must be unique");
    }
    if (!broadcasts_to(test_case.inputs[index], test_case.output)) {
      throw std::invalid_argument("pointwise input does not broadcast to output");
    }
  }
  validate_tensor(test_case.output, "output");
  if (!uids.emplace(test_case.output.uid).second) {
    throw std::invalid_argument("pointwise tensor UIDs must be unique");
  }

  const flagdnnDataType_t value_type = test_case.inputs.front().data_type;
  if (test_case.mode == FLAGDNN_POINTWISE_BINARY_SELECT) {
    if (test_case.inputs[1].data_type != value_type ||
        test_case.inputs[2].data_type != FLAGDNN_DATA_BOOLEAN ||
        test_case.output.data_type != value_type) {
      throw std::invalid_argument("binary-select tensor data types are invalid");
    }
  } else if (test_case.mode == FLAGDNN_POINTWISE_LOGICAL_NOT ||
             is_logical_binary_mode(test_case.mode)) {
    for (const TestTensor& input : test_case.inputs) {
      if (input.data_type != FLAGDNN_DATA_BOOLEAN) {
        throw std::invalid_argument("logical pointwise inputs must be BOOLEAN");
      }
    }
    if (test_case.output.data_type != FLAGDNN_DATA_BOOLEAN) {
      throw std::invalid_argument("logical pointwise output must be BOOLEAN");
    }
  } else {
    for (const TestTensor& input : test_case.inputs) {
      if (input.data_type != value_type ||
          input.data_type == FLAGDNN_DATA_BOOLEAN) {
        throw std::invalid_argument("numeric pointwise input types must match");
      }
    }
    const flagdnnDataType_t expected_output =
        is_comparison_mode(test_case.mode) ? FLAGDNN_DATA_BOOLEAN : value_type;
    if (test_case.output.data_type != expected_output) {
      throw std::invalid_argument("pointwise output data type is invalid");
    }
  }
}

int run_unary_pointwise_functional_test(
    int argc,
    char** argv,
    const PointwiseCaseDefinition& definition,
    std::string_view suite_name) {
  const auto cases = make_unary_pointwise_cases(definition);
  return run_pointwise_functional_test(argc, argv, cases, suite_name);
}

int run_binary_pointwise_functional_test(
    int argc,
    char** argv,
    const PointwiseCaseDefinition& definition,
    std::string_view suite_name) {
  const auto cases = make_binary_pointwise_cases(definition);
  return run_pointwise_functional_test(argc, argv, cases, suite_name);
}

int run_binary_select_functional_test(
    int argc,
    char** argv,
    const PointwiseCaseDefinition& definition,
    std::string_view suite_name) {
  const auto cases = make_binary_select_cases(definition);
  return run_pointwise_functional_test(argc, argv, cases, suite_name);
}

std::unique_ptr<PointwiseExecutable> build_flagdnn_pointwise(
    flagdnn::Handle& handle,
    const PointwiseTestCase& test_case) {
  return std::make_unique<FlagdnnPointwiseExecutable>(handle, test_case);
}

}  // namespace flagdnn::testing
