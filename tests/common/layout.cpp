/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/layout.hpp"

#include <flagdnn/flagdnn.hpp>
#include <flagdnn_frontend.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
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

constexpr std::array<flagdnnDataType_t, 3> kDataTypes = {
    FLAGDNN_DATA_FLOAT32,
    FLAGDNN_DATA_FLOAT16,
    FLAGDNN_DATA_BFLOAT16,
};

std::vector<std::int64_t> contiguous_strides(const Shape& dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    const std::int64_t dimension = dimensions[axis - 1];
    if (dimension <= 0 ||
        stride > std::numeric_limits<std::int64_t>::max() / dimension) {
      throw std::invalid_argument("Layout shape is invalid or too large");
    }
    result[axis - 1] = stride;
    stride *= dimension;
  }
  return result;
}

std::size_t element_count(const Shape& dimensions) {
  return std::accumulate(
      dimensions.begin(),
      dimensions.end(),
      std::size_t{1},
      [](std::size_t result, std::int64_t dimension) {
        return result * static_cast<std::size_t>(dimension);
      });
}

TestTensor tensor(std::int64_t uid,
                  Shape dimensions,
                  flagdnnDataType_t data_type) {
  auto strides = contiguous_strides(dimensions);
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
  throw std::invalid_argument("unsupported Layout data type");
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

void validate_tensor(const TestTensor& tensor_specification,
                     std::string_view name) {
  if (tensor_specification.uid <= 0 ||
      tensor_specification.dimensions.empty() ||
      tensor_specification.dimensions.size() > 8 ||
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
        std::string(name) + " data type is not supported by Layout");
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
  throw std::invalid_argument("unsupported FlagDNN Layout data type");
}

void check_frontend(fe::error_t status, std::string_view operation) {
  if (status.is_bad()) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + status.get_message());
  }
}

class FlagdnnLayoutExecutable final : public LayoutExecutable {
 public:
  FlagdnnLayoutExecutable(flagdnn::Handle& handle,
                          const LayoutTestCase& test_case)
      : handle_(handle), graph_(std::make_shared<fe::graph::Graph>()) {
    validate_layout_case(test_case);
    const fe::DataType_t io_type = frontend_data_type(test_case.input.data_type);
    graph_->set_name(test_case.name)
        .set_io_data_type(io_type)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_autotune(test_case.autotune);
    const auto input = graph_->tensor(
        fe::graph::Tensor_attributes()
            .set_name("input")
            .set_uid(test_case.input.uid)
            .set_data_type(io_type)
            .set_dim(test_case.input.dimensions)
            .set_stride(test_case.input.strides));

    std::shared_ptr<fe::graph::Tensor_attributes> output;
    switch (test_case.operation) {
      case LayoutOperation::kReshape:
        output = graph_->reshape(
            input,
            fe::graph::Reshape_attributes()
                .set_name("reshape")
                .set_compute_data_type(fe::DataType_t::FLOAT)
                .set_dim(test_case.output.dimensions)
                .set_stride(test_case.output.strides)
                .set_reshape_mode(fe::ReshapeMode_t::LOGICAL));
        break;
      case LayoutOperation::kTranspose:
        output = graph_->transpose(
            input,
            fe::graph::Transpose_attributes()
                .set_name("transpose")
                .set_compute_data_type(fe::DataType_t::FLOAT)
                .set_permutation(test_case.permutation));
        break;
      case LayoutOperation::kSlice:
        output = graph_->slice(
            input,
            fe::graph::Slice_attributes()
                .set_name("slice")
                .set_compute_data_type(fe::DataType_t::FLOAT)
                .set_slices(test_case.slices)
                .set_strides(test_case.slice_strides));
        break;
    }
    output->set_name("output")
        .set_uid(test_case.output.uid)
        .set_data_type(frontend_data_type(test_case.output.data_type))
        .set_dim(test_case.output.dimensions)
        .set_stride(test_case.output.strides)
        .set_output(true);

    check_frontend(graph_->build(handle_, {fe::HeurMode_t::A}),
                   "FlagDNN Layout graph build");
    std::int64_t workspace_size = 0;
    check_frontend(graph_->get_workspace_size(workspace_size),
                   "FlagDNN Layout workspace query");
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
      throw std::invalid_argument("FlagDNN Layout workspace is too small");
    }
    check_frontend(
        graph_->execute(handle_, bindings, workspace, workspace_size, stream),
        "FlagDNN Layout graph execute");
  }

 private:
  flagdnn::Handle& handle_;
  std::shared_ptr<fe::graph::Graph> graph_;
  std::size_t workspace_size_ = 0;
};

std::vector<LayoutTestCase> reshape_cases() {
  const std::array<std::pair<Shape, Shape>, 3> shapes = {
      std::pair{Shape{2, 3, 4}, Shape{6, 4}},
      std::pair{Shape{1, 8, 16}, Shape{4, 32}},
      std::pair{Shape{4, 5, 6}, Shape{2, 3, 20}},
  };
  std::vector<LayoutTestCase> result;
  std::int64_t uid = 53000;
  for (const auto& [input_shape, output_shape] : shapes) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      LayoutTestCase test_case;
      test_case.name = "reshape_" + data_type_name(data_type) + "_" +
                       shape_name(input_shape) + "_to_" +
                       shape_name(output_shape);
      test_case.operation = LayoutOperation::kReshape;
      test_case.input = tensor(uid, input_shape, data_type);
      test_case.output = tensor(uid + 1, output_shape, data_type);
      test_case.autotune = result.empty();
      result.push_back(std::move(test_case));
      uid += 2;
    }
  }
  return result;
}

std::vector<LayoutTestCase> transpose_cases() {
  const std::array<std::pair<Shape, Shape>, 3> shapes = {
      std::pair{Shape{2, 3, 4}, Shape{2, 0, 1}},
      std::pair{Shape{1, 8, 16}, Shape{0, 2, 1}},
      std::pair{Shape{2, 3, 4, 5}, Shape{0, 2, 3, 1}},
  };
  std::vector<LayoutTestCase> result;
  std::int64_t uid = 55000;
  for (const auto& [input_shape, permutation] : shapes) {
    const Shape input_strides = contiguous_strides(input_shape);
    Shape output_shape(input_shape.size());
    Shape output_strides(input_shape.size());
    for (std::size_t axis = 0; axis < permutation.size(); ++axis) {
      const std::size_t source =
          static_cast<std::size_t>(permutation[axis]);
      output_shape[axis] = input_shape[source];
      output_strides[axis] = input_strides[source];
    }
    for (const flagdnnDataType_t data_type : kDataTypes) {
      LayoutTestCase test_case;
      test_case.name = "transpose_" + data_type_name(data_type) + "_" +
                       shape_name(input_shape);
      test_case.operation = LayoutOperation::kTranspose;
      test_case.input = tensor(uid, input_shape, data_type);
      test_case.output =
          {uid + 1, data_type, output_shape, output_strides};
      test_case.permutation = permutation;
      test_case.autotune = result.empty();
      result.push_back(std::move(test_case));
      uid += 2;
    }
  }
  return result;
}

struct SliceDefinition {
  Shape input;
  std::vector<std::pair<std::int64_t, std::int64_t>> slices;
  Shape strides;
};

std::vector<LayoutTestCase> slice_cases() {
  const std::array<SliceDefinition, 3> shapes = {
      SliceDefinition{{2, 4, 5}, {{0, 2}, {1, 4}, {0, 5}}, {1, 2, 1}},
      SliceDefinition{{4, 6, 8}, {{1, 4}, {0, 6}, {2, 8}}, {1, 2, 3}},
      SliceDefinition{{3, 5, 7, 2},
                      {{0, 3}, {1, 5}, {0, 7}, {0, 2}},
                      {1, 2, 1, 1}},
  };
  std::vector<LayoutTestCase> result;
  std::int64_t uid = 57000;
  for (std::size_t case_index = 0; case_index < shapes.size(); ++case_index) {
    const SliceDefinition& definition = shapes[case_index];
    const Shape input_strides = contiguous_strides(definition.input);
    Shape output_shape(definition.input.size());
    Shape output_strides(definition.input.size());
    for (std::size_t axis = 0; axis < definition.input.size(); ++axis) {
      const auto [start, limit] = definition.slices[axis];
      const std::int64_t step = definition.strides[axis];
      output_shape[axis] = (limit - start + step - 1) / step;
      output_strides[axis] = input_strides[axis] * step;
    }
    for (const flagdnnDataType_t data_type : kDataTypes) {
      LayoutTestCase test_case;
      test_case.name = "slice_" + data_type_name(data_type) + "_case" +
                       std::to_string(case_index) + "_" +
                       shape_name(definition.input);
      test_case.operation = LayoutOperation::kSlice;
      test_case.input = tensor(uid, definition.input, data_type);
      test_case.output =
          {uid + 1, data_type, output_shape, output_strides};
      test_case.slices = definition.slices;
      test_case.slice_strides = definition.strides;
      test_case.autotune = result.empty();
      result.push_back(std::move(test_case));
      uid += 2;
    }
  }
  return result;
}

}  // namespace

std::vector<LayoutTestCase> make_layout_cases(LayoutOperation operation) {
  std::vector<LayoutTestCase> result;
  switch (operation) {
    case LayoutOperation::kReshape:
      result = reshape_cases();
      break;
    case LayoutOperation::kTranspose:
      result = transpose_cases();
      break;
    case LayoutOperation::kSlice:
      result = slice_cases();
      break;
  }
  for (const LayoutTestCase& test_case : result) {
    validate_layout_case(test_case);
  }
  return result;
}

void validate_layout_case(const LayoutTestCase& test_case) {
  if (test_case.name.empty() || test_case.input.uid == test_case.output.uid) {
    throw std::invalid_argument("Layout case metadata is invalid");
  }
  validate_tensor(test_case.input, "Layout input");
  validate_tensor(test_case.output, "Layout output");
  if (test_case.input.data_type != test_case.output.data_type) {
    throw std::invalid_argument("Layout input/output data types must match");
  }

  const std::size_t rank = test_case.input.dimensions.size();
  switch (test_case.operation) {
    case LayoutOperation::kReshape:
      if (element_count(test_case.input.dimensions) !=
              element_count(test_case.output.dimensions) ||
          !test_case.permutation.empty() || !test_case.slices.empty() ||
          !test_case.slice_strides.empty()) {
        throw std::invalid_argument("Reshape case attributes are invalid");
      }
      break;
    case LayoutOperation::kTranspose: {
      if (test_case.output.dimensions.size() != rank ||
          test_case.permutation.size() != rank ||
          !test_case.slices.empty() || !test_case.slice_strides.empty()) {
        throw std::invalid_argument("Transpose case attributes are invalid");
      }
      std::unordered_set<std::int64_t> axes;
      for (std::size_t axis = 0; axis < rank; ++axis) {
        const std::int64_t source = test_case.permutation[axis];
        if (source < 0 || source >= static_cast<std::int64_t>(rank) ||
            !axes.insert(source).second ||
            test_case.output.dimensions[axis] !=
                test_case.input.dimensions[static_cast<std::size_t>(source)]) {
          throw std::invalid_argument("Transpose permutation is invalid");
        }
      }
      break;
    }
    case LayoutOperation::kSlice:
      if (test_case.output.dimensions.size() != rank ||
          test_case.slices.size() != rank ||
          test_case.slice_strides.size() != rank ||
          !test_case.permutation.empty()) {
        throw std::invalid_argument("Slice case attributes are invalid");
      }
      for (std::size_t axis = 0; axis < rank; ++axis) {
        const auto [start, limit] = test_case.slices[axis];
        const std::int64_t step = test_case.slice_strides[axis];
        if (start < 0 || limit <= start ||
            limit > test_case.input.dimensions[axis] || step <= 0 ||
            test_case.output.dimensions[axis] !=
                (limit - start + step - 1) / step) {
          throw std::invalid_argument("Slice range is invalid");
        }
      }
      break;
  }
}

std::unique_ptr<LayoutExecutable> build_flagdnn_layout(
    flagdnn::Handle& handle,
    const LayoutTestCase& test_case) {
  return std::make_unique<FlagdnnLayoutExecutable>(handle, test_case);
}

}  // namespace flagdnn::testing
