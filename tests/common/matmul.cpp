/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/matmul.hpp"

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
#include <utility>
#include <vector>

namespace flagdnn::testing {
namespace {

namespace fe = ::flagdnn_frontend;
using Shape = std::vector<std::int64_t>;
using ShapePair = std::pair<Shape, Shape>;

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
      throw std::invalid_argument("MatMul shape is invalid or too large");
    }
    result[axis - 1] = stride;
    stride *= dimension;
  }
  return result;
}

Shape broadcast_batch(const Shape& a, const Shape& b) {
  const std::size_t a_rank = a.size() - 2;
  const std::size_t b_rank = b.size() - 2;
  const std::size_t rank = std::max(a_rank, b_rank);
  Shape result(rank, 1);
  for (std::size_t trailing = 0; trailing < rank; ++trailing) {
    const std::int64_t a_dimension =
        trailing < a_rank ? a[a_rank - 1 - trailing] : 1;
    const std::int64_t b_dimension =
        trailing < b_rank ? b[b_rank - 1 - trailing] : 1;
    if (a_dimension != b_dimension && a_dimension != 1 && b_dimension != 1) {
      throw std::invalid_argument("MatMul batch dimensions are incompatible");
    }
    result[rank - 1 - trailing] = std::max(a_dimension, b_dimension);
  }
  return result;
}

Shape output_shape(const Shape& a, const Shape& b) {
  if (a.size() < 2 || b.size() < 2 || a.back() != b[b.size() - 2]) {
    throw std::invalid_argument("MatMul shapes are invalid");
  }
  Shape result = broadcast_batch(a, b);
  result.push_back(a[a.size() - 2]);
  result.push_back(b.back());
  return result;
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
  throw std::invalid_argument("unsupported MatMul data type");
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

void set_tolerance(MatmulTestCase& test_case) {
  const std::int64_t k = test_case.a.dimensions.back();
  if (test_case.output.data_type == FLAGDNN_DATA_FLOAT16) {
    test_case.absolute_tolerance = 5.0e-2;
    test_case.relative_tolerance = 5.0e-2;
  } else if (test_case.output.data_type == FLAGDNN_DATA_BFLOAT16) {
    test_case.absolute_tolerance = 1.0e-1;
    test_case.relative_tolerance = 5.0e-2;
  } else {
    test_case.absolute_tolerance =
        5.0e-3 * std::sqrt(std::max(1.0, static_cast<double>(k) / 512.0));
    test_case.relative_tolerance = 5.0e-3;
  }
}

void validate_tensor(const TestTensor& tensor_specification,
                     std::string_view name) {
  if (tensor_specification.uid <= 0 ||
      tensor_specification.dimensions.size() < 2 ||
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
        std::string(name) + " data type is not supported by MatMul");
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
  throw std::invalid_argument("unsupported FlagDNN MatMul data type");
}

void check_frontend(fe::error_t status, std::string_view operation) {
  if (status.is_bad()) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + status.get_message());
  }
}

class FlagdnnMatmulExecutable final : public MatmulExecutable {
 public:
  FlagdnnMatmulExecutable(flagdnn::Handle& handle,
                          const MatmulTestCase& test_case)
      : handle_(handle), graph_(std::make_shared<fe::graph::Graph>()) {
    validate_matmul_case(test_case);
    const fe::DataType_t io_type = frontend_data_type(test_case.a.data_type);
    graph_->set_name(test_case.name)
        .set_io_data_type(io_type)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_autotune(test_case.autotune);
    const auto a = graph_->tensor(
        fe::graph::Tensor_attributes()
            .set_name("a")
            .set_uid(test_case.a.uid)
            .set_data_type(io_type)
            .set_dim(test_case.a.dimensions)
            .set_stride(test_case.a.strides));
    const auto b = graph_->tensor(
        fe::graph::Tensor_attributes()
            .set_name("b")
            .set_uid(test_case.b.uid)
            .set_data_type(io_type)
            .set_dim(test_case.b.dimensions)
            .set_stride(test_case.b.strides));
    auto output = graph_->matmul(
        a,
        b,
        fe::graph::Matmul_attributes()
            .set_name("matmul")
            .set_compute_data_type(fe::DataType_t::FLOAT));
    output->set_name("output")
        .set_uid(test_case.output.uid)
        .set_data_type(frontend_data_type(test_case.output.data_type))
        .set_dim(test_case.output.dimensions)
        .set_stride(test_case.output.strides)
        .set_output(true);

    check_frontend(graph_->build(handle_, {fe::HeurMode_t::A}),
                   "FlagDNN MatMul graph build");
    std::int64_t workspace_size = 0;
    check_frontend(graph_->get_workspace_size(workspace_size),
                   "FlagDNN MatMul workspace query");
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
      throw std::invalid_argument("FlagDNN MatMul workspace is too small");
    }
    check_frontend(
        graph_->execute(handle_, bindings, workspace, workspace_size, stream),
        "FlagDNN MatMul graph execute");
  }

 private:
  flagdnn::Handle& handle_;
  std::shared_ptr<fe::graph::Graph> graph_;
  std::size_t workspace_size_ = 0;
};

MatmulTestCase make_case(const ShapePair& shapes,
                         flagdnnDataType_t data_type,
                         std::int64_t uid) {
  MatmulTestCase result;
  result.name = "matmul_" + data_type_name(data_type) + "_" +
                shape_name(shapes.first) + "_by_" +
                shape_name(shapes.second);
  result.a = tensor(uid, shapes.first, data_type);
  result.b = tensor(uid + 1, shapes.second, data_type);
  result.output = tensor(
      uid + 2, output_shape(shapes.first, shapes.second), data_type);
  set_tolerance(result);
  return result;
}

}  // namespace

std::vector<MatmulTestCase> make_matmul_cases() {
  const std::array<ShapePair, 8> shapes = {
      ShapePair{{4, 16, 32}, {4, 32, 24}},
      ShapePair{{8, 32, 64}, {8, 64, 32}},
      ShapePair{{16, 32, 128}, {16, 128, 64}},
      ShapePair{{4, 17, 30}, {4, 30, 23}},
      ShapePair{{2, 65, 130}, {2, 130, 33}},
      ShapePair{{1, 64, 64}, {1, 64, 64}},
      ShapePair{{32, 64}, {64, 24}},
      ShapePair{{2, 1, 17, 30}, {3, 30, 23}},
  };
  std::vector<MatmulTestCase> result;
  result.reserve(27);
  std::int64_t uid = 51000;
  for (const ShapePair& shape : shapes) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_case(shape, data_type, uid));
      uid += 3;
    }
  }
  for (const flagdnnDataType_t data_type : kDataTypes) {
    MatmulTestCase strided = make_case(
        {{2, 17, 30}, {2, 30, 23}}, data_type, uid);
    strided.name = "matmul_" + data_type_name(data_type) +
                   "_strided_2x17x30_by_2x30x23";
    strided.a.strides = {600, 31, 1};
    strided.b.strides = {800, 1, 32};
    strided.output.strides = {500, 25, 1};
    result.push_back(std::move(strided));
    uid += 3;
  }
  result.front().autotune = true;
  for (const MatmulTestCase& test_case : result) {
    validate_matmul_case(test_case);
  }
  return result;
}

void validate_matmul_case(const MatmulTestCase& test_case) {
  if (test_case.name.empty() || test_case.a.uid == test_case.b.uid ||
      test_case.a.uid == test_case.output.uid ||
      test_case.b.uid == test_case.output.uid ||
      !std::isfinite(test_case.absolute_tolerance) ||
      !std::isfinite(test_case.relative_tolerance) ||
      test_case.absolute_tolerance < 0.0 ||
      test_case.relative_tolerance < 0.0) {
    throw std::invalid_argument("MatMul case metadata is invalid");
  }
  validate_tensor(test_case.a, "MatMul A");
  validate_tensor(test_case.b, "MatMul B");
  validate_tensor(test_case.output, "MatMul output");
  if (test_case.a.data_type != test_case.b.data_type ||
      test_case.a.data_type != test_case.output.data_type ||
      test_case.output.dimensions !=
          output_shape(test_case.a.dimensions, test_case.b.dimensions)) {
    throw std::invalid_argument("MatMul data types or output shape are invalid");
  }
}


std::unique_ptr<MatmulExecutable> build_flagdnn_matmul(
    flagdnn::Handle& handle,
    const MatmulTestCase& test_case) {
  return std::make_unique<FlagdnnMatmulExecutable>(handle, test_case);
}

}  // namespace flagdnn::testing
