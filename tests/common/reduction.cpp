/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/reduction.hpp"

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

constexpr std::array<flagdnnDataType_t, 3> kDataTypes = {
    FLAGDNN_DATA_FLOAT32,
    FLAGDNN_DATA_FLOAT16,
    FLAGDNN_DATA_BFLOAT16,
};

constexpr std::array<flagdnnReductionMode_t, 3> kModes = {
    FLAGDNN_REDUCTION_ADD,
    FLAGDNN_REDUCTION_AVG,
    FLAGDNN_REDUCTION_MUL,
};

std::vector<std::int64_t> contiguous_strides(const Shape& dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    const std::int64_t dimension = dimensions[axis - 1];
    if (dimension <= 0 ||
        stride > std::numeric_limits<std::int64_t>::max() / dimension) {
      throw std::invalid_argument("reduction shape is invalid or too large");
    }
    result[axis - 1] = stride;
    stride *= dimension;
  }
  return result;
}

TestTensor tensor(std::int64_t uid,
                  Shape dimensions,
                  flagdnnDataType_t data_type,
                  std::size_t binding_byte_offset = 0) {
  auto strides = contiguous_strides(dimensions);
  return {uid,
          data_type,
          std::move(dimensions),
          std::move(strides),
          binding_byte_offset};
}

std::int64_t binding_alignment(const TestTensor& tensor_specification) {
  if (tensor_specification.binding_byte_offset == 0) {
    return 16;
  }
  std::size_t alignment = 1;
  while (alignment < 16 &&
         tensor_specification.binding_byte_offset % (alignment * 2) == 0) {
    alignment *= 2;
  }
  return static_cast<std::int64_t>(alignment);
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
  throw std::invalid_argument("unsupported Reduction data type");
}

std::string mode_name(flagdnnReductionMode_t mode) {
  switch (mode) {
    case FLAGDNN_REDUCTION_ADD:
      return "sum";
    case FLAGDNN_REDUCTION_AVG:
      return "avg";
    case FLAGDNN_REDUCTION_MUL:
      return "mul";
  }
  throw std::invalid_argument("unsupported Reduction mode");
}

void set_tolerance(ReductionTestCase& test_case) {
  switch (test_case.output.data_type) {
    case FLAGDNN_DATA_FLOAT32:
      test_case.absolute_tolerance = 2.0e-5;
      test_case.relative_tolerance = 1.0e-5;
      return;
    case FLAGDNN_DATA_FLOAT16:
      test_case.absolute_tolerance = 5.0e-2;
      test_case.relative_tolerance = 1.0e-2;
      return;
    case FLAGDNN_DATA_BFLOAT16:
      test_case.absolute_tolerance = 8.0e-2;
      test_case.relative_tolerance = 1.0e-2;
      return;
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
    case FLAGDNN_DATA_BOOLEAN:
      break;
  }
  throw std::invalid_argument("unsupported Reduction output data type");
}

void validate_tensor(const TestTensor& tensor_specification,
                     std::string_view name,
                     bool allow_scalar) {
  if (tensor_specification.uid <= 0 ||
      tensor_specification.dimensions.size() !=
          tensor_specification.strides.size() ||
      tensor_specification.dimensions.size() > 8 ||
      (!allow_scalar && tensor_specification.dimensions.empty())) {
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
        std::string(name) + " data type is not supported by Reduction");
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
  throw std::invalid_argument("unsupported FlagDNN Reduction data type");
}

fe::ReductionMode_t frontend_reduction_mode(flagdnnReductionMode_t mode) {
  switch (mode) {
    case FLAGDNN_REDUCTION_ADD:
      return fe::ReductionMode_t::ADD;
    case FLAGDNN_REDUCTION_AVG:
      return fe::ReductionMode_t::AVG;
    case FLAGDNN_REDUCTION_MUL:
      return fe::ReductionMode_t::MUL;
  }
  throw std::invalid_argument("unsupported FlagDNN Reduction mode");
}

void check_frontend(fe::error_t status, std::string_view operation) {
  if (status.is_bad()) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + status.get_message());
  }
}

class FlagdnnReductionExecutable final : public ReductionExecutable {
 public:
  FlagdnnReductionExecutable(flagdnn::Handle& handle,
                             const ReductionTestCase& test_case)
      : handle_(handle), graph_(std::make_shared<fe::graph::Graph>()) {
    validate_reduction_case(test_case);
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
            .set_stride(test_case.input.strides)
            .set_alignment(binding_alignment(test_case.input)));
    auto output = graph_->reduction(
        input,
        fe::graph::Reduction_attributes()
            .set_name("reduction")
            .set_mode(frontend_reduction_mode(test_case.mode))
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_axis(test_case.axis)
            .set_keep_dimensions(test_case.keep_dimensions));
    output->set_name("output")
        .set_uid(test_case.output.uid)
        .set_data_type(frontend_data_type(test_case.output.data_type))
        .set_dim(test_case.output.dimensions)
        .set_stride(test_case.output.strides)
        .set_output(true);

    check_frontend(graph_->build(handle_, {fe::HeurMode_t::A}),
                   "FlagDNN Reduction graph build");
    std::int64_t workspace_size = 0;
    check_frontend(graph_->get_workspace_size(workspace_size),
                   "FlagDNN Reduction workspace query");
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
      throw std::invalid_argument("FlagDNN Reduction workspace is too small");
    }
    check_frontend(
        graph_->execute(handle_, bindings, workspace, workspace_size, stream),
        "FlagDNN Reduction graph execute");
  }

 private:
  flagdnn::Handle& handle_;
  std::shared_ptr<fe::graph::Graph> graph_;
  std::size_t workspace_size_ = 0;
};

ReductionTestCase regular_case(flagdnnReductionMode_t mode,
                               flagdnnDataType_t data_type,
                               std::int64_t uid) {
  ReductionTestCase result;
  result.name = "reduction_" + mode_name(mode) + "_" +
                data_type_name(data_type) + "_axis1_keepdim_2x4x8x8";
  result.input = tensor(uid, {2, 4, 8, 8}, data_type);
  result.output = tensor(uid + 1, {2, 1, 8, 8}, data_type);
  result.mode = mode;
  result.axis = 1;
  result.keep_dimensions = true;
  set_tolerance(result);
  return result;
}

ReductionTestCase channels_last_case(flagdnnReductionMode_t mode,
                                     flagdnnDataType_t data_type,
                                     std::int64_t uid) {
  ReductionTestCase result;
  result.name = "reduction_" + mode_name(mode) + "_" +
                data_type_name(data_type) +
                "_channels_last_axis1_keepdim_2x3x5x5";
  result.input = {uid, data_type, {2, 3, 5, 5}, {75, 1, 15, 3}};
  result.output = tensor(uid + 1, {2, 1, 5, 5}, data_type);
  result.mode = mode;
  result.axis = 1;
  result.keep_dimensions = true;
  set_tolerance(result);
  return result;
}

ReductionTestCase unaligned_case(flagdnnDataType_t data_type,
                                 std::int64_t uid) {
  ReductionTestCase result;
  result.name = "reduction_sum_" + data_type_name(data_type) +
                "_unaligned_input_entrance_axis1_2x4x8x8";
  result.input = {uid,
                  data_type,
                  {2, 4, 8, 8},
                  {288, 72, 9, 1},
                  data_type == FLAGDNN_DATA_FLOAT32 ? 4U : 2U};
  result.output = tensor(uid + 1, {2, 1, 8, 8}, data_type);
  result.mode = FLAGDNN_REDUCTION_ADD;
  result.axis = 1;
  result.keep_dimensions = true;
  set_tolerance(result);
  return result;
}

}  // namespace

std::vector<ReductionTestCase> make_reduction_cases() {
  std::vector<ReductionTestCase> result;
  result.reserve(23);

  ReductionTestCase last_axis;
  last_axis.name = "reduction_sum_fp32_7x256";
  last_axis.input = tensor(6, {7, 256}, FLAGDNN_DATA_FLOAT32);
  last_axis.output = tensor(7, {7}, FLAGDNN_DATA_FLOAT32);
  last_axis.mode = FLAGDNN_REDUCTION_ADD;
  last_axis.axis = -1;
  last_axis.keep_dimensions = false;
  last_axis.autotune = true;
  set_tolerance(last_axis);
  result.push_back(std::move(last_axis));

  std::int64_t uid = 100;
  for (const flagdnnReductionMode_t mode : kModes) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      ReductionTestCase regular = regular_case(mode, data_type, uid);
      regular.autotune = data_type == kDataTypes.front() &&
                         mode != FLAGDNN_REDUCTION_ADD;
      result.push_back(std::move(regular));
      uid += 2;
      result.push_back(channels_last_case(mode, data_type, uid));
      uid += 2;
    }
  }
  for (const flagdnnDataType_t data_type : kDataTypes) {
    result.push_back(unaligned_case(data_type, uid));
    uid += 2;
  }

  ReductionTestCase scalar;
  scalar.name = "reduction_sum_fp32_scalar_8";
  scalar.input = tensor(uid, {8}, FLAGDNN_DATA_FLOAT32);
  scalar.output = tensor(uid + 1, {}, FLAGDNN_DATA_FLOAT32);
  scalar.mode = FLAGDNN_REDUCTION_ADD;
  scalar.axis = 0;
  scalar.keep_dimensions = false;
  set_tolerance(scalar);
  result.push_back(std::move(scalar));

  for (const ReductionTestCase& test_case : result) {
    validate_reduction_case(test_case);
  }
  return result;
}

void validate_reduction_case(const ReductionTestCase& test_case) {
  if (test_case.name.empty() || test_case.input.uid == test_case.output.uid ||
      !std::isfinite(test_case.absolute_tolerance) ||
      !std::isfinite(test_case.relative_tolerance) ||
      test_case.absolute_tolerance < 0.0 ||
      test_case.relative_tolerance < 0.0) {
    throw std::invalid_argument("Reduction case metadata is invalid");
  }
  validate_tensor(test_case.input, "Reduction input", false);
  validate_tensor(test_case.output, "Reduction output", true);
  if (test_case.input.data_type != test_case.output.data_type) {
    throw std::invalid_argument("Reduction input/output data types must match");
  }
  (void)mode_name(test_case.mode);

  std::int32_t axis = test_case.axis;
  const std::int32_t rank =
      static_cast<std::int32_t>(test_case.input.dimensions.size());
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= rank) {
    throw std::invalid_argument("Reduction axis is out of range");
  }
  Shape expected = test_case.input.dimensions;
  if (test_case.keep_dimensions) {
    expected[static_cast<std::size_t>(axis)] = 1;
  } else {
    expected.erase(expected.begin() + axis);
  }
  if (test_case.output.dimensions != expected) {
    throw std::invalid_argument("Reduction output shape is invalid");
  }
}

std::unique_ptr<ReductionExecutable> build_flagdnn_reduction(
    flagdnn::Handle& handle,
    const ReductionTestCase& test_case) {
  return std::make_unique<FlagdnnReductionExecutable>(handle, test_case);
}

}  // namespace flagdnn::testing
