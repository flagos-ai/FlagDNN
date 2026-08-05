/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/add.hpp"

#include <flagdnn/flagdnn.hpp>
#include <flagdnn_frontend.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
      break;
  }
  throw std::invalid_argument("Add requires a floating data type");
}

void validate_tensor(const TestTensor& tensor, std::string_view name) {
  if (tensor.uid <= 0) {
    throw std::invalid_argument(std::string(name) + " UID must be positive");
  }
  if (tensor.dimensions.empty() ||
      tensor.dimensions.size() != tensor.strides.size()) {
    throw std::invalid_argument(
        std::string(name) + " dimensions and strides are invalid");
  }
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0) {
      throw std::invalid_argument(
          std::string(name) + " dimensions and strides must be positive");
    }
  }
  (void)frontend_data_type(tensor.data_type);
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

std::shared_ptr<fe::graph::Tensor_attributes> make_tensor(
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

std::vector<std::int64_t> contiguous_strides(const Shape& dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    result[axis - 1] = stride;
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

Shape broadcast_shape(const Shape& left, const Shape& right) {
  const std::size_t rank = std::max(left.size(), right.size());
  Shape result(rank, 1);
  for (std::size_t trailing = 0; trailing < rank; ++trailing) {
    const std::int64_t left_dimension =
        trailing < left.size() ? left[left.size() - 1 - trailing] : 1;
    const std::int64_t right_dimension =
        trailing < right.size() ? right[right.size() - 1 - trailing] : 1;
    result[rank - 1 - trailing] =
        std::max(left_dimension, right_dimension);
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
      return "bf16";
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
    case FLAGDNN_DATA_BOOLEAN:
      break;
  }
  return "invalid";
}

TestTensor test_tensor(std::int64_t uid,
                       Shape dimensions,
                       flagdnnDataType_t data_type) {
  TestTensor result;
  result.uid = uid;
  result.data_type = data_type;
  result.dimensions = std::move(dimensions);
  result.strides = pointwise_strides(result.dimensions);
  return result;
}

AddTestCase make_case(std::string_view feature,
                      const Shape& left,
                      const Shape& right,
                      flagdnnDataType_t data_type,
                      std::int64_t uid,
                      double alpha = 1.0,
                      bool autotune = false) {
  AddTestCase result;
  result.name = "add_" + std::string(feature) + "_" +
                data_type_name(data_type);
  result.left = test_tensor(uid, left, data_type);
  result.right = test_tensor(uid + 1, right, data_type);
  result.output =
      test_tensor(uid + 2, broadcast_shape(left, right), data_type);
  result.alpha = alpha;
  result.absolute_tolerance =
      data_type == FLAGDNN_DATA_FLOAT32
          ? 1.0e-6
          : (data_type == FLAGDNN_DATA_BFLOAT16 ? 5.0e-2 : 2.0e-2);
  result.relative_tolerance =
      data_type == FLAGDNN_DATA_FLOAT32 ? 1.0e-6 : 1.0e-2;
  result.autotune = autotune;
  return result;
}

class FlagdnnAddExecutable final : public AddExecutable {
 public:
  FlagdnnAddExecutable(flagdnn::Handle& handle,
                       const AddTestCase& test_case)
      : handle_(handle), graph_(std::make_shared<fe::graph::Graph>()) {
    validate_add_case(test_case);

    graph_->set_name(test_case.name)
        .set_io_data_type(frontend_data_type(test_case.left.data_type))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_autotune(test_case.autotune);

    const auto left = make_tensor(graph_, test_case.left, "left");
    const auto right = make_tensor(graph_, test_case.right, "right");
    auto output = graph_->pointwise(
        left,
        right,
        fe::graph::Pointwise_attributes()
            .set_name("add")
            .set_mode(fe::PointwiseMode_t::ADD)
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_alpha(test_case.alpha));
    output->set_name("output")
        .set_uid(test_case.output.uid)
        .set_data_type(frontend_data_type(test_case.output.data_type))
        .set_dim(test_case.output.dimensions)
        .set_stride(test_case.output.strides)
        .set_output(true);

    check_frontend(graph_->build(handle_, {fe::HeurMode_t::A}),
                   "FlagDNN Add graph build");
    std::int64_t workspace_size = 0;
    check_frontend(graph_->get_workspace_size(workspace_size),
                   "FlagDNN Add workspace query");
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
      throw std::invalid_argument("FlagDNN Add workspace is too small");
    }
    check_frontend(
        graph_->execute(
            handle_, bindings, workspace, workspace_size, stream),
        "FlagDNN Add graph execute");
  }

 private:
  flagdnn::Handle& handle_;
  std::shared_ptr<fe::graph::Graph> graph_;
  std::size_t workspace_size_ = 0;
};

}  // namespace

std::vector<AddTestCase> make_add_cases() {
  std::vector<AddTestCase> result;
  result.push_back(make_case("contiguous_autotune",
                             {2, 4, 8},
                             {2, 4, 8},
                             FLAGDNN_DATA_FLOAT32,
                             100,
                             1.0,
                             true));
  result.push_back(make_case("odd_extent",
                             {3, 5, 7},
                             {3, 5, 7},
                             FLAGDNN_DATA_FLOAT16,
                             110));
  result.push_back(make_case("nhwc_layout",
                             {2, 3, 5, 7},
                             {2, 3, 5, 7},
                             FLAGDNN_DATA_BFLOAT16,
                             120));
  result.push_back(make_case("alpha_half",
                             {2, 4, 8},
                             {2, 4, 8},
                             FLAGDNN_DATA_FLOAT32,
                             150,
                             0.5));
  result.push_back(make_case("alpha_negative",
                             {2, 4, 8},
                             {2, 4, 8},
                             FLAGDNN_DATA_BFLOAT16,
                             160,
                             -2.0));
  for (const AddTestCase& test_case : result) {
    validate_add_case(test_case);
  }
  return result;
}

void validate_add_case(const AddTestCase& test_case) {
  if (test_case.name.empty()) {
    throw std::invalid_argument("Add test case name must not be empty");
  }
  validate_tensor(test_case.left, "left");
  validate_tensor(test_case.right, "right");
  validate_tensor(test_case.output, "output");
  if (test_case.left.uid == test_case.right.uid ||
      test_case.left.uid == test_case.output.uid ||
      test_case.right.uid == test_case.output.uid) {
    throw std::invalid_argument("Add tensor UIDs must be unique");
  }
  if (test_case.left.data_type != test_case.right.data_type ||
      test_case.left.data_type != test_case.output.data_type) {
    throw std::invalid_argument("Add tensor data types must match");
  }
  if (!broadcasts_to(test_case.left, test_case.output) ||
      !broadcasts_to(test_case.right, test_case.output)) {
    throw std::invalid_argument("Add input shapes do not broadcast to output");
  }
  if (!std::isfinite(test_case.alpha)) {
    throw std::invalid_argument("Add alpha must be finite");
  }
  if (test_case.absolute_tolerance < 0.0 ||
      test_case.relative_tolerance < 0.0) {
    throw std::invalid_argument("Add tolerances must be non-negative");
  }
}

std::unique_ptr<AddExecutable> build_flagdnn_add(
    flagdnn::Handle& handle,
    const AddTestCase& test_case) {
  return std::make_unique<FlagdnnAddExecutable>(handle, test_case);
}

}  // namespace flagdnn::testing
