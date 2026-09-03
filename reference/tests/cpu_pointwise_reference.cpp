/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "reference/cpu/pointwise.hpp"

#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

namespace cfe = cudnn_frontend;
namespace cpu = flagdnn::reference::cpu;

using Shape = std::vector<std::int64_t>;

constexpr std::array<flagdnnPointwiseMode_t, 4> kSupportedModes = {
    FLAGDNN_POINTWISE_DIV,
    FLAGDNN_POINTWISE_POW,
    FLAGDNN_POINTWISE_MOD,
    FLAGDNN_POINTWISE_CMP_EQ,
};

struct BroadcastCase {
  std::string name;
  Shape left_dimensions;
  Shape right_dimensions;
  Shape output_dimensions;
  bool materialize_for_cudnn = false;
};

struct TensorLayout {
  Shape dimensions;
  Shape strides;
};

void check_cuda(cudaError_t status, std::string_view operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + " failed: " +
                             cudaGetErrorString(status));
  }
}

void check_cudnn(cudnnStatus_t status, std::string_view operation) {
  if (status != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed: " +
                             cudnnGetErrorString(status));
  }
}

void check_frontend(cfe::error_t status, std::string_view operation) {
  if (status.is_bad()) {
    throw std::runtime_error(std::string(operation) + " failed: " +
                             status.get_message());
  }
}

class CudnnHandle {
 public:
  CudnnHandle() { check_cudnn(cudnnCreate(&handle_), "cudnnCreate"); }

  ~CudnnHandle() {
    if (handle_ != nullptr) {
      (void)cudnnDestroy(handle_);
    }
  }

  CudnnHandle(const CudnnHandle&) = delete;
  CudnnHandle& operator=(const CudnnHandle&) = delete;

  [[nodiscard]] cudnnHandle_t get() const noexcept { return handle_; }

 private:
  cudnnHandle_t handle_ = nullptr;
};

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t byte_count) : byte_count_(byte_count) {
    if (byte_count_ != 0) {
      check_cuda(cudaMalloc(&pointer_, byte_count_), "cudaMalloc");
    }
  }

  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] void* get() const noexcept { return pointer_; }
  [[nodiscard]] std::size_t size() const noexcept { return byte_count_; }

  void copy_from_host(const void* source, std::size_t byte_count) {
    if (byte_count != byte_count_ || (byte_count != 0 && source == nullptr)) {
      throw std::invalid_argument("host-to-device copy size is invalid");
    }
    if (byte_count != 0) {
      check_cuda(cudaMemcpy(pointer_, source, byte_count,
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(host-to-device)");
    }
  }

  void copy_to_host(void* destination, std::size_t byte_count) const {
    if (byte_count != byte_count_ ||
        (byte_count != 0 && destination == nullptr)) {
      throw std::invalid_argument("device-to-host copy size is invalid");
    }
    if (byte_count != 0) {
      check_cuda(cudaMemcpy(destination, pointer_, byte_count,
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(device-to-host)");
    }
  }

 private:
  void* pointer_ = nullptr;
  std::size_t byte_count_ = 0;
};

std::size_t element_count(std::span<const std::int64_t> dimensions) {
  if (dimensions.empty()) {
    throw std::invalid_argument("test tensor dimensions must not be empty");
  }
  std::size_t result = 1;
  for (const std::int64_t dimension : dimensions) {
    if (dimension <= 0) {
      throw std::invalid_argument("test tensor dimensions must be positive");
    }
    const std::size_t value = static_cast<std::size_t>(dimension);
    if (result > std::numeric_limits<std::size_t>::max() / value) {
      throw std::overflow_error("test tensor element count overflows");
    }
    result *= value;
  }
  return result;
}

Shape contiguous_strides(std::span<const std::int64_t> dimensions) {
  Shape result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    const std::size_t current = axis - 1;
    result[current] = stride;
    if (dimensions[current] <= 0 ||
        stride > std::numeric_limits<std::int64_t>::max() /
                     dimensions[current]) {
      throw std::invalid_argument("cuDNN test tensor shape is invalid");
    }
    stride *= dimensions[current];
  }
  return result;
}

std::vector<float> materialize_broadcast(
    std::span<const float> values,
    std::span<const std::int64_t> input_dimensions,
    std::span<const std::int64_t> output_dimensions) {
  if (values.size() != element_count(input_dimensions) ||
      input_dimensions.size() > output_dimensions.size()) {
    throw std::invalid_argument("broadcast materialization input is invalid");
  }
  const std::size_t leading =
      output_dimensions.size() - input_dimensions.size();
  for (std::size_t axis = 0; axis < input_dimensions.size(); ++axis) {
    const std::int64_t input_dimension = input_dimensions[axis];
    const std::int64_t output_dimension = output_dimensions[leading + axis];
    if (input_dimension != 1 && input_dimension != output_dimension) {
      throw std::invalid_argument(
          "broadcast materialization shapes are incompatible");
    }
  }

  const Shape input_strides = contiguous_strides(input_dimensions);
  Shape coordinates(output_dimensions.size(), 0);
  std::vector<float> result(element_count(output_dimensions));
  for (float& output : result) {
    std::size_t input_index = 0;
    for (std::size_t axis = 0; axis < input_dimensions.size(); ++axis) {
      if (input_dimensions[axis] != 1) {
        input_index +=
            static_cast<std::size_t>(coordinates[leading + axis]) *
            static_cast<std::size_t>(input_strides[axis]);
      }
    }
    output = values[input_index];

    for (std::size_t axis = coordinates.size(); axis != 0; --axis) {
      const std::size_t current = axis - 1;
      ++coordinates[current];
      if (coordinates[current] < output_dimensions[current]) {
        break;
      }
      coordinates[current] = 0;
    }
  }
  return result;
}

TensorLayout canonical_cudnn_layout(
    std::span<const std::int64_t> dimensions, std::size_t logical_rank) {
  if (logical_rank == 0 || logical_rank > 4 || dimensions.empty() ||
      dimensions.size() > logical_rank) {
    throw std::invalid_argument("cuDNN test tensor rank is invalid");
  }

  Shape aligned_dimensions(logical_rank - dimensions.size(), 1);
  aligned_dimensions.insert(aligned_dimensions.end(), dimensions.begin(),
                            dimensions.end());
  Shape aligned_strides(logical_rank - dimensions.size(),
                        static_cast<std::int64_t>(element_count(dimensions)));
  const Shape input_strides = contiguous_strides(dimensions);
  aligned_strides.insert(aligned_strides.end(), input_strides.begin(),
                         input_strides.end());

  TensorLayout result;
  if (logical_rank == 1) {
    result.dimensions = {1, aligned_dimensions[0], 1, 1};
    result.strides = {
        aligned_strides[0], aligned_strides[0], aligned_strides[0],
        aligned_strides[0]};
  } else if (logical_rank == 2) {
    result.dimensions = {
        aligned_dimensions[0], aligned_dimensions[1], 1, 1};
    result.strides = {
        aligned_strides[0], aligned_strides[1], aligned_strides[0],
        aligned_strides[0]};
  } else if (logical_rank == 3) {
    result.dimensions = {
        aligned_dimensions[0], aligned_dimensions[2],
        aligned_dimensions[1], 1};
    result.strides = {
        aligned_strides[0], aligned_strides[2], aligned_strides[1],
        aligned_strides[1]};
  } else {
    result.dimensions = {
        aligned_dimensions[0], aligned_dimensions[3],
        aligned_dimensions[1], aligned_dimensions[2]};
    result.strides = {
        aligned_strides[0], aligned_strides[3], aligned_strides[1],
        aligned_strides[2]};
  }
  return result;
}

cfe::PointwiseMode_t cudnn_mode(flagdnnPointwiseMode_t mode) {
  switch (mode) {
    case FLAGDNN_POINTWISE_DIV:
      return cfe::PointwiseMode_t::DIV;
    case FLAGDNN_POINTWISE_POW:
      return cfe::PointwiseMode_t::POW;
    case FLAGDNN_POINTWISE_MOD:
      return cfe::PointwiseMode_t::MOD;
    case FLAGDNN_POINTWISE_CMP_EQ:
      return cfe::PointwiseMode_t::CMP_EQ;
    default:
      throw std::invalid_argument("mode has no cuDNN oracle in this test");
  }
}

std::shared_ptr<cfe::graph::Tensor_attributes> make_tensor(
    const std::shared_ptr<cfe::graph::Graph>& graph, std::int64_t uid,
    const TensorLayout& layout, cfe::DataType_t data_type,
    std::string_view name) {
  return graph->tensor(
      cfe::graph::Tensor_attributes()
          .set_name(std::string(name))
          .set_uid(uid)
          .set_data_type(data_type)
          .set_dim(layout.dimensions)
          .set_stride(layout.strides));
}

std::vector<float> run_cudnn(
    flagdnnPointwiseMode_t mode, std::span<const float> left,
    std::span<const std::int64_t> left_dimensions,
    std::span<const float> right,
    std::span<const std::int64_t> right_dimensions,
    std::span<const std::int64_t> output_dimensions, CudnnHandle& handle) {
  if (left.size() != element_count(left_dimensions) ||
      right.size() != element_count(right_dimensions)) {
    throw std::invalid_argument("cuDNN oracle input size is invalid");
  }
  const std::size_t output_count = element_count(output_dimensions);
  const bool comparison = mode == FLAGDNN_POINTWISE_CMP_EQ;
  if (comparison && output_count % 8U != 0) {
    throw std::invalid_argument(
        "cuDNN BOOLEAN test output must fill complete storage bytes");
  }

  const std::size_t logical_rank = output_dimensions.size();
  const TensorLayout left_layout =
      canonical_cudnn_layout(left_dimensions, logical_rank);
  const TensorLayout right_layout =
      canonical_cudnn_layout(right_dimensions, logical_rank);
  const TensorLayout output_layout =
      canonical_cudnn_layout(output_dimensions, logical_rank);

  auto graph = std::make_shared<cfe::graph::Graph>();
  graph->set_name("cpu-reference::cudnn")
      .set_io_data_type(cfe::DataType_t::FLOAT)
      .set_intermediate_data_type(cfe::DataType_t::FLOAT)
      .set_compute_data_type(cfe::DataType_t::FLOAT);

  constexpr std::int64_t kLeftUid = 1;
  constexpr std::int64_t kRightUid = 2;
  constexpr std::int64_t kOutputUid = 3;
  const auto left_tensor = make_tensor(graph, kLeftUid, left_layout,
                                       cfe::DataType_t::FLOAT, "left");
  const auto right_tensor = make_tensor(graph, kRightUid, right_layout,
                                        cfe::DataType_t::FLOAT, "right");
  auto output_tensor = graph->pointwise(
      left_tensor, right_tensor,
      cfe::graph::Pointwise_attributes()
          .set_name("pointwise")
          .set_mode(cudnn_mode(mode))
          // Floating-input comparisons also use FLOAT compute; only their
          // result tensor uses cuDNN's packed BOOLEAN representation.
          .set_compute_data_type(cfe::DataType_t::FLOAT));
  output_tensor->set_name("output")
      .set_uid(kOutputUid)
      .set_data_type(comparison ? cfe::DataType_t::BOOLEAN
                                : cfe::DataType_t::FLOAT)
      .set_dim(output_layout.dimensions)
      .set_stride(output_layout.strides)
      .set_output(true);

  check_frontend(graph->build(handle.get(), {cfe::HeurMode_t::A}),
                 "cuDNN pointwise graph build");
  std::int64_t signed_workspace_size = 0;
  check_frontend(graph->get_workspace_size(signed_workspace_size),
                 "cuDNN pointwise workspace query");
  if (signed_workspace_size < 0 ||
      static_cast<std::uint64_t>(signed_workspace_size) >
          std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("cuDNN returned an invalid workspace size");
  }

  DeviceBuffer left_device(left.size_bytes());
  DeviceBuffer right_device(right.size_bytes());
  const std::size_t output_bytes =
      comparison ? output_count / 8U : output_count * sizeof(float);
  DeviceBuffer output_device(output_bytes);
  DeviceBuffer workspace(static_cast<std::size_t>(signed_workspace_size));
  left_device.copy_from_host(left.data(), left.size_bytes());
  right_device.copy_from_host(right.data(), right.size_bytes());
  check_cuda(cudaMemset(output_device.get(), 0, output_device.size()),
             "cudaMemset(output)");

  std::unordered_map<std::int64_t, void*> pointers = {
      {kLeftUid, left_device.get()},
      {kRightUid, right_device.get()},
      {kOutputUid, output_device.get()},
  };
  check_frontend(graph->execute(handle.get(), pointers, workspace.get()),
                 "cuDNN pointwise graph execute");

  if (!comparison) {
    std::vector<float> result(output_count);
    output_device.copy_to_host(result.data(), output_device.size());
    return result;
  }

  std::vector<std::uint8_t> packed(output_bytes);
  output_device.copy_to_host(packed.data(), packed.size());
  std::vector<float> result(output_count);
  for (std::size_t index = 0; index < output_count; ++index) {
    result[index] = static_cast<float>(
        (packed[index / 8U] >> (index % 8U)) & 1U);
  }
  return result;
}

std::vector<float> make_input(flagdnnPointwiseMode_t mode,
                              std::size_t count, std::size_t operand) {
  std::vector<float> result(count);
  constexpr std::array<float, 8> kExponents = {
      -2.0F, -1.0F, -0.5F, 0.0F, 0.5F, 1.0F, 1.5F, 2.0F};
  constexpr std::array<float, 8> kModuloLeft = {
      -7.5F, -5.0F, -3.0F, -0.5F, 0.5F, 3.0F, 5.0F, 7.5F};
  constexpr std::array<float, 8> kModuloRight = {
      2.0F, -2.0F, 2.25F, -2.25F, 3.0F, -3.0F, 1.25F, -1.25F};

  for (std::size_t index = 0; index < count; ++index) {
    switch (mode) {
      case FLAGDNN_POINTWISE_DIV: {
        if (operand == 0) {
          const int centered = static_cast<int>((index * 7U) % 23U) - 11;
          result[index] = static_cast<float>(centered) / 4.0F;
        } else {
          const float magnitude =
              static_cast<float>((index * 5U) % 9U + 1U) / 3.0F;
          result[index] = index % 2U == 0 ? magnitude : -magnitude;
        }
        break;
      }
      case FLAGDNN_POINTWISE_POW:
        result[index] = operand == 0
                            ? 0.25F + static_cast<float>(index % 13U) / 4.0F
                            : kExponents[index % kExponents.size()];
        break;
      case FLAGDNN_POINTWISE_MOD:
        result[index] = operand == 0
                            ? kModuloLeft[index % kModuloLeft.size()]
                            : kModuloRight[index % kModuloRight.size()];
        break;
      case FLAGDNN_POINTWISE_CMP_EQ: {
        const std::size_t value =
            operand == 0 ? index % 7U : (index * 3U) % 7U;
        result[index] = static_cast<float>(static_cast<int>(value) - 3);
        break;
      }
      default:
        throw std::invalid_argument("unsupported input-generation mode");
    }
  }
  return result;
}

std::string mode_name(flagdnnPointwiseMode_t mode) {
  switch (mode) {
    case FLAGDNN_POINTWISE_DIV:
      return "div";
    case FLAGDNN_POINTWISE_POW:
      return "pow";
    case FLAGDNN_POINTWISE_MOD:
      return "mod";
    case FLAGDNN_POINTWISE_CMP_EQ:
      return "cmp_eq";
    default:
      return "unknown";
  }
}

flagdnnPointwiseMode_t parse_mode(std::string_view name) {
  for (const flagdnnPointwiseMode_t mode : kSupportedModes) {
    if (mode_name(mode) == name) {
      return mode;
    }
  }
  throw std::invalid_argument("unknown mode '" + std::string(name) +
                              "'; expected div, pow, mod, or cmp_eq");
}

void compare_outputs(std::span<const float> actual,
                     std::span<const float> expected,
                     flagdnnPointwiseMode_t mode,
                     std::string_view test_name) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error(std::string(test_name) +
                             " output sizes differ");
  }
  constexpr double kAbsoluteTolerance = 5.0e-5;
  constexpr double kRelativeTolerance = 5.0e-5;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double left = actual[index];
    const double right = expected[index];
    const double absolute = std::abs(left - right);
    const double relative =
        absolute / std::max({std::abs(left), std::abs(right), 1.0e-30});
    const bool equal = mode == FLAGDNN_POINTWISE_CMP_EQ
                           ? left == right
                           : std::isfinite(absolute) &&
                                 (absolute <= kAbsoluteTolerance ||
                                  relative <= kRelativeTolerance);
    if (!equal) {
      std::ostringstream message;
      message << test_name << " differs at element " << index
              << ": CPU=" << left << ", cuDNN=" << right
              << ", abs=" << absolute << ", rel=" << relative;
      throw std::runtime_error(message.str());
    }
  }
}

void verify_cpu_contracts(std::span<const flagdnnPointwiseMode_t> modes) {
  if (modes.empty()) {
    throw std::invalid_argument("at least one pointwise mode is required");
  }
  for (const flagdnnPointwiseMode_t mode : modes) {
    if (!cpu::supports_binary_pointwise(mode)) {
      throw std::runtime_error(mode_name(mode) +
                               " is missing from the CPU support contract");
    }
  }
  if (cpu::supports_binary_pointwise(FLAGDNN_POINTWISE_ADD)) {
    throw std::runtime_error("unsupported ADD mode was reported as supported");
  }

  const auto expect_invalid = [](std::string_view name, auto&& operation) {
    try {
      operation();
    } catch (const std::invalid_argument&) {
      return;
    }
    throw std::runtime_error(std::string(name) +
                             " did not throw std::invalid_argument");
  };
  const flagdnnPointwiseMode_t contract_mode = modes.front();
  const std::vector<float> one_value = {1.0F};
  const std::vector<float> two_values = {1.0F, 2.0F};
  const Shape empty_shape;
  const Shape one_dimension = {1};
  const Shape two_dimensions = {1, 1};
  const Shape two_elements = {2};
  const Shape three_elements = {3};
  const Shape negative_dimension = {-1};

  const auto expect_materialization = [](
      std::string_view name, const std::vector<float>& values,
      const Shape& input_dimensions, const Shape& output_dimensions,
      const std::vector<float>& expected) {
    if (materialize_broadcast(values, input_dimensions, output_dimensions) !=
        expected) {
      throw std::runtime_error(std::string(name) +
                               " broadcast materialization failed");
    }
  };

  const std::vector<float> matrix_source = {10.0F, 20.0F};
  const std::vector<float> matrix_expected = {
      10.0F, 10.0F, 10.0F, 20.0F, 20.0F, 20.0F};
  expect_materialization("matrix", matrix_source, {2, 1}, {2, 3},
                         matrix_expected);

  const std::vector<float> rank_four_source = {1.0F, 2.0F, 3.0F, 4.0F};
  const std::vector<float> rank_four_expected = {
      1.0F, 1.0F, 2.0F, 2.0F, 1.0F, 1.0F, 2.0F, 2.0F,
      3.0F, 3.0F, 4.0F, 4.0F, 3.0F, 3.0F, 4.0F, 4.0F};
  expect_materialization("rank-four multi-axis", rank_four_source,
                         {2, 1, 2, 1}, {2, 2, 2, 2},
                         rank_four_expected);

  expect_invalid("unsupported mode", [&] {
    (void)cpu::evaluate_binary_pointwise(
        FLAGDNN_POINTWISE_ADD, one_value, one_dimension, one_value,
        one_dimension, one_dimension);
  });
  expect_invalid("empty output shape", [&] {
    (void)cpu::evaluate_binary_pointwise(
        contract_mode, one_value, one_dimension, one_value,
        one_dimension, empty_shape);
  });
  expect_invalid("empty input shape", [&] {
    (void)cpu::evaluate_binary_pointwise(
        contract_mode, one_value, empty_shape, one_value,
        one_dimension, one_dimension);
  });
  expect_invalid("input value-count mismatch", [&] {
    (void)cpu::evaluate_binary_pointwise(
        contract_mode, two_values, one_dimension, one_value,
        one_dimension, one_dimension);
  });
  expect_invalid("input rank exceeds output rank", [&] {
    (void)cpu::evaluate_binary_pointwise(
        contract_mode, one_value, two_dimensions, one_value,
        one_dimension, one_dimension);
  });
  expect_invalid("incompatible broadcast shape", [&] {
    (void)cpu::evaluate_binary_pointwise(
        contract_mode, two_values, two_elements, one_value,
        one_dimension, three_elements);
  });
  expect_invalid("non-positive dimension", [&] {
    (void)cpu::evaluate_binary_pointwise(
        contract_mode, one_value, negative_dimension, one_value,
        one_dimension, one_dimension);
  });

  const Shape overflowing_dimensions = {
      std::numeric_limits<std::int64_t>::max(), 3};
  try {
    (void)cpu::evaluate_binary_pointwise(
        contract_mode, one_value, one_dimension, one_value,
        one_dimension, overflowing_dimensions);
  } catch (const std::overflow_error&) {
    return;
  }
  throw std::runtime_error(
      "overflowing output shape did not throw std::overflow_error");
}

std::size_t verify_against_cudnn(
    std::span<const flagdnnPointwiseMode_t> modes) {
  const std::vector<BroadcastCase> cases = {
      {"same_shape", {2, 3, 8}, {2, 3, 8}, {2, 3, 8}},
      {"scalar_right", {2, 3, 8}, {1}, {2, 3, 8}},
      {"scalar_left", {1}, {2, 3, 8}, {2, 3, 8}},
      {"right_aligned_rank", {2, 3, 8}, {8}, {2, 3, 8}},
      {"two_sided_broadcast", {2, 1, 8}, {1, 3, 1}, {2, 3, 8}, true},
      {"rank_four_broadcast",
       {2, 1, 3, 8},
       {1, 4, 1, 8},
       {2, 4, 3, 8},
       true},
  };

  CudnnHandle handle;
  std::size_t executed = 0;
  for (const flagdnnPointwiseMode_t mode : modes) {
    for (const BroadcastCase& test_case : cases) {
      const std::vector<float> left =
          make_input(mode, element_count(test_case.left_dimensions), 0);
      const std::vector<float> right =
          make_input(mode, element_count(test_case.right_dimensions), 1);
      // cuDNN rejects graphs where both operands expand complementary axes.
      // Materialize only those cases using the independently checked helper;
      // all other cases exercise cuDNN's native broadcasting directly.
      const std::vector<float> cudnn_left =
          test_case.materialize_for_cudnn
              ? materialize_broadcast(left, test_case.left_dimensions,
                                      test_case.output_dimensions)
              : left;
      const std::vector<float> cudnn_right =
          test_case.materialize_for_cudnn
              ? materialize_broadcast(right, test_case.right_dimensions,
                                      test_case.output_dimensions)
              : right;
      const Shape& cudnn_left_dimensions =
          test_case.materialize_for_cudnn ? test_case.output_dimensions
                                          : test_case.left_dimensions;
      const Shape& cudnn_right_dimensions =
          test_case.materialize_for_cudnn ? test_case.output_dimensions
                                          : test_case.right_dimensions;
      const std::vector<float> cpu_output =
          cpu::evaluate_binary_pointwise(
              mode, left, test_case.left_dimensions, right,
              test_case.right_dimensions, test_case.output_dimensions);
      std::vector<float> cudnn_output;
      try {
        cudnn_output = run_cudnn(
            mode, cudnn_left, cudnn_left_dimensions, cudnn_right,
            cudnn_right_dimensions, test_case.output_dimensions, handle);
      } catch (const std::exception& error) {
        throw std::runtime_error(mode_name(mode) + "::" + test_case.name +
                                 " cuDNN oracle failed: " + error.what());
      }
      compare_outputs(cpu_output, cudnn_output, mode,
                      mode_name(mode) + "::" + test_case.name);
      ++executed;
    }
  }
  return executed;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::vector<flagdnnPointwiseMode_t> modes(kSupportedModes.begin(),
                                              kSupportedModes.end());
    std::string selection = "all";
    if (argc == 3 && std::string_view(argv[1]) == "--mode") {
      selection = argv[2];
      modes = {parse_mode(selection)};
    } else if (argc != 1) {
      throw std::invalid_argument(
          "usage: flagdnn_reference_cpu_pointwise_cudnn "
          "[--mode div|pow|mod|cmp_eq]");
    }

    verify_cpu_contracts(modes);
    const std::size_t executed = verify_against_cudnn(modes);
    std::cout << "CPU_POINTWISE_CUDNN_REFERENCE: PASS cases=" << executed
              << " mode=" << selection
              << " cudnn_version=" << cudnnGetVersion() << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CPU_POINTWISE_CUDNN_REFERENCE_FAILED: " << error.what()
              << std::endl;
    return 1;
  }
}
