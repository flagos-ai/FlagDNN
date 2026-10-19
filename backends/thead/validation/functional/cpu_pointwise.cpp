// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "functional/cpu_pointwise.hpp"

#include "functional/pointwise_runner_support.hpp"
#include "numeric_types.hpp"
#include "reference/cpu/pointwise.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace flagdnn::validation::thead::functional {
namespace {

using flagdnn::testing::PointwiseTestCase;
using flagdnn::testing::TestTensor;
namespace cpu = flagdnn::reference::cpu;

std::size_t physical_index(std::size_t logical, const TestTensor& tensor) {
  // storage_element_count has already checked the geometry and overflow.
  std::size_t result = 0;
  for (std::size_t axis = tensor.dimensions.size(); axis != 0; --axis) {
    const auto extent = static_cast<std::size_t>(tensor.dimensions[axis - 1]);
    result += (logical % extent) *
              static_cast<std::size_t>(tensor.strides[axis - 1]);
    logical /= extent;
  }
  return result;
}

std::size_t allocation_size(const TestTensor& tensor) {
  const auto width = element_size(tensor.data_type);
  const auto elements = storage_element_count(tensor);
  if (tensor.binding_byte_offset % width != 0) {
    throw std::invalid_argument("CPU pointwise binding is misaligned");
  }
  if (elements > (std::numeric_limits<std::size_t>::max() -
                  tensor.binding_byte_offset) / width) {
    throw std::overflow_error("CPU pointwise allocation size overflows");
  }
  return tensor.binding_byte_offset + elements * width;
}

BoundTensor upload_raw(const TestTensor& tensor,
                       std::span<const std::byte> bytes, CUstream stream) {
  if (bytes.size() != allocation_size(tensor)) {
    throw std::invalid_argument("CPU pointwise raw buffer size mismatch");
  }
  auto buffer = std::make_unique<DeviceBuffer>(bytes.size());
  copy_to_device_async(*buffer, bytes, 0, stream);
  // Keep the host staging memory alive until its asynchronous transfer finishes.
  check_driver(cuStreamSynchronize(stream),
               "cuStreamSynchronize(CPU pointwise upload)");
  return {tensor, std::move(buffer)};
}

void require_raw_padding_unchanged(std::span<const std::byte> actual,
                                   std::span<const std::byte> initial,
                                   const PointwiseTestCase& test_case) {
  const auto& tensor = test_case.output;
  const auto width = element_size(tensor.data_type);
  std::vector<bool> occupied(actual.size(), false);
  for (std::size_t index = 0; index < element_count(tensor); ++index) {
    const auto offset = tensor.binding_byte_offset +
                        physical_index(index, tensor) * width;
    for (std::size_t byte = 0; byte < width; ++byte) {
      occupied.at(offset + byte) = true;
    }
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!occupied[index] && actual[index] != initial[index]) {
      throw std::runtime_error(test_case.name +
                               ": FlagDNN modified output padding at byte " +
                               std::to_string(index));
    }
  }
}

template <typename Expected>
void compare_exact(std::span<const std::byte> payload,
                   std::span<const Expected> expected,
                   const PointwiseTestCase& test_case) {
  for (std::size_t index = 0; index < expected.size(); ++index) {
    std::int32_t actual = 0;
    const auto physical = physical_index(index, test_case.output);
    if (test_case.output.data_type == FLAGDNN_DATA_BOOLEAN) {
      actual = std::to_integer<std::uint8_t>(payload[physical]);
    } else {
      std::memcpy(&actual, payload.data() + physical * sizeof(actual),
                  sizeof(actual));
    }
    if (actual != expected[index]) {
      std::ostringstream message;
      message << test_case.name << " differs at output element " << index
              << ": FlagDNN=" << actual << " CPU=" << expected[index];
      throw std::runtime_error(message.str());
    }
  }
}

void compare_floating(std::span<const float> actual,
                      std::span<const float> expected,
                      const PointwiseTestCase& test_case) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error("FlagDNN and CPU output sizes differ");
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double left = actual[index];
    const double right = expected[index];
    if (left == right || (std::isnan(left) && std::isnan(right))) {
      continue;
    }
    const double absolute = std::abs(left - right);
    const double relative =
        absolute / std::max({std::abs(left), std::abs(right), 1.0e-30});
    if (!std::isfinite(left) || !std::isfinite(right) ||
        (absolute > test_case.absolute_tolerance &&
         relative > test_case.relative_tolerance)) {
      std::ostringstream message;
      message << test_case.name << " differs at output element " << index
              << ": FlagDNN=" << left << " CPU=" << right
              << " abs=" << absolute << " rel=" << relative
              << " atol=" << test_case.absolute_tolerance
              << " rtol=" << test_case.relative_tolerance;
      throw std::runtime_error(message.str());
    }
  }
}

bool comparison_mode(flagdnnPointwiseMode_t mode) {
  return mode == FLAGDNN_POINTWISE_CMP_EQ || mode == FLAGDNN_POINTWISE_CMP_NEQ ||
         mode == FLAGDNN_POINTWISE_CMP_GT || mode == FLAGDNN_POINTWISE_CMP_GE ||
         mode == FLAGDNN_POINTWISE_CMP_LT || mode == FLAGDNN_POINTWISE_CMP_LE;
}

bool supports_cpu_pointwise_case(const PointwiseTestCase& test_case) {
  if (!cpu::supports_binary_pointwise(test_case.mode) ||
      test_case.inputs.size() != 2 ||
      test_case.inputs[0].data_type != test_case.inputs[1].data_type) {
    return false;
  }
  const auto type = test_case.inputs[0].data_type;
  if (type != FLAGDNN_DATA_FLOAT32 && type != FLAGDNN_DATA_FLOAT16 &&
      type != FLAGDNN_DATA_BFLOAT16) {
    return false;
  }
  return test_case.output.data_type ==
         (comparison_mode(test_case.mode) ? FLAGDNN_DATA_BOOLEAN : type);
}

}  // namespace

void run_cpu_pointwise_case(const PointwiseTestCase& test_case,
                            flagdnn::testing::TestExecutable& production,
                            DeviceStream& stream,
                            std::string_view fallback_reason, bool add_square) {
  flagdnn::testing::validate_pointwise_case(test_case);
  if (!supports_cpu_pointwise_case(test_case) || fallback_reason.empty() ||
      (add_square && (test_case.mode != FLAGDNN_POINTWISE_ADD ||
                      test_case.alpha != 1.0))) {
    throw std::invalid_argument("unsupported CPU pointwise fallback case");
  }
  std::vector<BoundTensor> inputs;
  std::array<std::vector<float>, 2> floating_inputs;
  inputs.reserve(test_case.inputs.size());
  for (std::size_t index = 0; index < 2; ++index) {
    const auto& specification = test_case.inputs[index];

    inputs.push_back(make_input_buffer(specification, index, stream.get(),
                                       test_case.input_domains.at(index)));
    // Reading back the uploaded values makes input dtype quantization explicit.
    floating_inputs[index] =
        gather(read_output(inputs.back(), stream.get()), specification);

  }

  // Input bytes, including offsets and padding, must remain unchanged.
  std::vector<std::vector<std::byte>> initial_inputs;
  initial_inputs.reserve(inputs.size());
  for (const auto& input : inputs) {
    initial_inputs.emplace_back(allocation_size(input.specification));
    copy_from_device_async(initial_inputs.back(), *input.buffer, 0, stream.get());
  }
  check_driver(cuStreamSynchronize(stream.get()),
               "cuStreamSynchronize(CPU pointwise inputs)");

  const auto& left_shape = test_case.inputs[0].dimensions;
  const auto& right_shape = test_case.inputs[1].dimensions;
  const auto& output_shape = test_case.output.dimensions;
  std::vector<float> floating_expected;

  const auto& right = floating_inputs[1];
  if (add_square) {
    floating_expected = cpu::evaluate_add_square(
        floating_inputs[0], left_shape, right, right_shape, output_shape,
        [&](std::span<const float> values) {
          return decode_floating(test_case.output.data_type,
              encode_floating(test_case.output.data_type, values));
        });
  } else {
    floating_expected = cpu::evaluate_binary_pointwise_with_alpha(
        test_case.mode, floating_inputs[0], left_shape, right, right_shape,
        output_shape, test_case.alpha);
  }
  if (test_case.output.data_type != FLAGDNN_DATA_BOOLEAN) {
    floating_expected = decode_floating(
        test_case.output.data_type,
        encode_floating(test_case.output.data_type, floating_expected));
  }

  std::vector<std::byte> initial(allocation_size(test_case.output),
                                  std::byte{0xa5});
  if (test_case.output.data_type != FLAGDNN_DATA_BOOLEAN) {
    // Raw 0xa5 bytes decode to tiny floats that can pass near-zero tolerances.
    // Use a representable sentinel for logical values; keep padding byte-exact.
    const std::array<float, 1> sentinel = {kPaddingSentinel};
    const auto encoded = encode_floating(test_case.output.data_type, sentinel);
    for (std::size_t index = 0; index < element_count(test_case.output); ++index) {
      const auto offset = test_case.output.binding_byte_offset +
                          physical_index(index, test_case.output) * encoded.size();
      std::memcpy(initial.data() + offset, encoded.data(), encoded.size());
    }
  }
  std::vector<BoundTensor> outputs;
  outputs.push_back(upload_raw(test_case.output, initial, stream.get()));
  const auto tensor_bindings = bindings(inputs, outputs);
  DeviceBuffer workspace(production.workspace_size());
  execute(production, tensor_bindings, workspace, stream);
  std::vector<std::byte> actual(initial.size());
  copy_from_device_async(actual, *outputs.front().buffer, 0, stream.get());
  check_driver(cuStreamSynchronize(stream.get()),
               "cuStreamSynchronize(CPU pointwise comparison)");
  require_raw_padding_unchanged(actual, initial, test_case);
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    std::vector<std::byte> after(initial_inputs[index].size());
    copy_from_device_async(after, *inputs[index].buffer, 0, stream.get());
    check_driver(cuStreamSynchronize(stream.get()),
                 "cuStreamSynchronize(CPU pointwise input guard)");
    if (after != initial_inputs[index]) {
      throw std::runtime_error(test_case.name + ": FlagDNN modified input " +
                               std::to_string(index));
    }
  }

  const auto payload = std::span<const std::byte>(actual).subspan(
      test_case.output.binding_byte_offset);
  if (test_case.output.data_type == FLAGDNN_DATA_BOOLEAN) {
    compare_exact<float>(payload, floating_expected, test_case);
  } else {
    compare_floating(
        gather(decode_floating(test_case.output.data_type, payload),
               test_case.output),
        floating_expected, test_case);
  }
  std::cout << test_case.name << ": FlagDNN Graph vs CPU reference PASS"
            << " fallback_reason=" << fallback_reason << '\n';
}

}  // namespace flagdnn::validation::thead::functional
