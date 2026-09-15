/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_NVIDIA_FUNCTIONAL_RAW_COPY_HPP_
#define FLAGDNN_NVIDIA_FUNCTIONAL_RAW_COPY_HPP_
#include <array>

#include "validation/benchmark/paired_measurement.hpp"
#include "validation/cuda_driver.hpp"
#include "validation/tensor_io.hpp"
namespace flagdnn::testing::cuda {
inline void run_raw_copy_graph(
    std::string_view name, const TestTensor& input, const TestTensor& output,
    TestExecutable& executable, TestExecutable& reference, Stream& stream,
    const timing::RuntimeMeasurements* measurements = nullptr,
    const timing::RuntimeMeasurements* reference_measurements = nullptr) {
  const auto width = data_type_size(input.data_type);
  std::vector<float> values(element_count(input));
  for (std::size_t index = 0; index < values.size(); ++index)
    values[index] = static_cast<float>(1U << (index % 6));
  auto source =
      encode(scatter(values, input), input.data_type, BooleanEncoding::kByte);
  if (input.data_type == FLAGDNN_DATA_INT32)
    for (std::size_t index = 0; index < element_count(input); ++index)
      for (std::size_t byte = 0; byte < width; ++byte)
        source[logical_offset(index, input) * width + byte] =
            static_cast<std::uint8_t>((index * 37 + byte * 83) % 256);
  std::vector<std::uint8_t> actual(storage_element_count(output) * width, 0xA5);
  auto expected = actual;
  DeviceBuffer device_input(input.binding_byte_offset + source.size()),
      device_output(output.binding_byte_offset + actual.size()),
      reference_output(output.binding_byte_offset + expected.size());
  device_input.copy_from_host_at(source.data(), source.size(),
                                 input.binding_byte_offset, stream.get());
  device_output.copy_from_host_at(actual.data(), actual.size(),
                                  output.binding_byte_offset, stream.get());
  reference_output.copy_from_host_at(expected.data(), expected.size(),
                                     output.binding_byte_offset, stream.get());
  const std::array<flagdnnBinding_t, 2> bindings = {
      flagdnnBinding_t{input.uid,
                       device_input.opaque_at(input.binding_byte_offset)},
      flagdnnBinding_t{output.uid,
                       device_output.opaque_at(output.binding_byte_offset)}};
  auto reference_bindings = bindings;
  reference_bindings[1].device_pointer =
      reference_output.opaque_at(output.binding_byte_offset);
  DeviceBuffer workspace(executable.workspace_size()),
      reference_workspace(reference.workspace_size());
  stream.synchronize();
  executable.execute(bindings, workspace.opaque(), executable.workspace_size(),
                     stream.opaque());
  reference.execute(reference_bindings, reference_workspace.opaque(),
                    reference.workspace_size(), stream.opaque());
  device_output.copy_to_host_at(actual.data(), actual.size(),
                                output.binding_byte_offset, stream.get());
  reference_output.copy_to_host_at(expected.data(), expected.size(),
                                   output.binding_byte_offset, stream.get());
  stream.synchronize();
  if (actual != expected)
    throw std::runtime_error(std::string(name) +
                             " differs from cuDNN tensor bits");
  std::vector<bool> logical(storage_element_count(output));
  for (std::size_t index = 0; index < element_count(output); ++index)
    logical[logical_offset(index, output)] = true;
  for (std::size_t index = 0; index < logical.size(); ++index)
    if (!logical[index])
      for (std::size_t byte = 0; byte < width; ++byte)
        if (actual[index * width + byte] != 0xA5 ||
            expected[index * width + byte] != 0xA5)
          throw std::runtime_error(std::string(name) +
                                   " changed output padding");
  if (measurements && reference_measurements)
    measure_paired_execution(name, executable, reference, bindings,
                             reference_bindings, workspace, reference_workspace,
                             stream, *measurements, *reference_measurements);
  std::cout << name << ": FlagDNN Graph vs cuDNN Graph bit-exact PASS"
            << std::endl;
}
}  // namespace flagdnn::testing::cuda
#endif
