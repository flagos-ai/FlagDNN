/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_NVIDIA_FUNCTIONAL_PAIRED_HPP_
#define FLAGDNN_NVIDIA_FUNCTIONAL_PAIRED_HPP_
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <flagdnn/flagdnn.hpp>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "validation/benchmark/paired_measurement.hpp"
#include "validation/cuda_driver.hpp"
#include "validation/tensor_io.hpp"
namespace flagdnn::testing::cuda {
struct PairedTolerance {
  double absolute, relative;
};

// Device allocation and storage quantization belong to the platform adapter;
// case generation and FlagDNN graph construction stay platform neutral.
template <class Case, class Builder, class Inputs, class Reference,
          class Tolerance>
int run_paired_cases(int argc, char** argv, std::span<const Case> cases,
                     const char* filter_name, Builder build, Inputs make_inputs,
                     Reference reference, Tolerance tolerance,
                     bool benchmark = false,
                     std::string_view reference_name = "cuDNN Graph") {
  if (argc != 3) return 2;
  char pattern[] = "/tmp/flagdnn-floating-functional-XXXXXX";
  const char* directory = mkdtemp(pattern);
  if (!directory) return 1;
  struct Cache {
    std::filesystem::path path;
    ~Cache() {
      std::error_code ignored;
      std::filesystem::remove_all(path, ignored);
    }
  } cache{directory};
  try {
    DriverContext driver;
    Stream stream;
    flagdnn::Handle handle(FLAGDNN_BACKEND_NVIDIA, 0);
    handle.set_compiler(argv[1], argv[2], cache.path.string());
    const char* filter = std::getenv(filter_name);
    std::size_t count = 0;
    for (const auto& test_case : cases) {
      if (filter && test_case.name.find(filter) == std::string::npos) continue;
      timing::RuntimeMeasurements measurements;
      measurements.build_cache = "fresh_artifact_cache";
      auto executable =
          timing::profile_build([&] { return build(handle, test_case); },
                                benchmark ? &measurements : nullptr);
      auto inputs = make_inputs(test_case);
      if (inputs.size() != test_case.inputs.size())
        throw std::runtime_error("incorrect test input count");
      std::vector<std::unique_ptr<DeviceBuffer>> buffers;
      std::vector<flagdnnBinding_t> bindings;
      for (std::size_t index = 0; index < test_case.inputs.size(); ++index) {
        const auto& tensor = test_case.inputs[index];
        const auto bytes = encode(scatter(inputs[index], tensor),
                                  tensor.data_type, BooleanEncoding::kByte);
        auto buffer = std::make_unique<DeviceBuffer>(
            tensor.binding_byte_offset + bytes.size());
        buffer->copy_from_host_at(bytes.data(), bytes.size(),
                                  tensor.binding_byte_offset, stream.get());
        bindings.push_back(
            {tensor.uid, buffer->opaque_at(tensor.binding_byte_offset)});
        buffers.push_back(std::move(buffer));
      }
      timing::RuntimeMeasurements reference_measurements;
      auto reference_executable =
          timing::profile_build([&] { return reference(test_case); },
                                benchmark ? &reference_measurements : nullptr);
      std::vector<std::unique_ptr<DeviceBuffer>> reference_buffers;
      auto reference_bindings = bindings;
      for (const auto& tensor : test_case.outputs) {
        const auto bytes =
            encode(std::vector<float>(storage_element_count(tensor),
                                      padding_sentinel()),
                   tensor.data_type, BooleanEncoding::kByte);
        auto buffer = std::make_unique<DeviceBuffer>(
            tensor.binding_byte_offset + bytes.size());
        buffer->copy_from_host_at(bytes.data(), bytes.size(),
                                  tensor.binding_byte_offset, stream.get());
        bindings.push_back(
            {tensor.uid, buffer->opaque_at(tensor.binding_byte_offset)});
        buffers.push_back(std::move(buffer));
        auto reference_buffer = std::make_unique<DeviceBuffer>(
            tensor.binding_byte_offset + bytes.size());
        reference_buffer->copy_from_host_at(bytes.data(), bytes.size(),
                                            tensor.binding_byte_offset,
                                            stream.get());
        reference_bindings.push_back(
            {tensor.uid,
             reference_buffer->opaque_at(tensor.binding_byte_offset)});
        reference_buffers.push_back(std::move(reference_buffer));
      }
      DeviceBuffer reference_workspace(reference_executable->workspace_size());
      reference_executable->execute(
          reference_bindings, reference_workspace.opaque(),
          reference_executable->workspace_size(), stream.opaque());
      std::vector<std::vector<float>> expected;
      for (std::size_t index = 0; index < test_case.outputs.size(); ++index) {
        const auto& output = test_case.outputs[index];
        std::vector<std::uint8_t> bytes(
            encoded_byte_count(output, BooleanEncoding::kByte));
        reference_buffers[index]->copy_to_host_at(bytes.data(), bytes.size(),
                                                  output.binding_byte_offset,
                                                  stream.get());
        stream.synchronize();
        const auto physical =
            decode(bytes, output.data_type, storage_element_count(output),
                   BooleanEncoding::kByte);
        require_padding_unchanged("cuDNN", physical, output);
        expected.push_back(gather(physical, output));
      }
      DeviceBuffer workspace(executable->workspace_size());
      stream.synchronize();
      std::vector<std::vector<std::uint8_t>> first_outputs;
      // Repeated execution must preserve results for fixed inputs and RNG
      // counters.
      for (int repeat = 0; repeat < 2; ++repeat) {
        executable->execute(bindings, workspace.opaque(),
                            executable->workspace_size(), stream.opaque());
        for (std::size_t index = 0; index < test_case.outputs.size(); ++index) {
          const auto& output = test_case.outputs[index];
          std::vector<std::uint8_t> bytes(
              encoded_byte_count(output, BooleanEncoding::kByte));
          buffers[test_case.inputs.size() + index]->copy_to_host_at(
              bytes.data(), bytes.size(), output.binding_byte_offset,
              stream.get());
          stream.synchronize();
          if (repeat == 1) {
            if (bytes != first_outputs[index])
              throw std::runtime_error(test_case.name +
                                       " repeated execution differs");
            continue;
          }
          first_outputs.push_back(bytes);
          const auto physical =
              decode(bytes, output.data_type, storage_element_count(output),
                     BooleanEncoding::kByte);
          require_padding_unchanged("FlagDNN", physical, output);
          const auto actual = gather(physical, output);
          const auto limit = tolerance(test_case, index);
          if (actual.size() != expected[index].size())
            throw std::runtime_error("reference output shape differs");
          for (std::size_t element = 0; element < actual.size(); ++element) {
            if (actual[element] == expected[index][element]) continue;
            const double difference =
                std::abs(static_cast<double>(actual[element]) -
                         expected[index][element]);
            if (!std::isfinite(difference) ||
                difference >
                    limit.absolute +
                        limit.relative * std::abs(expected[index][element]))
              throw std::runtime_error(
                  test_case.name + " mismatch at output " +
                  std::to_string(index) + " element " +
                  std::to_string(element) +
                  " actual=" + std::to_string(actual[element]) +
                  " expected=" + std::to_string(expected[index][element]));
          }
        }
      }
      if (benchmark)
        measure_paired_execution(
            test_case.name, *executable, *reference_executable, bindings,
            reference_bindings, workspace, reference_workspace, stream,
            measurements, reference_measurements);
      std::cout << test_case.name << ": FlagDNN Graph vs " << reference_name
                << " PASS" << std::endl;
      ++count;
    }
    if (!count)
      throw std::runtime_error(std::string(filter_name) + " matched no cases");
    std::cout << "FLAGDNN_PAIRED_FUNCTIONAL: PASS cases=" << count << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FLAGDNN_PAIRED_FUNCTIONAL_FAILED: " << error.what()
              << std::endl;
    return 1;
  }
}
}  // namespace flagdnn::testing::cuda
#endif
