/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <flagdnn/flagdnn.hpp>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "backends/mthreads/validation/functional/tensor_io_adapter.hpp"
#include "backends/mthreads/validation/paired_timing.hpp"

namespace flagdnn::testing::mthreads {
namespace mv = validation::mthreads;
namespace io = mv::tensor_io;
struct Tolerance {
  double absolute, relative;
};

template <class Case, class Builder, class Inputs, class Reference,
          class Limit>
int run_paired_cases(int argc, char** argv, std::span<const Case> cases,
                     const char* filter_name, const std::string& operation,
                     Builder build, Inputs make_inputs, Reference reference,
                     Limit tolerance, bool benchmark = false,
                     bool capture_graph = true) {
  if (argc != 3) return 2;
  if (benchmark) setenv("FLAGDNN_MTHREADS_NATIVE_BENCHMARK", "1", 1);
  std::string marker = "FLAGDNN_" + operation + "_FUNCTIONAL";
  for (char& c : marker) c = static_cast<char>(std::toupper(c));
  try {
    char pattern[] = "/tmp/flagdnn-mthreads-paired-XXXXXX";
    const char* directory = mkdtemp(pattern);
    if (!directory) throw std::runtime_error("cannot create artifact cache");
    struct Cache {
      std::filesystem::path path;
      ~Cache() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
      }
    } cache{directory};
    mv::check_musa(musaSetDevice(0), "musaSetDevice");
    mv::Stream stream;
    flagdnn::Handle handle("mthreads", 0);
    handle.set_compiler(argv[1], argv[2], cache.path.string());
    const char* filter = std::getenv(filter_name);
    std::size_t executed = 0, skipped = 0;
    for (const auto& test_case : cases) {
      if (filter && test_case.name.find(filter) == std::string::npos) continue;
      try {
        auto reference_executable = reference(test_case);
        const auto inputs = make_inputs(test_case);
        if (inputs.size() != test_case.inputs.size())
          throw std::runtime_error("incorrect test input count");
        std::vector<std::unique_ptr<mv::DeviceBuffer>> buffers,
            reference_buffers;
        std::vector<std::vector<std::uint8_t>> initial_inputs;
        std::vector<flagdnnBinding_t> bindings;
        const auto allocate = [&](const TestTensor& tensor,
                                  const std::vector<float>& values) {
          const auto bytes = io::encode(values, tensor.data_type);
          auto buffer = std::make_unique<mv::DeviceBuffer>(
              tensor.binding_byte_offset + bytes.size());
          buffer->copy_from_host_at(bytes.data(), bytes.size(),
                                    tensor.binding_byte_offset, stream.get());
          return buffer;
        };
        for (std::size_t i = 0; i < inputs.size(); ++i) {
          const auto& tensor = test_case.inputs[i];
          const auto physical = io::scatter(inputs[i], tensor);
          initial_inputs.push_back(io::encode(physical, tensor.data_type));
          auto buffer = allocate(tensor, physical);
          bindings.push_back(
              {tensor.uid, buffer->opaque_at(tensor.binding_byte_offset)});
          buffers.push_back(std::move(buffer));
        }
        auto reference_bindings = bindings;
        for (const auto& tensor : test_case.outputs) {
          const std::vector<float> padding(io::storage_element_count(tensor),
                                           io::kPaddingSentinel);
          auto buffer = allocate(tensor, padding);
          bindings.push_back(
              {tensor.uid, buffer->opaque_at(tensor.binding_byte_offset)});
          buffers.push_back(std::move(buffer));
          auto reference_buffer = allocate(tensor, padding);
          reference_bindings.push_back(
              {tensor.uid,
               reference_buffer->opaque_at(tensor.binding_byte_offset)});
          reference_buffers.push_back(std::move(reference_buffer));
        }
        mv::DeviceBuffer reference_workspace(
            reference_executable->workspace_size(), 256);
        const auto enqueue_reference = [&] {
          reference_executable->prepare(reference_bindings, stream.opaque());
          reference_executable->execute(
              reference_bindings, reference_workspace.opaque(),
              reference_executable->workspace_size(), stream.opaque());
        };
        // Establish the native library capability before compiling production.
        enqueue_reference();
        stream.synchronize();
        auto executable = build(handle, test_case);
        mv::DeviceBuffer workspace(executable->workspace_size(), 256);
        const auto enqueue = [&] {
          executable->prepare(bindings, stream.opaque());
          executable->execute(bindings, workspace.opaque(),
                              executable->workspace_size(), stream.opaque());
        };
        enqueue();
        mv::timing::paired(test_case.name, stream, enqueue, enqueue_reference,
                           capture_graph);
        const auto read = [&](const mv::DeviceBuffer& buffer,
                              const TestTensor& tensor) {
          std::vector<std::uint8_t> bytes(
              io::encode(std::vector<float>(io::storage_element_count(tensor)),
                         tensor.data_type)
                  .size());
          buffer.copy_to_host_at(bytes.data(), bytes.size(),
                                 tensor.binding_byte_offset, stream.get());
          stream.synchronize();
          return bytes;
        };
        std::vector<std::vector<std::uint8_t>> first;
        for (std::size_t i = 0; i < test_case.outputs.size(); ++i) {
          const auto& tensor = test_case.outputs[i];
          const auto actual_bytes = read(*buffers[inputs.size() + i], tensor);
          const auto expected_bytes = read(*reference_buffers[i], tensor);
          io::require_padding_unchanged("FlagDNN", actual_bytes, tensor);
          io::require_padding_unchanged("muDNN", expected_bytes, tensor);
          const auto actual =
              io::gather(io::decode(actual_bytes, tensor.data_type), tensor);
          const auto expected =
              io::gather(io::decode(expected_bytes, tensor.data_type), tensor);
          const auto limit = tolerance(test_case, i);
          for (std::size_t j = 0; j < actual.size(); ++j) {
            if (actual[j] == expected[j]) continue;
            const double difference =
                std::abs(static_cast<double>(actual[j]) - expected[j]);
            if (!std::isfinite(difference) ||
                difference >
                    limit.absolute + limit.relative * std::abs(expected[j]))
              throw std::runtime_error(
                  test_case.name + " output=" + std::to_string(i) +
                  " element=" + std::to_string(j) +
                  " actual=" + std::to_string(actual[j]) +
                  " expected=" + std::to_string(expected[j]));
          }
          first.push_back(actual_bytes);
        }
        enqueue();
        for (std::size_t i = 0; i < first.size(); ++i)
          io::require_bytes_equal(
              "repeated FlagDNN output",
              read(*buffers[inputs.size() + i], test_case.outputs[i]),
              first[i]);
        for (std::size_t i = 0; i < inputs.size(); ++i)
          io::require_bytes_equal("paired input",
                                  read(*buffers[i], test_case.inputs[i]),
                                  initial_inputs[i]);
        ++executed;
        std::cout << test_case.name << ": FlagDNN Graph vs direct muDNN PASS"
                  << std::endl;
      } catch (const mv::ReferenceUnsupported& error) {
        stream.synchronize();
        ++skipped;
        mv::report_skip(test_case.name, error);
      }
    }
    return mv::report_cases(marker, executed, skipped);
  } catch (const std::exception& error) {
    std::cerr << marker << "_FAILED: " << error.what() << std::endl;
    return 1;
  }
}
}  // namespace flagdnn::testing::mthreads
