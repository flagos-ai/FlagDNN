// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "acdnn_layout_reference.hpp"
#include "capability.hpp"
#include "common/layout.hpp"
#include "pointwise_runner_support.hpp"

#include <acdnn.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG
#define FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG "capability.json"
#endif

#ifndef FLAGDNN_THEAD_PPU_SDK_VERSION
#define FLAGDNN_THEAD_PPU_SDK_VERSION "unknown"
#endif

namespace flagdnn::testing {
namespace {

namespace tv = flagdnn::validation::thead;
namespace functional = tv::functional;

std::string operation_name(LayoutOperation operation) {
  switch (operation) {
    case LayoutOperation::kReshape:
      return "reshape";
    case LayoutOperation::kTranspose:
      return "transpose";
    case LayoutOperation::kSlice:
      return "slice";
  }
  throw std::invalid_argument("unknown THead Layout operation");
}

std::string filter_name(LayoutOperation operation) {
  switch (operation) {
    case LayoutOperation::kReshape:
      return "FLAGDNN_RESHAPE_CASE";
    case LayoutOperation::kTranspose:
      return "FLAGDNN_TRANSPOSE_CASE";
    case LayoutOperation::kSlice:
      return "FLAGDNN_SLICE_CASE";
  }
  throw std::invalid_argument("unknown THead Layout operation");
}

std::vector<std::int64_t>
contiguous_strides(std::span<const std::int64_t> dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    result[axis - 1] = stride;
    stride *= dimensions[axis - 1];
  }
  return result;
}

void emit_skip(const LayoutTestCase &test_case,
               const tv::CapabilityRecord &record,
               std::string_view operation, std::string_view target) {
  if (record.status != tv::CapabilityStatus::kUnsupported ||
      record.reason_code.empty()) {
    throw std::runtime_error("invalid acDNN Layout skip capability");
  }
  std::cout << "[SKIP][acdnn]"
            << " op=" << operation << " case=" << test_case.name
            << " reason=" << record.reason_code
            << " sdk=" << FLAGDNN_THEAD_PPU_SDK_VERSION
            << " acdnn_header=" << ACDNN_VERSION
            << " acdnn_runtime=" << acdnnGetVersion()
            << " target=" << target
            << " dtype=" << functional::data_type_name(
                   test_case.input.data_type)
            << " layout=" << functional::layout_name(test_case.input)
            << " shape=" << functional::shape_name(test_case.output) << '\n';
}

void compare(std::span<const float> actual,
             std::span<const float> expected,
             const LayoutTestCase &test_case) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error("FlagDNN and acDNN Layout sizes differ");
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (actual[index] == expected[index] ||
        (std::isnan(actual[index]) && std::isnan(expected[index]))) {
      continue;
    }
    std::ostringstream message;
    message << test_case.name << " differs at output element " << index
            << ": FlagDNN=" << actual[index]
            << " acDNN=" << expected[index];
    throw std::runtime_error(message.str());
  }
}

}  // namespace

int run_layout_functional_test(int argc, char **argv,
                               std::span<const LayoutTestCase> cases,
                               std::string_view suite_name) {
  try {
    if (argc != 3 || cases.empty()) {
      throw std::invalid_argument(
          "THead Layout test requires compiler arguments and cases");
    }
    const LayoutOperation operation = cases.front().operation;
    const std::string operation_text = operation_name(operation);
    const tv::CapabilityCatalog catalog = tv::CapabilityCatalog::load(
        FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
    const auto selected_cases = catalog.select_cases(operation_text, cases);
    cases = selected_cases;
    catalog.validate_versions(FLAGDNN_THEAD_PPU_SDK_VERSION, ACDNN_VERSION,
                              static_cast<std::int64_t>(acdnnGetVersion()));

    tv::check_driver(cuInit(0), "cuInit");
    CUdevice device = 0;
    tv::check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    tv::PrimaryContext primary(device);
    tv::ScopedCurrentContext current(primary.get());
    tv::DeviceStream stream;
    functional::TemporaryCache cache;
    flagdnn::Handle handle("thead", 0);
    handle.set_compiler(argv[1], argv[2], cache.path().string());
    const std::string target(handle.target_fingerprint());

    const std::string case_filter_name = filter_name(operation);
    const char *filter = std::getenv(case_filter_name.c_str());
    const bool qualify_probes =
        std::getenv("FLAGDNN_THEAD_QUALIFY_PROBES") != nullptr;
    std::size_t selected = 0;
    std::size_t executed = 0;
    std::size_t skipped = 0;
    std::cout << std::setprecision(9);
    for (const LayoutTestCase &test_case : cases) {
      if (test_case.operation != operation) {
        throw std::runtime_error("mixed Layout operations reached runner");
      }
      if (filter != nullptr && filter[0] != '\0' &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      ++selected;
      validate_layout_case(test_case);
      const tv::CapabilityRecord &record =
          catalog.lookup(operation_text, test_case.name);
      if (record.status == tv::CapabilityStatus::kUnsupported) {
        emit_skip(test_case, record, operation_text, target);
        ++skipped;
        continue;
      }
      if (record.status == tv::CapabilityStatus::kProbeRequired &&
          !qualify_probes) {
        throw std::runtime_error(
            "unqualified acDNN capability reached Layout run");
      }

      auto production = build_flagdnn_layout(handle, test_case);
      auto reference =
          tv::make_acdnn_layout_reference(test_case, record);
      std::vector<functional::BoundTensor> inputs;
      inputs.push_back(functional::make_input_buffer(
          test_case.input, 0, stream.get()));
      std::vector<functional::BoundTensor> production_outputs;
      production_outputs.push_back(
          functional::make_output_buffer(test_case.output, stream.get()));
      TestTensor reference_output = test_case.output;
      reference_output.strides =
          contiguous_strides(reference_output.dimensions);
      std::vector<functional::BoundTensor> reference_outputs;
      reference_outputs.push_back(
          functional::make_output_buffer(reference_output, stream.get()));
      const auto production_bindings =
          functional::bindings(inputs, production_outputs);
      const auto reference_bindings =
          functional::bindings(inputs, reference_outputs);
      tv::DeviceBuffer production_workspace(production->workspace_size());
      tv::DeviceBuffer reference_workspace(reference->workspace_size());

      tv::check_driver(cuStreamSynchronize(stream.get()),
                       "cuStreamSynchronize(before Layout)");
      functional::execute(*production, production_bindings,
                          production_workspace, stream);
      functional::execute(*reference, reference_bindings,
                          reference_workspace, stream);
      tv::check_driver(cuStreamSynchronize(stream.get()),
                       "cuStreamSynchronize(after Layout)");

      const std::vector<float> production_physical =
          functional::read_output(production_outputs.front(), stream.get());
      const std::vector<float> reference_physical =
          functional::read_output(reference_outputs.front(), stream.get());
      functional::require_padding_unchanged(
          "FlagDNN", production_physical, test_case.output);
      functional::require_padding_unchanged(
          "acDNN", reference_physical, reference_output);
      compare(functional::gather(production_physical, test_case.output),
              functional::gather(reference_physical, reference_output),
              test_case);
      ++executed;
      std::cout << test_case.name
                << ": FlagDNN Graph vs acDNN Layout PASS\n";
    }
    if (selected == 0 || selected != executed + skipped) {
      throw std::runtime_error("THead Layout case accounting mismatch");
    }
    std::cout << suite_name << ": PASS cases=" << selected
              << " executed=" << executed << " skipped=" << skipped
              << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << suite_name << ": FAIL reason=" << error.what() << '\n';
    return 1;
  }
}

}  // namespace flagdnn::testing
