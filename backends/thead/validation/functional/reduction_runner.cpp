// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "acdnn_reduction_reference.hpp"
#include "capability.hpp"
#include "common/reduction.hpp"
#include "pointwise_runner_support.hpp"

#include <acdnn.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
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

void emit_skip(const ReductionTestCase &test_case,
               const tv::CapabilityRecord &record,
               std::string_view target) {
  if (record.status != tv::CapabilityStatus::kUnsupported ||
      record.reason_code.empty()) {
    throw std::runtime_error("invalid acDNN reduction skip capability");
  }
  std::cout << "[SKIP][acdnn]"
            << " op=reduction case=" << test_case.name
            << " reason=" << record.reason_code
            << " sdk=" << FLAGDNN_THEAD_PPU_SDK_VERSION
            << " acdnn_header=" << ACDNN_VERSION
            << " acdnn_runtime=" << acdnnGetVersion()
            << " target=" << target
            << " dtype=" << functional::data_type_name(
                   test_case.input.data_type)
            << " layout=" << functional::layout_name(test_case.input)
            << " shape=" << functional::shape_name(test_case.input) << '\n';
}

void compare(std::span<const float> actual,
             std::span<const float> expected,
             const ReductionTestCase &test_case) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error("FlagDNN and acDNN reduction sizes differ");
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
    if (!std::isfinite(absolute) ||
        (absolute > test_case.absolute_tolerance &&
         relative > test_case.relative_tolerance)) {
      std::ostringstream message;
      message << test_case.name << " differs at output element " << index
              << ": FlagDNN=" << left << " acDNN=" << right
              << " abs=" << absolute << " rel=" << relative;
      throw std::runtime_error(message.str());
    }
  }
}

}  // namespace

int run_reduction_functional_test(
    int argc, char **argv, std::span<const ReductionTestCase> cases) {
  constexpr std::string_view kSuite = "FLAGDNN_REDUCTION_FUNCTIONAL";
  try {
    if (argc != 3 || cases.empty()) {
      throw std::invalid_argument(
          "THead reduction test requires compiler arguments and cases");
    }
    const tv::CapabilityCatalog catalog = tv::CapabilityCatalog::load(
        FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
    const auto selected_cases = catalog.select_cases("reduction", cases);
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
    const char *filter = std::getenv("FLAGDNN_REDUCTION_CASE");
    const bool qualify_probes =
        std::getenv("FLAGDNN_THEAD_QUALIFY_PROBES") != nullptr;
    std::size_t selected = 0;
    std::size_t executed = 0;
    std::size_t skipped = 0;
    std::cout << std::setprecision(9);
    for (const ReductionTestCase &test_case : cases) {
      if (filter != nullptr && filter[0] != '\0' &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      ++selected;
      validate_reduction_case(test_case);
      const tv::CapabilityRecord &record =
          catalog.lookup("reduction", test_case.name);
      if (record.status == tv::CapabilityStatus::kUnsupported) {
        emit_skip(test_case, record, target);
        ++skipped;
        continue;
      }
      if (record.status == tv::CapabilityStatus::kProbeRequired &&
          !qualify_probes) {
        throw std::runtime_error(
            "unqualified acDNN reduction capability reached execution");
      }
      auto production = build_flagdnn_reduction(handle, test_case);
      auto reference = tv::make_acdnn_reduction_reference(test_case, record);
      std::vector<functional::BoundTensor> inputs;
      inputs.push_back(functional::make_input_buffer(
          test_case.input, 0, stream.get(),
          test_case.mode == FLAGDNN_REDUCTION_MUL
              ? PointwiseInputDomain::kTan
              : PointwiseInputDomain::kReal));
      std::vector<functional::BoundTensor> production_outputs;
      production_outputs.push_back(
          functional::make_output_buffer(test_case.output, stream.get()));
      std::vector<functional::BoundTensor> reference_outputs;
      reference_outputs.push_back(
          functional::make_output_buffer(test_case.output, stream.get()));
      const auto production_bindings =
          functional::bindings(inputs, production_outputs);
      const auto reference_bindings =
          functional::bindings(inputs, reference_outputs);
      tv::DeviceBuffer production_workspace(production->workspace_size());
      tv::DeviceBuffer reference_workspace(reference->workspace_size());
      tv::check_driver(cuStreamSynchronize(stream.get()),
                       "cuStreamSynchronize(before reduction)");
      functional::execute(*production, production_bindings,
                          production_workspace, stream);
      functional::execute(*reference, reference_bindings,
                          reference_workspace, stream);
      tv::check_driver(cuStreamSynchronize(stream.get()),
                       "cuStreamSynchronize(after reduction)");
      const std::vector<float> production_physical =
          functional::read_output(production_outputs.front(), stream.get());
      const std::vector<float> reference_physical =
          functional::read_output(reference_outputs.front(), stream.get());
      functional::require_padding_unchanged(
          "FlagDNN", production_physical, test_case.output);
      functional::require_padding_unchanged(
          "acDNN", reference_physical, test_case.output);
      compare(functional::gather(production_physical, test_case.output),
              functional::gather(reference_physical, test_case.output),
              test_case);
      ++executed;
      std::cout << test_case.name
                << ": FlagDNN Graph vs acDNN Reduction PASS\n";
    }
    if (selected == 0 || selected != executed + skipped) {
      throw std::runtime_error("THead reduction case accounting mismatch");
    }
    std::cout << kSuite << ": PASS cases=" << selected
              << " executed=" << executed << " skipped=" << skipped
              << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << kSuite << ": FAIL reason=" << error.what() << '\n';
    return 1;
  }
}

}  // namespace flagdnn::testing
