// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "acdnn_composite_reference.hpp"
#include "acdnn_convolution_reference.hpp"
#include "acdnn_reference.hpp"
#include "capability.hpp"
#include "common/composite.hpp"
#include "pointwise_runner_support.hpp"
#include "cpu_pointwise.hpp"

#include <acdnn.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
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

void run_cpu_case(const AddSquareTestCase &test_case,
                  flagdnn::Handle &handle, tv::DeviceStream &stream,
                  std::string_view reason) {
  const PointwiseTestCase pointwise{
      .name = test_case.name,
      .mode = FLAGDNN_POINTWISE_ADD,
      .inputs = {test_case.left, test_case.right},
      .output = test_case.output,
      .input_domains = {PointwiseInputDomain::kReal, PointwiseInputDomain::kReal},
      .absolute_tolerance = test_case.absolute_tolerance,
      .relative_tolerance = test_case.relative_tolerance};
  auto production = build_flagdnn_add_square(handle, test_case);
  functional::run_cpu_pointwise_case(pointwise, *production, stream, reason,
                                    true);
}

void emit_skip(const ConvBiasReluTestCase &test_case,
               const tv::CapabilityRecord &record,
               std::string_view target) {
  if (record.status != tv::CapabilityStatus::kUnsupported ||
      record.reason_code.empty()) {
    throw std::runtime_error("invalid acDNN ConvBiasRelu skip capability");
  }
  std::cout << "[SKIP][acdnn]"
            << " op=conv_bias_relu case=" << test_case.name
            << " reason=" << record.reason_code
            << " sdk=" << FLAGDNN_THEAD_PPU_SDK_VERSION
            << " acdnn_header=" << ACDNN_VERSION
            << " acdnn_runtime=" << acdnnGetVersion()
            << " target=" << target
            << " dtype=" << functional::data_type_name(test_case.x.data_type)
            << " layout=" << functional::layout_name(test_case.x)
            << " shape=" << functional::shape_name(test_case.output) << '\n';
}

void compare(std::span<const float> actual,
             std::span<const float> expected,
             const AddSquareTestCase &test_case) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error("FlagDNN and acDNN AddSquare sizes differ");
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

void compare(std::span<const float> actual,
             std::span<const float> expected,
             const ConvBiasReluTestCase &test_case) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error("FlagDNN and acDNN ConvBiasRelu sizes differ");
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

std::unique_ptr<CompositeExecutable> build_add_square_reference(
    const AddSquareTestCase &test_case) {
  static const tv::CapabilityCatalog catalog =
      tv::CapabilityCatalog::load(FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
  const tv::CapabilityRecord &record =
      catalog.lookup("add_square", test_case.name);
  const tv::ReferenceSelection selection = tv::select_reference(record);
  if (!std::holds_alternative<tv::ReferencePlan>(selection)) {
    throw std::invalid_argument(
        "unsupported AddSquare case reached acDNN reference builder");
  }
  const tv::ReferencePlan &plan = std::get<tv::ReferencePlan>(selection);
  const std::vector<std::string> stable_primitives = {
      "acdnnOpTensor(MUL)", "acdnnOpTensor(ADD)"};
  const std::vector<std::string> backend_primitives = {
      "acdnnBackendExecute(POINTWISE_MUL)",
      "acdnnBackendExecute(POINTWISE_ADD)"};
  const bool stable = plan.path == tv::ReferencePath::kStablePrimitive &&
                      plan.primitives == stable_primitives;
  const bool backend = plan.path == tv::ReferencePath::kBackendDescriptor &&
                       plan.primitives == backend_primitives;
  if (!stable && !backend) {
    throw std::invalid_argument("THead AddSquare reference plan mismatch");
  }
  return tv::make_acdnn_add_square_reference(
      test_case.left, test_case.right, test_case.output, record);
}

std::unique_ptr<CompositeExecutable> build_conv_bias_relu_reference(
    const ConvBiasReluTestCase &test_case) {
  static const tv::CapabilityCatalog catalog =
      tv::CapabilityCatalog::load(FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
  const tv::CapabilityRecord &record =
      catalog.lookup("conv_bias_relu", test_case.name);
  return tv::make_acdnn_conv_bias_relu_reference(test_case, record);
}

int run_add_square_functional_test(
    int argc, char **argv, std::span<const AddSquareTestCase> cases) {
  constexpr std::string_view kSuite = "FLAGDNN_ADD_SQUARE_FUNCTIONAL";
  try {
    if (argc != 3 || cases.empty()) {
      throw std::invalid_argument(
          "THead AddSquare test requires compiler arguments and cases");
    }
    const tv::CapabilityCatalog catalog = tv::CapabilityCatalog::load(
        FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
    const auto selected_cases = catalog.select_cases("add_square", cases);
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
    const char *filter = std::getenv("FLAGDNN_ADD_SQUARE_CASE");
    const bool qualify_probes =
        std::getenv("FLAGDNN_THEAD_QUALIFY_PROBES") != nullptr;
    std::size_t selected = 0;
    std::size_t executed = 0;
    std::cout << std::setprecision(9);
    for (const AddSquareTestCase &test_case : cases) {
      if (filter != nullptr && filter[0] != '\0' &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      ++selected;
      validate_composite_case(test_case);
      const tv::CapabilityRecord &record =
          catalog.lookup("add_square", test_case.name);
      if (record.status == tv::CapabilityStatus::kUnsupported) {
        run_cpu_case(test_case, handle, stream, record.reason_code);
        ++executed;
        continue;
      }
      if (record.status == tv::CapabilityStatus::kProbeRequired &&
          !qualify_probes) {
        throw std::runtime_error(
            "unqualified acDNN capability reached AddSquare run");
      }
      try {
        auto production = build_flagdnn_add_square(handle, test_case);
        auto reference = build_add_square_reference(test_case);
        std::vector<functional::BoundTensor> inputs;
        inputs.push_back(functional::make_input_buffer(
            test_case.left, 0, stream.get(), PointwiseInputDomain::kReal));
        inputs.push_back(functional::make_input_buffer(
            test_case.right, 1, stream.get(), PointwiseInputDomain::kReal));
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
                         "cuStreamSynchronize(before AddSquare)");
        functional::execute(*production, production_bindings,
                            production_workspace, stream);
        functional::execute(*reference, reference_bindings,
                            reference_workspace, stream);
        tv::check_driver(cuStreamSynchronize(stream.get()),
                         "cuStreamSynchronize(after AddSquare)");
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
        std::cout << test_case.name
                  << ": FlagDNN Graph vs acDNN AddSquare PASS\n";
      } catch (const tv::AcdnnStatusError &error) {
        if (error.status() != ACDNN_STATUS_NOT_SUPPORTED) throw;
        run_cpu_case(test_case, handle, stream, "ACDNN_STATUS_NOT_SUPPORTED");
      }
      ++executed;
    }
    if (selected == 0 || selected != executed) {
      throw std::runtime_error("THead AddSquare case accounting mismatch");
    }
    std::cout << kSuite << ": PASS cases=" << selected
              << " executed=" << executed << " skipped=0\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << kSuite << ": FAIL reason=" << error.what() << '\n';
    return 1;
  }
}

int run_conv_bias_relu_functional_test(
    int argc, char **argv, std::span<const ConvBiasReluTestCase> cases) {
  constexpr std::string_view kSuite = "FLAGDNN_CONV_BIAS_RELU_FUNCTIONAL";
  try {
    if (argc != 3 || cases.empty()) {
      throw std::invalid_argument(
          "THead ConvBiasRelu test requires compiler arguments and cases");
    }
    const tv::CapabilityCatalog catalog = tv::CapabilityCatalog::load(
        FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
    const auto selected_cases = catalog.select_cases("conv_bias_relu", cases);
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
    const char *filter = std::getenv("FLAGDNN_CONV_BIAS_RELU_CASE");
    const bool qualify_probes =
        std::getenv("FLAGDNN_THEAD_QUALIFY_PROBES") != nullptr;
    std::size_t selected = 0;
    std::size_t executed = 0;
    std::size_t skipped = 0;
    std::cout << std::setprecision(9);
    for (const ConvBiasReluTestCase &test_case : cases) {
      if (filter != nullptr && filter[0] != '\0' &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      ++selected;
      validate_composite_case(test_case);
      const tv::CapabilityRecord &record =
          catalog.lookup("conv_bias_relu", test_case.name);
      if (record.status == tv::CapabilityStatus::kUnsupported) {
        emit_skip(test_case, record, target);
        ++skipped;
        continue;
      }
      if (record.status == tv::CapabilityStatus::kProbeRequired &&
          !qualify_probes) {
        throw std::runtime_error(
            "unqualified acDNN capability reached ConvBiasRelu run");
      }
      auto production = build_flagdnn_conv_bias_relu(handle, test_case);
      auto reference = tv::make_acdnn_conv_bias_relu_reference(
          test_case, record);
      std::vector<functional::BoundTensor> inputs;
      for (const TestTensor *input :
           {&test_case.x, &test_case.w, &test_case.bias}) {
        inputs.push_back(functional::make_input_buffer(
            *input, inputs.size(), stream.get(), PointwiseInputDomain::kReal));
      }
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
                       "cuStreamSynchronize(before ConvBiasRelu)");
      functional::execute(*production, production_bindings,
                          production_workspace, stream);
      functional::execute(*reference, reference_bindings,
                          reference_workspace, stream);
      tv::check_driver(cuStreamSynchronize(stream.get()),
                       "cuStreamSynchronize(after ConvBiasRelu)");
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
                << ": FlagDNN Graph vs acDNN ConvBiasRelu PASS\n";
    }
    if (selected == 0 || selected != executed + skipped) {
      throw std::runtime_error("THead ConvBiasRelu case accounting mismatch");
    }
    std::cout << kSuite << ": " << (executed == 0 ? "SKIP" : "PASS")
              << " cases=" << selected << " executed=" << executed
              << " skipped=" << skipped << '\n';
    return executed == 0 ? 77 : 0;
  } catch (const std::exception &error) {
    std::cerr << kSuite << ": FAIL reason=" << error.what() << '\n';
    return 1;
  }
}

}  // namespace flagdnn::testing
