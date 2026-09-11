// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "acdnn_normalization_reference.hpp"
#include "capability.hpp"
#include "common/normalization.hpp"
#include "pointwise_runner_support.hpp"

#include <acdnn.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
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

template <typename Case>
void emit_skip(const Case &test_case, const tv::CapabilityRecord &record,
               std::string_view operation, const TestTensor &representative,
               std::string_view target) {
  if (record.status != tv::CapabilityStatus::kUnsupported ||
      record.reason_code.empty()) {
    throw std::runtime_error("invalid acDNN BatchNorm skip capability");
  }
  std::cout << "[SKIP][acdnn]"
            << " op=" << operation << " case=" << test_case.name
            << " reason=" << record.reason_code
            << " sdk=" << FLAGDNN_THEAD_PPU_SDK_VERSION
            << " acdnn_header=" << ACDNN_VERSION
            << " acdnn_runtime=" << acdnnGetVersion()
            << " target=" << target
            << " dtype=" << functional::data_type_name(
                   representative.data_type)
            << " layout=" << functional::layout_name(representative)
            << " shape=" << functional::shape_name(representative) << '\n';
}

void compare_output(const functional::BoundTensor &actual,
                    const functional::BoundTensor &expected,
                    std::string_view case_name, std::string_view output_name,
                    double absolute_tolerance, double relative_tolerance,
                    tv::DeviceStream &stream) {
  const std::vector<float> actual_physical =
      functional::read_output(actual, stream.get());
  const std::vector<float> expected_physical =
      functional::read_output(expected, stream.get());
  functional::require_padding_unchanged("FlagDNN", actual_physical,
                                        actual.specification);
  functional::require_padding_unchanged("acDNN", expected_physical,
                                        expected.specification);
  const std::vector<float> actual_logical =
      functional::gather(actual_physical, actual.specification);
  const std::vector<float> expected_logical =
      functional::gather(expected_physical, expected.specification);
  if (actual_logical.size() != expected_logical.size()) {
    throw std::runtime_error("FlagDNN and acDNN BatchNorm sizes differ");
  }
  for (std::size_t index = 0; index < actual_logical.size(); ++index) {
    const double left = actual_logical[index];
    const double right = expected_logical[index];
    if (left == right || (std::isnan(left) && std::isnan(right))) {
      continue;
    }
    const double absolute = std::abs(left - right);
    const double relative =
        absolute / std::max({std::abs(left), std::abs(right), 1.0e-30});
    if (std::isnan(left) != std::isnan(right) || !std::isfinite(absolute) ||
        (absolute > absolute_tolerance && relative > relative_tolerance)) {
      std::ostringstream message;
      message << case_name << ' ' << output_name << " differs at element "
              << index << ": FlagDNN=" << left << " acDNN=" << right
              << " abs=" << absolute << " rel=" << relative
              << " atol=" << absolute_tolerance
              << " rtol=" << relative_tolerance;
      throw std::runtime_error(message.str());
    }
  }
}

void execute_pair(NormalizationExecutable &production,
                  NormalizationExecutable &reference,
                  std::span<const functional::BoundTensor> inputs,
                  std::span<functional::BoundTensor> production_outputs,
                  std::span<functional::BoundTensor> reference_outputs,
                  tv::DeviceStream &stream) {
  const auto production_bindings =
      functional::bindings(inputs, production_outputs);
  const auto reference_bindings =
      functional::bindings(inputs, reference_outputs);
  tv::DeviceBuffer production_workspace(production.workspace_size());
  tv::DeviceBuffer reference_workspace(reference.workspace_size());
  tv::check_driver(cuStreamSynchronize(stream.get()),
                   "cuStreamSynchronize(before BatchNorm)");
  functional::execute(production, production_bindings, production_workspace,
                      stream);
  functional::execute(reference, reference_bindings, reference_workspace,
                      stream);
  tv::check_driver(cuStreamSynchronize(stream.get()),
                   "cuStreamSynchronize(after BatchNorm)");
}

void run_layernorm_case(const LayernormTestCase &test_case,
                        const tv::CapabilityRecord &record,
                        flagdnn::Handle &handle,
                        tv::DeviceStream &stream) {
  auto production = build_flagdnn_layernorm(handle, test_case);
  auto reference = tv::make_acdnn_layernorm_reference(test_case, record);
  std::vector<functional::BoundTensor> inputs;
  inputs.push_back(functional::make_input_buffer(
      test_case.x, 0, stream.get(), PointwiseInputDomain::kReal));
  inputs.push_back(functional::make_input_buffer(
      test_case.scale, 1, stream.get(), PointwiseInputDomain::kReal));
  inputs.push_back(functional::make_input_buffer(
      test_case.bias, 2, stream.get(), PointwiseInputDomain::kReal));
  const std::array<const TestTensor *, 3> specifications = {
      &test_case.y, &test_case.mean, &test_case.inv_variance};
  std::vector<functional::BoundTensor> production_outputs;
  std::vector<functional::BoundTensor> reference_outputs;
  for (const TestTensor *specification : specifications) {
    production_outputs.push_back(
        functional::make_output_buffer(*specification, stream.get()));
    reference_outputs.push_back(
        functional::make_output_buffer(*specification, stream.get()));
  }
  execute_pair(*production, *reference, inputs, production_outputs,
               reference_outputs, stream);
  constexpr std::array<std::string_view, 3> names = {
      "y", "mean", "inv_variance"};
  for (std::size_t index = 0; index < names.size(); ++index) {
    compare_output(production_outputs[index], reference_outputs[index],
                   test_case.name, names[index],
                   test_case.absolute_tolerance,
                   test_case.relative_tolerance, stream);
  }
}

void run_rmsnorm_case(const RmsnormTestCase &test_case,
                      const tv::CapabilityRecord &record,
                      flagdnn::Handle &handle,
                      tv::DeviceStream &stream) {
  auto production = build_flagdnn_rmsnorm(handle, test_case);
  auto reference = tv::make_acdnn_rmsnorm_reference(test_case, record);
  std::vector<functional::BoundTensor> inputs;
  inputs.push_back(functional::make_input_buffer(
      test_case.x, 0, stream.get(), PointwiseInputDomain::kReal));
  inputs.push_back(functional::make_input_buffer(
      test_case.scale, 1, stream.get(), PointwiseInputDomain::kReal));
  inputs.push_back(functional::make_input_buffer(
      test_case.bias, 2, stream.get(), PointwiseInputDomain::kReal));
  const std::array<const TestTensor *, 2> specifications = {
      &test_case.y, &test_case.inv_variance};
  std::vector<functional::BoundTensor> production_outputs;
  std::vector<functional::BoundTensor> reference_outputs;
  for (const TestTensor *specification : specifications) {
    production_outputs.push_back(
        functional::make_output_buffer(*specification, stream.get()));
    reference_outputs.push_back(
        functional::make_output_buffer(*specification, stream.get()));
  }
  execute_pair(*production, *reference, inputs, production_outputs,
               reference_outputs, stream);
  constexpr std::array<std::string_view, 2> names = {
      "y", "inv_variance"};
  for (std::size_t index = 0; index < names.size(); ++index) {
    compare_output(production_outputs[index], reference_outputs[index],
                   test_case.name, names[index],
                   test_case.absolute_tolerance,
                   test_case.relative_tolerance, stream);
  }
}

void run_training_case(const BatchnormTestCase &test_case,
                       const tv::CapabilityRecord &record,
                       flagdnn::Handle &handle, tv::DeviceStream &stream) {
  auto production = build_flagdnn_batchnorm(handle, test_case);
  auto reference = tv::make_acdnn_batchnorm_reference(test_case, record);
  std::vector<functional::BoundTensor> inputs;
  inputs.push_back(functional::make_input_buffer(
      test_case.x, 0, stream.get(), PointwiseInputDomain::kReal));
  inputs.push_back(functional::make_input_buffer(
      test_case.scale, 1, stream.get(), PointwiseInputDomain::kReal));
  inputs.push_back(functional::make_input_buffer(
      test_case.bias, 2, stream.get(), PointwiseInputDomain::kReal));
  inputs.push_back(functional::make_input_buffer(
      test_case.previous_running_mean, 3, stream.get(),
      PointwiseInputDomain::kReal));
  inputs.push_back(functional::make_input_buffer(
      test_case.previous_running_variance, 4, stream.get(),
      PointwiseInputDomain::kPositive));

  const std::array<const TestTensor *, 5> specifications = {
      &test_case.y, &test_case.mean, &test_case.inv_variance,
      &test_case.next_running_mean, &test_case.next_running_variance};
  std::vector<functional::BoundTensor> production_outputs;
  std::vector<functional::BoundTensor> reference_outputs;
  for (const TestTensor *specification : specifications) {
    production_outputs.push_back(
        functional::make_output_buffer(*specification, stream.get()));
    reference_outputs.push_back(
        functional::make_output_buffer(*specification, stream.get()));
  }
  execute_pair(*production, *reference, inputs, production_outputs,
               reference_outputs, stream);
  constexpr std::array<std::string_view, 5> names = {
      "y", "mean", "inv_variance", "next_running_mean",
      "next_running_variance"};
  for (std::size_t index = 0; index < names.size(); ++index) {
    compare_output(production_outputs[index], reference_outputs[index],
                   test_case.name, names[index],
                   test_case.absolute_tolerance,
                   test_case.relative_tolerance, stream);
  }
}

void run_inference_case(const BatchnormInferenceTestCase &test_case,
                        const tv::CapabilityRecord &record,
                        flagdnn::Handle &handle, tv::DeviceStream &stream) {
  auto production = build_flagdnn_batchnorm_inference(handle, test_case);
  auto reference =
      tv::make_acdnn_batchnorm_inference_reference(test_case, record);
  std::vector<functional::BoundTensor> inputs;
  inputs.push_back(functional::make_input_buffer(
      test_case.x, 0, stream.get(), PointwiseInputDomain::kReal));
  inputs.push_back(functional::make_input_buffer(
      test_case.mean, 1, stream.get(), PointwiseInputDomain::kReal));
  inputs.push_back(functional::make_input_buffer(
      test_case.inv_variance, 2, stream.get(),
      PointwiseInputDomain::kPositive));
  inputs.push_back(functional::make_input_buffer(
      test_case.scale, 3, stream.get(), PointwiseInputDomain::kReal));
  inputs.push_back(functional::make_input_buffer(
      test_case.bias, 4, stream.get(), PointwiseInputDomain::kReal));
  std::vector<functional::BoundTensor> production_outputs;
  production_outputs.push_back(
      functional::make_output_buffer(test_case.y, stream.get()));
  std::vector<functional::BoundTensor> reference_outputs;
  reference_outputs.push_back(
      functional::make_output_buffer(test_case.y, stream.get()));
  execute_pair(*production, *reference, inputs, production_outputs,
               reference_outputs, stream);
  compare_output(production_outputs.front(), reference_outputs.front(),
                 test_case.name, "y", test_case.absolute_tolerance,
                 test_case.relative_tolerance, stream);
}

template <typename Case, typename RunCase>
int run_suite(int argc, char **argv, std::span<const Case> cases,
              std::string_view operation, std::string_view suite,
              RunCase &&run_case) {
  try {
    if (argc != 3 || cases.empty()) {
      throw std::invalid_argument(
          "THead BatchNorm test requires compiler arguments and cases");
    }
    const tv::CapabilityCatalog catalog = tv::CapabilityCatalog::load(
        FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
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
    std::string filter_name = "FLAGDNN_";
    for (const char character : operation) {
      filter_name.push_back(character == '_' ? '_' : static_cast<char>(
          std::toupper(static_cast<unsigned char>(character))));
    }
    filter_name += "_CASE";
    const char *filter = std::getenv(filter_name.c_str());
    const bool qualify_probes =
        std::getenv("FLAGDNN_THEAD_QUALIFY_PROBES") != nullptr;
    std::size_t selected = 0;
    std::size_t executed = 0;
    std::size_t skipped = 0;
    std::cout << std::setprecision(9);
    for (const Case &test_case : cases) {
      if (filter != nullptr && filter[0] != '\0' &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      ++selected;
      validate_normalization_case(test_case);
      const tv::CapabilityRecord &record =
          catalog.lookup(operation, test_case.name);
      if (record.status == tv::CapabilityStatus::kUnsupported) {
        emit_skip(test_case, record, operation, test_case.x, target);
        ++skipped;
        continue;
      }
      if (record.status == tv::CapabilityStatus::kProbeRequired &&
          !qualify_probes) {
        throw std::runtime_error(
            "unqualified acDNN BatchNorm capability reached execution");
      }
      run_case(test_case, record, handle, stream);
      ++executed;
      std::cout << test_case.name
                << ": FlagDNN Graph vs acDNN BatchNorm PASS\n";
    }
    if (selected == 0 || selected != executed + skipped) {
      throw std::runtime_error("THead BatchNorm case accounting mismatch");
    }
    std::cout << suite << ": PASS cases=" << selected
              << " executed=" << executed << " skipped=" << skipped << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << suite << ": FAIL reason=" << error.what() << '\n';
    return 1;
  }
}

}  // namespace

int run_layernorm_functional_test(
    int argc, char **argv, std::span<const LayernormTestCase> cases) {
  return run_suite(argc, argv, cases, "layernorm",
                   "FLAGDNN_LAYERNORM_FUNCTIONAL", run_layernorm_case);
}

int run_rmsnorm_functional_test(
    int argc, char **argv, std::span<const RmsnormTestCase> cases) {
  return run_suite(argc, argv, cases, "rmsnorm",
                   "FLAGDNN_RMSNORM_FUNCTIONAL", run_rmsnorm_case);
}

int run_batchnorm_functional_test(
    int argc, char **argv, std::span<const BatchnormTestCase> cases) {
  return run_suite(argc, argv, cases, "batchnorm",
                   "FLAGDNN_BATCHNORM_FUNCTIONAL", run_training_case);
}

int run_batchnorm_inference_functional_test(
    int argc, char **argv,
    std::span<const BatchnormInferenceTestCase> cases) {
  return run_suite(argc, argv, cases, "batchnorm_inference",
                   "FLAGDNN_BATCHNORM_INFERENCE_FUNCTIONAL",
                   run_inference_case);
}

}  // namespace flagdnn::testing
