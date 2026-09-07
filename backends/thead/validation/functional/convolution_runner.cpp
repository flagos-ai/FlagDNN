// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "acdnn_convolution_reference.hpp"
#include "capability.hpp"
#include "common/convolution.hpp"
#include "pointwise_runner_support.hpp"

#include <acdnn.h>

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
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

constexpr int kSkipReturnCode = 77;
constexpr char kIsolatedBackwardChild[] =
    "FLAGDNN_THEAD_ISOLATED_BACKWARD_FUNCTIONAL_CHILD";

std::string operation_name(ConvolutionDirection direction) {
  switch (direction) {
    case ConvolutionDirection::kFprop:
      return "conv_fprop";
    case ConvolutionDirection::kDgrad:
      return "conv_dgrad";
    case ConvolutionDirection::kWgrad:
      return "conv_wgrad";
  }
  throw std::invalid_argument("unknown convolution direction");
}

std::string uppercase(std::string value) {
  std::ranges::transform(value, value.begin(), [](unsigned char character) {
    return static_cast<char>(std::toupper(character));
  });
  return value;
}

void emit_skip(const ConvolutionTestCase &test_case,
               const tv::CapabilityRecord &record,
               std::string_view operation, std::string_view target) {
  if (record.status != tv::CapabilityStatus::kUnsupported ||
      record.reason_code.empty()) {
    throw std::runtime_error("invalid acDNN convolution skip capability");
  }
  std::cout << "[SKIP][acdnn]"
            << " op=" << operation << " case=" << test_case.name
            << " reason=" << record.reason_code
            << " sdk=" << FLAGDNN_THEAD_PPU_SDK_VERSION
            << " acdnn_header=" << ACDNN_VERSION
            << " acdnn_runtime=" << acdnnGetVersion()
            << " target=" << target
            << " dtype=" << functional::data_type_name(test_case.x.data_type)
            << " layout=" << functional::layout_name(test_case.x)
            << " shape=" << functional::shape_name(test_case.x) << '\n';
}

void compare(std::span<const float> actual,
             std::span<const float> expected,
             const ConvolutionTestCase &test_case) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error("FlagDNN and acDNN convolution sizes differ");
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
    if (std::isnan(left) != std::isnan(right) ||
        !std::isfinite(absolute) ||
        (absolute > test_case.absolute_tolerance &&
         relative > test_case.relative_tolerance)) {
      std::ostringstream message;
      message << test_case.name << " differs at output element " << index
              << ": FlagDNN=" << left << " acDNN=" << right
              << " abs=" << absolute << " rel=" << relative
              << " atol=" << test_case.absolute_tolerance
              << " rtol=" << test_case.relative_tolerance;
      throw std::runtime_error(message.str());
    }
  }
}

std::vector<const TestTensor *> input_specs(
    const ConvolutionTestCase &test_case) {
  switch (test_case.direction) {
    case ConvolutionDirection::kFprop:
      return {&test_case.x, &test_case.w};
    case ConvolutionDirection::kDgrad:
      return {&test_case.y, &test_case.w};
    case ConvolutionDirection::kWgrad:
      return {&test_case.y, &test_case.x};
  }
  throw std::invalid_argument("unknown convolution direction");
}

// acDNN runtime 1400 retains backward-convolution geometry state across
// plans. A fresh process keeps each reference case independent; the parent
// shares one production artifact cache so isolation does not recompile it.
int run_isolated_backward_suite(
    char **argv, std::span<const ConvolutionTestCase> cases,
    std::string_view operation, std::string_view suite,
    const tv::CapabilityCatalog &catalog, const std::string &filter_name,
    std::string_view filter, bool qualify_probes) {
  functional::TemporaryCache cache;
  if (::setenv("FLAGDNN_CACHE_PATH", cache.path().c_str(), 1) != 0 ||
      ::setenv(kIsolatedBackwardChild, "1", 1) != 0) {
    throw std::runtime_error(
        "cannot configure isolated backward functional environment");
  }

  std::size_t selected = 0;
  std::size_t executed = 0;
  std::size_t skipped = 0;
  for (const ConvolutionTestCase &test_case : cases) {
    if (!filter.empty() &&
        test_case.name.find(filter) == std::string::npos) {
      continue;
    }
    ++selected;
    const tv::CapabilityRecord &record =
        catalog.lookup(operation, test_case.name);
    if (record.status == tv::CapabilityStatus::kProbeRequired &&
        !qualify_probes) {
      throw std::runtime_error(
          "unqualified acDNN convolution capability reached execution");
    }
    if (::setenv(filter_name.c_str(), test_case.name.c_str(), 1) != 0) {
      throw std::runtime_error(
          "cannot select isolated backward functional case");
    }
    std::cout.flush();
    std::cerr.flush();
    const pid_t child = ::fork();
    if (child < 0) {
      throw std::runtime_error(
          "fork failed for isolated backward functional case");
    }
    if (child == 0) {
      ::execv(argv[0], argv);
      ::_exit(126);
    }

    int status = 0;
    pid_t waited = -1;
    do {
      waited = ::waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    const int expected =
        record.status == tv::CapabilityStatus::kUnsupported
            ? kSkipReturnCode
            : 0;
    if (waited != child || !WIFEXITED(status) ||
        WEXITSTATUS(status) != expected) {
      std::cerr << suite << ": FAIL isolated_case=" << test_case.name
                << " expected_exit=" << expected;
      if (waited == child && WIFEXITED(status)) {
        std::cerr << " actual_exit=" << WEXITSTATUS(status);
      } else if (waited == child && WIFSIGNALED(status)) {
        std::cerr << " signal=" << WTERMSIG(status);
      } else {
        std::cerr << " waitpid_error=" << errno;
      }
      std::cerr << '\n';
      return 1;
    }
    if (record.status == tv::CapabilityStatus::kUnsupported) {
      ++skipped;
    } else {
      ++executed;
    }
  }
  if (selected == 0 || selected != executed + skipped) {
    throw std::runtime_error("THead convolution case accounting mismatch");
  }
  std::cout << suite << ": " << (executed == 0 ? "SKIP" : "PASS")
            << " cases=" << selected << " executed=" << executed
            << " skipped=" << skipped << '\n';
  return executed == 0 ? kSkipReturnCode : 0;
}

}  // namespace

std::unique_ptr<ConvolutionExecutable> build_convolution_reference(
    const ConvolutionTestCase &test_case) {
  static const tv::CapabilityCatalog catalog = tv::CapabilityCatalog::load(
      FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
  const std::string operation = operation_name(test_case.direction);
  const tv::CapabilityRecord &record =
      catalog.lookup(operation, test_case.name);
  if (record.status == tv::CapabilityStatus::kUnsupported) {
    throw std::invalid_argument(
        "unsupported convolution reached acDNN reference builder");
  }
  return tv::make_acdnn_convolution_reference(test_case, record);
}

int run_convolution_functional_test(
    int argc, char **argv, std::span<const ConvolutionTestCase> cases,
    ConvolutionDirection expected_direction) {
  const std::string operation = operation_name(expected_direction);
  const std::string suite =
      "FLAGDNN_" + uppercase(operation) + "_FUNCTIONAL";
  try {
    if (argc != 3 || cases.empty()) {
      throw std::invalid_argument(
          "THead convolution test requires compiler arguments and cases");
    }
    const tv::CapabilityCatalog catalog = tv::CapabilityCatalog::load(
        FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
    catalog.validate_versions(FLAGDNN_THEAD_PPU_SDK_VERSION, ACDNN_VERSION,
                              static_cast<std::int64_t>(acdnnGetVersion()));
    const std::string filter_name =
        "FLAGDNN_" + uppercase(operation) + "_CASE";
    const char *filter_environment = std::getenv(filter_name.c_str());
    const std::string filter =
        filter_environment == nullptr ? "" : filter_environment;
    const bool qualify_probes =
        std::getenv("FLAGDNN_THEAD_QUALIFY_PROBES") != nullptr;
    if (expected_direction != ConvolutionDirection::kFprop &&
        std::getenv(kIsolatedBackwardChild) == nullptr) {
      return run_isolated_backward_suite(
          argv, cases, operation, suite, catalog, filter_name, filter,
          qualify_probes);
    }
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
    std::size_t selected = 0;
    std::size_t executed = 0;
    std::size_t skipped = 0;
    std::cout << std::setprecision(9);
    for (const ConvolutionTestCase &test_case : cases) {
      if (!filter.empty() &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      ++selected;
      validate_convolution_case(test_case);
      if (test_case.direction != expected_direction) {
        throw std::runtime_error("convolution suite direction mismatch");
      }
      const tv::CapabilityRecord &record =
          catalog.lookup(operation, test_case.name);
      if (record.status == tv::CapabilityStatus::kUnsupported) {
        emit_skip(test_case, record, operation, target);
        ++skipped;
        continue;
      }
      if (record.status == tv::CapabilityStatus::kProbeRequired &&
          !qualify_probes) {
        throw std::runtime_error(
            "unqualified acDNN convolution capability reached execution");
      }

      auto production = build_flagdnn_convolution(handle, test_case);
      auto reference =
          tv::make_acdnn_convolution_reference(test_case, record);
      std::vector<functional::BoundTensor> inputs;
      const auto specifications = input_specs(test_case);
      for (std::size_t index = 0; index < specifications.size(); ++index) {
        inputs.push_back(functional::make_input_buffer(
            *specifications[index], index, stream.get(),
            PointwiseInputDomain::kTan));
      }
      const TestTensor &output_specification =
          convolution_output_tensor(test_case);
      std::vector<functional::BoundTensor> production_outputs;
      production_outputs.push_back(
          functional::make_output_buffer(output_specification, stream.get()));
      std::vector<functional::BoundTensor> reference_outputs;
      reference_outputs.push_back(
          functional::make_output_buffer(output_specification, stream.get()));
      const auto production_bindings =
          functional::bindings(inputs, production_outputs);
      const auto reference_bindings =
          functional::bindings(inputs, reference_outputs);
      tv::DeviceBuffer production_workspace(production->workspace_size());
      tv::DeviceBuffer reference_workspace(reference->workspace_size());
      tv::check_driver(cuStreamSynchronize(stream.get()),
                       "cuStreamSynchronize(before convolution)");
      functional::execute(*production, production_bindings,
                          production_workspace, stream);
      functional::execute(*reference, reference_bindings,
                          reference_workspace, stream);
      tv::check_driver(cuStreamSynchronize(stream.get()),
                       "cuStreamSynchronize(after convolution)");
      const std::vector<float> production_physical =
          functional::read_output(production_outputs.front(), stream.get());
      const std::vector<float> reference_physical =
          functional::read_output(reference_outputs.front(), stream.get());
      functional::require_padding_unchanged(
          "FlagDNN", production_physical, output_specification);
      functional::require_padding_unchanged(
          "acDNN", reference_physical, output_specification);
      compare(functional::gather(production_physical, output_specification),
              functional::gather(reference_physical, output_specification),
              test_case);
      ++executed;
      std::cout << test_case.name
                << ": FlagDNN Graph vs acDNN convolution PASS\n";
    }
    if (selected == 0 || selected != executed + skipped) {
      throw std::runtime_error("THead convolution case accounting mismatch");
    }
    if (std::getenv(kIsolatedBackwardChild) == nullptr) {
      std::cout << suite << ": " << (executed == 0 ? "SKIP" : "PASS")
                << " cases=" << selected << " executed=" << executed
                << " skipped=" << skipped << '\n';
    }
    return executed == 0 ? kSkipReturnCode : 0;
  } catch (const std::exception &error) {
    std::cerr << suite << ": FAIL reason=" << error.what() << '\n';
    return 1;
  }
}

}  // namespace flagdnn::testing
