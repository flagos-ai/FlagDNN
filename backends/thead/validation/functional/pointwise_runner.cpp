// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "common/pointwise.hpp"
#include "functional/pointwise_runner_support.hpp"

#include "capability.hpp"
#include "pointwise_reference.hpp"
#include "ppu_driver.hpp"

#include <acdnn.h>
#include <cuda.h>
#include <flagdnn/flagdnn.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
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

std::string uppercase(std::string_view value) {
  std::string result(value);
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::toupper(character));
                 });
  return result;
}

std::string operation_from_suite(std::string_view suite_name) {
  constexpr std::string_view prefix = "FLAGDNN_";
  constexpr std::string_view suffix = "_FUNCTIONAL";
  if (!suite_name.starts_with(prefix) || !suite_name.ends_with(suffix) ||
      suite_name.size() <= prefix.size() + suffix.size()) {
    throw std::invalid_argument("malformed THead pointwise suite name");
  }
  std::string operation(
      suite_name.substr(prefix.size(),
                        suite_name.size() - prefix.size() - suffix.size()));
  std::transform(operation.begin(), operation.end(), operation.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  return operation;
}

std::string operation_from_mode(flagdnnPointwiseMode_t mode) {
  switch (mode) {
    case FLAGDNN_POINTWISE_LOGICAL_NOT:
      return "logical_not";
    case FLAGDNN_POINTWISE_LOGICAL_AND:
      return "logical_and";
    case FLAGDNN_POINTWISE_LOGICAL_OR:
      return "logical_or";
    case FLAGDNN_POINTWISE_ERF:
      return "erf";
    case FLAGDNN_POINTWISE_BINARY_SELECT:
      return "binary_select";
    case FLAGDNN_POINTWISE_ADD:
      return "add";
    case FLAGDNN_POINTWISE_SUB:
      return "sub";
    case FLAGDNN_POINTWISE_MUL:
      return "mul";
    case FLAGDNN_POINTWISE_MIN:
      return "min";
    case FLAGDNN_POINTWISE_MAX:
      return "max";
    case FLAGDNN_POINTWISE_RELU_FWD:
      return "relu";
    case FLAGDNN_POINTWISE_SIGMOID_FWD:
      return "sigmoid";
    case FLAGDNN_POINTWISE_TANH_FWD:
      return "tanh";
    case FLAGDNN_POINTWISE_ELU_FWD:
      return "elu";
    case FLAGDNN_POINTWISE_IDENTITY:
      return "identity";
    case FLAGDNN_POINTWISE_GELU_FWD:
      return "gelu";
    case FLAGDNN_POINTWISE_SQRT:
      return "sqrt";
    case FLAGDNN_POINTWISE_NEG:
      return "neg";
    case FLAGDNN_POINTWISE_ABS:
      return "abs";
    case FLAGDNN_POINTWISE_CEIL:
      return "ceil";
    case FLAGDNN_POINTWISE_FLOOR:
      return "floor";
    case FLAGDNN_POINTWISE_EXP:
      return "exp";
    case FLAGDNN_POINTWISE_LOG:
      return "log";
    case FLAGDNN_POINTWISE_COS:
      return "cos";
    case FLAGDNN_POINTWISE_RSQRT:
      return "rsqrt";
    case FLAGDNN_POINTWISE_SIN:
      return "sin";
    case FLAGDNN_POINTWISE_TAN:
      return "tan";
    case FLAGDNN_POINTWISE_SOFTPLUS_FWD:
      return "softplus";
    case FLAGDNN_POINTWISE_SWISH_FWD:
      return "swish";
    case FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD:
      return "gelu_approx_tanh";
    case FLAGDNN_POINTWISE_DIV:
      return "div";
    case FLAGDNN_POINTWISE_POW:
      return "pow";
    case FLAGDNN_POINTWISE_MOD:
      return "mod";
    case FLAGDNN_POINTWISE_SIGMOID_BWD:
      return "sigmoid_backward";
    case FLAGDNN_POINTWISE_RECIPROCAL:
      return "reciprocal";
    case FLAGDNN_POINTWISE_CMP_EQ:
      return "cmp_eq";
    case FLAGDNN_POINTWISE_CMP_NEQ:
      return "cmp_neq";
    case FLAGDNN_POINTWISE_CMP_GT:
      return "cmp_gt";
    case FLAGDNN_POINTWISE_CMP_GE:
      return "cmp_ge";
    case FLAGDNN_POINTWISE_CMP_LT:
      return "cmp_lt";
    case FLAGDNN_POINTWISE_CMP_LE:
      return "cmp_le";
    default:
      throw std::invalid_argument(
          "THead acDNN pointwise builder has no qualified operation mode");
  }
}

std::string operation_from_case(const PointwiseTestCase &test_case) {
  if (test_case.mode == FLAGDNN_POINTWISE_MUL &&
      test_case.name.starts_with("scale_")) {
    return "scale";
  }
  if (test_case.mode == FLAGDNN_POINTWISE_RELU_FWD &&
      test_case.name.starts_with("leaky_relu_")) {
    return "leaky_relu";
  }
  return operation_from_mode(test_case.mode);
}

struct Accuracy {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
};

Accuracy compare(std::span<const float> actual,
                 std::span<const float> expected,
                 const PointwiseTestCase &test_case) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error("FlagDNN and acDNN output sizes differ");
  }
  Accuracy result;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double left = actual[index];
    const double right = expected[index];
    if (left == right || (std::isnan(left) && std::isnan(right))) {
      continue;
    }
    const double absolute = std::abs(left - right);
    const double relative =
        absolute / std::max({std::abs(left), std::abs(right), 1.0e-30});
    result.maximum_absolute = std::max(result.maximum_absolute, absolute);
    result.maximum_relative = std::max(result.maximum_relative, relative);
    if (std::isnan(left) != std::isnan(right) || !std::isfinite(absolute) ||
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
  return result;
}

void emit_skip(const PointwiseTestCase &test_case,
               const tv::CapabilityRecord &record,
               std::string_view operation, std::string_view target) {
  if (record.status != tv::CapabilityStatus::kUnsupported ||
      record.reason_code.empty()) {
    throw std::runtime_error("invalid acDNN pointwise skip capability");
  }
  if (test_case.inputs.empty()) {
    throw std::runtime_error("pointwise skip case has no input");
  }
  const TestTensor &representative = test_case.inputs.front();
  std::cout << "[SKIP][acdnn]"
            << " op=" << operation << " case=" << test_case.name
            << " reason=" << record.reason_code
            << " sdk=" << FLAGDNN_THEAD_PPU_SDK_VERSION
            << " acdnn_header=" << ACDNN_VERSION
            << " acdnn_runtime=" << acdnnGetVersion()
            << " target=" << target
            << " dtype="
            << functional::data_type_name(representative.data_type)
            << " layout=" << functional::layout_name(representative)
            << " shape=" << functional::shape_name(test_case.output) << '\n';
}

void run_case(const PointwiseTestCase &test_case, flagdnn::Handle &handle,
              tv::DeviceStream &stream) {
  auto production = build_flagdnn_pointwise(handle, test_case);
  auto reference = build_pointwise_reference(test_case);

  std::vector<functional::BoundTensor> inputs;
  inputs.reserve(test_case.inputs.size());
  for (std::size_t index = 0; index < test_case.inputs.size(); ++index) {
    inputs.push_back(functional::make_input_buffer(
        test_case.inputs[index], index, stream.get(),
        test_case.input_domains.at(index)));
  }
  std::vector<functional::BoundTensor> production_outputs;
  production_outputs.push_back(
      functional::make_output_buffer(test_case.output, stream.get()));
  std::vector<functional::BoundTensor> reference_outputs;
  reference_outputs.push_back(
      functional::make_output_buffer(test_case.output, stream.get()));

  const std::vector<flagdnnBinding_t> production_bindings =
      functional::bindings(inputs, production_outputs);
  const std::vector<flagdnnBinding_t> reference_bindings =
      functional::bindings(inputs, reference_outputs);
  tv::DeviceBuffer production_workspace(production->workspace_size());
  tv::DeviceBuffer reference_workspace(reference->workspace_size());

  tv::check_driver(cuStreamSynchronize(stream.get()),
                   "cuStreamSynchronize(before pointwise)");
  functional::execute(*production, production_bindings,
                      production_workspace, stream);
  functional::execute(*reference, reference_bindings, reference_workspace,
                      stream);
  tv::check_driver(cuStreamSynchronize(stream.get()),
                   "cuStreamSynchronize(after pointwise)");

  const std::vector<float> production_physical =
      functional::read_output(production_outputs.front(), stream.get());
  const std::vector<float> reference_physical =
      functional::read_output(reference_outputs.front(), stream.get());
  functional::require_padding_unchanged(
      "FlagDNN", production_physical, test_case.output);
  functional::require_padding_unchanged(
      "acDNN", reference_physical, test_case.output);
  const Accuracy accuracy = compare(
      functional::gather(production_physical, test_case.output),
      functional::gather(reference_physical, test_case.output), test_case);
  std::cout << test_case.name << ": FlagDNN Graph vs acDNN PASS"
            << " max_abs=" << accuracy.maximum_absolute
            << " max_rel=" << accuracy.maximum_relative << '\n';
}

}  // namespace

std::unique_ptr<PointwiseExecutable> build_pointwise_reference(
    const PointwiseTestCase &test_case) {
  static const tv::CapabilityCatalog catalog =
      tv::CapabilityCatalog::load(FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
  const std::string operation = operation_from_case(test_case);
  const tv::CapabilityRecord &record =
      catalog.lookup(operation, test_case.name);
  if (record.status == tv::CapabilityStatus::kUnsupported) {
    throw std::invalid_argument(
        "unsupported pointwise case reached acDNN reference builder");
  }
  return tv::make_acdnn_pointwise_reference(
      {.mode = test_case.mode,
       .inputs = test_case.inputs,
       .output = test_case.output,
       .alpha = test_case.alpha,
       .attributes = test_case.attributes},
      record);
}

int run_pointwise_functional_test(int argc, char **argv,
                                  std::span<const PointwiseTestCase> cases,
                                  std::string_view suite_name) {
  try {
    if (argc != 3) {
      throw std::invalid_argument(
          "THead pointwise test requires COMPILER_EXECUTABLE COMPILER_ENTRY");
    }
    if (cases.empty()) {
      throw std::invalid_argument("THead pointwise workload has no cases");
    }
    const tv::CapabilityCatalog catalog = tv::CapabilityCatalog::load(
        FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
    catalog.validate_versions(FLAGDNN_THEAD_PPU_SDK_VERSION, ACDNN_VERSION,
                              static_cast<std::int64_t>(acdnnGetVersion()));

    tv::check_driver(cuInit(0), "cuInit");
    int device_count = 0;
    tv::check_driver(cuDeviceGetCount(&device_count), "cuDeviceGetCount");
    if (device_count <= 0) {
      throw std::runtime_error("PPU driver reported no devices");
    }
    CUdevice device = 0;
    tv::check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    tv::PrimaryContext primary(device);
    tv::ScopedCurrentContext current(primary.get());
    tv::DeviceStream stream;
    functional::TemporaryCache cache;
    flagdnn::Handle handle("thead", 0);
    handle.set_compiler(argv[1], argv[2], cache.path().string());
    const std::string target(handle.target_fingerprint());

    const std::string operation = operation_from_suite(suite_name);
    const std::string filter_name =
        "FLAGDNN_" + uppercase(operation) + "_CASE";
    const char *filter = std::getenv(filter_name.c_str());
    const bool qualify_probes =
        std::getenv("FLAGDNN_THEAD_QUALIFY_PROBES") != nullptr;
    std::size_t selected = 0;
    std::size_t executed = 0;
    std::size_t skipped = 0;
    std::cout << std::setprecision(9);
    for (const PointwiseTestCase &test_case : cases) {
      if (filter != nullptr && filter[0] != '\0' &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      ++selected;
      validate_pointwise_case(test_case);
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
            "unqualified acDNN capability reached pointwise run: " +
            operation + "/" + test_case.name);
      }
      if (operation_from_case(test_case) != operation) {
        throw std::runtime_error(
            "pointwise suite operation and case mode disagree");
      }
      run_case(test_case, handle, stream);
      ++executed;
    }
    if (selected == 0) {
      throw std::runtime_error(filter_name + " matched no test cases");
    }
    if (selected != executed + skipped) {
      throw std::runtime_error("THead pointwise accounting invariant failed");
    }
    std::cout << suite_name << ": " << (executed == 0 ? "SKIP" : "PASS")
              << " cases=" << selected << " executed=" << executed
              << " skipped=" << skipped << '\n';
    return executed == 0 ? 77 : 0;
  } catch (const std::exception &error) {
    std::cerr << suite_name << ": FAIL reason=" << error.what() << '\n';
    return 1;
  }
}

}  // namespace flagdnn::testing
