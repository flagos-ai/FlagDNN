// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "capability.hpp"

#include "common/runner.hpp"

#include <acdnn.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

#ifndef FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG
#define FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG "capability.json"
#endif

#ifndef FLAGDNN_THEAD_PPU_SDK_VERSION
#define FLAGDNN_THEAD_PPU_SDK_VERSION "unknown"
#endif

namespace flagdnn::benchmarking {
namespace {

namespace tv = flagdnn::validation::thead;

constexpr int kSkipReturnCode = 77;

std::string operation_from_suite(std::string_view suite_name) {
  constexpr std::string_view prefix = "FLAGDNN_";
  constexpr std::string_view suffix = "_BENCHMARK";
  if (!suite_name.starts_with(prefix) || !suite_name.ends_with(suffix) ||
      suite_name.size() <= prefix.size() + suffix.size()) {
    throw std::invalid_argument("malformed THead benchmark suite name");
  }
  std::string operation(
      suite_name.substr(prefix.size(),
                        suite_name.size() - prefix.size() - suffix.size()));
  std::ranges::transform(operation, operation.begin(),
                         [](unsigned char character) {
                           return static_cast<char>(std::tolower(character));
                         });
  return operation;
}

std::string data_type_name(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_FLOAT32:
      return "fp32";
    case FLAGDNN_DATA_FLOAT16:
      return "fp16";
    case FLAGDNN_DATA_BFLOAT16:
      return "bf16";
    case FLAGDNN_DATA_BOOLEAN:
      return "bool";
    case FLAGDNN_DATA_FP8_E4M3:
      return "fp8_e4m3";
    case FLAGDNN_DATA_FP8_E5M2:
      return "fp8_e5m2";
  }
  return "unknown";
}

bool is_contiguous(const TensorSpec &tensor) {
  if (tensor.dimensions.size() != tensor.strides.size()) {
    return false;
  }
  std::int64_t expected = 1;
  for (std::size_t axis = tensor.dimensions.size(); axis != 0; --axis) {
    const std::int64_t dimension = tensor.dimensions[axis - 1];
    if (dimension <= 0 || tensor.strides[axis - 1] != expected ||
        dimension > std::numeric_limits<std::int64_t>::max() / expected) {
      return false;
    }
    expected *= dimension;
  }
  return true;
}

std::string layout_name(const TensorSpec &tensor) {
  if (is_contiguous(tensor)) {
    return "contiguous";
  }
  if (tensor.dimensions.size() == 4 && tensor.strides.size() == 4 &&
      tensor.strides[1] == 1) {
    return "nhwc";
  }
  return "explicit_strided";
}

std::string shape_name(const TensorSpec &tensor) {
  std::string result;
  for (const std::int64_t dimension : tensor.dimensions) {
    if (!result.empty()) {
      result.push_back('x');
    }
    result += std::to_string(dimension);
  }
  return result.empty() ? "scalar" : result;
}

std::string unsupported_reason(const tv::CapabilityCatalog &catalog,
                               std::string_view operation) {
  const auto operation_iterator = catalog.records().find(operation);
  if (operation_iterator == catalog.records().end() ||
      operation_iterator->second.empty()) {
    throw std::runtime_error(
        "THead benchmark operator is missing from capability catalog");
  }
  std::string reason;
  for (const auto &[case_name, record] : operation_iterator->second) {
    (void)case_name;
    if (record.status != tv::CapabilityStatus::kUnsupported ||
        record.reason_code.empty()) {
      throw std::runtime_error(
          "qualified capability reached the THead benchmark stub");
    }
    if (reason.empty()) {
      reason = record.reason_code;
    } else if (reason != record.reason_code) {
      throw std::runtime_error(
          "THead benchmark stub cannot collapse mixed capability reasons");
    }
  }
  return reason;
}

void emit_skip(const BenchmarkCase &specification, std::string_view operation,
               std::string_view reason, std::int64_t runtime_version) {
  if (specification.tensors.empty()) {
    throw std::runtime_error("THead benchmark case has no tensor metadata");
  }
  const TensorSpec &tensor = specification.tensors.front();
  std::cout << "[SKIP][acdnn]"
            << " op=" << operation << " case=" << specification.name
            << " reason=" << reason
            << " sdk=" << FLAGDNN_THEAD_PPU_SDK_VERSION
            << " acdnn_header=" << ACDNN_VERSION
            << " acdnn_runtime=" << runtime_version
            << " target=ppu_static_capability"
            << " dtype=" << data_type_name(tensor.data_type)
            << " layout=" << layout_name(tensor)
            << " shape=" << shape_name(tensor) << '\n';
}

}  // namespace

int run_benchmark_suite(int argc, char **argv,
                        std::span<const BenchmarkCase> cases,
                        std::string_view suite_name) {
  try {
    if (argc != 3) {
      throw std::invalid_argument(
          "THead benchmark requires COMPILER_EXECUTABLE COMPILER_ENTRY");
    }
    (void)argv;
    if (cases.empty()) {
      throw std::invalid_argument("THead benchmark workload has no cases");
    }
    const std::string operation = operation_from_suite(suite_name);
    const tv::CapabilityCatalog catalog = tv::CapabilityCatalog::load(
        FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
    const std::int64_t runtime_version =
        static_cast<std::int64_t>(acdnnGetVersion());
    catalog.validate_versions(FLAGDNN_THEAD_PPU_SDK_VERSION, ACDNN_VERSION,
                              runtime_version);
    const std::string reason = unsupported_reason(catalog, operation);

    const char *filter = std::getenv("FLAGDNN_BENCHMARK_CASE");
    std::size_t selected = 0;
    for (const BenchmarkCase &specification : cases) {
      if (filter != nullptr && filter[0] != '\0' &&
          specification.name != filter) {
        continue;
      }
      ++selected;
      emit_skip(specification, operation, reason, runtime_version);
    }
    if (selected == 0) {
      throw std::runtime_error(
          "FLAGDNN_BENCHMARK_CASE did not match any case");
    }
    std::cout << suite_name << ": SKIP cases=" << selected
              << " comparable_executed=0 reference_skipped=" << selected
              << '\n';
    return kSkipReturnCode;
  } catch (const std::exception &error) {
    std::cerr << suite_name << ": FAIL reason=" << error.what() << '\n';
    return 1;
  }
}

}  // namespace flagdnn::benchmarking
