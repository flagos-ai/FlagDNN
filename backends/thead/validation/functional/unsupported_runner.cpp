// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "capability.hpp"

#include "common/attention.hpp"
#include "common/composite.hpp"
#include "common/normalization.hpp"

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

namespace flagdnn::testing {
namespace {

namespace tv = flagdnn::validation::thead;

constexpr int kSkipReturnCode = 77;

std::string uppercase(std::string_view value) {
  std::string result(value);
  std::ranges::transform(result, result.begin(), [](unsigned char character) {
    return static_cast<char>(std::toupper(character));
  });
  return result;
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

bool is_contiguous(const TestTensor &tensor) {
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

std::string layout_name(const TestTensor &tensor) {
  if (is_contiguous(tensor)) {
    return "contiguous";
  }
  if (tensor.dimensions.size() == 4 && tensor.strides.size() == 4 &&
      tensor.strides[1] == 1) {
    return "nhwc";
  }
  return "explicit_strided";
}

std::string shape_name(const TestTensor &tensor) {
  std::string result;
  for (const std::int64_t dimension : tensor.dimensions) {
    if (!result.empty()) {
      result.push_back('x');
    }
    result += std::to_string(dimension);
  }
  return result.empty() ? "scalar" : result;
}

const TestTensor &representative(const SdpaTestCase &test_case) {
  return test_case.q;
}

const TestTensor &representative(const SdpaBackwardTestCase &test_case) {
  return test_case.q;
}

const TestTensor &representative(const SdpaFp8TestCase &test_case) {
  return test_case.q;
}

const TestTensor &representative(
    const SdpaFp8BackwardTestCase &test_case) {
  return test_case.q;
}

std::string case_filter_name(std::string_view operation) {
  return "FLAGDNN_" + uppercase(operation) + "_CASE";
}

template <typename Case>
void emit_skip(const Case &test_case, const tv::CapabilityRecord &record,
               std::string_view operation, std::int64_t runtime_version) {
  if (record.status != tv::CapabilityStatus::kUnsupported ||
      record.reason_code.empty()) {
    throw std::runtime_error(
        "non-unsupported capability reached the THead functional stub");
  }
  const TestTensor &tensor = representative(test_case);
  std::cout << "[SKIP][acdnn]"
            << " op=" << operation << " case=" << test_case.name
            << " reason=" << record.reason_code
            << " sdk=" << FLAGDNN_THEAD_PPU_SDK_VERSION
            << " acdnn_header=" << ACDNN_VERSION
            << " acdnn_runtime=" << runtime_version
            << " target=ppu_static_capability"
            << " dtype=" << data_type_name(tensor.data_type)
            << " layout=" << layout_name(tensor)
            << " shape=" << shape_name(tensor) << '\n';
}

template <typename Case>
int run_unsupported(int argc, char **argv, std::span<const Case> cases,
                    std::string_view operation, std::string_view suite_name) {
  try {
    if (argc != 3) {
      throw std::invalid_argument(
          "THead functional test requires COMPILER_EXECUTABLE COMPILER_ENTRY");
    }
    (void)argv;
    if (cases.empty()) {
      throw std::invalid_argument("THead functional workload has no cases");
    }
    const tv::CapabilityCatalog catalog = tv::CapabilityCatalog::load(
        FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
    const std::int64_t runtime_version =
        static_cast<std::int64_t>(acdnnGetVersion());
    catalog.validate_versions(FLAGDNN_THEAD_PPU_SDK_VERSION, ACDNN_VERSION,
                              runtime_version);

    const std::string filter_name = case_filter_name(operation);
    const char *filter = std::getenv(filter_name.c_str());
    std::size_t selected = 0;
    for (const Case &test_case : cases) {
      if (filter != nullptr && filter[0] != '\0' &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      ++selected;
      emit_skip(test_case, catalog.lookup(operation, test_case.name), operation,
                runtime_version);
    }
    if (selected == 0) {
      throw std::runtime_error(filter_name + " matched no test cases");
    }
    std::cout << suite_name << ": SKIP cases=" << selected
              << " executed=0 skipped=" << selected << '\n';
    return kSkipReturnCode;
  } catch (const std::exception &error) {
    std::cerr << suite_name << ": FAIL reason=" << error.what() << '\n';
    return 1;
  }
}

}  // namespace

int run_sdpa_functional_test(int argc, char **argv,
                             std::span<const SdpaTestCase> cases) {
  return run_unsupported(argc, argv, cases, "sdpa",
                         "FLAGDNN_SDPA_FUNCTIONAL");
}

int run_sdpa_backward_functional_test(
    int argc, char **argv, std::span<const SdpaBackwardTestCase> cases) {
  return run_unsupported(argc, argv, cases, "sdpa_backward",
                         "FLAGDNN_SDPA_BACKWARD_FUNCTIONAL");
}

int run_sdpa_fp8_functional_test(
    int argc, char **argv, std::span<const SdpaFp8TestCase> cases) {
  return run_unsupported(argc, argv, cases, "sdpa_fp8",
                         "FLAGDNN_SDPA_FP8_FUNCTIONAL");
}

int run_sdpa_fp8_backward_functional_test(
    int argc, char **argv, std::span<const SdpaFp8BackwardTestCase> cases) {
  return run_unsupported(argc, argv, cases, "sdpa_fp8_backward",
                         "FLAGDNN_SDPA_FP8_BACKWARD_FUNCTIONAL");
}

}  // namespace flagdnn::testing
