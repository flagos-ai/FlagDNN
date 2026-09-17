// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "benchmark/capability_contract.hpp"

#include "benchmark/corex_cudnn_provider.hpp"
#include "common/cases.hpp"

#include <algorithm>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::iluvatar::validation::benchmark {
namespace {

using flagdnn::benchmarking::BenchmarkCase;
using flagdnn::benchmarking::ProviderCapability;

void require_capability(const BenchmarkCase &test_case,
                        const ProviderCapability &capability,
                        bool expected_supported,
                        std::string_view expected_reason) {
  if (capability.supported != expected_supported ||
      capability.reason != expected_reason) {
    throw std::runtime_error(
        "benchmark capability mismatch for " + test_case.name +
        ": supported=" + (capability.supported ? "true" : "false") +
        " reason=" + capability.reason);
  }
}

template <typename Expectation>
void check_cases(CorexCudnnProvider &provider,
                 const std::vector<BenchmarkCase> &cases,
                 Expectation &&expectation) {
  for (const BenchmarkCase &test_case : cases) {
    const auto [supported, reason] = expectation(test_case);
    require_capability(test_case, provider.capability(test_case), supported,
                       reason);
  }
}

std::pair<bool, std::string_view> fp32_only(const BenchmarkCase &test_case) {
  if (test_case.tensors.front().data_type == FLAGDNN_DATA_FLOAT32) {
    return {true, {}};
  }
  return {false, "DTYPE_UNSUPPORTED"};
}

std::pair<bool, std::string_view> no_bfloat16(const BenchmarkCase &test_case) {
  if (test_case.tensors.front().data_type != FLAGDNN_DATA_BFLOAT16) {
    return {true, {}};
  }
  return {false, "DTYPE_UNSUPPORTED"};
}

std::pair<bool, std::string_view>
vendor_rejects_bfloat16(const BenchmarkCase &test_case) {
  if (test_case.tensors.front().data_type != FLAGDNN_DATA_BFLOAT16) {
    return {true, {}};
  }
  return {false, "CUDNN_STATUS_NOT_SUPPORTED"};
}

} // namespace

void run_capability_contract() {
  using namespace flagdnn::benchmarking;
  CorexCudnnProvider provider;

  // The qualified INT8 ADD/MUL/MAX compositions implement boolean inputs.
  // Floating-input truth conversion still has no matching DNN contract.
  const auto boolean_only = [](const BenchmarkCase &test_case) {
    if (test_case.tensors.front().data_type == FLAGDNN_DATA_BOOLEAN)
      return std::pair<bool, std::string_view>{true, {}};
    return std::pair<bool, std::string_view>{false, "DTYPE_UNSUPPORTED"};
  };
  check_cases(provider,
              binary_pointwise_benchmark_cases(FLAGDNN_POINTWISE_LOGICAL_AND,
                                               "logical_and",
                                               InputDomain::kLogical),
              boolean_only);
  check_cases(provider,
              binary_pointwise_benchmark_cases(FLAGDNN_POINTWISE_LOGICAL_OR,
                                               "logical_or",
                                               InputDomain::kLogical),
              boolean_only);
  check_cases(provider,
              unary_pointwise_benchmark_cases(FLAGDNN_POINTWISE_LOGICAL_NOT,
                                              "logical_not",
                                              InputDomain::kLogical),
              boolean_only);

  check_cases(provider, batchnorm_benchmark_cases(), fp32_only);
  check_cases(provider, batchnorm_inference_benchmark_cases(),
              [](const BenchmarkCase &) {
                return std::pair<bool, std::string_view>{true, {}};
              });
  check_cases(provider, reduction_benchmark_cases(), vendor_rejects_bfloat16);
  check_cases(provider, conv_dgrad_benchmark_cases(), vendor_rejects_bfloat16);
  check_cases(provider, conv_wgrad_benchmark_cases(),
              [](const BenchmarkCase &test_case) {
                // New shared channels-last BF16 cases execute successfully.
                if (test_case.tensors.front().strides[1] == 1)
                  return std::pair<bool, std::string_view>{true, {}};
                return vendor_rejects_bfloat16(test_case);
              });
  check_cases(
      provider, conv_fprop_benchmark_cases(),
      [](const BenchmarkCase &test_case) {
        if (test_case.convolution.spatial_rank == 1 &&
            test_case.convolution.pre_padding !=
                test_case.convolution.post_padding)
          return std::pair<bool, std::string_view>{false, "SEMANTIC_MISMATCH"};
        if (test_case.tensors.front().data_type == FLAGDNN_DATA_BFLOAT16 &&
            test_case.convolution.spatial_rank == 1) {
          return std::pair<bool, std::string_view>{false, "DTYPE_UNSUPPORTED"};
        }
        return std::pair<bool, std::string_view>{true, {}};
      });
  check_cases(provider, conv_bias_relu_benchmark_cases(),
              vendor_rejects_bfloat16);
  check_cases(
      provider,
      unary_pointwise_benchmark_cases(FLAGDNN_POINTWISE_IDENTITY, "identity"),
      no_bfloat16);
  check_cases(provider,
              unary_pointwise_benchmark_cases(FLAGDNN_POINTWISE_NEG, "neg"),
              no_bfloat16);
  auto neg_correctness_cases =
      unary_pointwise_cases(FLAGDNN_POINTWISE_NEG, "neg");
  const auto neg_strided =
      std::find_if(neg_correctness_cases.begin(), neg_correctness_cases.end(),
                   [](const BenchmarkCase &test_case) {
                     return test_case.name == "neg_strided_fp32_2x3x4";
                   });
  if (neg_strided == neg_correctness_cases.end()) {
    throw std::runtime_error(
        "neg explicit-stride capability fixture is missing");
  }
  BenchmarkCase neg_strided_benchmark = *neg_strided;
  neg_strided_benchmark.name = "neg_perf_strided_fp32_2x3x4";
  require_capability(neg_strided_benchmark,
                     provider.capability(neg_strided_benchmark), false,
                     "LAYOUT_UNSUPPORTED");
  check_cases(provider, reshape_benchmark_cases(), no_bfloat16);
  check_cases(provider, slice_benchmark_cases(), no_bfloat16);
  check_cases(provider, transpose_benchmark_cases(), no_bfloat16);
}

} // namespace flagdnn::iluvatar::validation::benchmark
