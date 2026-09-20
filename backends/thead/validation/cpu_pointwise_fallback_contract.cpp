// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "common/pointwise.hpp"
#include "functional/cpu_pointwise.hpp"
#include "functional/pointwise_runner_support.hpp"

#include <flagdnn/flagdnn.hpp>

#include <array>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

namespace tv = flagdnn::validation::thead;
namespace functional = tv::functional;
using flagdnn::testing::PointwiseInputDomain;
using flagdnn::testing::PointwiseTestCase;

class NoOpExecutable final : public flagdnn::testing::TestExecutable {
 public:
  std::size_t workspace_size() const noexcept override { return 0; }
  void execute(std::span<const flagdnnBinding_t>, void*, std::size_t,
               flagdnnStream_t) override {}
};

std::array<PointwiseTestCase, 3> cases() {
  return {{
      {.name = "cpu_fallback_sub_fp16_alpha_neg2",
       .mode = FLAGDNN_POINTWISE_SUB,
       .inputs = {{1, FLAGDNN_DATA_FLOAT16, {1, 3, 8}, {24, 8, 1}},
                  {2, FLAGDNN_DATA_FLOAT16, {1, 3, 8}, {24, 8, 1}}},
       .output = {3, FLAGDNN_DATA_FLOAT16, {1, 3, 8}, {24, 8, 1}},
       .input_domains = {PointwiseInputDomain::kReal,
                         PointwiseInputDomain::kReal},
       .alpha = -2.0,
       .absolute_tolerance = 2.0e-2,
       .relative_tolerance = 1.0e-2},
      {.name = "cpu_fallback_add_bf16_strided_alpha_half",
       .mode = FLAGDNN_POINTWISE_ADD,
       .inputs = {{11, FLAGDNN_DATA_BFLOAT16, {2, 3, 4}, {31, 9, 1}, 16},
                  {12, FLAGDNN_DATA_BFLOAT16, {2, 3, 4}, {37, 11, 1}, 32}},
       .output = {13, FLAGDNN_DATA_BFLOAT16, {2, 3, 4}, {43, 14, 1}, 48},
       .input_domains = {PointwiseInputDomain::kReal,
                         PointwiseInputDomain::kReal},
       .alpha = 0.5,
       .absolute_tolerance = 5.0e-2,
       .relative_tolerance = 1.0e-2},
      {.name = "cpu_fallback_cmp_eq_fp32_bool",
       .mode = FLAGDNN_POINTWISE_CMP_EQ,
       .inputs = {{21, FLAGDNN_DATA_FLOAT32, {1, 1, 16}, {16, 16, 1}},
                  {22, FLAGDNN_DATA_FLOAT32, {1, 1, 16}, {16, 16, 1}}},
       .output = {23, FLAGDNN_DATA_BOOLEAN, {1, 1, 16}, {16, 16, 1}},
       .input_domains = {PointwiseInputDomain::kComparison,
                         PointwiseInputDomain::kComparison}},
  }};
}

}  // namespace

int main(int argc, char** argv) {
  constexpr std::string_view suite =
      "FLAGDNN_THEAD_CPU_POINTWISE_FALLBACK_CONTRACT";
  try {
    if (argc != 3) {
      throw std::invalid_argument(
          "CPU fallback contract requires COMPILER_EXECUTABLE COMPILER_ENTRY");
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
    const auto test_cases = cases();
    for (const auto& test_case : test_cases) {
      auto production =
          flagdnn::testing::build_flagdnn_pointwise(handle, test_case);
      functional::run_cpu_pointwise_case(test_case, *production, stream,
                                         "contract_probe");
    }
    auto near_zero = test_cases.front();
    near_zero.name = "cpu_fallback_reject_unwritten_near_zero";
    near_zero.alpha = 20.0 / 9.0;
    near_zero.input_domains = {PointwiseInputDomain::kScaled,
                               PointwiseInputDomain::kScaled};
    for (auto* tensor : {&near_zero.inputs[0], &near_zero.inputs[1],
                         &near_zero.output}) {
      tensor->data_type = FLAGDNN_DATA_FLOAT32;
      tensor->dimensions = {1, 1, 1};
      tensor->strides = {1, 1, 1};
    }
    NoOpExecutable no_op;
    bool rejected = false;
    try {
      functional::run_cpu_pointwise_case(near_zero, no_op, stream,
                                         "contract_probe");
    } catch (const std::runtime_error& error) {
      if (std::string_view(error.what()).find("differs at output element") ==
          std::string_view::npos) throw;
      rejected = true;
    }
    if (!rejected) throw std::runtime_error("unwritten near-zero output passed");
    std::cout << suite << ": PASS cases=" << test_cases.size()
              << " rejected_noop=1\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << suite << ": FAIL reason=" << error.what() << '\n';
    return 1;
  }
}
