// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "benchmark/equivalent_runner.hpp"
#include "common/index.hpp"
#include "extended_reference.hpp"
#include "functional/cpu_reference.hpp"
#include "reference/cpu/index.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <iomanip>
#include <iostream>

namespace flagdnn::testing {
namespace f = iluvatar::validation::functional;
namespace b = benchmarking;

int run_index_equivalent_benchmark(int argc, char **argv,
                                   std::span<const IndexTestCase> cases) {
  if (cases.empty()) throw std::invalid_argument("empty index benchmark");
  const auto operation = cases.front().operation;
  std::string marker = operation;
  std::transform(marker.begin(), marker.end(), marker.begin(),
                  [](unsigned char c) { return std::toupper(c); });
  f::FunctionalSuite suite(argc, argv, operation,
                            "FLAGDNN_" + marker + "_BENCHMARK");
  std::cout << std::setprecision(17);
  for (const auto &test_case : cases) {
    const char *filter = std::getenv("FLAGDNN_BENCHMARK_CASE");
    if (filter && *filter && test_case.name != filter) continue;
    f::CasePlan plan{operation, test_case.name, {},
                     {{test_case.output, 0, 0, "output"}}};
    reference::cpu::IndexParameters parameters{
        operation, {}, test_case.output.dimensions, test_case.axis};
    for (const auto &input : test_case.inputs) {
      plan.inputs.push_back({input, f::InputDomain::kReal, {}});
      parameters.input_shapes.push_back(input.dimensions);
    }
    const auto oracle = [&](const std::vector<std::vector<float>> &inputs) {
      std::vector<std::vector<std::byte>> bytes;
      for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto encoded = f::encode(inputs[i], test_case.inputs[i].data_type);
        bytes.emplace_back(encoded.size());
        std::memcpy(bytes.back().data(), encoded.data(), encoded.size());
      }
      const auto output = reference::cpu::evaluate_index_bytes(
          parameters, bytes, test_case.output.data_type);
      const auto storage = std::span(
          reinterpret_cast<const std::uint8_t *>(output.data()), output.size());
      return std::vector<std::vector<float>>{f::decode(
          storage, test_case.output.data_type, f::element_count(test_case.output))};
    };
    const auto reason = iluvatar::validation::extended_reference_skip(test_case);
    if (reason.empty()) {
      for (auto &input : plan.inputs) {
        input.exact_values.resize(f::element_count(input.tensor));
        for (std::size_t i = 0; i < input.exact_values.size(); ++i)
          input.exact_values[i] = static_cast<float>(static_cast<int>(i % 61) - 30);
      }
      suite.run(plan,
          [&] { return build_flagdnn_index(suite.handle(), test_case); },
          [&] { return iluvatar::validation::build_extended_reference(test_case); },
          oracle, true);
      continue;
    }
    auto specification = f::cpu_case(plan);
    specification.operation = b::Operation::kGraph;
    if (!b::supports_equivalent_baseline(specification)) {
      suite.skip_benchmark_case(plan, reason);
      std::cout << "[unavailable-equivalent] case=" << specification.name
                << " reason=OUTSIDE_NVIDIA_REFERENCE_COVERAGE\n";
      continue;
    }
    suite.run_external_benchmark(plan, [&] {
      b::run_equivalent_native(specification,
          [&] { return build_flagdnn_index(suite.handle(), test_case); },
          oracle, {operation, test_case.axis});
    });
  }
  return suite.finish();
}
} // namespace flagdnn::testing
