// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "benchmark/equivalent_baseline.hpp"
#include "benchmark/shared_cases.hpp"
#include "common/cases.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace b = flagdnn::benchmarking;
template <class Action> void rejects(Action action) {
  try { action(); } catch (const std::exception &) { return; }
  throw std::runtime_error("accepted inconsistent equivalent baseline");
}
int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::invalid_argument("expected baseline path");
    b::PerformanceBaseline baseline(argv[1]);
    const auto cases = b::binary_pointwise_benchmark_cases(
        FLAGDNN_POINTWISE_DIV, "div", b::InputDomain::kDivisor);
    if (cases.empty()) throw std::runtime_error("empty div catalog");
    const auto lookup = [&](const b::BenchmarkCase &spec) {
      return baseline.lookup(spec, "Iluvatar BI-V150", "corex_71");
    };
    for (const auto &spec : cases) {
      const auto result = lookup(spec);
      const double expected = result.reference_time_us * result.reference_peak /
                              result.target_peak;
      if (result.equivalent_time_us() != expected ||
          result.limit_us() != expected / result.min_relative_efficiency)
        throw std::runtime_error("incorrect equivalent-performance formula");
    }
    // Exercise the actual shared adapter and both supplemental batches.
    // A missing record must fail; an in-scope case must never become a skip.
    const auto reductions =
        flagdnn::iluvatar::validation::benchmark::shared_benchmark_cases(
            "FLAGDNN_REDUCTION_BENCHMARK");
    if (reductions.size() != 97)
      throw std::runtime_error("unexpected retained mixed-reduction catalog");
    for (auto spec : reductions) {
      if (!spec.name.ends_with("_shared"))
        throw std::runtime_error("missing shared adapter suffix");
      spec.name.resize(spec.name.size() - 7);
      spec.benchmark = b::equivalent_benchmark_config(spec.operation);
      (void)lookup(spec);
      ++spec.reduction_axis;
      rejects([&] { (void)lookup(spec); });
    }
    std::size_t added_core = 0;
    for (auto spec : b::unary_pointwise_benchmark_cases(
             FLAGDNN_POINTWISE_IDENTITY, "identity")) {
      if (spec.tensors.front().dimensions !=
          std::vector<std::int64_t>{1, 1, 1}) continue;
      spec.benchmark = b::equivalent_benchmark_config(spec.operation);
      (void)lookup(spec);
      ++added_core;
    }
    for (const auto factory : {b::reshape_benchmark_cases,
                                b::transpose_benchmark_cases,
                                b::slice_benchmark_cases}) {
      for (auto spec : factory()) {
        if (spec.tensors.front().data_type != FLAGDNN_DATA_BFLOAT16) continue;
        // Shared layout timing is intentionally different from the newly
        // measured native reference; reject it before selecting that profile.
        rejects([&] { (void)lookup(spec); });
        spec.benchmark = b::equivalent_benchmark_config(spec.operation);
        (void)lookup(spec);
        ++added_core;
        spec.tensors.back().strides.back() += 1;
        rejects([&] { (void)lookup(spec); });
      }
    }
    if (added_core != 17)
      throw std::runtime_error("incomplete supplemented core catalog");

    const auto original = cases.front();
    rejects([&] { (void)baseline.lookup(original, "different device", "corex_71"); });
    rejects([&] { (void)baseline.lookup(original, "Iluvatar BI-V150", "corex_80"); });
    auto altered = original;
    altered.tensors[0].dimensions.back() += 1;
    rejects([&] { lookup(altered); });
    altered = original;
    altered.tensors[0].strides.back() += 1;
    rejects([&] { lookup(altered); });
    altered = original;
    altered.tensors[0].data_type = FLAGDNN_DATA_FLOAT16;
    rejects([&] { lookup(altered); });
    altered = original;
    altered.tensors[0].binding_byte_offset += 16;
    rejects([&] { lookup(altered); });
    altered = original;
    altered.input_domain = b::InputDomain::kReal;
    rejects([&] { lookup(altered); });
    altered = original;
    altered.benchmark.iterations_per_sample += 1;
    rejects([&] { lookup(altered); });
    altered = original;
    altered.name += "_unmeasured";
    rejects([&] { lookup(altered); });
    auto numeric = lookup(original);
    numeric.target_peak = 0;
    rejects([&] { (void)numeric.limit_us(); });
    numeric = lookup(original);
    numeric.reference_time_us = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { (void)numeric.limit_us(); });
    numeric = lookup(original);
    numeric.min_relative_efficiency = 1.1;
    rejects([&] { (void)numeric.limit_us(); });
    numeric = lookup(original);
    numeric.traffic_ratio = 2;
    rejects([&] { (void)numeric.limit_us(); });
    std::cout << "PASS Iluvatar equivalent baseline contract\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
