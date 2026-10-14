/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "validation/benchmark/equivalent_baseline.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <locale>
#include <stdexcept>

namespace {
using namespace flagdnn::testing;
namespace baseline = flagdnn::testing::ascend;
void require(bool condition, const char *reason) {
  if (!condition) throw std::runtime_error(reason);
}
template <class Action> void rejects(Action action) {
  bool rejected = false;
  try { action(); } catch (const std::exception &) { rejected = true; }
  require(rejected, "invalid evidence was accepted");
}
struct CommaDecimal final : std::numpunct<char> {
  char do_decimal_point() const override { return ','; }
};
}  // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2 || argc == 3, "expected fixture path and optional --reject");
    const auto catalog = make_unary_pointwise_cases(
        {.operation_name = "identity", .mode = FLAGDNN_POINTWISE_IDENTITY});
    std::vector<PointwiseTestCase> cases;
    for (const auto &spec : catalog)
      if (baseline::supports_equivalent_baseline(spec)) cases.push_back(spec);
    require(cases.size() == 24, "identity FP8 catalog no longer has 24 cases");
    if (argc == 3 && std::string_view(argv[2]) == "--metadata") {
      const baseline::PerformanceBaseline source(argv[1]);
      const auto result = source.lookup(cases.front(), "contract-device", "contract-architecture");
      std::cout << result.json(result.limit_us() * 2) << '\n';
      return 0;
    }
    if (argc == 3) {
      require(std::string_view(argv[2]) == "--reject", "unknown contract mode");
      rejects([&] {
        const baseline::PerformanceBaseline source(argv[1]);
        (void)source.lookup(cases.front(), "contract-device", "contract-architecture");
      });
      return 0;
    }
    const baseline::PerformanceBaseline source(argv[1]);
    // JSON must retain a decimal point even in a process with a localized
    // global C++ locale; a malformed record would otherwise abort reporting.
    const auto previous_locale = std::locale();
    std::locale::global(std::locale(previous_locale, new CommaDecimal));
    const auto localized_result = source.lookup(cases.front(), "contract-device", "contract-architecture");
    const auto localized_metadata = flagdnn::native::json::parse(localized_result.json(2.5));
    require(localized_metadata.at("reference_time_us").as_double() == localized_result.reference_time_us,
            "localized decimal separator leaked into JSON");
    std::locale::global(previous_locale);
    for (const auto &spec : cases) {
      const auto result = source.lookup(spec, "contract-device", "contract-architecture");
      require(std::abs(result.equivalent_time_us() -
                           result.reference_time_us * 2.858 / 1.25) < 1e-12,
              "equivalent formula differs");
      require(result.limit_us() == result.equivalent_time_us() / 0.8,
              "performance limit differs");
      const auto metadata = flagdnn::native::json::parse(result.json(result.limit_us() * 2));
      require(!metadata.at("passed").as_bool() &&
                  metadata.at("target_platform").as_string() == "ascend" &&
                  metadata.at("reference_validation").as_string() == "bit_exact",
              "baseline metadata hides failed performance or validation provenance");
      const auto fixture = baseline::identity_fp8_logical_bytes(spec.inputs[0]);
      const std::vector<std::uint8_t> six =
          spec.inputs[0].data_type == FLAGDNN_DATA_FP8_E4M3
              ? std::vector<std::uint8_t>{0x38, 0x40, 0x48, 0x50, 0x58, 0x60}
              : std::vector<std::uint8_t>{0x3c, 0x40, 0x44, 0x48, 0x4c, 0x50};
      for (std::size_t i = 0; i < fixture.size(); ++i)
        require(fixture[i] == six[i % six.size()], "six-value FP8 fixture differs");
      auto invalid = spec;
      invalid.output.strides.back() += 1;
      rejects([&] { (void)source.lookup(invalid, "contract-device", "contract-architecture"); });
      invalid = spec;
      invalid.alpha = 2;
      rejects([&] { (void)source.lookup(invalid, "contract-device", "contract-architecture"); });
      invalid = spec;
      invalid.inputs[0].binding_byte_offset = 1;
      rejects([&] { (void)source.lookup(invalid, "contract-device", "contract-architecture"); });
    }
    rejects([&] { (void)source.lookup(cases.front(), "other-device", "contract-architecture"); });
    rejects([&] { (void)source.lookup(cases.front(), "contract-device", "other-architecture"); });
    auto invalid = cases.front();
    invalid.name += "_missing";
    rejects([&] { (void)source.lookup(invalid, "contract-device", "contract-architecture"); });
    invalid = cases.front();
    invalid.inputs[0].dimensions[0] = std::numeric_limits<std::int64_t>::max();
    rejects([&] { (void)baseline::identity_fp8_logical_bytes(invalid.inputs[0]); });
    std::cout << "Ascend FP8 equivalent baseline: 24 catalog cases and drift rejection PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
