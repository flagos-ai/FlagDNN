/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "benchmark/common/cases.hpp"
#include "backends/mthreads/validation/benchmark/equivalent_baseline.hpp"
#include "tests/common/pointwise.hpp"
#include "backends/mthreads/validation/native_pointwise_case.hpp"

namespace {
namespace bench = flagdnn::benchmarking;
namespace baseline_api = flagdnn::benchmarking::mthreads;
namespace test = flagdnn::testing;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
template <typename Action>
void rejects(Action action, std::string_view label) {
  bool failed = false;
  try { action(); } catch (const std::exception&) { failed = true; }
  require(failed, "accepted invalid " + std::string(label));
}
void near(double actual, double expected) {
  require(std::isfinite(actual) &&
              std::abs(actual - expected) <= 1e-12 * std::max(1.0, expected),
          "equivalent baseline formula changed");
}
std::string replace_field(std::string source, std::string_view anchor,
                          std::string_view key, std::string_view value) {
  const auto begin = source.find(anchor);
  require(begin != std::string::npos, "fixture anchor absent");
  const auto field = source.find("\"" + std::string(key) + "\":", begin);
  require(field != std::string::npos, "fixture field absent");
  const auto start = source.find(':', field) + 1;
  const auto end = source.find_first_of(",}", start);
  require(end != std::string::npos, "fixture value absent");
  source.replace(start, end - start, " " + std::string(value));
  return source;
}
std::string replace_dtype_list(std::string source, std::string_view value) {
  const auto target = source.find("\"mthreads\"");
  const auto field = source.find("\"supported_dtypes\":", target);
  require(target != std::string::npos && field != std::string::npos,
          "fixture dtype policy absent");
  const auto start = source.find('[', field);
  const auto end = source.find(']', start);
  require(start != std::string::npos && end != std::string::npos,
          "fixture dtype policy is not an array");
  source.replace(start, end - start + 1, value);
  return source;
}
class Fixture {
 public:
  Fixture() {
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         "flagdnn-mthreads-baseline-contract-XXXXXX").string();
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    const int fd = mkstemp(buffer.data());
    if (fd == -1) throw std::runtime_error("mkstemp failed");
    close(fd);
    path_ = buffer.data();
  }
  ~Fixture() {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
  }
  void write(const std::string& data) const {
    std::ofstream file(path_);
    file << data;
    require(static_cast<bool>(file), "fixture write failed");
  }
  const std::filesystem::path& path() const { return path_; }
 private:
  std::filesystem::path path_;
};
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "usage: mthreads_baseline_contract BASELINE_JSON");
    std::ifstream input(argv[1]);
    require(input.is_open(), "cannot read baseline");
    const std::string raw((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
    const auto document = flagdnn::native::json::parse(raw);
    const auto& target = document.at("platforms").at("mthreads");
    const auto device = target.at("device").as_string();
    const auto arch = target.at("architecture").as_string();
    const baseline_api::PerformanceBaseline baseline(argv[1], "mthreads");
    std::vector<bench::BenchmarkCase> cases;
    for (auto [name, mode] :
         {std::pair{"elu_backward", FLAGDNN_POINTWISE_ELU_BWD},
          std::pair{"softplus_backward", FLAGDNN_POINTWISE_SOFTPLUS_BWD}}) {
      const auto core = bench::binary_pointwise_benchmark_cases(mode, name);
      require(core.size() == 45, "core catalog size changed");
      cases.insert(cases.end(), core.begin(), core.end());
    }
    for (auto [name, mode] :
         {std::pair{"relu_backward", FLAGDNN_POINTWISE_RELU_BWD},
          std::pair{"leaky_relu_backward", FLAGDNN_POINTWISE_RELU_BWD},
          std::pair{"swish_backward", FLAGDNN_POINTWISE_SWISH_BWD}}) {
      test::PointwiseCaseDefinition definition;
      definition.operation_name = name;
      definition.mode = mode;
      if (std::string_view(name) == "leaky_relu_backward") {
        definition.attributes.flags =
            FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE;
        definition.attributes.relu_lower_clip_slope = 0.2;
      }
      for (const auto& value : test::make_binary_pointwise_cases(definition))
        if (value.name.find("_attributes_") != std::string::npos)
          cases.push_back(flagdnn::validation::mthreads::equivalent::as_benchmark(value));
    }
    require(cases.size() == 99, "expected 90 core + 9 native attribute cases");
    for (const auto& spec : cases) {
      require(baseline_api::supports_equivalent_baseline(spec, "mthreads"),
              "catalog case outside scope: " + spec.name);
      const auto value = baseline.lookup(spec, device, arch);
      if (spec.name.find("_attributes_") != std::string::npos) {
        require(spec.input_domains ==
                    std::vector<bench::InputDomain>{bench::InputDomain::kReal,
                                                   bench::InputDomain::kScaled},
                "native adapter lost Real/Scaled attribute input domains");
        auto bad = spec;
        bad.input_domains.clear();
        rejects([&] { (void)baseline.lookup(bad, device, arch); },
                "implicit native attribute domains");
        bad = spec;
        bad.input_domains[1] = bench::InputDomain::kReal;
        rejects([&] { (void)baseline.lookup(bad, device, arch); },
                "unscaled native attribute activation");
        bad = spec;
        std::swap(bad.input_domains[0], bad.input_domains[1]);
        rejects([&] { (void)baseline.lookup(bad, device, arch); },
                "swapped native attribute domains");
      }
      const auto& entry = document.at("reference").at("cases").at(spec.name);
      const auto& policy =
          target.at("scaling").at("case_policies").at(entry.at("op").as_string());
      require(value.scale_metric == policy.at("scale_metric").as_string(),
              "target metric differs");
      near(value.reference_time_us, entry.at("time_us").as_double());
      near(value.traffic_ratio, 1.0);
      near(value.equivalent_time_us(),
           value.reference_time_us * value.reference_peak / value.target_peak);
      near(value.limit_us(),
           value.equivalent_time_us() / value.min_relative_efficiency);
    }
    const auto first = cases.front();
    const auto invalid_case = [&](auto change, std::string_view label) {
      auto spec = first;
      change(spec);
      rejects([&] { (void)baseline.lookup(spec, device, arch); }, label);
    };
    invalid_case([](auto& s) { s.benchmark.sample_count = 7; }, "sample count");
    invalid_case([](auto& s) { s.benchmark.warmup_iterations = 5; }, "warmup");
    invalid_case([](auto& s) { s.benchmark.iterations_per_sample = 20; }, "batch");
    invalid_case([](auto& s) { s.tensors[0].strides.back() = 2; }, "strides");
    invalid_case([](auto& s) { ++s.tensors[0].dimensions.back(); }, "shape");
    invalid_case([](auto& s) { s.tensors[0].binding_byte_offset = 4; }, "offset");
    invalid_case([](auto& s) { s.tensors[0].uid = s.tensors[1].uid; }, "UID");
    invalid_case([](auto& s) { s.pointwise_attributes.elu_alpha = 2; }, "alpha");
    invalid_case([](auto& s) { s.add_alpha = 2; }, "add alpha");
    invalid_case([](auto& s) { s.input_domain = bench::InputDomain::kPositive; },
                 "input domain");
    invalid_case([](auto& s) { s.name += "_not_measured"; }, "missing evidence");
    rejects([&] { (void)baseline.lookup(first, device + "_wrong", arch); },
            "device");
    rejects([&] { (void)baseline.lookup(first, device, arch + "_wrong"); },
            "architecture");
    rejects([&] { baseline_api::PerformanceBaseline bad(argv[1], "unconfigured"); },
            "unknown target");
    // Actual missing native catalogs must not silently use a core FLOAT
    // record, a neighboring shape, or another target's scaling policy.
    std::size_t missing = 0;
    for (auto [name, mode] :
         {std::pair{"elu_backward", FLAGDNN_POINTWISE_ELU_BWD},
          std::pair{"softplus_backward", FLAGDNN_POINTWISE_SOFTPLUS_BWD},
          std::pair{"pow", FLAGDNN_POINTWISE_POW}}) {
      test::PointwiseCaseDefinition definition;
      definition.operation_name = name;
      definition.mode = mode;
      if (mode == FLAGDNN_POINTWISE_POW)
        definition.input_domain = test::PointwiseInputDomain::kPower;
      for (const auto& value : test::make_binary_pointwise_cases(definition)) {
        if (mode == FLAGDNN_POINTWISE_POW &&
            value.output.data_type != FLAGDNN_DATA_INT32) continue;
        const auto spec = flagdnn::validation::mthreads::equivalent::as_benchmark(value);
        require(!document.at("reference").at("cases").as_object()
                     .contains(spec.name),
                "new NVIDIA evidence needs contract coverage: " + spec.name);
        rejects([&] { (void)baseline.lookup(spec, device, arch); },
                "unmeasured native case " + spec.name);
        ++missing;
      }
    }
    require(missing == 105, "missing native evidence count changed");

    Fixture fixture;
    const auto invalid_document = [&](std::string data, std::string_view label) {
      (void)flagdnn::native::json::parse(data);  // Reject for the contract, not malformed JSON.
      fixture.write(data);
      rejects([&] {
        baseline_api::PerformanceBaseline bad(fixture.path(), "mthreads");
        (void)bad.lookup(first, device, arch);
      }, label);
    };
    invalid_document(replace_field(raw, "\"mthreads\"", "vector_fp32", "null"),
                     "missing target peak");
    invalid_document(replace_field(raw, "\"mthreads\"", "vector_fp32", "0"),
                     "nonpositive target peak");
    invalid_document(replace_field(raw, "\"mthreads\"", "measured_on", "\"\""),
                     "missing target provenance");
    invalid_document(replace_field(raw, "\"mthreads\"", "scale_metric",
                                   "\"hbm_copy\""), "wrong target policy");
    invalid_document(replace_field(raw, "\"" + first.name + "\"",
                                   "time_us", "0"), "invalid NVIDIA duration");
    invalid_document(replace_field(raw, "\"" + first.name + "\"",
                                   "validation", "\"sanity\""),
                     "insufficient pointwise validation");
    std::string absent_policy = raw;
    auto position = absent_policy.find("\"case_policies\"",
                                      absent_policy.find("\"mthreads\""));
    require(position != std::string::npos, "target policies absent");
    absent_policy.replace(position, 15, "\"absent_policy\"");
    invalid_document(absent_policy, "no implicit Hygon policy");
    invalid_document(replace_field(raw, "\"schema_version\"",
                                   "min_relative_efficiency", "0.1"),
                     "MThreads efficiency below mandatory 0.8");
    for (std::string_view value :
         {"[\"bfloat16\"]", "[]", "\"fp32\"", "[123]",
          "[\"unknown_dtype\"]", "[\"fp32\", \"fp32\"]"})
      invalid_document(replace_dtype_list(raw, value),
                       "invalid or restricted target dtype policy");
    // A valid narrower policy and a stricter efficiency are permitted.
    fixture.write(replace_dtype_list(raw, "[\"fp32\"]"));
    const baseline_api::PerformanceBaseline fp32_only(fixture.path(), "mthreads");
    (void)fp32_only.lookup(first, device, arch);
    rejects([&] { (void)fp32_only.lookup(cases.at(1), device, arch); },
            "FP16 outside an FP32-only policy");
    fixture.write(replace_field(raw, "\"schema_version\"",
                                "min_relative_efficiency", "0.9"));
    const baseline_api::PerformanceBaseline stricter(fixture.path(), "mthreads");
    const auto strict_value = stricter.lookup(first, device, arch);
    near(strict_value.min_relative_efficiency, 0.9);
    near(strict_value.limit_us(), strict_value.equivalent_time_us() / 0.9);
    // Historical baseline fixtures retain their explicit platform policy.
    const baseline_api::PerformanceBaseline hygon(argv[1], "hygon");
    const auto& hygon_device = document.at("platforms").at("hygon");
    const auto div = bench::binary_pointwise_benchmark_cases(
        FLAGDNN_POINTWISE_DIV, "div", bench::InputDomain::kDivisor);
    for (const auto& spec : div)
      (void)hygon.lookup(spec, hygon_device.at("device").as_string(),
                        hygon_device.at("architecture").as_string());
    fixture.write(replace_field(raw, "\"schema_version\"",
                                "min_relative_efficiency", "0.1"));
    const baseline_api::PerformanceBaseline old_hygon_policy(fixture.path(), "hygon");
    near(old_hygon_policy.lookup(
             div.front(), hygon_device.at("device").as_string(),
             hygon_device.at("architecture").as_string())
             .min_relative_efficiency, 0.1);
    std::cout << "MThreads equivalent baseline contract PASS: "
              << cases.size() << " matched, " << missing
              << " missing evidence rejected, 45 historical div fixtures compatible\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "MTHREADS_BASELINE_CONTRACT_FAILED: " << error.what() << '\n';
    return 1;
  }
}
