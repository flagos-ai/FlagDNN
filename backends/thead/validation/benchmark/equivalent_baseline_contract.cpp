/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "benchmark/equivalent_baseline.hpp"
#include "common/cases.hpp"
#include "common/index.hpp"

#include <unistd.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>

namespace {
namespace bench = flagdnn::benchmarking;
namespace baseline = bench::thead;
using flagdnn::native::json::Value;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::size_t rejected = 0;
template <class Action>
void rejects(Action action, std::string_view label) {
  bool failed = false;
  try { action(); } catch (const std::exception&) { failed = true; }
  require(failed, "accepted invalid " + std::string(label));
  ++rejected;
}

void near(double actual, double expected) {
  require(std::isfinite(actual) &&
              std::abs(actual - expected) <= 1.0e-12 * std::abs(expected),
          "equivalent-performance formula differs");
}

bench::BenchmarkCase index_spec(const flagdnn::testing::IndexTestCase& test) {
  bench::BenchmarkCase spec;
  spec.name = test.name;
  spec.operation = bench::Operation::kGraph;
  for (const auto& tensor : test.inputs)
    spec.tensors.push_back({tensor.uid, tensor.data_type, tensor.dimensions,
                           tensor.strides, tensor.binding_byte_offset});
  const auto& tensor = test.output;
  spec.tensors.push_back({tensor.uid, tensor.data_type, tensor.dimensions,
                         tensor.strides, tensor.binding_byte_offset});
  return spec;
}

// Keep only the case under test and the two platforms it actually uses.
Value fixture_document(const Value& source, std::string_view name) {
  auto document = source.as_object();
  auto reference = source.at("reference").as_object();
  reference["cases"] = Value(Value::Object{{std::string(name), source.at("reference").at("cases").at(name)}});
  reference.erase("unavailable_cases");
  const auto& measured = reference.at("measurement");
  reference["measurement"] = Value(Value::Object{
      {"timing_method", measured.at("timing_method")},
      {"statistic", measured.at("statistic")},
      {"timing_profiles", measured.at("timing_profiles")}});
  document["reference"] = Value(std::move(reference));
  document["platforms"] = Value(Value::Object{
      {"thead", source.at("platforms").at("thead")},
      {"nvidia", source.at("platforms").at("nvidia")}});
  return Value(std::move(document));
}

Value replace(const Value& root, std::span<const std::string_view> path, Value value) {
  if (path.empty()) return value;
  auto object = root.as_object();
  const std::string key(path.front());
  object[key] = replace(object.at(key), path.subspan(1), std::move(value));
  return Value(std::move(object));
}

void write_string(std::ostream& out, std::string_view value) {
  constexpr char hex[] = "0123456789abcdef";
  out << '"';
  for (const unsigned char byte : value) {
    if (byte == '"' || byte == '\\') out << '\\' << static_cast<char>(byte);
    else if (byte < 0x20) out << "\\u00" << hex[byte >> 4] << hex[byte & 15];
    else out << static_cast<char>(byte);
  }
  out << '"';
}

void write_json(std::ostream& out, const Value& value) {
  if (value.is_null()) { out << "null"; return; }
  const Value::Object* object = nullptr;
  try { object = &value.as_object(); } catch (const std::exception&) {}
  if (object) {
    out << '{'; bool first = true;
    for (const auto& [key, item] : *object) {
      if (!first) out << ',';
      first = false;
      write_string(out, key);
      out << ':';
      write_json(out, item);
    }
    out << '}'; return;
  }
  const Value::Array* array = nullptr;
  try { array = &value.as_array(); } catch (const std::exception&) {}
  if (array) {
    out << '['; bool first = true;
    for (const auto& item : *array) {
      if (!first) out << ',';
      first = false;
      write_json(out, item);
    }
    out << ']'; return;
  }
  try { write_string(out, value.as_string()); return; } catch (const std::exception&) {}
  try { out << (value.as_bool() ? "true" : "false"); return; } catch (const std::exception&) {}
  out << std::setprecision(17) << value.as_double();
}

class Fixture {
 public:
  Fixture() {
    std::string name = (std::filesystem::temp_directory_path() /
                        "flagdnn-thead-baseline-contract-XXXXXX").string();
    const int fd = ::mkstemp(name.data());
    require(fd >= 0, "cannot create baseline fixture");
    ::close(fd);
    path = name;
  }
  ~Fixture() {
    std::error_code error;
    std::filesystem::remove(path, error);
  }
  void write(const Value& content) const {
    std::ofstream file(path, std::ios::trunc);
    write_json(file, content);
    file.close();
    require(!file.fail(), "cannot write baseline fixture");
  }
  std::filesystem::path path;
};
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "usage: equivalent_baseline_contract BASELINE_JSON");
    std::ifstream input(argv[1]);
    require(input.is_open(), "cannot read production baseline");
    const std::string raw((std::istreambuf_iterator<char>(input)),
                          std::istreambuf_iterator<char>());
    const auto document = flagdnn::native::json::parse(raw);
    const auto& target = document.at("platforms").at("thead");
    const auto& device = target.at("device").as_string();
    const auto& architecture = target.at("architecture").as_string();
    const baseline::PerformanceBaseline production(argv[1]);
    std::size_t measured = 0, missing = 0;
    const auto check = [&](const bench::BenchmarkCase& spec,
                           const baseline::NativeBaselineMetadata* metadata) {
      require(baseline::supports_equivalent_baseline(spec), "measured case outside scope");
      const auto actual = production.lookup(spec, device, architecture, metadata);
      const double time = document.at("reference").at("cases").at(spec.name).at("time_us").as_double();
      const double peak = document.at("platforms").at("nvidia").at("peaks").at(actual.scale_metric).as_double();
      const double target_peak = target.at("peaks").at(actual.scale_metric).as_double();
      near(actual.equivalent_time_us(), time * peak / target_peak);
      near(actual.limit_us(), time * peak / (target_peak * 0.8));
      require(actual.traffic_ratio == 1 && actual.timing_profile == "native", "Thead metadata differs");
      ++measured;
    };
    for (const auto operation : {"concatenate", "gen_index"}) {
      for (const auto& test : flagdnn::testing::make_index_cases(operation)) {
        const auto spec = index_spec(test);
        const baseline::NativeBaselineMetadata metadata{operation, test.axis};
        if (document.at("reference").at("unavailable_cases").as_object().contains(spec.name)) {
          require(!baseline::supports_equivalent_baseline(spec), "missing NVIDIA dtype accepted");
          rejects([&] { (void)production.lookup(spec, device, architecture, &metadata); }, "missing NVIDIA reference");
          ++missing;
        } else {
          check(spec, &metadata);
          auto wrong = metadata;
          ++wrong.axis;
          rejects([&] { (void)production.lookup(spec, device, architecture, &wrong); }, "axis mismatch");
        }
      }
    }
    const auto gelu = bench::binary_pointwise_benchmark_cases(
        FLAGDNN_POINTWISE_GELU_APPROX_TANH_BWD, "gelu_approx_tanh_backward");
    for (const auto& spec : gelu) check(spec, nullptr);
    require(measured == 115 && missing == 70 && gelu.size() == 45, "Thead baseline catalog changed");

    const auto& first = gelu.front();
    const auto lookup = [&](const bench::BenchmarkCase& spec) {
      return production.lookup(spec, device, architecture);
    };
    const auto mutate_spec = [&](auto mutate, std::string_view label) {
      auto spec = first;
      mutate(spec);
      rejects([&] { (void)lookup(spec); }, label);
    };
    mutate_spec([](auto& s) { s.operation = bench::Operation::kRelu; }, "operation");
    mutate_spec([](auto& s) { s.pointwise_mode = FLAGDNN_POINTWISE_DIV; }, "pointwise mode");
    mutate_spec([](auto& s) { s.name += "_unknown"; }, "unknown reference");
    mutate_spec([](auto& s) { s.output_count = 2; }, "output count");
    mutate_spec([](auto& s) { s.tensors.erase(s.tensors.begin()); }, "input count");
    mutate_spec([](auto& s) { s.input_domain = bench::InputDomain::kScaled; }, "input domain");
    mutate_spec([](auto& s) { s.input_domains = {bench::InputDomain::kScaled, bench::InputDomain::kReal}; }, "per-input domain");
    mutate_spec([](auto& s) { s.tensors[1].uid = s.tensors[0].uid; }, "duplicate UID");
    mutate_spec([](auto& s) { ++s.tensors[0].dimensions.back(); }, "shape");
    mutate_spec([](auto& s) { ++s.tensors[0].strides.back(); }, "strides");
    mutate_spec([](auto& s) { s.tensors[0].binding_byte_offset = 4; }, "binding offset");
    mutate_spec([](auto& s) { s.tensors[0].data_type = FLAGDNN_DATA_FLOAT16; }, "dtype");
    mutate_spec([](auto& s) { ++s.pointwise_attributes.version; }, "attribute ABI");
    mutate_spec([](auto& s) { ++s.pointwise_attributes.flags; }, "attribute flags");
    mutate_spec([](auto& s) { s.pointwise_attributes.swish_beta = 2; }, "attribute value");
    mutate_spec([](auto& s) { s.add_alpha = 2; }, "alpha");
    mutate_spec([](auto& s) { --s.benchmark.warmup_iterations; }, "warmup");
    mutate_spec([](auto& s) { --s.benchmark.sample_count; }, "sample count");
    mutate_spec([](auto& s) { --s.benchmark.iterations_per_sample; }, "batch iterations");
    rejects([&] { (void)production.lookup(first, "foreign-device", architecture); }, "foreign device");
    rejects([&] { (void)production.lookup(first, device, "foreign-architecture"); }, "foreign architecture");
    const baseline::NativeBaselineMetadata metadata{"concatenate", 0};
    rejects([&] { (void)production.lookup(first, device, architecture, &metadata); }, "metadata on GELU");
    const auto index = index_spec(flagdnn::testing::make_index_cases("concatenate").front());
    rejects([&] { (void)production.lookup(index, device, architecture); }, "missing index metadata");
    const baseline::NativeBaselineMetadata wrong_operation{"gen_index", 0};
    rejects([&] { (void)production.lookup(index, device, architecture, &wrong_operation); }, "wrong index operation");
    auto graph = index;
    graph.graph.nodes.emplace_back();
    rejects([&] { (void)production.lookup(graph, device, architecture, &metadata); }, "unexpected index graph");
    require(!baseline::supports_equivalent_baseline(bench::binary_pointwise_benchmark_cases(
                FLAGDNN_POINTWISE_DIV, "div", bench::InputDomain::kDivisor).front()),
            "foreign platform DIV scope leaked into Thead");

    Fixture fixture;
    const auto minimal = fixture_document(document, first.name);
    fixture.write(minimal);
    const baseline::PerformanceBaseline isolated(fixture.path);
    near(isolated.lookup(first, device, architecture).limit_us(), lookup(first).limit_us());
    const auto invalid = [&](std::initializer_list<std::string_view> path,
                             Value value, std::string_view label) {
      fixture.write(replace(minimal, std::span(path.begin(), path.size()), std::move(value)));
      rejects([&] {
        const baseline::PerformanceBaseline loader(fixture.path);
        (void)loader.lookup(first, device, architecture);
      }, label);
    };
    invalid({"platforms", "thead"}, Value(), "missing Thead platform");
    invalid({"platforms", "thead", "scaling", "case_policies"}, Value(Value::Object{}), "missing policy");
    invalid({"platforms", "thead", "scaling", "case_policies", "gelu_approx_tanh_backward", "scale_metric"},
            Value(std::string("hbm_copy")), "wrong scale metric");
    for (const auto& value : {Value(), Value(0.0), Value(-1.0), Value(true)})
      invalid({"platforms", "thead", "peaks", "vector_fp32"}, value, "invalid target peak");
    invalid({"platforms", "nvidia", "peaks", "vector_fp32"}, Value(0.0), "invalid reference peak");
    invalid({"metric_units", "vector_fp32"}, Value(std::string("TB/s")), "wrong unit");
    invalid({"min_relative_efficiency"}, Value(0.5), "efficiency threshold");
    invalid({"reference", "cases", first.name, "time_us"}, Value(0.0), "invalid reference time");
    invalid({"reference", "cases", first.name, "validation"}, Value(std::string("sanity")), "insufficient reference validation");
    invalid({"reference", "cases", first.name, "timing_profile"}, Value(std::string("attention")), "timing profile");
    invalid({"reference", "measurement", "timing_method"}, Value(std::string("host_clock")), "timing method");
    invalid({"reference", "measurement", "statistic"}, Value(std::string("mean")), "timing statistic");
    invalid({"reference", "measurement", "timing_profiles", "native", "graph_warmup_replays"}, Value(std::int64_t(0)), "graph warmup");
    invalid({"reference", "provider"}, Value(std::string()), "reference provenance");
    std::cout << "Thead equivalent baseline: " << measured << " measured cases, "
              << missing << " unavailable references, " << rejected
              << " negative checks and isolated platform lookup passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
