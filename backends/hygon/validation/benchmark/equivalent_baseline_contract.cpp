/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "equivalent_baseline.hpp"
#include "cases.hpp"
#include "common/cases.hpp"
#include "common/index.hpp"
#include "common/layout.hpp"
#include "common/pointwise.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>

namespace {

namespace bench = flagdnn::benchmarking;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Action>
void rejects(Action action, std::string_view label) {
  bool rejected = false;
  try {
    action();
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, "accepted invalid " + std::string(label));
}

void near(double actual, double expected) {
  require(std::isfinite(actual) &&
              std::abs(actual - expected) <=
                  1.0e-12 * std::max(1.0, std::abs(expected)),
          "equivalent-performance formula differs");
}

std::string replace_after(std::string source, std::string_view anchor,
                          std::string_view needle,
                          std::string_view replacement) {
  const auto begin = source.find(anchor);
  require(begin != std::string::npos, "missing fixture anchor");
  const auto pos = source.find(needle, begin);
  require(pos != std::string::npos, "missing fixture value");
  source.replace(pos, needle.size(), replacement);
  return source;
}

std::string replace_field(std::string source, std::string_view anchor,
                          std::string_view key, std::string_view value) {
  const auto begin = source.find(anchor);
  require(begin != std::string::npos, "missing fixture field anchor");
  const std::string marker = "\"" + std::string(key) + "\":";
  const auto field = source.find(marker, begin);
  require(field != std::string::npos, "missing fixture field");
  const auto start = field + marker.size();
  const auto end = source.find_first_of(",}", start);
  require(end != std::string::npos, "missing fixture field delimiter");
  source.replace(start, end - start, " " + std::string(value));
  return source;
}

class Fixture {
 public:
  Fixture() {
    std::string name =
        (std::filesystem::temp_directory_path() /
         "flagdnn-div-baseline-contract-XXXXXX").string();
    const int fd = ::mkstemp(name.data());
    require(fd >= 0, "cannot create baseline fixture");
    ::close(fd);
    path = name;
  }
  ~Fixture() {
    std::error_code error;
    std::filesystem::remove(path, error);
  }
  void write(const std::string& content) {
    std::ofstream file(path, std::ios::trunc);
    file << content;
    file.close();
    require(!file.fail(), "cannot write baseline fixture");
  }
  std::filesystem::path path;
};

struct CatalogCase {
  bench::BenchmarkCase specification;
  std::optional<bench::NativeBaselineMetadata> metadata;
};

bench::TensorSpec tensor(const flagdnn::testing::TestTensor& value) {
  return {value.uid, value.data_type, value.dimensions, value.strides,
          value.binding_byte_offset};
}

bool floating(flagdnnDataType_t type) {
  return type == FLAGDNN_DATA_FLOAT32 || type == FLAGDNN_DATA_FLOAT16 ||
         type == FLAGDNN_DATA_BFLOAT16;
}

std::vector<CatalogCase> full_catalog() {
  using Domain = bench::InputDomain;
  struct Pointwise {
    flagdnnPointwiseMode_t mode;
    std::string_view name;
    Domain domain;
    bool binary;
  };
  const Pointwise operations[] = {
      {FLAGDNN_POINTWISE_DIV, "div", Domain::kDivisor, true},
      {FLAGDNN_POINTWISE_POW, "pow", Domain::kPower, true},
      {FLAGDNN_POINTWISE_MOD, "mod", Domain::kModulo, true},
      {FLAGDNN_POINTWISE_CMP_EQ, "cmp_eq", Domain::kComparison, true},
      {FLAGDNN_POINTWISE_CMP_NEQ, "cmp_neq", Domain::kComparison, true},
      {FLAGDNN_POINTWISE_CMP_GT, "cmp_gt", Domain::kComparison, true},
      {FLAGDNN_POINTWISE_CMP_GE, "cmp_ge", Domain::kComparison, true},
      {FLAGDNN_POINTWISE_CMP_LT, "cmp_lt", Domain::kComparison, true},
      {FLAGDNN_POINTWISE_CMP_LE, "cmp_le", Domain::kComparison, true},
      {FLAGDNN_POINTWISE_LOGICAL_AND, "logical_and", Domain::kLogical, true},
      {FLAGDNN_POINTWISE_LOGICAL_OR, "logical_or", Domain::kLogical, true},
      {FLAGDNN_POINTWISE_NEG, "neg", Domain::kReal, false},
      {FLAGDNN_POINTWISE_SQRT, "sqrt", Domain::kPositive, false},
      {FLAGDNN_POINTWISE_RSQRT, "rsqrt", Domain::kPositive, false},
      {FLAGDNN_POINTWISE_RECIPROCAL, "reciprocal", Domain::kPositive, false},
      {FLAGDNN_POINTWISE_CEIL, "ceil", Domain::kScaled, false},
      {FLAGDNN_POINTWISE_FLOOR, "floor", Domain::kScaled, false},
      {FLAGDNN_POINTWISE_EXP, "exp", Domain::kReal, false},
      {FLAGDNN_POINTWISE_LOG, "log", Domain::kPositive, false},
      {FLAGDNN_POINTWISE_ERF, "erf", Domain::kReal, false},
      {FLAGDNN_POINTWISE_SIN, "sin", Domain::kReal, false},
      {FLAGDNN_POINTWISE_COS, "cos", Domain::kReal, false},
      {FLAGDNN_POINTWISE_TAN, "tan", Domain::kTan, false},
      {FLAGDNN_POINTWISE_RELU_FWD, "leaky_relu", Domain::kReal, false},
      {FLAGDNN_POINTWISE_RELU_BWD, "leaky_relu_backward", Domain::kReal, true},
      {FLAGDNN_POINTWISE_ELU_BWD, "elu_backward", Domain::kReal, true},
      {FLAGDNN_POINTWISE_GELU_FWD, "gelu", Domain::kReal, false},
      {FLAGDNN_POINTWISE_GELU_BWD, "gelu_backward", Domain::kReal, true},
      {FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD, "gelu_approx_tanh", Domain::kReal, false},
      {FLAGDNN_POINTWISE_GELU_APPROX_TANH_BWD, "gelu_approx_tanh_backward", Domain::kReal, true},
      {FLAGDNN_POINTWISE_SOFTPLUS_FWD, "softplus", Domain::kReal, false},
      {FLAGDNN_POINTWISE_SOFTPLUS_BWD, "softplus_backward", Domain::kReal, true},
  };
  std::vector<CatalogCase> result;
  const auto append = [&](const std::vector<bench::BenchmarkCase>& cases) {
    for (auto spec : bench::aligned_benchmark_cases(cases))
      result.push_back({std::move(spec), std::nullopt});
  };
  for (const auto& operation : operations) {
    auto cases = operation.binary
                     ? bench::binary_pointwise_benchmark_cases(
                           operation.mode, operation.name, operation.domain)
                     : bench::unary_pointwise_benchmark_cases(
                           operation.mode, operation.name, operation.domain);
    for (auto& spec : cases) {
      if (operation.name == "leaky_relu" ||
          operation.name == "leaky_relu_backward") {
        spec.pointwise_attributes.flags =
            FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE;
        spec.pointwise_attributes.relu_lower_clip_slope = 0.2;
      } else if (operation.name == "softplus") {
        spec.pointwise_attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA;
      }
    }
    append(cases);
  }
  append(bench::binary_select_benchmark_cases());

  namespace testing = flagdnn::testing;
  testing::PointwiseCaseDefinition identity;
  identity.operation_name = "identity";
  identity.mode = FLAGDNN_POINTWISE_IDENTITY;
  std::size_t fp8_identity_count = 0;
  for (const auto& value : testing::make_unary_pointwise_cases(identity)) {
    if (value.output.data_type != FLAGDNN_DATA_FP8_E4M3 &&
        value.output.data_type != FLAGDNN_DATA_FP8_E5M2) continue;
    bench::BenchmarkCase spec;
    spec.name = value.name;
    spec.operation = bench::Operation::kPointwise;
    spec.pointwise_mode = value.mode;
    spec.pointwise_attributes = value.attributes;
    spec.add_alpha = value.alpha;
    spec.absolute_tolerance = value.absolute_tolerance;
    spec.relative_tolerance = value.relative_tolerance;
    for (const auto& input : value.inputs) spec.tensors.push_back(tensor(input));
    spec.tensors.push_back(tensor(value.output));
    result.push_back({std::move(spec), std::nullopt});
    ++fp8_identity_count;
  }
  require(fp8_identity_count == 24, "expected exactly 24 native FP8 identity cases");
  testing::PointwiseCaseDefinition logical;
  logical.operation_name = "logical_not";
  logical.mode = FLAGDNN_POINTWISE_LOGICAL_NOT;
  logical.input_domain = testing::PointwiseInputDomain::kLogical;
  for (const auto& value : testing::make_unary_pointwise_cases(logical)) {
    require(value.output.data_type == FLAGDNN_DATA_BOOLEAN,
            "logical_not native catalog dtype changed");
    bench::BenchmarkCase spec;
    spec.name = value.name;
    spec.operation = bench::Operation::kPointwise;
    spec.pointwise_mode = value.mode;
    spec.pointwise_attributes = value.attributes;
    spec.add_alpha = value.alpha;
    spec.input_domain = Domain::kLogical;
    for (const auto& input : value.inputs) spec.tensors.push_back(tensor(input));
    spec.tensors.push_back(tensor(value.output));
    result.push_back({std::move(spec), std::nullopt});
  }
  for (auto operation : {testing::LayoutOperation::kReshape,
                         testing::LayoutOperation::kTranspose,
                         testing::LayoutOperation::kSlice}) {
    for (const auto& value : testing::make_layout_cases(operation)) {
      if (!floating(value.input.data_type)) continue;
      bench::BenchmarkCase spec;
      spec.name = value.name;
      spec.operation = operation == testing::LayoutOperation::kReshape
                           ? bench::Operation::kReshape
                           : operation == testing::LayoutOperation::kTranspose
                                 ? bench::Operation::kTranspose
                                 : bench::Operation::kSlice;
      spec.tensors = {tensor(value.input), tensor(value.output)};
      spec.reshape = {value.output.dimensions, value.output.strides, true};
      spec.transpose.permutation = value.permutation;
      spec.slice = {value.slices, value.slice_strides};
      result.push_back({std::move(spec), std::nullopt});
    }
  }
  for (std::string_view operation : {"concatenate", "gen_index"}) {
    for (const auto& value : testing::make_index_cases(operation)) {
      if (operation == "concatenate" && !floating(value.output.data_type)) continue;
      bench::BenchmarkCase spec;
      spec.name = value.name;
      spec.operation = bench::Operation::kGraph;
      for (const auto& input : value.inputs) spec.tensors.push_back(tensor(input));
      spec.tensors.push_back(tensor(value.output));
      result.push_back({std::move(spec),
                        bench::NativeBaselineMetadata{value.operation, value.axis}});
    }
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "usage: equivalent_baseline_contract BASELINE_JSON");
    const std::filesystem::path path = argv[1];
    std::ifstream file(path);
    require(file.is_open(), "cannot open baseline for contract");
    const std::string raw((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());
    const auto document = flagdnn::native::json::parse(raw);
    const auto& target = document.at("platforms").at("hygon");
    const auto& reference = document.at("reference");
    const auto& policies = target.at("scaling").at("case_policies");
    require(policies.as_object().size() == 40,
            "expected 40 Hygon-specific scaling policies");
    for (const auto& [name, entry] : reference.at("cases").as_object())
      require(entry.at("scale_metric").is_null(),
              "shared reference metric must remain unknown: " + name);
    const std::string device = target.at("device").as_string();
    const std::string architecture = target.at("architecture").as_string();
    const double p_nv =
        document.at("platforms").at("nvidia").at("peaks")
            .at("vector_fp32").as_double();
    const double p_target = target.at("peaks").at("vector_fp32").as_double();
    const double alpha = document.at("min_relative_efficiency").as_double();
    const bench::DivPerformanceBaseline baseline(path);
    const auto cases = bench::binary_pointwise_benchmark_cases(
        FLAGDNN_POINTWISE_DIV, "div", bench::InputDomain::kDivisor);
    require(cases.size() == 45, "expected 45 catalog Div performance cases");
    for (const auto& spec : cases) {
      const auto value = baseline.lookup(spec, device, architecture);
      const double time =
          reference.at("cases").at(spec.name).at("time_us").as_double();
      require(value.scale_metric == "vector_fp32" &&
                  value.timing_profile == "native",
              "unexpected metric or timing profile");
      near(value.reference_time_us, time);
      near(value.reference_peak, p_nv);
      near(value.target_peak, p_target);
      near(value.min_relative_efficiency, alpha);
      near(value.equivalent_time_us(), time * p_nv / p_target);
      near(value.limit_us(), time * p_nv / (alpha * p_target));
    }

    const auto catalog = full_catalog();
    require(catalog.size() == 1570, "expected 1570 current equivalent catalog cases");
    std::set<std::string> operations;
    std::set<std::string> names;
    for (const auto& item : catalog) {
      const auto& spec = item.specification;
      require(bench::supports_equivalent_baseline(spec),
              "missing equivalent scope for " + spec.name);
      require(names.insert(spec.name).second, "duplicate contract catalog case");
      const auto* metadata = item.metadata ? &*item.metadata : nullptr;
      const auto value = baseline.lookup(spec, device, architecture, metadata);
      const auto& entry = reference.at("cases").at(spec.name);
      const auto operation = entry.at("op").as_string();
      operations.insert(operation);
      require(value.scale_metric ==
                  policies.at(operation).at("scale_metric").as_string(),
              "lookup did not use the Hygon operation policy");
      require(value.reference_validation == entry.at("validation").as_string(),
              "reference validation provenance was lost");
      near(value.reference_peak,
           document.at("platforms").at("nvidia").at("peaks")
               .at(value.scale_metric).as_double());
      near(value.target_peak, target.at("peaks").at(value.scale_metric).as_double());
      const bool comparison = operation.starts_with("cmp_");
      const auto dtype = spec.tensors.front().data_type;
      const double float_bits = dtype == FLAGDNN_DATA_FLOAT32 ? 32.0 : 16.0;
      const double ratio = comparison ? (2 * float_bits + 8) / (2 * float_bits + 1)
                           : operation.starts_with("logical_") ? 8.0
                           : operation == "binary_select"
                               ? (3 * float_bits + 8) / (3 * float_bits + 1)
                               : 1.0;
      near(value.traffic_ratio, ratio);
      near(value.equivalent_time_us(),
           entry.at("time_us").as_double() * value.reference_peak /
               value.target_peak * ratio);
      near(value.limit_us(), value.equivalent_time_us() / alpha);

      auto unknown = spec;
      unknown.name += "_unknown_evidence";
      require(bench::supports_equivalent_baseline(unknown),
              "missing evidence was misclassified as out of scope");
      rejects([&] { (void)baseline.lookup(unknown, device, architecture, metadata); },
              "unknown reference " + spec.name);
    }
    require(operations.size() == 40, "not all 40 operations were validated");

    const auto find_case = [&](std::string_view prefix) -> const CatalogCase& {
      const auto found = std::find_if(catalog.begin(), catalog.end(),
          [&](const auto& item) { return item.specification.name.starts_with(prefix); });
      require(found != catalog.end(), "missing contract case " + std::string(prefix));
      return *found;
    };
    const auto invalid_catalog_spec = [&](std::string_view prefix, auto mutate,
                                          std::string_view label) {
      auto item = find_case(prefix);
      mutate(item);
      rejects([&] {
        (void)baseline.lookup(item.specification, device, architecture,
                               item.metadata ? &*item.metadata : nullptr);
      }, label);
    };
    invalid_catalog_spec("leaky_relu_perf_", [](auto& item) {
      item.specification.pointwise_attributes.relu_lower_clip_slope = 0.3;
    }, "leaky ReLU slope");
    invalid_catalog_spec("softplus_perf_", [](auto& item) {
      item.specification.pointwise_attributes.softplus_beta = 2;
    }, "softplus beta");
    invalid_catalog_spec("binary_select_", [](auto& item) {
      item.specification.input_domains.clear();
    }, "binary select predicate domain");
    invalid_catalog_spec("transpose_", [](auto& item) {
      std::reverse(item.specification.transpose.permutation.begin(),
                   item.specification.transpose.permutation.end());
    }, "transpose permutation");
    invalid_catalog_spec("slice_", [](auto& item) {
      ++item.specification.slice.slices[0].first;
    }, "slice bounds");
    invalid_catalog_spec("reshape_", [](auto& item) {
      item.specification.reshape.logical = false;
    }, "reshape logical flag");
    invalid_catalog_spec("concatenate_", [](auto& item) {
      item.metadata->operation = "gen_index";
    }, "native operation");
    invalid_catalog_spec("concatenate_", [](auto& item) {
      ++item.metadata->axis;
    }, "concatenate axis");
    invalid_catalog_spec("gen_index_", [](auto& item) {
      item.metadata.reset();
    }, "missing native metadata");
    invalid_catalog_spec("gen_index_", [](auto& item) {
      item.specification.graph.nodes.emplace_back();
    }, "nonempty native graph carrier");
    invalid_catalog_spec("logical_not_", [](auto& item) {
      item.specification.tensors[0].data_type = FLAGDNN_DATA_INT32;
    }, "logical datatype");
    for (const auto type : {FLAGDNN_DATA_FLOAT32, FLAGDNN_DATA_FLOAT16,
                            FLAGDNN_DATA_BFLOAT16, FLAGDNN_DATA_INT32,
                            FLAGDNN_DATA_BOOLEAN, FLAGDNN_DATA_FP8_E8M0}) {
      auto item = find_case("identity_fp8_e4m3_");
      for (auto& tensor : item.specification.tensors) tensor.data_type = type;
      require(!bench::supports_equivalent_baseline(item.specification),
              "identity broadened beyond the two evidenced FP8 dtypes");
    }
    invalid_catalog_spec("identity_fp8_e4m3_", [](auto& item) {
      item.specification.tensors.back().data_type = FLAGDNN_DATA_FP8_E5M2;
    }, "mixed FP8 identity dtypes");
    invalid_catalog_spec("identity_fp8_e4m3_", [](auto& item) {
      item.specification.absolute_tolerance = 0.01;
    }, "nonzero identity absolute tolerance");
    invalid_catalog_spec("identity_fp8_e5m2_", [](auto& item) {
      item.specification.relative_tolerance = 0.01;
    }, "nonzero identity relative tolerance");
    invalid_catalog_spec("identity_fp8_e4m3_", [](auto& item) {
      item.specification.pointwise_attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA;
    }, "identity flags");
    invalid_catalog_spec("identity_fp8_e5m2_", [](auto& item) {
      item.specification.add_alpha = 2;
    }, "identity alpha");
    invalid_catalog_spec("identity_fp8_e4m3_", [](auto& item) {
      item.specification.pointwise_attributes.elu_alpha = 2;
    }, "identity implicit attribute");
    invalid_catalog_spec("identity_fp8_e4m3_", [](auto& item) {
      ++item.specification.benchmark.iterations_per_sample;
    }, "identity native timing");
    auto fp8_div = cases.front();
    for (auto& tensor : fp8_div.tensors) tensor.data_type = FLAGDNN_DATA_FP8_E4M3;
    require(!bench::supports_equivalent_baseline(fp8_div),
            "FP8 dtype escaped identity-only scope");
    auto ordinary_relu = cases.front();
    ordinary_relu.pointwise_mode = FLAGDNN_POINTWISE_RELU_FWD;
    ordinary_relu.name = "relu_perf_fp32_1x1x1024";
    require(!bench::supports_equivalent_baseline(ordinary_relu),
            "ordinary ReLU must retain its original benchmark policy");

    const auto& first = cases.front();
    const auto invalid_spec = [&](auto mutate, std::string_view label) {
      auto spec = first;
      mutate(spec);
      rejects([&] { (void)baseline.lookup(spec, device, architecture); }, label);
    };
    invalid_spec([](auto& s) { s.name += "_missing"; }, "case name");
    invalid_spec([](auto& s) { s.operation = bench::Operation::kAdd; },
                 "operation");
    invalid_spec([](auto& s) { s.pointwise_mode = FLAGDNN_POINTWISE_MUL; },
                 "pointwise mode");
    invalid_spec([](auto& s) { ++s.tensors[0].dimensions.back(); }, "shape");
    invalid_spec([](auto& s) { ++s.tensors[1].strides.front(); }, "strides");
    invalid_spec([](auto& s) {
      s.tensors[2].data_type = FLAGDNN_DATA_FLOAT16;
    }, "dtype");
    invalid_spec([](auto& s) { s.tensors[1].binding_byte_offset = 4; },
                 "binding offset");
    invalid_spec([](auto& s) { s.output_count = 2; }, "output count");
    invalid_spec([](auto& s) { s.tensors[1].uid = s.tensors[0].uid; },
                 "duplicate tensor UIDs");
    invalid_spec([](auto& s) { s.input_domain = bench::InputDomain::kReal; },
                 "input domain");
    invalid_spec([](auto& s) {
      s.input_domains = {bench::InputDomain::kReal, bench::InputDomain::kDivisor};
    }, "per-input domain");
    invalid_spec([](auto& s) {
      s.pointwise_attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA;
    }, "pointwise attributes");
    invalid_spec([](auto& s) { s.pointwise_attributes.elu_alpha = 2; },
                 "unrecorded pointwise attribute");
    invalid_spec([](auto& s) { ++s.pointwise_attributes.version; },
                 "pointwise attribute ABI");
    invalid_spec([](auto& s) { ++s.benchmark.warmup_iterations; }, "warmup");
    invalid_spec([](auto& s) { ++s.benchmark.sample_count; }, "sample count");
    invalid_spec([](auto& s) { ++s.benchmark.iterations_per_sample; },
                 "iterations per sample");
    rejects([&] { (void)baseline.lookup(first, device + "_other", architecture); },
            "device");
    rejects([&] { (void)baseline.lookup(first, device, architecture + "_other"); },
            "architecture");

    Fixture fixture;
    const auto bad_json = [&](const std::string& content,
                              std::string_view label) {
      fixture.write(content);
      rejects([&] {
        const bench::DivPerformanceBaseline invalid(fixture.path);
        (void)invalid.lookup(first, device, architecture);
      }, label);
    };
    const std::string anchor = "\"" + first.name + "\"";
    for (const std::string value :
         {"null", "0", "-1", "NaN", "1e999", "true", "\"14.6\""}) {
      bad_json(replace_field(raw, "\"hygon\"", "vector_fp32", value),
               "target peak " + value);
    }
    bad_json(replace_field(raw, "\"nvidia\"", "vector_fp32", "0"),
             "NVIDIA peak");
    bad_json(replace_field(raw, "", "min_relative_efficiency", "1.1"),
             "alpha above one");
    bad_json(replace_field(raw, "", "min_relative_efficiency", "0"),
             "zero alpha");
    bad_json(replace_after(raw, "", "\"schema_version\": 4",
                           "\"schema_version\": 3"), "schema");
    bad_json(replace_after(raw, "\"reference\"", "\"platform\": \"nvidia\"",
                           "\"platform\": \"hygon\""), "reference platform");
    bad_json(replace_field(raw, "\"hygon\"", "measured_on", "\"\""),
             "measurement date");
    bad_json(replace_field(raw, anchor, "time_us", "0"), "reference time");
    const std::string div_policy = "\"div\": {";
    for (const std::string value :
         {"null", "0", "true", "[]", "\"\"", "\"vector_fp16\"", "\"unknown\""}) {
      bad_json(replace_field(raw, div_policy, "scale_metric", value),
               "Hygon scale metric " + value);
    }
    const auto with_foreign_policies = [&](std::string content,
                                          std::string_view metric) {
      const std::string quoted_metric = "\"" + std::string(metric) + "\"";
      content = replace_field(std::move(content), anchor, "scale_metric",
                              quoted_metric);
      const std::string policy =
          "\"case_policies\": {\"div\": {\"scale_metric\": " +
          quoted_metric + "}},";
      content = replace_after(std::move(content), "\"platforms\"",
                              "\"nvidia\": {",
                              "\"nvidia\": {\"scaling\": {" + policy +
                                  "\"status\": \"foreign_platform_policy\"},");
      return replace_after(std::move(content), "\"reference\"",
                           "\"scaling\": {", "\"scaling\": {" + policy);
    };
    const std::pair<std::string, std::string_view> missing_policies[] = {
        {replace_after(raw, "\"hygon\"", "\"scaling\":",
                       "\"missing_scaling\":"), "Hygon scaling object"},
        {replace_after(raw, "\"hygon\"", "\"case_policies\":",
                       "\"missing_case_policies\":"), "Hygon policy map"},
        {replace_after(raw, "\"hygon\"", div_policy, "\"missing_div\": {"),
         "Hygon operation policy"},
        {replace_after(raw, div_policy, "\"scale_metric\":",
                       "\"missing_scale_metric\":"), "Hygon scale metric"},
    };
    for (const auto& [content, label] : missing_policies) {
      bad_json(content, label);
      bad_json(with_foreign_policies(content, "vector_fp32"),
               std::string(label) + " despite matching foreign policies");
    }
    // Neither a shared case metric nor another platform's matching/conflicting
    // policy may substitute for or override the explicit Hygon policy.
    for (std::string_view metric : {"vector_fp32", "hbm_copy"}) {
      fixture.write(with_foreign_policies(raw, metric));
      const bench::PerformanceBaseline isolated(fixture.path);
      const auto value = isolated.lookup(first, device, architecture);
      require(value.scale_metric == "vector_fp32",
              "foreign policy overrode the Hygon scale metric");
      near(value.limit_us(),
           baseline.lookup(first, device, architecture).limit_us());
    }
    bad_json(replace_after(raw, anchor, "\"validation\": \"cpu_full\"",
                           "\"validation\": \"sanity\""), "reference validation");
    bad_json(replace_after(raw, anchor, "\"role\": \"input\"",
                           "\"role\": \"output\""), "tensor role");
    bad_json(replace_after(raw, anchor, "\"op\": \"div\"",
                           "\"op\": \"mul\""), "reference operation");
    bad_json(replace_after(raw, anchor, "\"timing_profile\": \"native\"",
                           "\"timing_profile\": \"attention\""), "timing profile");
    bad_json(replace_after(raw, "", "\"graph_warmup_replays\": 1",
                           "\"graph_warmup_replays\": 2"), "graph warmup");
    bad_json(replace_after(raw, "\"reference\"", "nearest_rank_p50",
                           "mean"), "timing statistic");
    bad_json(replace_after(raw, "", "device_events_over_cuda_graph",
                           "host_clock"), "timing method");
    bad_json(replace_after(raw, "", "\"vector_fp32\": \"TFLOP/s\"",
                           "\"vector_fp32\": \"GFLOP/s\""), "peak unit");
    bad_json(replace_after(raw, "", anchor, "\"missing_case\""),
             "missing reference case");
    bad_json("{", "malformed JSON");

    const auto bad_catalog_json = [&](std::string_view prefix,
                                      std::string_view needle,
                                      std::string_view replacement,
                                      std::string_view label,
                                      std::string_view policy_anchor = {}) {
      const auto& item = find_case(prefix);
      const auto& spec = item.specification;
      const std::string mutation_anchor = policy_anchor.empty()
                                              ? "\"" + spec.name + "\""
                                              : std::string(policy_anchor);
      fixture.write(replace_after(raw, mutation_anchor, needle, replacement));
      rejects([&] {
        const bench::PerformanceBaseline invalid(fixture.path);
        (void)invalid.lookup(spec, device, architecture,
                             item.metadata ? &*item.metadata : nullptr);
      }, label);
    };
    bad_catalog_json("cmp_gt_perf_", "\"reference_storage_bits\": 1",
                     "\"reference_storage_bits\": 8", "wrong packed BOOL width");
    bad_catalog_json("logical_not_", "\"reference_storage_bits\": 1",
                     "\"unknown_storage_bits\": 1", "missing packed BOOL width");
    bad_catalog_json("binary_select_", "\"scale_metric\": \"hbm_triad\"",
                     "\"scale_metric\": \"vector_fp32\"", "BOOL compute proxy",
                     "\"binary_select\": {");
    bad_catalog_json("leaky_relu_perf_", "\"relu_lower_clip_slope\": 0.2",
                     "\"relu_lower_clip_slope\": 0.3", "reference slope");
    bad_catalog_json("reshape_", "\"validation\": \"sanity\"",
                     "\"validation\": \"unknown\"", "unknown layout validation");
    bad_catalog_json("gen_index_", "\"axis\": 0", "\"axis\": -1",
                     "reference index axis");
    bad_catalog_json("sqrt_perf_", "\"pointwise_flags\": 0",
                     "\"unknown_flags\": 0", "unknown attribute schema");
    for (std::string_view validation : {"cpu_full", "cpu_sampled", "sanity", "unknown"})
      bad_catalog_json("identity_fp8_e4m3_", "\"validation\": \"bit_exact\"",
                       "\"validation\": \"" + std::string(validation) + "\"",
                       "identity reference must be bit-exact");
    bad_catalog_json("identity_fp8_e5m2_", "\"dtype\": \"fp8_e5m2\"",
                     "\"dtype\": \"fp8_e4m3\"", "identity reference dtype");
    bad_catalog_json("identity_fp8_e4m3_", "\"alpha\": 1", "\"alpha\": 2",
                     "identity reference alpha");
    bad_catalog_json("identity_fp8_e4m3_", "\"scale_metric\": \"hbm_copy\"",
                     "\"scale_metric\": \"vector_fp32\"", "identity metric",
                     "\"identity\": {");

    bench::EquivalentBaseline arithmetic{
        "vector_fp32", "native", 2.0, 60.0, 15.0, 0.8, 1.0, "cpu_full"};
    near(arithmetic.equivalent_time_us(), 8.0);
    near(arithmetic.limit_us(), 10.0);
    arithmetic.traffic_ratio = 8;
    rejects([&] { (void)arithmetic.limit_us(); }, "compute proxy traffic correction");
    arithmetic.scale_metric = "hbm_copy";
    near(arithmetic.equivalent_time_us(), 64.0);
    near(arithmetic.limit_us(), 80.0);
    for (double invalid : {0.0, -1.0,
                            std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()}) {
      arithmetic.traffic_ratio = invalid;
      rejects([&] { (void)arithmetic.limit_us(); }, "invalid traffic ratio");
    }
    arithmetic.traffic_ratio = 1;
    arithmetic.reference_time_us = std::numeric_limits<double>::max();
    rejects([&] { (void)arithmetic.limit_us(); }, "overflow");
    arithmetic.reference_time_us = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { (void)arithmetic.limit_us(); }, "nonfinite arithmetic");
    std::cout << "PASS: " << catalog.size()
              << " baseline cases across 40 operations, formula, traffic, provenance, "
                 "platform isolation, mismatch and invalid-data contracts\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
