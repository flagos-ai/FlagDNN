/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "equivalent_baseline.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <set>
#include <stdexcept>

namespace flagdnn::benchmarking {
namespace {

using native::json::Value;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

double positive(double value, std::string_view name) {
  require(std::isfinite(value) && value > 0.0,
          std::string(name) + " must be finite and positive");
  return value;
}

double positive(const Value& object, std::string_view field) {
  return positive(object.at(field).as_double(), field);
}

const Value* optional(const Value& object, std::string_view field) {
  const auto& fields = object.as_object();
  const auto found = fields.find(field);
  return found == fields.end() ? nullptr : &found->second;
}

std::string_view dtype_name(flagdnnDataType_t type) {
  switch (type) {
    case FLAGDNN_DATA_FLOAT32: return "fp32";
    case FLAGDNN_DATA_FLOAT16: return "fp16";
    case FLAGDNN_DATA_BFLOAT16: return "bfloat16";
    case FLAGDNN_DATA_BOOLEAN: return "bool";
    case FLAGDNN_DATA_INT32: return "int32";
    case FLAGDNN_DATA_FP8_E4M3: return "fp8_e4m3";
    case FLAGDNN_DATA_FP8_E5M2: return "fp8_e5m2";
    default: throw std::runtime_error("unsupported equivalent-baseline dtype");
  }
}

bool same_vector(const Value& value,
                 const std::vector<std::int64_t>& expected) {
  const auto& actual = value.as_array();
  if (actual.size() != expected.size()) return false;
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (actual[i].as_int() != expected[i]) return false;
  return true;
}

struct Policy {
  std::string_view operation;
  std::string_view metric;
  InputDomain domain = InputDomain::kReal;
  std::size_t inputs = 1;
};

Policy policy(const BenchmarkCase& spec) {
  switch (spec.operation) {
    case Operation::kReshape: return {"reshape", "hbm_copy"};
    case Operation::kTranspose: return {"transpose", "hbm_copy"};
    case Operation::kSlice: return {"slice", "hbm_copy"};
    case Operation::kGraph:
      if (!spec.graph.nodes.empty() || !spec.graph.intermediates.empty())
        return {};
      if (spec.name.starts_with("concatenate_"))
        return {"concatenate", "hbm_copy"};
      if (spec.name.starts_with("gen_index_"))
        return {"gen_index", "hbm_write", InputDomain::kReal, 0};
      return {};
    case Operation::kPointwise: break;
    default: return {};
  }
  switch (spec.pointwise_mode) {
    case FLAGDNN_POINTWISE_IDENTITY: return {"identity", "hbm_copy"};
    case FLAGDNN_POINTWISE_DIV:
      return {"div", "vector_fp32", InputDomain::kDivisor, 2};
    case FLAGDNN_POINTWISE_POW:
      return {"pow", "vector_fp32", InputDomain::kPower, 2};
    case FLAGDNN_POINTWISE_MOD:
      return {"mod", "vector_fp32", InputDomain::kModulo, 2};
    case FLAGDNN_POINTWISE_CMP_EQ:
      return {"cmp_eq", "hbm_triad", InputDomain::kComparison, 2};
    case FLAGDNN_POINTWISE_CMP_NEQ:
      return {"cmp_neq", "hbm_triad", InputDomain::kComparison, 2};
    case FLAGDNN_POINTWISE_CMP_GT:
      return {"cmp_gt", "hbm_triad", InputDomain::kComparison, 2};
    case FLAGDNN_POINTWISE_CMP_GE:
      return {"cmp_ge", "hbm_triad", InputDomain::kComparison, 2};
    case FLAGDNN_POINTWISE_CMP_LT:
      return {"cmp_lt", "hbm_triad", InputDomain::kComparison, 2};
    case FLAGDNN_POINTWISE_CMP_LE:
      return {"cmp_le", "hbm_triad", InputDomain::kComparison, 2};
    case FLAGDNN_POINTWISE_LOGICAL_AND:
      return {"logical_and", "hbm_triad", InputDomain::kLogical, 2};
    case FLAGDNN_POINTWISE_LOGICAL_OR:
      return {"logical_or", "hbm_triad", InputDomain::kLogical, 2};
    case FLAGDNN_POINTWISE_LOGICAL_NOT:
      return {"logical_not", "hbm_copy", InputDomain::kLogical};
    case FLAGDNN_POINTWISE_BINARY_SELECT:
      return {"binary_select", "hbm_triad", InputDomain::kReal, 3};
    case FLAGDNN_POINTWISE_NEG: return {"neg", "hbm_copy"};
    case FLAGDNN_POINTWISE_CEIL:
      return {"ceil", "hbm_copy", InputDomain::kScaled};
    case FLAGDNN_POINTWISE_FLOOR:
      return {"floor", "hbm_copy", InputDomain::kScaled};
    case FLAGDNN_POINTWISE_SQRT:
      return {"sqrt", "vector_fp32", InputDomain::kPositive};
    case FLAGDNN_POINTWISE_RSQRT:
      return {"rsqrt", "vector_fp32", InputDomain::kPositive};
    case FLAGDNN_POINTWISE_RECIPROCAL:
      return {"reciprocal", "vector_fp32", InputDomain::kPositive};
    case FLAGDNN_POINTWISE_LOG:
      return {"log", "vector_fp32", InputDomain::kPositive};
    case FLAGDNN_POINTWISE_TAN:
      return {"tan", "vector_fp32", InputDomain::kTan};
    case FLAGDNN_POINTWISE_EXP: return {"exp", "vector_fp32"};
    case FLAGDNN_POINTWISE_ERF: return {"erf", "vector_fp32"};
    case FLAGDNN_POINTWISE_SIN: return {"sin", "vector_fp32"};
    case FLAGDNN_POINTWISE_COS: return {"cos", "vector_fp32"};
    case FLAGDNN_POINTWISE_RELU_FWD:
      if (spec.name.starts_with("leaky_relu_") &&
          !spec.name.starts_with("leaky_relu_backward_"))
        return {"leaky_relu", "hbm_copy"};
      return {};
    case FLAGDNN_POINTWISE_RELU_BWD:
      if (spec.name.starts_with("leaky_relu_backward_"))
        return {"leaky_relu_backward", "hbm_triad", InputDomain::kReal, 2};
      return {};
    case FLAGDNN_POINTWISE_ELU_BWD:
      return {"elu_backward", "vector_fp32", InputDomain::kReal, 2};
    case FLAGDNN_POINTWISE_GELU_FWD: return {"gelu", "vector_fp32"};
    case FLAGDNN_POINTWISE_GELU_BWD:
      return {"gelu_backward", "vector_fp32", InputDomain::kReal, 2};
    case FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD:
      return {"gelu_approx_tanh", "vector_fp32"};
    case FLAGDNN_POINTWISE_GELU_APPROX_TANH_BWD:
      return {"gelu_approx_tanh_backward", "vector_fp32", InputDomain::kReal, 2};
    case FLAGDNN_POINTWISE_SOFTPLUS_FWD: return {"softplus", "vector_fp32"};
    case FLAGDNN_POINTWISE_SOFTPLUS_BWD:
      return {"softplus_backward", "vector_fp32", InputDomain::kReal, 2};
    default: return {};
  }
}

void exact_keys(const Value& object,
                std::initializer_list<std::string_view> names) {
  require(object.as_object().size() == names.size(),
          "unexpected reference attributes");
  for (auto name : names) (void)object.at(name);
}

void same_number(const Value& object, std::string_view key, double expected) {
  const double actual = object.at(key).as_double();
  require(std::isfinite(actual) && std::isfinite(expected) && actual == expected,
          std::string(key) + " differs from reference");
}

void pointwise_attributes(const BenchmarkCase& spec, const Value& entry,
                          const Policy& chosen) {
  const auto& a = spec.pointwise_attributes;
  const auto defaults = default_pointwise_attributes();
  require(a.struct_size == defaults.struct_size && a.version == defaults.version,
          "unsupported pointwise attribute ABI");
  const auto* attributes = optional(entry, "attributes");
  if (!attributes || attributes->as_object().empty()) {
    require(chosen.operation == "div" || chosen.operation == "pow" ||
                chosen.operation == "mod" || chosen.operation == "cmp_eq" ||
                chosen.operation == "logical_not",
            "reference lacks pointwise attributes");
    require(a.flags == 0 && a.relu_lower_clip == 0 && a.relu_upper_clip == 0 &&
                a.relu_lower_clip_slope == 0 && a.swish_beta == 1 &&
                a.elu_alpha == 1 && a.softplus_beta == 1 && spec.add_alpha == 1,
            "pointwise attributes differ from implicit reference defaults");
    return;
  }
  const auto& attrs = *attributes;
  if (optional(attrs, "pointwise_mode")) {
    exact_keys(attrs, {"pointwise_mode", "pointwise_flags", "relu_lower_clip",
                       "relu_upper_clip", "relu_lower_clip_slope", "swish_beta",
                       "elu_alpha", "softplus_beta"});
    require(attrs.at("pointwise_mode").as_int() == spec.pointwise_mode &&
                attrs.at("pointwise_flags").as_int() ==
                    static_cast<std::int64_t>(a.flags),
            "pointwise mode or flags differ from reference");
    same_number(attrs, "elu_alpha", a.elu_alpha);
    same_number(attrs, "softplus_beta", a.softplus_beta);
    require(spec.add_alpha == 1, "pointwise alpha differs from reference");
  } else {
    // Native INT32 comparison and attribute probes use the native catalog's
    // explicit attribute representation, not the common benchmark serializer.
    const bool integer_comparison = chosen.operation == "cmp_gt" ||
                                    chosen.operation == "cmp_le";
    if (integer_comparison) {
      exact_keys(attrs, {"flags", "relu_lower_clip", "relu_upper_clip",
                         "relu_lower_clip_slope", "swish_beta", "alpha",
                         "reference_compute_dtype", "semantic_scope"});
      require(spec.tensors.front().data_type == FLAGDNN_DATA_INT32 &&
                  attrs.at("reference_compute_dtype").as_string() == "fp32" &&
                  attrs.at("semantic_scope").as_string() == "shared_fixture_only",
              "INT32 comparison reference semantics differ");
    } else {
      require(chosen.operation == "leaky_relu_backward" ||
                  chosen.operation == "identity",
              "unknown native pointwise attribute representation");
      exact_keys(attrs, {"flags", "relu_lower_clip", "relu_upper_clip",
                         "relu_lower_clip_slope", "swish_beta", "alpha"});
    }
    require(attrs.at("flags").as_int() == static_cast<std::int64_t>(a.flags) &&
                a.elu_alpha == 1 && a.softplus_beta == 1,
            "native pointwise flags or defaults differ");
    same_number(attrs, "alpha", spec.add_alpha);
  }
  same_number(attrs, "relu_lower_clip", a.relu_lower_clip);
  same_number(attrs, "relu_upper_clip", a.relu_upper_clip);
  same_number(attrs, "relu_lower_clip_slope", a.relu_lower_clip_slope);
  same_number(attrs, "swish_beta", a.swish_beta);
}

void layout_attributes(const BenchmarkCase& spec, const Value& entry) {
  const auto& attrs = entry.at("attributes");
  exact_keys(attrs, {"permutation", "slices", "slice_strides"});
  require(same_vector(attrs.at("permutation"), spec.transpose.permutation) &&
              same_vector(attrs.at("slice_strides"), spec.slice.strides),
          "layout permutation or slice strides differ");
  const auto& slices = attrs.at("slices").as_array();
  require(slices.size() == spec.slice.slices.size(), "slice count differs");
  for (std::size_t i = 0; i < slices.size(); ++i)
    require(same_vector(slices[i], {spec.slice.slices[i].first,
                                    spec.slice.slices[i].second}),
            "slice bounds differ");
  if (spec.operation == Operation::kReshape)
    require(spec.reshape.logical &&
                spec.reshape.dimensions == spec.tensors.back().dimensions &&
                spec.reshape.strides == spec.tensors.back().strides,
            "reshape attributes differ from output descriptor");
}

std::int64_t storage_bits(flagdnnDataType_t type) {
  switch (type) {
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
    case FLAGDNN_DATA_BOOLEAN: return 8;
    case FLAGDNN_DATA_FLOAT16:
    case FLAGDNN_DATA_BFLOAT16: return 16;
    case FLAGDNN_DATA_FLOAT32:
    case FLAGDNN_DATA_INT32: return 32;
    default: throw std::runtime_error("unknown tensor storage width");
  }
}

}  // namespace

bool supports_equivalent_baseline(const BenchmarkCase& specification) {
  const auto chosen = policy(specification);
  if (chosen.operation.empty()) return false;
  // Only the 24 native FP8 identity cases have matching NVIDIA evidence.
  // FP32/FP16 retain their native pair; BF16/INT32/BOOL cannot borrow it.
  if (chosen.operation == "identity") {
    if (specification.tensors.size() != 2) return false;
    const auto type = specification.tensors.front().data_type;
    return (type == FLAGDNN_DATA_FP8_E4M3 || type == FLAGDNN_DATA_FP8_E5M2) &&
           specification.tensors.back().data_type == type;
  }
  return std::all_of(specification.tensors.begin(), specification.tensors.end(),
                     [&](const TensorSpec& tensor) {
    return tensor.data_type == FLAGDNN_DATA_FLOAT32 ||
           tensor.data_type == FLAGDNN_DATA_FLOAT16 ||
           tensor.data_type == FLAGDNN_DATA_BFLOAT16 ||
           tensor.data_type == FLAGDNN_DATA_BOOLEAN ||
           (tensor.data_type == FLAGDNN_DATA_INT32 &&
            (chosen.operation == "gen_index" || chosen.operation == "cmp_gt" ||
             chosen.operation == "cmp_le"));
  });
}

double EquivalentBaseline::equivalent_time_us() const {
  require(traffic_ratio == 1.0 || scale_metric.starts_with("hbm_"),
          "traffic correction requires a bandwidth scaling metric");
  return positive(positive(reference_time_us, "reference_time_us") *
                      positive(reference_peak, "reference_peak") /
                      positive(target_peak, "target_peak") *
                      positive(traffic_ratio, "traffic_ratio"),
                  "equivalent_time_us");
}

double EquivalentBaseline::limit_us() const {
  const double alpha =
      positive(min_relative_efficiency, "min_relative_efficiency");
  require(alpha <= 1.0, "min_relative_efficiency must not exceed 1");
  return positive(equivalent_time_us() / alpha, "limit_us");
}

PerformanceBaseline::PerformanceBaseline(
    const std::filesystem::path& path) : path_(path) {
  try {
    std::ifstream file(path);
    require(file.is_open(), "cannot open baseline");
    const std::string content((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
    require(!file.bad(), "cannot read baseline");
    document_ = native::json::parse(content);
    require(document_.at("schema_version").as_int() == 4,
            "expected baseline schema_version 4");
    require(document_.at("reference").at("platform").as_string() == "nvidia",
            "expected NVIDIA reference platform");
    const auto& measured_on =
        document_.at("reference").at("measured_on").as_array();
    require(!measured_on.empty() &&
                std::all_of(measured_on.begin(), measured_on.end(),
                    [](const Value& date) { return !date.as_string().empty(); }) &&
                !document_.at("reference").at("provider").as_string().empty() &&
                !document_.at("platforms").at("nvidia").at("device").as_string().empty() &&
                !document_.at("platforms").at("nvidia").at("measured_on").as_string().empty(),
            "NVIDIA measurement provenance is missing");
    const double alpha = positive(document_, "min_relative_efficiency");
    require(alpha <= 1.0, "min_relative_efficiency must not exceed 1");
    const auto& target = document_.at("platforms").at("hygon");
    require(!target.at("device").as_string().empty(),
            "Hygon device is missing");
    require(!target.at("architecture").as_string().empty(),
            "Hygon architecture is missing");
    require(!target.at("measured_on").as_string().empty(),
            "Hygon measurement date is missing");
  } catch (const std::exception& error) {
    throw std::runtime_error("performance baseline " + path_.string() +
                             ": " + error.what());
  }
}

EquivalentBaseline PerformanceBaseline::lookup(
    const BenchmarkCase& specification, std::string_view device_name,
    std::string_view architecture, const NativeBaselineMetadata* metadata) const {
  try {
    require(supports_equivalent_baseline(specification),
            "case is outside equivalent-baseline scope");
    const auto chosen = policy(specification);
    require(specification.name.starts_with(std::string(chosen.operation) + "_"),
            "case name and operation differ");
    require(specification.output_count == 1, "expected exactly one output");
    const std::size_t input_count = input_tensor_count(specification);
    require(chosen.operation == "concatenate" ? input_count >= 1
                                              : input_count == chosen.inputs,
            "input count differs from operation");
    require(specification.input_domain == chosen.domain,
            "input domain differs from the NVIDIA catalog");
    std::vector<InputDomain> domains(input_count, chosen.domain);
    if (chosen.operation == "binary_select") domains[2] = InputDomain::kLogical;
    require(specification.input_domains.empty()
                ? chosen.operation != "binary_select"
                : specification.input_domains == domains,
            "per-input domains differ from the NVIDIA catalog");
    require((specification.operation == Operation::kGraph) == (metadata != nullptr),
            "native metadata is required only for native index operations");

    const auto& target = document_.at("platforms").at("hygon");
    require(target.at("device").as_string() == device_name,
            "Hygon device does not match the measured baseline");
    require(target.at("architecture").as_string() == architecture,
            "Hygon architecture does not match the measured baseline");

    const auto& reference = document_.at("reference");
    const auto& entry = reference.at("cases").at(specification.name);
    require(entry.at("op").as_string() == chosen.operation,
            "reference operation differs");
    const bool pointwise = specification.operation == Operation::kPointwise;
    const std::string validation = entry.at("validation").as_string();
    require(chosen.operation == "identity"
                ? validation == "bit_exact" &&
                      specification.absolute_tolerance == 0 &&
                      specification.relative_tolerance == 0
                : validation == "cpu_full" ||
                      (!pointwise && validation == "sanity"),
            "reference validation is insufficient or unknown");
    if (pointwise) {
      pointwise_attributes(specification, entry, chosen);
    } else if (metadata) {
      require(metadata->operation == chosen.operation,
              "native operation differs from reference");
      exact_keys(entry.at("attributes"), {"axis"});
      require(entry.at("attributes").at("axis").as_int() == metadata->axis,
              "native axis differs from reference");
      const auto rank = static_cast<std::int64_t>(specification.tensors.back().dimensions.size());
      require(rank > 0 && metadata->axis >= -rank && metadata->axis < rank,
              "invalid native axis");
    } else {
      layout_attributes(specification, entry);
    }
    if (const auto* domain = optional(entry, "input_domain"))
      require(domain->as_int() == static_cast<int>(chosen.domain),
              "reference input domain differs from the NVIDIA catalog");
    if (const auto* input_domains = optional(entry, "input_domains")) {
      std::vector<std::int64_t> expected;
      for (auto domain : domains) expected.push_back(static_cast<int>(domain));
      require(same_vector(*input_domains, expected),
              "reference per-input domains differ");
    }
    if (const auto* count = optional(entry, "output_count"))
      require(count->as_int() == 1, "reference output count is not one");

    const auto& tensors = entry.at("tensors").as_array();
    require(tensors.size() == specification.tensors.size(),
            "reference tensor count differs");
    std::set<std::int64_t> uids;
    double reference_bits = 0;
    double target_bits = 0;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
      const auto& tensor = specification.tensors[i];
      const auto& expected = tensors[i];
      const std::string label = "tensor " + std::to_string(i);
      require(uids.insert(tensor.uid).second, "tensor UIDs are not unique");
      require(expected.at("role").as_string() ==
                  (i < input_count ? "input" : "output"),
              label + " role differs");
      require(expected.at("dtype").as_string() == dtype_name(tensor.data_type),
              label + " dtype differs");
      require(same_vector(expected.at("shape"), tensor.dimensions),
              label + " shape differs");
      require(same_vector(expected.at("strides"), tensor.strides),
              label + " strides differ");
      const auto* offset = optional(expected, "binding_byte_offset");
      const auto expected_offset = offset ? offset->as_int() : 0;
      require(expected_offset >= 0 &&
                  static_cast<std::uint64_t>(expected_offset) ==
                      tensor.binding_byte_offset,
              label + " binding offset differs");
      if (const auto* strides = optional(expected, "reference_strides"))
        require(same_vector(*strides, tensor.strides),
                label + " converted reference strides differ");
      require(tensor.dimensions.size() == tensor.strides.size(),
              label + " rank and strides differ");
      double count = 1;
      for (std::size_t d = 0; d < tensor.dimensions.size(); ++d) {
        require(tensor.dimensions[d] > 0 && tensor.strides[d] > 0,
                label + " dimensions and strides must be positive");
        count = positive(count * static_cast<double>(tensor.dimensions[d]),
                         "logical element count");
      }
      const auto target_width = storage_bits(tensor.data_type);
      const auto* bits = optional(expected, "reference_storage_bits");
      const auto reference_width = bits ? bits->as_int() : target_width;
      require(tensor.data_type == FLAGDNN_DATA_BOOLEAN
                  ? bits && reference_width == 1
                  : reference_width == target_width,
              label + " reference storage width is unknown or inconsistent");
      target_bits += count * static_cast<double>(target_width);
      reference_bits += count * static_cast<double>(reference_width);
    }

    EquivalentBaseline result;
    // The shared NVIDIA records contain measurements, not a target's choice
    // of scaling proxy. Require Hygon's explicit policy; never inherit one
    // from the reference case or another platform.
    const auto& scaling = target.at("scaling").at("case_policies")
                             .at(chosen.operation);
    result.scale_metric = scaling.at("scale_metric").as_string();
    require(result.scale_metric == chosen.metric,
            "Hygon scaling metric differs from the operation policy");
    const bool bandwidth = result.scale_metric.starts_with("hbm_");
    require(document_.at("metric_units").at(result.scale_metric).as_string() ==
                (bandwidth ? "TB/s" : "TFLOP/s"), "peak unit differs");
    result.reference_validation = validation;
    result.traffic_ratio = positive(target_bits, "target logical traffic") /
                           positive(reference_bits, "reference logical traffic");
    require(bandwidth || result.traffic_ratio == 1,
            "storage conversion is unsupported for a compute proxy");
    result.timing_profile = entry.at("timing_profile").as_string();
    require(result.timing_profile == "native",
            "equivalent baseline requires the native timing profile");
    const auto& measurement = reference.at("measurement");
    require(measurement.at("timing_method").as_string() ==
                "device_events_over_cuda_graph",
            "reference timing method differs");
    require(measurement.at("statistic").as_string() == "nearest_rank_p50",
            "reference timing statistic differs");
    const auto& timing =
        measurement.at("timing_profiles").at(result.timing_profile);
    const auto& config = specification.benchmark;
    if (chosen.operation == "identity")
      require(config.warmup_iterations == 10 && config.sample_count == 20 &&
                  config.iterations_per_sample == 50,
              "FP8 identity requires the recorded native timing configuration");
    require(config.warmup_iterations >= 0 && config.sample_count > 0 &&
                config.iterations_per_sample > 0,
            "invalid benchmark timing configuration");
    require(timing.at("warmup_iterations").as_int() ==
                    config.warmup_iterations &&
                timing.at("sample_count").as_int() == config.sample_count &&
                timing.at("iterations_per_sample").as_int() ==
                    config.iterations_per_sample &&
                timing.at("graph_warmup_replays").as_int() == 1,
            "benchmark timing configuration differs from reference");

    result.reference_time_us = positive(entry, "time_us");
    result.reference_peak = positive(
        document_.at("platforms").at("nvidia").at("peaks"),
        result.scale_metric);
    result.target_peak = positive(target.at("peaks"), result.scale_metric);
    result.min_relative_efficiency =
        positive(document_, "min_relative_efficiency");
    (void)result.limit_us();
    return result;
  } catch (const std::exception& error) {
    throw std::runtime_error("performance baseline " + path_.string() +
                             " case " + specification.name + ": " +
                             error.what());
  }
}

}  // namespace flagdnn::benchmarking
