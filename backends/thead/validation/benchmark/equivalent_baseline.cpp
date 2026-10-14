/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "equivalent_baseline.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>

namespace flagdnn::benchmarking::thead {
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

bool same_vector(const Value& value, const std::vector<std::int64_t>& expected) {
  const auto& actual = value.as_array();
  if (actual.size() != expected.size()) return false;
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (actual[i].as_int() != expected[i]) return false;
  return true;
}

std::string_view dtype_name(flagdnnDataType_t type) {
  switch (type) {
    case FLAGDNN_DATA_FLOAT32: return "fp32";
    case FLAGDNN_DATA_FLOAT16: return "fp16";
    case FLAGDNN_DATA_BFLOAT16: return "bfloat16";
    case FLAGDNN_DATA_INT32: return "int32";
    default: throw std::runtime_error("unsupported Thead baseline dtype");
  }
}

struct Policy {
  std::string_view operation;
  std::string_view metric;
};

Policy policy(const BenchmarkCase& spec) {
  if (spec.operation == Operation::kPointwise &&
      spec.pointwise_mode == FLAGDNN_POINTWISE_GELU_APPROX_TANH_BWD &&
      spec.name.starts_with("gelu_approx_tanh_backward_perf_"))
    return {"gelu_approx_tanh_backward", "vector_fp32"};
  if (spec.operation == Operation::kGraph && spec.graph.nodes.empty() &&
      spec.graph.intermediates.empty()) {
    if (spec.name.starts_with("concatenate_")) return {"concatenate", "hbm_copy"};
    if (spec.name.starts_with("gen_index_")) return {"gen_index", "hbm_write"};
  }
  return {};
}

void exact_keys(const Value& object,
                std::initializer_list<std::string_view> names) {
  require(object.as_object().size() == names.size(), "unexpected reference attributes");
  for (const auto name : names) (void)object.at(name);
}

void same_number(const Value& object, std::string_view key, double expected) {
  const double actual = object.at(key).as_double();
  require(std::isfinite(actual) && std::isfinite(expected) && actual == expected,
          std::string(key) + " differs from reference");
}

void gelu_attributes(const BenchmarkCase& spec, const Value& entry) {
  const auto& a = spec.pointwise_attributes;
  const auto defaults = default_pointwise_attributes();
  require(a.struct_size == defaults.struct_size && a.version == defaults.version,
          "unsupported pointwise attribute ABI");
  const auto& attrs = entry.at("attributes");
  exact_keys(attrs, {"pointwise_mode", "pointwise_flags", "relu_lower_clip",
                    "relu_upper_clip", "relu_lower_clip_slope", "swish_beta",
                    "elu_alpha", "softplus_beta"});
  require(attrs.at("pointwise_mode").as_int() == spec.pointwise_mode &&
              attrs.at("pointwise_flags").as_int() == static_cast<std::int64_t>(a.flags),
          "pointwise mode or flags differ from reference");
  same_number(attrs, "relu_lower_clip", a.relu_lower_clip);
  same_number(attrs, "relu_upper_clip", a.relu_upper_clip);
  same_number(attrs, "relu_lower_clip_slope", a.relu_lower_clip_slope);
  same_number(attrs, "swish_beta", a.swish_beta);
  same_number(attrs, "elu_alpha", a.elu_alpha);
  same_number(attrs, "softplus_beta", a.softplus_beta);
  require(spec.add_alpha == 1, "pointwise alpha differs from reference");
}
}  // namespace

bool supports_equivalent_baseline(const BenchmarkCase& spec) {
  const auto chosen = policy(spec);
  if (chosen.operation.empty() || spec.tensors.empty()) return false;
  return std::all_of(spec.tensors.begin(), spec.tensors.end(), [&](const TensorSpec& t) {
    if (chosen.operation == "gen_index")
      return t.data_type == FLAGDNN_DATA_FLOAT32 || t.data_type == FLAGDNN_DATA_INT32;
    return t.data_type == FLAGDNN_DATA_FLOAT32 || t.data_type == FLAGDNN_DATA_FLOAT16 ||
           t.data_type == FLAGDNN_DATA_BFLOAT16;
  });
}

double EquivalentBaseline::equivalent_time_us() const {
  require(traffic_ratio == 1.0, "Thead baseline requires identical storage widths");
  return positive(positive(reference_time_us, "reference_time_us") *
                      positive(reference_peak, "reference_peak") /
                      positive(target_peak, "target_peak"), "equivalent_time_us");
}

double EquivalentBaseline::limit_us() const {
  const double alpha = positive(min_relative_efficiency, "min_relative_efficiency");
  require(alpha == 0.8, "Thead baseline requires min_relative_efficiency 0.8");
  return positive(equivalent_time_us() / alpha, "limit_us");
}

PerformanceBaseline::PerformanceBaseline(const std::filesystem::path& path)
    : path_(path) {
  try {
    std::ifstream file(path);
    require(file.is_open(), "cannot open baseline");
    const std::string content((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
    require(!file.bad(), "cannot read baseline");
    document_ = native::json::parse(content);
    require(document_.at("schema_version").as_int() == 4, "expected baseline schema_version 4");
    const auto& reference = document_.at("reference");
    require(reference.at("platform").as_string() == "nvidia", "expected NVIDIA reference platform");
    const auto& dates = reference.at("measured_on").as_array();
    const auto& nvidia = document_.at("platforms").at("nvidia");
    require(!dates.empty() && std::all_of(dates.begin(), dates.end(),
                [](const Value& date) { return !date.as_string().empty(); }) &&
                !reference.at("provider").as_string().empty() &&
                !nvidia.at("device").as_string().empty() &&
                !nvidia.at("measured_on").as_string().empty(),
            "NVIDIA measurement provenance is missing");
    require(positive(document_, "min_relative_efficiency") == 0.8,
            "Thead baseline requires min_relative_efficiency 0.8");
    const auto& target = document_.at("platforms").at("thead");
    require(!target.at("device").as_string().empty(), "Thead device is missing");
    require(!target.at("architecture").as_string().empty(), "Thead architecture is missing");
    require(!target.at("measured_on").as_string().empty(), "Thead measurement date is missing");
  } catch (const std::exception& error) {
    throw std::runtime_error("performance baseline " + path_.string() + ": " + error.what());
  }
}

EquivalentBaseline PerformanceBaseline::lookup(
    const BenchmarkCase& spec, std::string_view device_name,
    std::string_view architecture, const NativeBaselineMetadata* metadata) const {
  try {
    require(supports_equivalent_baseline(spec), "case is outside Thead equivalent-baseline scope");
    const auto chosen = policy(spec);
    const bool index = spec.operation == Operation::kGraph;
    require(spec.output_count == 1, "output count differs from operation");
    const auto inputs = input_tensor_count(spec);
    require(chosen.operation == "concatenate" ? inputs >= 1
                : inputs == (index ? 0U : 2U), "input count differs from operation");
    require(spec.input_domain == InputDomain::kReal, "input domain differs from NVIDIA catalog");
    require(spec.input_domains.empty() ||
                spec.input_domains == std::vector<InputDomain>(inputs, InputDomain::kReal),
            "per-input domains differ from NVIDIA catalog");
    require(index == (metadata != nullptr), "native metadata is required only for index operations");
    const auto& target = document_.at("platforms").at("thead");
    require(target.at("device").as_string() == device_name,
            "Thead device does not match measured baseline");
    require(target.at("architecture").as_string() == architecture,
            "Thead architecture does not match measured baseline");
    const auto& reference = document_.at("reference");
    const auto& entry = reference.at("cases").at(spec.name);
    require(entry.at("op").as_string() == chosen.operation, "reference operation differs");
    const auto& validation = entry.at("validation").as_string();
    require(validation == (index ? "sanity" : "cpu_full"), "reference validation differs");
    if (index) {
      require(metadata->operation == chosen.operation, "native operation differs from reference");
      exact_keys(entry.at("attributes"), {"axis"});
      require(entry.at("attributes").at("axis").as_int() == metadata->axis,
              "native axis differs from reference");
      const auto rank = static_cast<std::int64_t>(spec.tensors.back().dimensions.size());
      require(rank > 0 && metadata->axis >= -rank && metadata->axis < rank, "invalid native axis");
    } else {
      gelu_attributes(spec, entry);
    }
    if (const auto* domain = optional(entry, "input_domain"))
      require(domain->as_int() == static_cast<int>(InputDomain::kReal), "reference input domain differs");
    if (const auto* domains = optional(entry, "input_domains"))
      require(same_vector(*domains, std::vector<std::int64_t>(inputs, 0)),
              "reference per-input domains differ");
    if (const auto* outputs = optional(entry, "output_count"))
      require(outputs->as_int() == 1, "reference output count differs");

    const auto& tensors = entry.at("tensors").as_array();
    require(tensors.size() == spec.tensors.size(), "reference tensor count differs");
    std::set<std::int64_t> uids;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
      const auto& tensor = spec.tensors[i];
      const auto& expected = tensors[i];
      const auto label = "tensor " + std::to_string(i);
      require(uids.insert(tensor.uid).second, "tensor UIDs are not unique");
      if (const auto* uid = optional(expected, "uid"))
        require(uid->as_int() == tensor.uid, label + " UID differs");
      require(expected.at("role").as_string() == (i < inputs ? "input" : "output"), label + " role differs");
      require(expected.at("dtype").as_string() == dtype_name(tensor.data_type), label + " dtype differs");
      require(same_vector(expected.at("shape"), tensor.dimensions), label + " shape differs");
      require(same_vector(expected.at("strides"), tensor.strides), label + " strides differ");
      const auto* offset = optional(expected, "binding_byte_offset");
      const auto expected_offset = offset ? offset->as_int() : 0;
      require(expected_offset >= 0 && static_cast<std::uint64_t>(expected_offset) == tensor.binding_byte_offset,
              label + " binding offset differs");
      if (const auto* strides = optional(expected, "reference_strides"))
        require(same_vector(*strides, tensor.strides), label + " converted reference strides differ");
      require(!tensor.dimensions.empty() && tensor.dimensions.size() == tensor.strides.size(),
              label + " rank and strides differ");
      std::uint64_t count = 1;
      for (std::size_t d = 0; d < tensor.dimensions.size(); ++d) {
        require(tensor.dimensions[d] > 0 && tensor.strides[d] > 0 &&
                    count <= std::numeric_limits<std::uint64_t>::max() /
                                 static_cast<std::uint64_t>(tensor.dimensions[d]),
                label + " dimensions or strides are invalid");
        count *= static_cast<std::uint64_t>(tensor.dimensions[d]);
      }
      const auto width = tensor.data_type == FLAGDNN_DATA_FLOAT16 ||
                         tensor.data_type == FLAGDNN_DATA_BFLOAT16 ? 16 : 32;
      if (const auto* bits = optional(expected, "reference_storage_bits"))
        require(bits->as_int() == width, label + " reference storage width differs");
    }

    EquivalentBaseline result;
    // The NVIDIA records contain measurements; only Thead selects its proxy.
    result.scale_metric = target.at("scaling").at("case_policies")
                               .at(chosen.operation).at("scale_metric").as_string();
    require(result.scale_metric == chosen.metric, "Thead scaling metric differs from operation policy");
    require(document_.at("metric_units").at(result.scale_metric).as_string() ==
                (index ? "TB/s" : "TFLOP/s"), "peak unit differs");
    result.reference_validation = validation;
    result.timing_profile = entry.at("timing_profile").as_string();
    require(result.timing_profile == "native", "equivalent baseline timing profile differs");
    const auto& measurement = reference.at("measurement");
    require(measurement.at("timing_method").as_string() == "device_events_over_cuda_graph",
            "reference timing method differs");
    require(measurement.at("statistic").as_string() == "nearest_rank_p50",
            "reference timing statistic differs");
    const auto& timing = measurement.at("timing_profiles").at("native");
    const auto& config = spec.benchmark;
    require(config.warmup_iterations == 10 && config.sample_count == 20 &&
                config.iterations_per_sample == 50, "benchmark timing configuration differs");
    require(timing.at("warmup_iterations").as_int() == config.warmup_iterations &&
                timing.at("sample_count").as_int() == config.sample_count &&
                timing.at("iterations_per_sample").as_int() == config.iterations_per_sample &&
                timing.at("graph_warmup_replays").as_int() == 1,
            "benchmark timing configuration differs from reference");
    result.reference_time_us = positive(entry, "time_us");
    result.reference_peak = positive(document_.at("platforms").at("nvidia").at("peaks"), result.scale_metric);
    result.target_peak = positive(target.at("peaks"), result.scale_metric);
    result.min_relative_efficiency = positive(document_, "min_relative_efficiency");
    (void)result.limit_us();
    return result;
  } catch (const std::exception& error) {
    throw std::runtime_error("performance baseline " + path_.string() + " case " + spec.name + ": " + error.what());
  }
}
}  // namespace flagdnn::benchmarking::thead
