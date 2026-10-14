/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "validation/benchmark/equivalent_baseline.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

#ifndef FLAGDNN_PERFORMANCE_BASELINE_FILE
#define FLAGDNN_PERFORMANCE_BASELINE_FILE "docs/performance-baseline.json"
#endif

namespace flagdnn::testing::ascend {
namespace {
using native::json::Value;

void require(bool condition, std::string_view reason) {
  if (!condition) throw std::runtime_error(std::string(reason));
}
double positive(double value, std::string_view field) {
  require(std::isfinite(value) && value > 0,
          std::string(field) + " must be finite and positive");
  return value;
}
const Value *optional(const Value &value, std::string_view field) {
  const auto &object = value.as_object();
  const auto found = object.find(field);
  return found == object.end() ? nullptr : &found->second;
}
bool fp8(flagdnnDataType_t type) {
  return type == FLAGDNN_DATA_FP8_E4M3 || type == FLAGDNN_DATA_FP8_E5M2;
}
std::string_view dtype(flagdnnDataType_t type) {
  require(fp8(type), "identity baseline requires FP8 storage");
  return type == FLAGDNN_DATA_FP8_E4M3 ? "fp8_e4m3" : "fp8_e5m2";
}
bool same_vector(const Value &value, const std::vector<std::int64_t> &expected) {
  const auto &actual = value.as_array();
  if (actual.size() != expected.size()) return false;
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (actual[i].as_int() != expected[i]) return false;
  return true;
}
std::size_t element_count(const TestTensor &tensor) {
  require(!tensor.dimensions.empty() &&
              tensor.dimensions.size() == tensor.strides.size(),
          "invalid identity tensor rank");
  std::size_t count = 1, last = 0;
  for (std::size_t i = 0; i < tensor.dimensions.size(); ++i) {
    const auto d = tensor.dimensions[i], s = tensor.strides[i];
    require(d > 0 && s > 0, "identity dimensions/strides must be positive");
    require(static_cast<std::uint64_t>(d) <=
                std::numeric_limits<std::size_t>::max() / count,
            "identity element count overflow");
    count *= static_cast<std::size_t>(d);
    require(static_cast<std::uint64_t>(d - 1) <=
                (std::numeric_limits<std::size_t>::max() - last - 1) /
                    static_cast<std::uint64_t>(s),
            "identity storage span overflow");
    last += static_cast<std::size_t>(d - 1) * static_cast<std::size_t>(s);
  }
  require(tensor.binding_byte_offset <=
              std::numeric_limits<std::size_t>::max() - last - 1,
          "identity binding offset overflow");
  return count;
}
void same_number(const Value &value, std::string_view field, double expected) {
  const double actual = value.at(field).as_double();
  require(std::isfinite(actual) && actual == expected,
          std::string(field) + " differs from reference");
}
std::string quote(std::string_view value) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32)
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
          << static_cast<unsigned>(c) << std::dec;
    else out << c;
  }
  out << '"';
  return out.str();
}
}  // namespace

bool supports_equivalent_baseline(const PointwiseTestCase &spec) {
  return spec.mode == FLAGDNN_POINTWISE_IDENTITY && spec.inputs.size() == 1 &&
         fp8(spec.inputs.front().data_type) &&
         spec.output.data_type == spec.inputs.front().data_type &&
         spec.name.starts_with("identity_" +
                               std::string(dtype(spec.inputs.front().data_type)) + "_");
}

std::filesystem::path performance_baseline_path() {
  const char *path = std::getenv("FLAGDNN_PERFORMANCE_BASELINE");
  return path && *path ? path : FLAGDNN_PERFORMANCE_BASELINE_FILE;
}

std::vector<std::uint8_t> identity_fp8_logical_bytes(const TestTensor &tensor) {
  require(fp8(tensor.data_type), "identity fixture requires FP8 storage");
  std::vector<std::uint8_t> result(element_count(tensor));
  const bool e4m3 = tensor.data_type == FLAGDNN_DATA_FP8_E4M3;
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = static_cast<std::uint8_t>((e4m3 ? 0x38 : 0x3c) +
                                         (e4m3 ? 8 : 4) * (i % 6));
  return result;
}

double EquivalentBaseline::equivalent_time_us() const {
  require(traffic_ratio == 1, "FP8 identity traffic ratio must equal one");
  return positive(positive(reference_time_us, "reference_time_us") *
                      positive(reference_peak, "reference_peak") /
                      positive(target_peak, "target_peak"),
                  "equivalent_time_us");
}
double EquivalentBaseline::limit_us() const {
  require(positive(min_relative_efficiency, "min_relative_efficiency") <= 1,
          "min_relative_efficiency must not exceed one");
  return positive(equivalent_time_us() / min_relative_efficiency, "limit_us");
}
std::string EquivalentBaseline::json(double measured_us) const {
  positive(measured_us, "measured_us");
  const double efficiency = positive(equivalent_time_us() / measured_us,
                                      "relative_efficiency");
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(17)
      << "{\"provider\":\"nvidia_equivalent\",\"reference_platform\":\"nvidia\","
         "\"target_platform\":\"ascend\",\"reference_case\":" << quote(reference_case)
      << ",\"scale_metric\":" << quote(scale_metric)
      << ",\"timing_profile\":" << quote(timing_profile)
      << ",\"reference_validation\":" << quote(reference_validation)
      << ",\"reference_time_us\":" << reference_time_us
      << ",\"reference_peak\":" << reference_peak
      << ",\"target_peak\":" << target_peak
      << ",\"min_relative_efficiency\":" << min_relative_efficiency
      << ",\"traffic_ratio\":" << traffic_ratio
      << ",\"equivalent_time_us\":" << equivalent_time_us()
      << ",\"limit_us\":" << limit_us()
      << ",\"relative_efficiency\":" << efficiency
      << ",\"passed\":" << (measured_us <= limit_us() ? "true" : "false") << '}';
  return out.str();
}

PerformanceBaseline::PerformanceBaseline(const std::filesystem::path &path)
    : path_(path) {
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
    const auto &reference = document_.at("reference");
    require(!reference.at("provider").as_string().empty() &&
                !reference.at("measurement").at("source_revision").as_string().empty(),
            "NVIDIA reference provenance is missing");
    const auto &dates = reference.at("measured_on").as_array();
    require(!dates.empty() && std::all_of(dates.begin(), dates.end(), [](const auto &date) {
              return !date.as_string().empty();
            }), "NVIDIA reference dates are missing");
    for (const auto platform : {"nvidia", "ascend"}) {
      const auto &target = document_.at("platforms").at(platform);
      require(!target.at("device").as_string().empty() &&
                  !target.at("measured_on").as_string().empty(),
              std::string(platform) + " measurement identity is missing");
    }
    require(!document_.at("platforms").at("ascend").at("architecture").as_string().empty(),
            "Ascend architecture is missing");
  } catch (const std::exception &error) {
    throw std::runtime_error("performance baseline " + path_.string() + ": " + error.what());
  }
}

EquivalentBaseline PerformanceBaseline::lookup(
    const PointwiseTestCase &spec, std::string_view device_name,
    std::string_view target_fingerprint) const {
  try {
    require(supports_equivalent_baseline(spec), "case is outside equivalent-baseline scope");
    const auto &target = document_.at("platforms").at("ascend");
    require(target.at("device").as_string() == device_name &&
                target.at("architecture").as_string() == target_fingerprint,
            "Ascend device/architecture differs from measured baseline");
    const auto &policy = target.at("scaling").at("case_policies").at("identity");
    require(policy.at("scale_metric").as_string() == "hbm_copy" &&
                !policy.at("reason").as_string().empty(),
            "Ascend identity requires an explicit logical copy bandwidth policy");
    const auto &dtypes = policy.at("supported_dtypes").as_array();
    require(std::any_of(dtypes.begin(), dtypes.end(), [&](const auto &value) {
              return value.as_string() == dtype(spec.output.data_type);
            }), "Ascend policy does not enable this FP8 dtype");
    require(spec.input_domains == std::vector{PointwiseInputDomain::kReal},
            "identity input domain differs from native catalog");
    const auto &a = spec.attributes;
    constexpr flagdnnPointwiseAttributes_t defaults = FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
    require(a.struct_size == defaults.struct_size && a.version == defaults.version &&
                a.flags == 0 && a.relu_lower_clip == 0 && a.relu_upper_clip == 0 &&
                a.relu_lower_clip_slope == 0 && a.swish_beta == 1 && a.elu_alpha == 1 &&
                a.softplus_beta == 1 && spec.alpha == 1,
            "identity attributes differ from native reference");
    const auto &reference = document_.at("reference");
    const auto &entry = reference.at("cases").at(spec.name);
    require(entry.at("op").as_string() == "identity" &&
                entry.at("validation").as_string() == "bit_exact" &&
                entry.at("timing_profile").as_string() == "native" &&
                entry.at("scale_metric").is_null(),
            "reference case identity, validation or timing profile differs");
    if (const auto *domain = optional(entry, "input_domain"))
      require(domain->as_int() == static_cast<int>(PointwiseInputDomain::kReal),
              "reference input domain differs");
    if (const auto *domains = optional(entry, "input_domains"))
      require(same_vector(*domains, {static_cast<int>(PointwiseInputDomain::kReal)}),
              "reference per-input domains differ");
    if (const auto *count = optional(entry, "output_count"))
      require(count->as_int() == 1, "reference output count differs");
    const auto &attributes = entry.at("attributes");
    require(attributes.as_object().size() == 6, "unexpected reference attributes");
    same_number(attributes, "flags", 0);
    same_number(attributes, "relu_lower_clip", 0);
    same_number(attributes, "relu_upper_clip", 0);
    same_number(attributes, "relu_lower_clip_slope", 0);
    same_number(attributes, "swish_beta", 1);
    same_number(attributes, "alpha", 1);
    const auto &tensors = entry.at("tensors").as_array();
    require(tensors.size() == 2 && spec.inputs[0].uid != spec.output.uid,
            "identity tensor count or UIDs differ");
    const TestTensor *actual[] = {&spec.inputs[0], &spec.output};
    for (std::size_t i = 0; i < 2; ++i) {
      const auto &tensor = *actual[i];
      const auto &recorded = tensors[i];
      (void)element_count(tensor);
      require(recorded.at("role").as_string() == (i ? "output" : "input") &&
                  recorded.at("dtype").as_string() == dtype(tensor.data_type) &&
                  same_vector(recorded.at("shape"), tensor.dimensions) &&
                  same_vector(recorded.at("strides"), tensor.strides),
              "identity tensor dtype, shape, strides or role differs");
      if (const auto *uid = optional(recorded, "uid"))
        require(uid->as_int() == tensor.uid, "identity reference UID differs");
      const auto *offset = optional(recorded, "binding_byte_offset");
      require((offset ? offset->as_int() : 0) >= 0 &&
                  static_cast<std::uint64_t>(offset ? offset->as_int() : 0) ==
                      tensor.binding_byte_offset,
              "identity binding offset differs");
      if (const auto *bits = optional(recorded, "reference_storage_bits"))
        require(bits->as_int() == 8, "identity reference storage width differs");
      if (const auto *strides = optional(recorded, "reference_strides"))
        require(same_vector(*strides, tensor.strides), "identity reference layout differs");
      if (const auto *conversion = optional(recorded, "reference_layout_conversion_timed"))
        require(!conversion->as_bool(), "identity reference includes layout conversion timing");
    }
    require(spec.inputs[0].dimensions == spec.output.dimensions,
            "identity input and output shapes differ");
    const auto &measurement = reference.at("measurement");
    require(measurement.at("timing_method").as_string() == "device_events_over_cuda_graph" &&
                measurement.at("statistic").as_string() == "nearest_rank_p50",
            "NVIDIA timing method/statistic differs");
    const auto &profile = measurement.at("timing_profiles").at("native");
    require(profile.at("warmup_iterations").as_int() == 10 &&
                profile.at("sample_count").as_int() == 20 &&
                profile.at("iterations_per_sample").as_int() == 50 &&
                profile.at("graph_warmup_replays").as_int() == 1,
            "NVIDIA native timing profile differs");
    require(document_.at("metric_units").at("hbm_copy").as_string() == "TB/s",
            "copy bandwidth peak unit differs");
    EquivalentBaseline result;
    result.reference_case = spec.name;
    result.reference_time_us = positive(entry.at("time_us").as_double(), "reference_time_us");
    result.reference_peak = positive(document_.at("platforms").at("nvidia").at("peaks").at("hbm_copy").as_double(), "reference_peak");
    result.target_peak = positive(target.at("peaks").at("hbm_copy").as_double(), "target_peak");
    result.min_relative_efficiency = positive(document_.at("min_relative_efficiency").as_double(), "min_relative_efficiency");
    (void)result.limit_us();
    return result;
  } catch (const std::exception &error) {
    throw std::runtime_error("performance baseline " + path_.string() + " case " + spec.name + ": " + error.what());
  }
}

}  // namespace flagdnn::testing::ascend
