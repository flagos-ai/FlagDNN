// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "capability.hpp"

#include "runtime/json.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace flagdnn::validation::thead {
namespace {

using JsonObject = flagdnn::native::json::Value::Object;

const std::set<std::string, std::less<>> kStableReasons = {
    "acdnn_semantic_mismatch",
    "acdnn_status_not_supported",
    "attribute_unsupported",
    "dtype_unsupported",
    "layout_unsupported",
    "no_certified_acdnn_primitive",
    "production_kernel_unavailable",
    "real_device_qualification_pending",
    "sdk_version_out_of_range",
    "shape_unsupported",
};

const std::set<std::string, std::less<>> kDataTypes = {
    "bf16",     "bool",     "fp16",     "fp32",
    "fp8_e4m3", "fp8_e5m2", "fp8_e8m0", "int32"};
const std::set<std::string, std::less<>> kLayouts = {
    "contiguous", "explicit_strided", "nchw", "nhwc"};
const std::set<std::string, std::less<>> kStridePolicies = {
    "broadcast_compatible", "dense_contiguous", "explicit_strides"};
const std::set<std::string, std::less<>> kBroadcastPolicies = {
    "none", "numpy_trailing"};

void require_keys(const JsonObject &object,
                  std::initializer_list<std::string_view> required,
                  std::string_view context) {
  for (const std::string_view key : required) {
    if (!object.contains(key)) {
      throw std::runtime_error(std::string(context) +
                               " is missing required key " +
                               std::string(key));
    }
  }
  for (const auto &[key, value] : object) {
    (void)value;
    const bool known = std::ranges::any_of(
        required, [&key](std::string_view candidate) { return key == candidate; });
    if (!known) {
      throw std::runtime_error(std::string(context) + " has unknown key " +
                               key);
    }
  }
}

std::vector<std::string>
parse_string_array(const flagdnn::native::json::Value &value,
                   std::string_view context, bool allow_empty = false) {
  std::vector<std::string> result;
  std::set<std::string, std::less<>> unique;
  for (const auto &item : value.as_array()) {
    std::string text = item.as_string();
    if (text.empty()) {
      throw std::runtime_error(std::string(context) +
                               " contains an empty value");
    }
    if (!unique.insert(text).second) {
      throw std::runtime_error(std::string(context) +
                               " contains a duplicate value");
    }
    result.push_back(std::move(text));
  }
  if (!allow_empty && result.empty()) {
    throw std::runtime_error(std::string(context) + " must not be empty");
  }
  return result;
}

CapabilityStatus parse_status(std::string_view value) {
  if (value == "supported") {
    return CapabilityStatus::kSupported;
  }
  if (value == "unsupported") {
    return CapabilityStatus::kUnsupported;
  }
  if (value == "probe_required") {
    return CapabilityStatus::kProbeRequired;
  }
  throw std::runtime_error("unknown THead capability status");
}

ReferencePath parse_path(const flagdnn::native::json::Value &value,
                         CapabilityStatus status) {
  if (status == CapabilityStatus::kUnsupported) {
    if (!value.is_null()) {
      throw std::runtime_error("unsupported capability must have null path");
    }
    return ReferencePath::kNone;
  }
  const std::string &path = value.as_string();
  if (path == "stable_primitive") {
    return ReferencePath::kStablePrimitive;
  }
  if (path == "backend_descriptor") {
    return ReferencePath::kBackendDescriptor;
  }
  throw std::runtime_error("unknown THead acDNN reference path");
}

CapabilityConstraints
parse_constraints(const flagdnn::native::json::Value &value,
                  std::string_view context) {
  const JsonObject &object = value.as_object();
  require_keys(object,
               {"dtypes", "compute_type", "rank", "shape", "layouts",
                "stride_policy", "broadcast", "attributes"},
               context);

  CapabilityConstraints result;
  result.dtypes = parse_string_array(object.at("dtypes"),
                                     std::string(context) + " dtypes");
  for (const std::string &dtype : result.dtypes) {
    if (!kDataTypes.contains(dtype)) {
      throw std::runtime_error(std::string(context) +
                               " contains an unknown dtype");
    }
  }
  result.compute_type = object.at("compute_type").as_string();
  if (!kDataTypes.contains(result.compute_type)) {
    throw std::runtime_error(std::string(context) +
                             " has an unknown compute type");
  }

  const JsonObject &rank = object.at("rank").as_object();
  require_keys(rank, {"min", "max"}, std::string(context) + " rank");
  result.rank.minimum = rank.at("min").as_int();
  result.rank.maximum = rank.at("max").as_int();
  if (result.rank.minimum <= 0 || result.rank.maximum < result.rank.minimum) {
    throw std::runtime_error(std::string(context) + " has an invalid rank");
  }

  result.shape = parse_string_array(object.at("shape"),
                                    std::string(context) + " shape");
  result.layouts = parse_string_array(object.at("layouts"),
                                      std::string(context) + " layouts");
  for (const std::string &layout : result.layouts) {
    if (!kLayouts.contains(layout)) {
      throw std::runtime_error(std::string(context) +
                               " contains an unknown layout");
    }
  }
  result.stride_policy = object.at("stride_policy").as_string();
  if (!kStridePolicies.contains(result.stride_policy)) {
    throw std::runtime_error(std::string(context) +
                             " has an unknown stride policy");
  }
  result.broadcast = object.at("broadcast").as_string();
  if (!kBroadcastPolicies.contains(result.broadcast)) {
    throw std::runtime_error(std::string(context) +
                             " has an unknown broadcast policy");
  }

  const JsonObject &attributes = object.at("attributes").as_object();
  for (const auto &[name, values] : attributes) {
    if (name.empty()) {
      throw std::runtime_error(std::string(context) +
                               " contains an empty attribute name");
    }
    result.attributes.emplace(
        name, parse_string_array(values, std::string(context) +
                                             " attribute " + name));
  }
  return result;
}

CapabilityRecord parse_record(const JsonObject &object,
                              std::string_view context) {
  require_keys(object,
               {"status", "path", "reference_plan", "constraints",
                "reason_code", "detail"},
               context);
  CapabilityRecord result;
  result.status = parse_status(object.at("status").as_string());
  result.path = parse_path(object.at("path"), result.status);
  result.reference_plan = parse_string_array(
      object.at("reference_plan"), std::string(context) + " reference plan",
      result.status == CapabilityStatus::kUnsupported);
  result.reason_code = object.at("reason_code").as_string();
  result.detail = object.at("detail").as_string();
  if (result.detail.empty()) {
    throw std::runtime_error(std::string(context) + " has an empty detail");
  }

  if (result.status == CapabilityStatus::kUnsupported) {
    if (!object.at("constraints").is_null()) {
      throw std::runtime_error(std::string(context) +
                               " unsupported record has constraints");
    }
    if (!result.reference_plan.empty()) {
      throw std::runtime_error(std::string(context) +
                               " unsupported record has a reference plan");
    }
    if (result.reason_code.empty() ||
        !kStableReasons.contains(result.reason_code) ||
        result.reason_code == "real_device_qualification_pending") {
      throw std::runtime_error(std::string(context) +
                               " has an invalid unsupported reason");
    }
    return result;
  }

  result.constraints = parse_constraints(
      object.at("constraints"), std::string(context) + " constraints");
  if (result.status == CapabilityStatus::kSupported) {
    if (!result.reason_code.empty()) {
      throw std::runtime_error(std::string(context) +
                               " supported record has a skip reason");
    }
  } else if (result.reason_code != "real_device_qualification_pending") {
    throw std::runtime_error(std::string(context) +
                             " probe record has no qualification reason");
  }
  return result;
}

struct ParsedVersion {
  std::vector<std::uint64_t> components;
  std::string suffix;
};

ParsedVersion parse_version(std::string_view value) {
  if (value.empty()) {
    throw std::runtime_error("PPU SDK version must not be empty");
  }
  ParsedVersion result;
  const std::size_t suffix_offset = value.find('-');
  const std::string_view core = value.substr(0, suffix_offset);
  if (suffix_offset != std::string_view::npos) {
    result.suffix = std::string(value.substr(suffix_offset + 1));
    if (result.suffix.empty() ||
        !std::ranges::all_of(result.suffix, [](unsigned char character) {
          return std::isalnum(character) != 0 || character == '.' ||
                 character == '_' || character == '+' || character == '-';
        })) {
      throw std::runtime_error("malformed PPU SDK version suffix");
    }
  }

  std::size_t start = 0;
  while (start <= core.size()) {
    const std::size_t end = core.find('.', start);
    const std::string_view component =
        core.substr(start, end == std::string_view::npos
                               ? std::string_view::npos
                               : end - start);
    if (component.empty() ||
        !std::ranges::all_of(component, [](unsigned char character) {
          return std::isdigit(character) != 0;
        })) {
      throw std::runtime_error("malformed PPU SDK version component");
    }
    std::uint64_t parsed = 0;
    for (const char character : component) {
      const auto digit = static_cast<std::uint64_t>(character - '0');
      if (parsed >
          (std::numeric_limits<std::uint64_t>::max() - digit) / 10U) {
        throw std::runtime_error("PPU SDK version component is too large");
      }
      parsed = parsed * 10U + digit;
    }
    result.components.push_back(parsed);
    if (end == std::string_view::npos) {
      break;
    }
    start = end + 1;
  }
  return result;
}

int compare_versions(std::string_view left, std::string_view right) {
  const ParsedVersion left_parsed = parse_version(left);
  const ParsedVersion right_parsed = parse_version(right);
  const std::size_t count =
      std::max(left_parsed.components.size(), right_parsed.components.size());
  for (std::size_t index = 0; index < count; ++index) {
    const std::uint64_t left_component =
        index < left_parsed.components.size() ? left_parsed.components[index]
                                              : 0;
    const std::uint64_t right_component =
        index < right_parsed.components.size() ? right_parsed.components[index]
                                               : 0;
    if (left_component != right_component) {
      return left_component < right_component ? -1 : 1;
    }
  }
  if (left_parsed.suffix == right_parsed.suffix) {
    return 0;
  }
  if (left_parsed.suffix.empty()) {
    return 1;
  }
  if (right_parsed.suffix.empty()) {
    return -1;
  }
  return left_parsed.suffix < right_parsed.suffix ? -1 : 1;
}

}  // namespace

CapabilityCatalog CapabilityCatalog::parse(std::string_view document) {
  const flagdnn::native::json::Value root_value =
      flagdnn::native::json::parse(document);
  const JsonObject &root = root_value.as_object();
  require_keys(root,
               {"schema_version", "platform", "reference_provider",
                "baseline", "operators"},
               "catalog root");
  if (root.at("schema_version").as_int() != 1) {
    throw std::runtime_error("unsupported THead capability schema");
  }
  if (root.at("platform").as_string() != "thead" ||
      root.at("reference_provider").as_string() != "acdnn") {
    throw std::runtime_error("THead capability catalog identity mismatch");
  }

  CapabilityCatalog catalog;
  const JsonObject &baseline = root.at("baseline").as_object();
  require_keys(baseline,
               {"ppu_sdk_min", "ppu_sdk_max", "acdnn_header_min",
                "acdnn_header_max", "acdnn_runtime_min",
                "acdnn_runtime_max"},
               "catalog baseline");
  catalog.baseline_.ppu_sdk_min = baseline.at("ppu_sdk_min").as_string();
  catalog.baseline_.ppu_sdk_max = baseline.at("ppu_sdk_max").as_string();
  catalog.baseline_.acdnn_header_min =
      baseline.at("acdnn_header_min").as_int();
  catalog.baseline_.acdnn_header_max =
      baseline.at("acdnn_header_max").as_int();
  catalog.baseline_.acdnn_runtime_min =
      baseline.at("acdnn_runtime_min").as_int();
  catalog.baseline_.acdnn_runtime_max =
      baseline.at("acdnn_runtime_max").as_int();
  if (compare_versions(catalog.baseline_.ppu_sdk_min,
                       catalog.baseline_.ppu_sdk_max) > 0 ||
      catalog.baseline_.acdnn_header_min <= 0 ||
      catalog.baseline_.acdnn_header_max <
          catalog.baseline_.acdnn_header_min ||
      catalog.baseline_.acdnn_runtime_min <= 0 ||
      catalog.baseline_.acdnn_runtime_max <
          catalog.baseline_.acdnn_runtime_min) {
    throw std::runtime_error("THead capability baseline range is invalid");
  }

  const JsonObject &operators = root.at("operators").as_object();
  if (operators.empty()) {
    throw std::runtime_error("THead capability catalog has no operators");
  }
  std::map<std::string, std::string, std::less<>> case_owners;
  for (const auto &[operation, operation_value] : operators) {
    if (operation.empty()) {
      throw std::runtime_error("THead capability catalog has an empty operator");
    }
    const JsonObject &operation_object = operation_value.as_object();
    require_keys(operation_object, {"case_names", "default", "overrides"},
                 "operator " + operation);
    const std::vector<std::string> case_names = parse_string_array(
        operation_object.at("case_names"), "operator " + operation + " cases");
    const CapabilityRecord default_record = parse_record(
        operation_object.at("default").as_object(),
        "operator " + operation + " default");
    const JsonObject &overrides =
        operation_object.at("overrides").as_object();
    std::map<std::string, CapabilityRecord, std::less<>> parsed_overrides;
    for (const auto &[case_name, record] : overrides) {
      if (std::ranges::find(case_names, case_name) == case_names.end()) {
        throw std::runtime_error("override is not owned by operator " +
                                 operation + ": " + case_name);
      }
      parsed_overrides.emplace(
          case_name, parse_record(record.as_object(),
                                  "operator " + operation + " case " +
                                      case_name));
    }

    auto &output_cases = catalog.records_[operation];
    for (const std::string &case_name : case_names) {
      const auto [owner, inserted] = case_owners.emplace(case_name, operation);
      if (!inserted) {
        throw std::runtime_error("capability case has multiple owners: " +
                                 case_name + " (" + owner->second + ", " +
                                 operation + ")");
      }
      const auto override_iterator = parsed_overrides.find(case_name);
      output_cases.emplace(case_name,
                           override_iterator == parsed_overrides.end()
                               ? default_record
                               : override_iterator->second);
    }
  }
  return catalog;
}

CapabilityCatalog CapabilityCatalog::load(const std::string &path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open THead acDNN capability catalog: " +
                             path);
  }
  return parse(std::string(std::istreambuf_iterator<char>(input),
                           std::istreambuf_iterator<char>()));
}

void CapabilityCatalog::require_exact_cases(
    const std::map<std::string, std::set<std::string>> &expected) const {
  if (records_.size() != expected.size()) {
    throw std::runtime_error("capability catalog operator count mismatch");
  }
  for (const auto &[operation, expected_cases] : expected) {
    const auto operation_iterator = records_.find(operation);
    if (operation_iterator == records_.end()) {
      throw std::runtime_error("capability catalog is missing operator " +
                               operation);
    }
    if (operation_iterator->second.size() != expected_cases.size()) {
      throw std::runtime_error("capability catalog case count mismatch for " +
                               operation);
    }
    for (const std::string &case_name : expected_cases) {
      if (!operation_iterator->second.contains(case_name)) {
        throw std::runtime_error("capability catalog is missing case " +
                                 operation + "/" + case_name);
      }
    }
  }
}

void CapabilityCatalog::validate_versions(
    std::string_view ppu_sdk_version, std::int64_t acdnn_header_version,
    std::int64_t acdnn_runtime_version) const {
  if (compare_versions(ppu_sdk_version, baseline_.ppu_sdk_min) < 0 ||
      compare_versions(ppu_sdk_version, baseline_.ppu_sdk_max) > 0) {
    throw std::runtime_error("PPU SDK version is outside capability baseline");
  }
  if (acdnn_header_version < baseline_.acdnn_header_min ||
      acdnn_header_version > baseline_.acdnn_header_max) {
    throw std::runtime_error(
        "acDNN header version is outside capability baseline");
  }
  if (acdnn_runtime_version < baseline_.acdnn_runtime_min ||
      acdnn_runtime_version > baseline_.acdnn_runtime_max) {
    throw std::runtime_error(
        "acDNN runtime version is outside capability baseline");
  }
}

const CapabilityRecord &
CapabilityCatalog::lookup(std::string_view operation,
                          std::string_view case_name) const {
  const auto operation_iterator = records_.find(operation);
  if (operation_iterator == records_.end()) {
    throw std::runtime_error("unknown THead capability operator");
  }
  const auto case_iterator = operation_iterator->second.find(case_name);
  if (case_iterator == operation_iterator->second.end()) {
    throw std::runtime_error("unknown THead capability case");
  }
  return case_iterator->second;
}

}  // namespace flagdnn::validation::thead
