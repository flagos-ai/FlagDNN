// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "corex_cudnn_reference.hpp"

#include "runtime/json.hpp"

#include <fstream>
#include <iterator>
#include <stdexcept>
#include <utility>

#ifndef FLAGDNN_ILUVATAR_CUDNN_CAPABILITY_CATALOG
#define FLAGDNN_ILUVATAR_CUDNN_CAPABILITY_CATALOG                              \
  "corex_cudnn_capabilities.json"
#endif

namespace flagdnn::iluvatar::validation {
namespace {

using JsonObject = flagdnn::native::json::Value::Object;

const std::set<std::string, std::less<>> kStableReasons = {
    "NO_EXACT_CUDNN_PRIMITIVE",   "SEMANTIC_MISMATCH",     "DTYPE_UNSUPPORTED",
    "LAYOUT_UNSUPPORTED",         "ATTRIBUTE_UNSUPPORTED", "SHAPE_UNSUPPORTED",
    "CUDNN_STATUS_NOT_SUPPORTED", "CAPTURE_UNSUPPORTED",
};

void require_keys(const JsonObject &object,
                  std::initializer_list<std::string_view> required,
                  std::initializer_list<std::string_view> optional,
                  std::string_view context) {
  for (const std::string_view key : required) {
    if (!object.contains(key)) {
      throw std::runtime_error(std::string(context) +
                               " is missing required key " + std::string(key));
    }
  }
  for (const auto &[key, value] : object) {
    (void)value;
    bool known = false;
    for (const std::string_view candidate : required) {
      known = known || key == candidate;
    }
    for (const std::string_view candidate : optional) {
      known = known || key == candidate;
    }
    if (!known) {
      throw std::runtime_error(std::string(context) + " has unknown key " +
                               key);
    }
  }
}

CorexCudnnCatalogClassification parse_classification(std::string_view value) {
  if (value == "qualified_supported") {
    return CorexCudnnCatalogClassification::kQualifiedSupported;
  }
  if (value == "candidate") {
    return CorexCudnnCatalogClassification::kCandidate;
  }
  if (value == "semantic_unsupported") {
    return CorexCudnnCatalogClassification::kSemanticUnsupported;
  }
  if (value == "vendor_unsupported") {
    return CorexCudnnCatalogClassification::kVendorUnsupported;
  }
  throw std::runtime_error("unknown CoreX cuDNN catalog classification");
}

CorexCudnnCatalogRecord parse_record(const JsonObject &object,
                                     std::string_view context) {
  require_keys(object, {"classification", "reason", "primitive_sequence"},
               {"detail"}, context);
  CorexCudnnCatalogRecord record;
  record.classification =
      parse_classification(object.at("classification").as_string());
  record.reason_code = object.at("reason").as_string();
  if (const auto iterator = object.find("detail"); iterator != object.end()) {
    record.detail = iterator->second.as_string();
  }
  for (const auto &primitive : object.at("primitive_sequence").as_array()) {
    std::string name = primitive.as_string();
    if (name.empty()) {
      throw std::runtime_error(std::string(context) +
                               " contains an empty primitive name");
    }
    record.primitive_sequence.push_back(std::move(name));
  }

  const bool supported = record.classification ==
                         CorexCudnnCatalogClassification::kQualifiedSupported;
  const bool candidate =
      record.classification == CorexCudnnCatalogClassification::kCandidate;
  const bool unsupported =
      record.classification ==
          CorexCudnnCatalogClassification::kSemanticUnsupported ||
      record.classification ==
          CorexCudnnCatalogClassification::kVendorUnsupported;
  if ((supported || candidate) && !record.reason_code.empty()) {
    throw std::runtime_error(std::string(context) +
                             " executable classification has a skip reason");
  }
  if ((supported || candidate) && record.primitive_sequence.empty()) {
    throw std::runtime_error(std::string(context) +
                             " executable classification has no primitive");
  }
  if (unsupported && record.reason_code.empty()) {
    throw std::runtime_error(std::string(context) +
                             " unsupported classification has no reason");
  }
  if (unsupported && !kStableReasons.contains(record.reason_code)) {
    throw std::runtime_error(std::string(context) +
                             " uses an unknown skip reason");
  }
  if (record.classification ==
          CorexCudnnCatalogClassification::kSemanticUnsupported &&
      !record.primitive_sequence.empty()) {
    throw std::runtime_error(std::string(context) +
                             " semantic SKIP must not claim a primitive");
  }
  if (record.classification ==
          CorexCudnnCatalogClassification::kVendorUnsupported &&
      (record.reason_code != "CUDNN_STATUS_NOT_SUPPORTED" ||
       record.primitive_sequence.empty())) {
    throw std::runtime_error(std::string(context) +
                             " vendor SKIP must record NOT_SUPPORTED and a "
                             "primitive sequence");
  }
  return record;
}

const CorexCudnnCapabilityCatalog &default_catalog() {
  static const CorexCudnnCapabilityCatalog catalog =
      CorexCudnnCapabilityCatalog::load(
          FLAGDNN_ILUVATAR_CUDNN_CAPABILITY_CATALOG);
  return catalog;
}

} // namespace

CorexCudnnCapabilityCatalog
CorexCudnnCapabilityCatalog::parse(std::string_view document) {
  const flagdnn::native::json::Value root_value =
      flagdnn::native::json::parse(document);
  const JsonObject &root = root_value.as_object();
  require_keys(root, {"schema_version", "baseline", "operators"}, {},
               "catalog root");
  if (root.at("schema_version").as_int() != 1) {
    throw std::runtime_error("unsupported CoreX cuDNN catalog schema");
  }

  const JsonObject &baseline = root.at("baseline").as_object();
  require_keys(baseline, {"corex", "target", "cudnn_header", "cudnn_runtime"},
               {}, "catalog baseline");
  if (baseline.at("corex").as_string() != "4.4.0" ||
      baseline.at("target").as_string() != "corex_71" ||
      baseline.at("cudnn_header").as_int() != 7605 ||
      baseline.at("cudnn_runtime").as_int() != 7605) {
    throw std::runtime_error("CoreX cuDNN catalog baseline mismatch");
  }

  CorexCudnnCapabilityCatalog catalog;
  const JsonObject &operators = root.at("operators").as_object();
  if (operators.empty()) {
    throw std::runtime_error("CoreX cuDNN catalog has no operators");
  }
  for (const auto &[operation, operation_value] : operators) {
    if (operation.empty()) {
      throw std::runtime_error("CoreX cuDNN catalog has an empty operator");
    }
    const JsonObject &operation_object = operation_value.as_object();
    require_keys(operation_object, {"cases"}, {}, "operator " + operation);
    const JsonObject &cases = operation_object.at("cases").as_object();
    if (cases.empty()) {
      throw std::runtime_error("operator " + operation + " has no cases");
    }
    auto &output_cases = catalog.records_[operation];
    for (const auto &[case_name, case_value] : cases) {
      if (case_name.empty()) {
        throw std::runtime_error("operator " + operation +
                                 " has an empty case name");
      }
      output_cases.emplace(case_name, parse_record(case_value.as_object(),
                                                   "operator " + operation +
                                                       " case " + case_name));
    }
  }
  return catalog;
}

CorexCudnnCapabilityCatalog
CorexCudnnCapabilityCatalog::load(const std::string &path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open CoreX cuDNN capability catalog: " +
                             path);
  }
  return parse(std::string(std::istreambuf_iterator<char>(input),
                           std::istreambuf_iterator<char>()));
}

void CorexCudnnCapabilityCatalog::require_exact_cases(
    const std::map<std::string, std::set<std::string>> &expected) const {
  if (records_.size() != expected.size()) {
    throw std::runtime_error("capability catalog operator count mismatch");
  }
  for (const auto &[operation, expected_cases] : expected) {
    const auto operation_iterator = records_.find(std::string(operation));
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
    for (const auto &[case_name, record] : operation_iterator->second) {
      (void)record;
      if (!expected_cases.contains(case_name)) {
        throw std::runtime_error("capability case is owned by the wrong "
                                 "operator: " +
                                 operation + "/" + case_name);
      }
    }
  }
}

const CorexCudnnCatalogRecord &
CorexCudnnCapabilityCatalog::lookup(std::string_view operation,
                                    std::string_view case_name) const {
  const auto operation_iterator = records_.find(std::string(operation));
  if (operation_iterator == records_.end()) {
    throw std::runtime_error("unknown CoreX cuDNN capability operator");
  }
  const auto case_iterator =
      operation_iterator->second.find(std::string(case_name));
  if (case_iterator == operation_iterator->second.end()) {
    throw std::runtime_error("unknown CoreX cuDNN capability case");
  }
  return case_iterator->second;
}

CorexCudnnCapability corex_cudnn_capability(std::string_view operation,
                                            std::string_view case_name) {
  const CorexCudnnCatalogRecord &record =
      default_catalog().lookup(operation, case_name);
  CorexCudnnCapability result;
  result.reason_code = record.reason_code;
  result.detail = record.detail;
  result.primitive_sequence = record.primitive_sequence;
  switch (record.classification) {
  case CorexCudnnCatalogClassification::kQualifiedSupported:
    result.classification = CorexCudnnCapabilityClass::kSupported;
    break;
  case CorexCudnnCatalogClassification::kSemanticUnsupported:
    result.classification = CorexCudnnCapabilityClass::kSemanticUnsupported;
    break;
  case CorexCudnnCatalogClassification::kVendorUnsupported:
    result.classification = CorexCudnnCapabilityClass::kVendorUnsupported;
    break;
  case CorexCudnnCatalogClassification::kCandidate:
    result.classification = CorexCudnnCapabilityClass::kInvalidAdapterContract;
    result.detail =
        "unqualified CoreX cuDNN candidate: " + std::string(operation) + "/" +
        std::string(case_name);
    break;
  }
  return result;
}

void require_valid_capability(const CorexCudnnCapability &capability) {
  if (capability.classification ==
      CorexCudnnCapabilityClass::kInvalidAdapterContract) {
    throw std::runtime_error("invalid CoreX cuDNN adapter contract: " +
                             capability.detail);
  }
  if (capability.classification == CorexCudnnCapabilityClass::kSupported &&
      (!capability.reason_code.empty() ||
       capability.primitive_sequence.empty())) {
    throw std::runtime_error("supported CoreX cuDNN capability is malformed");
  }
  if (capability.classification != CorexCudnnCapabilityClass::kSupported &&
      capability.reason_code.empty()) {
    throw std::runtime_error("skipped CoreX cuDNN capability has no reason");
  }
}

} // namespace flagdnn::iluvatar::validation
