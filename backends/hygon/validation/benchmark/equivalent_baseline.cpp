/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "equivalent_baseline.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
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
    case Operation::kAdd: return {"add", "hbm_triad", InputDomain::kReal, 2};
    case Operation::kRelu: return {"relu", "hbm_copy"};
    case Operation::kBatchnorm:
      return {"batchnorm", "vector_fp32", InputDomain::kReal, 5};
    case Operation::kReduction: return {"reduction", "vector_fp32"};
    case Operation::kReshape: return {"reshape", "hbm_copy"};
    case Operation::kTranspose: return {"transpose", "hbm_copy"};
    case Operation::kSlice: return {"slice", "hbm_copy"};
    case Operation::kGraph:
      if (spec.name.starts_with("add_square_"))
        return {"add_square", "vector_fp32", InputDomain::kReal, 2};
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
    case FLAGDNN_POINTWISE_ABS: return {"abs", "hbm_copy"};
    case FLAGDNN_POINTWISE_SUB:
      return {"sub", "hbm_triad", InputDomain::kReal, 2};
    case FLAGDNN_POINTWISE_MUL:
      return {spec.name.starts_with("scale_") ? "scale" : "mul", "hbm_triad",
              InputDomain::kReal, 2};
    case FLAGDNN_POINTWISE_MIN:
      return {"min", "hbm_triad", InputDomain::kReal, 2};
    case FLAGDNN_POINTWISE_MAX:
      return {"max", "hbm_triad", InputDomain::kReal, 2};
    case FLAGDNN_POINTWISE_ELU_FWD: return {"elu", "vector_fp32"};
    case FLAGDNN_POINTWISE_SIGMOID_FWD: return {"sigmoid", "vector_fp32"};
    case FLAGDNN_POINTWISE_SIGMOID_BWD:
      return {"sigmoid_backward", "vector_fp32", InputDomain::kReal, 2};
    case FLAGDNN_POINTWISE_SWISH_FWD: return {"swish", "vector_fp32"};
    case FLAGDNN_POINTWISE_SWISH_BWD:
      return {"swish_backward", "vector_fp32", InputDomain::kReal, 2};
    case FLAGDNN_POINTWISE_TANH_FWD: return {"tanh", "vector_fp32"};
    case FLAGDNN_POINTWISE_TANH_BWD:
      return {"tanh_backward", "vector_fp32", InputDomain::kReal, 2};
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
      return {"relu", "hbm_copy"};
    case FLAGDNN_POINTWISE_RELU_BWD:
      if (spec.name.starts_with("leaky_relu_backward_"))
        return {"leaky_relu_backward", "hbm_triad", InputDomain::kReal, 2};
      return {"relu_backward", "hbm_triad", InputDomain::kReal, 2};
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

bool supplemental_pointwise(std::string_view operation) {
  constexpr std::string_view names[] = {
      "abs",   "add",          "add_square",
      "elu",   "max",          "min",
      "mul",   "relu",         "relu_backward",
      "scale", "sigmoid",      "sigmoid_backward",
      "sub",   "swish",        "swish_backward",
      "tanh",  "tanh_backward"};
  return std::find(std::begin(names), std::end(names), operation) !=
         std::end(names);
}

bool supplemental_case(const BenchmarkCase& spec, const Policy& chosen) {
  return supplemental_pointwise(chosen.operation) ||
         chosen.operation == "batchnorm" || chosen.operation == "reduction" ||
         (chosen.operation == "identity" &&
          spec.tensors.front().data_type != FLAGDNN_DATA_FP8_E4M3 &&
          spec.tensors.front().data_type != FLAGDNN_DATA_FP8_E5M2) ||
         spec.name.starts_with("logical_and_bool_") ||
         spec.name.starts_with("logical_or_bool_");
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

std::int64_t checked_element_count(const TensorSpec& tensor) {
  std::int64_t count = 1;
  for (const auto dimension : tensor.dimensions) {
    require(dimension > 0 &&
                count <= std::numeric_limits<std::int64_t>::max() / dimension,
            "invalid flattened reference extent");
    count *= dimension;
  }
  return count;
}

void flattened_descriptor(const TensorSpec& tensor, const Value& entry) {
  const auto count = checked_element_count(tensor);
  require(
      same_vector(entry.at("reference_descriptor_shape"), {1, count, 1, 1}) &&
          same_vector(entry.at("reference_descriptor_strides"),
                      {count, 1, 1, 1}),
      "flattened NVIDIA graph descriptor differs");
}

void graph_attributes(const BenchmarkCase& spec, const Value& entry,
                      const Policy& chosen) {
  pointwise_attributes(spec, entry, chosen);
  const auto& nodes = entry.at("graph_nodes").as_array();
  const auto& intermediates = entry.at("intermediate_tensors").as_array();
  require(spec.graph.nodes.size() == 2 &&
              spec.graph.intermediates.size() == 1 && nodes.size() == 2 &&
              intermediates.size() == 1,
          "add_square graph topology differs");
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const auto& node = spec.graph.nodes[i];
    const auto& recorded = nodes[i];
    exact_keys(
        recorded,
        {"name", "operation_kind", "input_uids", "output_uid", "pointwise_mode",
         "alpha", "pointwise_flags", "relu_lower_clip", "relu_upper_clip",
         "relu_lower_clip_slope", "swish_beta", "elu_alpha", "softplus_beta"});
    require(recorded.at("name").as_string() == node.name &&
                recorded.at("operation_kind").as_int() ==
                    static_cast<int>(node.operation) &&
                node.operation == Operation::kPointwise &&
                same_vector(recorded.at("input_uids"), node.input_uids) &&
                recorded.at("output_uid").as_int() == node.output_uid,
            "add_square graph node differs");
    const auto& a = node.pointwise_attributes;
    const auto defaults = default_pointwise_attributes();
    require(a.struct_size == defaults.struct_size &&
                a.version == defaults.version &&
                recorded.at("pointwise_mode").as_int() == node.pointwise_mode &&
                recorded.at("pointwise_flags").as_int() ==
                    static_cast<std::int64_t>(a.flags),
            "add_square node attributes differ");
    same_number(recorded, "alpha", node.alpha);
    same_number(recorded, "relu_lower_clip", a.relu_lower_clip);
    same_number(recorded, "relu_upper_clip", a.relu_upper_clip);
    same_number(recorded, "relu_lower_clip_slope", a.relu_lower_clip_slope);
    same_number(recorded, "swish_beta", a.swish_beta);
    same_number(recorded, "elu_alpha", a.elu_alpha);
    same_number(recorded, "softplus_beta", a.softplus_beta);
  }
  const auto& tensor = spec.graph.intermediates.front();
  const auto& recorded = intermediates.front();
  require(
      recorded.at("uid").as_int() == tensor.uid &&
          recorded.at("role").as_string() == "intermediate" &&
          recorded.at("semantic_role").as_string() ==
              "intermediate_" + std::to_string(tensor.uid) &&
          recorded.at("dtype").as_string() == dtype_name(tensor.data_type) &&
          same_vector(recorded.at("shape"), tensor.dimensions) &&
          same_vector(recorded.at("strides"), tensor.strides) &&
          recorded.at("binding_byte_offset").as_int() == 0 &&
          tensor.binding_byte_offset == 0 && recorded.at("virtual").as_bool(),
      "add_square intermediate differs");
  require(std::none_of(
              spec.tensors.begin(), spec.tensors.end(),
              [&](const TensorSpec& bound) { return bound.uid == tensor.uid; }),
          "add_square intermediate aliases a binding UID");
  flattened_descriptor(tensor, recorded);
}

void supplemental_inputs(const BenchmarkCase& spec, const Value& entry,
                         const Value& measurement,
                         const std::vector<InputDomain>& domains) {
  constexpr std::string_view id = "nvidia-missing-20261012";
  require(entry.at("measurement_id").as_string() == id,
          "supplemental measurement identity differs");
  const auto& run = measurement.at("supplemental_runs").at(id);
  require(run.at("case_count").as_int() == 371 &&
              !run.at("source_revision").as_string().empty() &&
              !run.at("manifest").as_string().empty(),
          "supplemental provenance is incomplete");
  const auto& generation = entry.at("input_generation");
  constexpr std::string_view real_formula =
      "float(int((index * 17 + input_index * 11) % 41) - 20) / float(13 + "
      "input_index)";
  const bool core = spec.name.find("_perf_") != std::string::npos;
  if (core) {
    require(entry.at("input_domain").as_int() ==
                    static_cast<int>(spec.input_domain) &&
                entry.at("output_count").as_int() ==
                    static_cast<std::int64_t>(spec.output_count),
            "core supplemental input/output contract differs");
    std::vector<std::int64_t> domain_values;
    for (auto domain : domains)
      domain_values.push_back(static_cast<int>(domain));
    require(same_vector(entry.at("input_domains"), domain_values),
            "core supplemental per-input domains differ");
    exact_keys(generation, {"source", "domain_definitions"});
    require(
        generation.at("source").as_string() ==
                "backends/nvidia/validation/benchmark/runner.cpp::make_input" &&
            generation.at("domain_definitions").as_string() ==
                "measurement.supplemental_runs.nvidia-missing-20261012.input_"
                "domains",
        "core supplemental input generator differs");
    const auto& definitions = run.at("input_domains");
    require(definitions.at("real_value").as_string() ==
                    "float(int((index * 17 + tensor_index * 11) % 41) - 20) / "
                    "float(13 + tensor_index)" &&
                definitions.at("0").as_string() == "real_value (kReal)" &&
                definitions.at("1").as_string() ==
                    "abs(real_value) + 0.5F (kPositive)",
            "core supplemental input domain formula differs");
    return;
  }
  exact_keys(generation, {"source", "inputs", "padding_value"});
  require(generation.at("source").as_string() ==
              "backends/hygon/validation/benchmark/runner.cpp:make_input",
          "auxiliary supplemental input generator differs");
  same_number(generation, "padding_value", -97);
  const auto& inputs = generation.at("inputs").as_array();
  require(inputs.size() == domains.size(), "supplemental input count differs");
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto& input = inputs[i];
    exact_keys(input,
               {"uid", "input_index", "domain", "formula", "conversion"});
    require(
        input.at("uid").as_int() == spec.tensors[i].uid &&
            input.at("input_index").as_int() == static_cast<std::int64_t>(i),
        "supplemental input identity differs");
    const bool logical = domains[i] == InputDomain::kLogical;
    require(
        (logical || domains[i] == InputDomain::kReal) &&
            input.at("domain").as_string() == (logical ? "logical" : "real") &&
            input.at("formula").as_string() ==
                (logical ? "((index * 17 + input_index * 11) % 3) != 0"
                         : real_formula),
        "supplemental input formula differs");
    const auto dtype = spec.tensors[i].data_type;
    const std::string_view conversion =
        dtype == FLAGDNN_DATA_BOOLEAN
            ? "nonzero to bool; pack little-endian bits"
        : dtype == FLAGDNN_DATA_INT32
            ? "truncate toward zero; equivalent integer rational evaluation "
              "without FLOAT storage"
            : "round to tensor dtype before CPU oracle";
    require(input.at("conversion").as_string() == conversion,
            "supplemental input conversion differs");
  }
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
  if (supplemental_pointwise(chosen.operation))
    return specification.name.starts_with(std::string(chosen.operation) +
                                          "_perf_bfloat16_") &&
           !specification.tensors.empty() &&
           std::all_of(specification.tensors.begin(),
                       specification.tensors.end(),
                       [](const TensorSpec& tensor) {
                         return tensor.data_type == FLAGDNN_DATA_BFLOAT16;
                       });
  if (chosen.operation == "identity") {
    if (specification.tensors.size() != 2) return false;
    const auto type = specification.tensors.front().data_type;
    const bool dtype_supported =
        type == FLAGDNN_DATA_FP8_E4M3 || type == FLAGDNN_DATA_FP8_E5M2 ||
        type == FLAGDNN_DATA_BFLOAT16 || type == FLAGDNN_DATA_INT32 ||
        type == FLAGDNN_DATA_BOOLEAN;
    return dtype_supported &&
           specification.tensors.back().data_type == type &&
           (specification.name.starts_with(
                "identity_" + std::string(dtype_name(type)) + "_") ||
            (type == FLAGDNN_DATA_BFLOAT16 &&
             specification.name.starts_with("identity_perf_bfloat16_")));
  }
  if (chosen.operation == "batchnorm") {
    if (specification.tensors.size() != 10 || specification.output_count != 5)
      return false;
    const auto type = specification.tensors.front().data_type;
    if ((type != FLAGDNN_DATA_FLOAT16 && type != FLAGDNN_DATA_BFLOAT16) ||
        !specification.name.starts_with("batchnorm_perf_" +
                                        std::string(dtype_name(type)) + "_"))
      return false;
    for (std::size_t i = 0; i < specification.tensors.size(); ++i)
      if (specification.tensors[i].data_type !=
          (i <= 2 || i == 5 ? type : FLAGDNN_DATA_FLOAT32))
        return false;
    return true;
  }
  if (chosen.operation == "reduction") {
    if (specification.tensors.size() != 2 || specification.output_count != 1 ||
        specification.tensors.back().data_type != FLAGDNN_DATA_FLOAT32)
      return false;
    const auto& input = specification.tensors.front();
    const auto rank = static_cast<std::int32_t>(input.dimensions.size());
    const auto axis = specification.reduction_axis < 0
                          ? specification.reduction_axis + rank
                          : specification.reduction_axis;
    if (input.data_type == FLAGDNN_DATA_BFLOAT16)
      return rank == 2 &&
             specification.reduction_mode == FLAGDNN_REDUCTION_ADD &&
             specification.name.starts_with("reduction_sum_bfloat16_to_fp32_");
    return input.data_type == FLAGDNN_DATA_FLOAT16 && rank > 2 && axis > 0 &&
           axis < rank - 1 &&
           (specification.reduction_mode == FLAGDNN_REDUCTION_ADD ||
            specification.reduction_mode == FLAGDNN_REDUCTION_AVG ||
            specification.reduction_mode == FLAGDNN_REDUCTION_MUL) &&
           specification.name.find("_fp16_to_fp32_") != std::string::npos;
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
    const bool batchnorm = chosen.operation == "batchnorm";
    const bool reduction = chosen.operation == "reduction";
    const bool add_square = chosen.operation == "add_square";
    const bool supplement = supplemental_case(specification, chosen);
    require(specification.output_count == (batchnorm ? 5U : 1U),
            "output count differs from operation");
    const std::size_t input_count = input_tensor_count(specification);
    require(chosen.operation == "concatenate" ? input_count >= 1
                                              : input_count == chosen.inputs,
            "input count differs from operation");
    require(specification.input_domain == chosen.domain,
            "input domain differs from the NVIDIA catalog");
    std::vector<InputDomain> domains(input_count, chosen.domain);
    if (chosen.operation == "binary_select") domains[2] = InputDomain::kLogical;
    if (batchnorm) domains[4] = InputDomain::kPositive;
    require(specification.input_domains.empty()
                ? chosen.operation != "binary_select" && !batchnorm
                : specification.input_domains == domains,
            "per-input domains differ from the NVIDIA catalog");
    require((specification.operation == Operation::kGraph && !add_square) == (metadata != nullptr),
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
    const bool pointwise = specification.operation == Operation::kPointwise ||
                           specification.operation == Operation::kAdd ||
                           specification.operation == Operation::kRelu;
    const std::string validation = entry.at("validation").as_string();
    require(chosen.operation == "identity"
                ? validation == (specification.name.starts_with("identity_perf_")
                                 ? "cpu_full"
                                 : "bit_exact") &&
                      specification.absolute_tolerance == 0 &&
                      specification.relative_tolerance == 0
                : validation == "cpu_full" ||
                      (!pointwise && !supplement && validation == "sanity"),
            "reference validation is insufficient or unknown");
    if (pointwise) {
      pointwise_attributes(specification, entry, chosen);
    } else if (add_square) {
      graph_attributes(specification, entry, chosen);
    } else if (batchnorm) {
      exact_keys(entry.at("attributes"), {"epsilon", "momentum"});
      same_number(entry.at("attributes"), "epsilon",
                  specification.normalization.epsilon);
      same_number(entry.at("attributes"), "momentum",
                  specification.normalization.momentum);
    } else if (reduction) {
      const auto& attrs = entry.at("attributes");
      exact_keys(attrs, {"mode", "mode_name", "axis", "keep_dimensions"});
      const auto mode = specification.reduction_mode;
      require(attrs.at("mode").as_int() == mode &&
                  attrs.at("mode_name").as_string() ==
                      (mode == FLAGDNN_REDUCTION_ADD   ? "sum"
                       : mode == FLAGDNN_REDUCTION_AVG ? "avg"
                                                       : "mul") &&
                  attrs.at("axis").as_int() == specification.reduction_axis &&
                  attrs.at("keep_dimensions").as_bool() ==
                      specification.keep_dimensions,
              "reduction attributes differ");
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
      require(count->as_int() ==
                  static_cast<std::int64_t>(specification.output_count),
              "reference output count differs");
    if (supplement)
      supplemental_inputs(specification, entry, reference.at("measurement"),
                          domains);

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
      if (supplement)
        require(expected.at("uid").as_int() == tensor.uid,
                label + " UID differs");
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
      if (batchnorm && (i == 0 || i == 5)) {
        const auto& shape = tensor.dimensions;
        // Validate every positive product before computing either layout.
        // Even adversarial matching source/target metadata must fail closed.
        (void)checked_element_count(tensor);
        require(shape.size() == 4 &&
                    tensor.strides == contiguous_strides(shape) &&
                    same_vector(expected.at("reference_strides"),
                                {shape[1] * shape[2] * shape[3], 1,
                                 shape[3] * shape[1], shape[1]}) &&
                    !expected.at("reference_layout_conversion_timed").as_bool(),
                label + " batchnorm NCHW-to-NHWC reference layout differs");
      } else if (const auto* strides = optional(expected, "reference_strides")) {
        require(same_vector(*strides, tensor.strides),
                label + " converted reference strides differ");
      }
      if (batchnorm) {
        constexpr std::string_view roles[] = {"x",
                                              "scale",
                                              "bias",
                                              "previous_running_mean",
                                              "previous_running_variance",
                                              "y",
                                              "saved_mean",
                                              "saved_inverse_variance",
                                              "next_running_mean",
                                              "next_running_variance"};
        require(expected.at("semantic_role").as_string() == roles[i],
                label + " batchnorm semantic role differs");
      } else if (supplement &&
                 specification.name.find("_perf_") != std::string::npos) {
        require(
            expected.at("semantic_role").as_string() ==
                (i < input_count ? "input_" + std::to_string(i)
                                 : "output_" + std::to_string(i - input_count)),
            label + " core semantic role differs");
      }
      if (add_square) flattened_descriptor(tensor, expected);
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
    require(result.timing_profile == (batchnorm ? "batchnorm_training" : "native"),
        "equivalent baseline timing profile differs");
    const auto& measurement = reference.at("measurement");
    require(measurement.at("timing_method").as_string() ==
                "device_events_over_cuda_graph",
            "reference timing method differs");
    require(measurement.at("statistic").as_string() == "nearest_rank_p50",
            "reference timing statistic differs");
    const auto& timing =
        measurement.at("timing_profiles").at(result.timing_profile);
    const auto& config = specification.benchmark;
    require(config.warmup_iterations == (batchnorm ? 5 : 10) && config.sample_count == (batchnorm ? 10 : 20) &&
                  config.iterations_per_sample == (batchnorm ? 20 : 50),
            "equivalent baseline requires the recorded timing configuration");
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
