// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "acdnn_reference.hpp"
#include "acdnn_layout_reference.hpp"
#include "capability.hpp"
#include "acdnn_pointwise_dag.hpp"
#include "acdnn_attention_reference.hpp"
#include "numeric_types.hpp"
#include "tensor_io.hpp"

#include "common/add.hpp"
#include "common/attention.hpp"
#include "common/composite.hpp"
#include "common/convolution.hpp"
#include "common/layout.hpp"
#include "common/matmul.hpp"
#include "common/normalization.hpp"
#include "common/pointwise.hpp"
#include "common/reduction.hpp"

#include <acdnn.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#ifndef FLAGDNN_THEAD_PPU_SDK_VERSION
#define FLAGDNN_THEAD_PPU_SDK_VERSION "unknown"
#endif

namespace flagdnn::testing {

int run_pointwise_functional_test(int, char **,
                                  std::span<const PointwiseTestCase>,
                                  std::string_view) {
  throw std::logic_error("catalog contract runner stub must not execute");
}

}  // namespace flagdnn::testing

namespace {

using CaseMap = std::map<std::string, std::set<std::string>>;
using flagdnn::testing::PointwiseCaseDefinition;
using flagdnn::testing::PointwiseInputDomain;
using flagdnn::validation::thead::AcdnnHandle;
using flagdnn::validation::thead::AcdnnActivationDescriptor;
using flagdnn::validation::thead::AcdnnOpTensorDescriptor;
using flagdnn::validation::thead::AcdnnStatusError;
using flagdnn::validation::thead::AcdnnTensorDescriptor;
using flagdnn::validation::thead::CapabilityCatalog;
using flagdnn::validation::thead::CapabilityRecord;
using flagdnn::validation::thead::CapabilityStatus;
using flagdnn::validation::thead::DeviceBuffer;
using flagdnn::validation::thead::DeviceEvent;
using flagdnn::validation::thead::ReferencePath;
using flagdnn::validation::thead::ReferencePlan;
using flagdnn::validation::thead::UnsupportedCapability;

static_assert(!std::is_copy_constructible_v<AcdnnHandle>);
static_assert(!std::is_copy_constructible_v<AcdnnActivationDescriptor>);
static_assert(!std::is_copy_constructible_v<AcdnnTensorDescriptor>);
static_assert(!std::is_copy_constructible_v<AcdnnOpTensorDescriptor>);
static_assert(!std::is_copy_constructible_v<DeviceBuffer>);
static_assert(!std::is_copy_constructible_v<DeviceEvent>);
static_assert(std::is_move_constructible_v<DeviceBuffer>);
static_assert(std::is_move_constructible_v<DeviceEvent>);
static_assert(std::is_move_constructible_v<AcdnnActivationDescriptor>);

template <typename Cases>
void add_cases(CaseMap &output, std::string operation, const Cases &cases) {
  auto &names = output[std::move(operation)];
  for (const auto &test_case : cases) {
    if (!names.insert(test_case.name).second) {
      throw std::runtime_error("neutral case generator returned a duplicate");
    }
  }
  if (names.empty()) {
    throw std::runtime_error("neutral case generator returned no cases");
  }
}

enum class PointwiseArity { kUnary, kBinary, kTernary };

struct PointwiseSpecification {
  const char *operation;
  flagdnnPointwiseMode_t mode;
  PointwiseArity arity;
};

CaseMap neutral_cases() {
  using namespace flagdnn::testing;
  CaseMap output;
  const std::vector<PointwiseSpecification> pointwise = {
      {"abs", FLAGDNN_POINTWISE_ABS, PointwiseArity::kUnary},
      {"binary_select", FLAGDNN_POINTWISE_BINARY_SELECT,
       PointwiseArity::kTernary},
      {"ceil", FLAGDNN_POINTWISE_CEIL, PointwiseArity::kUnary},
      {"cmp_eq", FLAGDNN_POINTWISE_CMP_EQ, PointwiseArity::kBinary},
      {"cmp_ge", FLAGDNN_POINTWISE_CMP_GE, PointwiseArity::kBinary},
      {"cmp_gt", FLAGDNN_POINTWISE_CMP_GT, PointwiseArity::kBinary},
      {"cmp_le", FLAGDNN_POINTWISE_CMP_LE, PointwiseArity::kBinary},
      {"cmp_lt", FLAGDNN_POINTWISE_CMP_LT, PointwiseArity::kBinary},
      {"cmp_neq", FLAGDNN_POINTWISE_CMP_NEQ, PointwiseArity::kBinary},
      {"cos", FLAGDNN_POINTWISE_COS, PointwiseArity::kUnary},
      {"div", FLAGDNN_POINTWISE_DIV, PointwiseArity::kBinary},
      {"elu", FLAGDNN_POINTWISE_ELU_FWD, PointwiseArity::kUnary},
      {"erf", FLAGDNN_POINTWISE_ERF, PointwiseArity::kUnary},
      {"exp", FLAGDNN_POINTWISE_EXP, PointwiseArity::kUnary},
      {"floor", FLAGDNN_POINTWISE_FLOOR, PointwiseArity::kUnary},
      {"gelu", FLAGDNN_POINTWISE_GELU_FWD, PointwiseArity::kUnary},
      {"gelu_approx_tanh", FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD,
       PointwiseArity::kUnary},
      {"identity", FLAGDNN_POINTWISE_IDENTITY, PointwiseArity::kUnary},
      {"leaky_relu", FLAGDNN_POINTWISE_RELU_FWD, PointwiseArity::kUnary},
      {"log", FLAGDNN_POINTWISE_LOG, PointwiseArity::kUnary},
      {"logical_and", FLAGDNN_POINTWISE_LOGICAL_AND,
       PointwiseArity::kBinary},
      {"logical_not", FLAGDNN_POINTWISE_LOGICAL_NOT,
       PointwiseArity::kUnary},
      {"logical_or", FLAGDNN_POINTWISE_LOGICAL_OR,
       PointwiseArity::kBinary},
      {"max", FLAGDNN_POINTWISE_MAX, PointwiseArity::kBinary},
      {"min", FLAGDNN_POINTWISE_MIN, PointwiseArity::kBinary},
      {"mod", FLAGDNN_POINTWISE_MOD, PointwiseArity::kBinary},
      {"mul", FLAGDNN_POINTWISE_MUL, PointwiseArity::kBinary},
      {"neg", FLAGDNN_POINTWISE_NEG, PointwiseArity::kUnary},
      {"pow", FLAGDNN_POINTWISE_POW, PointwiseArity::kBinary},
      {"reciprocal", FLAGDNN_POINTWISE_RECIPROCAL, PointwiseArity::kUnary},
      {"relu", FLAGDNN_POINTWISE_RELU_FWD, PointwiseArity::kUnary},
      {"rsqrt", FLAGDNN_POINTWISE_RSQRT, PointwiseArity::kUnary},
      {"scale", FLAGDNN_POINTWISE_MUL, PointwiseArity::kBinary},
      {"sigmoid", FLAGDNN_POINTWISE_SIGMOID_FWD, PointwiseArity::kUnary},
      {"sigmoid_backward", FLAGDNN_POINTWISE_SIGMOID_BWD,
       PointwiseArity::kBinary},
      {"sin", FLAGDNN_POINTWISE_SIN, PointwiseArity::kUnary},
      {"softplus", FLAGDNN_POINTWISE_SOFTPLUS_FWD, PointwiseArity::kUnary},
      {"sqrt", FLAGDNN_POINTWISE_SQRT, PointwiseArity::kUnary},
      {"sub", FLAGDNN_POINTWISE_SUB, PointwiseArity::kBinary},
      {"swish", FLAGDNN_POINTWISE_SWISH_FWD, PointwiseArity::kUnary},
      {"tan", FLAGDNN_POINTWISE_TAN, PointwiseArity::kUnary},
      {"tanh", FLAGDNN_POINTWISE_TANH_FWD, PointwiseArity::kUnary},
  };
  for (const PointwiseSpecification &specification : pointwise) {
    const PointwiseCaseDefinition definition{
        .operation_name = specification.operation,
        .mode = specification.mode,
        .input_domain = PointwiseInputDomain::kReal,
    };
    switch (specification.arity) {
      case PointwiseArity::kUnary:
        add_cases(output, specification.operation,
                  make_unary_pointwise_cases(definition));
        break;
      case PointwiseArity::kBinary:
        add_cases(output, specification.operation,
                  make_binary_pointwise_cases(definition));
        break;
      case PointwiseArity::kTernary:
        add_cases(output, specification.operation,
                  make_binary_select_cases(definition));
        break;
    }
  }

  add_cases(output, "add", make_add_cases());
  add_cases(output, "add_square", make_add_square_cases());
  add_cases(output, "batchnorm", make_batchnorm_cases());
  add_cases(output, "batchnorm_inference", make_batchnorm_inference_cases());
  add_cases(output, "conv_bias_relu", make_conv_bias_relu_cases());
  add_cases(output, "conv_dgrad",
            make_convolution_cases(ConvolutionDirection::kDgrad));
  add_cases(output, "conv_fprop",
            make_convolution_cases(ConvolutionDirection::kFprop));
  add_cases(output, "conv_wgrad",
            make_convolution_cases(ConvolutionDirection::kWgrad));
  add_cases(output, "layernorm", make_layernorm_cases());
  add_cases(output, "matmul", make_matmul_cases());
  add_cases(output, "reduction", make_reduction_cases());
  add_cases(output, "reshape", make_layout_cases(LayoutOperation::kReshape));
  add_cases(output, "rmsnorm", make_rmsnorm_cases());
  add_cases(output, "sdpa", make_sdpa_cases());
  add_cases(output, "sdpa_backward", make_sdpa_backward_cases());
  add_cases(output, "sdpa_fp8", make_sdpa_fp8_cases());
  add_cases(output, "sdpa_fp8_backward", make_sdpa_fp8_backward_cases());
  add_cases(output, "slice", make_layout_cases(LayoutOperation::kSlice));
  add_cases(output, "transpose",
            make_layout_cases(LayoutOperation::kTranspose));
  return output;
}

std::set<std::string> parse_operator_block(const std::string &source,
                                           std::string_view variable) {
  const std::string marker = "set(" + std::string(variable);
  const std::size_t start = source.find(marker);
  if (start == std::string::npos) {
    throw std::runtime_error("operator manifest is missing " +
                             std::string(variable));
  }
  const std::size_t end = source.find(')', start + marker.size());
  if (end == std::string::npos) {
    throw std::runtime_error("operator manifest block is unterminated");
  }
  std::istringstream input(
      source.substr(start + marker.size(), end - start - marker.size()));
  std::set<std::string> result;
  std::string token;
  while (input >> token) {
    if (token.starts_with('#')) {
      std::getline(input, token);
    } else if (!token.starts_with("${")) {
      result.insert(std::move(token));
    }
  }
  return result;
}

void require_operator_manifest(const CaseMap &cases, const std::string &path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open neutral operator manifest");
  }
  const std::string source{std::istreambuf_iterator<char>(input),
                           std::istreambuf_iterator<char>()};
  std::set<std::string> expected =
      parse_operator_block(source, "FLAGDNN_BENCHMARK_OPERATORS");
  const auto functional =
      parse_operator_block(source, "FLAGDNN_FUNCTIONAL_OPERATORS");
  expected.insert(functional.begin(), functional.end());
  std::set<std::string> actual;
  for (const auto &[operation, operation_cases] : cases) {
    (void)operation_cases;
    actual.insert(operation);
  }
  if (actual != expected) {
    throw std::runtime_error(
        "generated capability operators do not match Operators.cmake");
  }
}

template <typename Function>
void require_rejected(std::string_view name, Function &&function) {
  try {
    function();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error("negative fixture passed: " + std::string(name));
}

constexpr std::string_view kConstraints =
    R"({"dtypes":["fp32"],"compute_type":"fp32","rank":{"min":1,"max":8},"shape":["same_shape","positive_extents"],"layouts":["contiguous"],"stride_policy":"dense_contiguous","broadcast":"none","attributes":{"alpha":["1"],"autotune":["false"]}})";

std::string executable_record(std::string_view status = "probe_required",
                              std::string_view path = "stable_primitive",
                              std::string_view reason =
                                  "real_device_qualification_pending",
                              std::string_view constraints = kConstraints) {
  return "{\"status\":\"" + std::string(status) +
         "\",\"path\":\"" + std::string(path) +
         "\",\"reference_plan\":[\"acdnnOpTensor(ADD)\"],"
         "\"constraints\":" +
         std::string(constraints) + ",\"reason_code\":\"" +
         std::string(reason) +
         "\",\"detail\":\"pending real-device qualification\"}";
}

std::string unsupported_record(
    std::string_view reason = "production_kernel_unavailable") {
  return "{\"status\":\"unsupported\",\"path\":null,"
         "\"reference_plan\":[],\"constraints\":null,"
         "\"reason_code\":\"" +
         std::string(reason) +
         "\",\"detail\":\"production implementation is unavailable\"}";
}

std::string fixture(std::string record,
                    std::string_view sdk_min = "2.0.0-715aa1",
                    std::string_view sdk_max = "2.0.0-715aa1",
                    std::string_view header_min = "1400",
                    std::string_view header_max = "1400") {
  return "{\"schema_version\":1,\"platform\":\"thead\","
         "\"reference_provider\":\"acdnn\",\"baseline\":{"
         "\"ppu_sdk_min\":\"" +
         std::string(sdk_min) + "\",\"ppu_sdk_max\":\"" +
         std::string(sdk_max) + "\",\"acdnn_header_min\":" +
         std::string(header_min) + ",\"acdnn_header_max\":" +
         std::string(header_max) +
         ",\"acdnn_runtime_min\":1400,\"acdnn_runtime_max\":1400},"
         "\"operators\":{\"add\":{\"case_names\":[\"candidate\","
         "\"skipped\"],\"default\":" +
         unsupported_record() +
         ",\"overrides\":{\"candidate\":" + record + "}}}}";
}

void run_capability_negative_fixtures() {
  require_rejected("duplicate object key", [] {
    (void)CapabilityCatalog::parse(
        "{\"schema_version\":1,\"schema_version\":1}");
  });
  require_rejected("unknown root key", [] {
    std::string value = fixture(executable_record());
    value.insert(value.size() - 1, ",\"unknown\":true");
    (void)CapabilityCatalog::parse(value);
  });
  require_rejected("inverted SDK range", [] {
    (void)CapabilityCatalog::parse(
        fixture(executable_record(), "2.1.0", "2.0.0"));
  });
  require_rejected("inverted acDNN range", [] {
    (void)CapabilityCatalog::parse(
        fixture(executable_record(), "2.0.0", "2.0.0", "1401", "1400"));
  });
  require_rejected("unknown status", [] {
    (void)CapabilityCatalog::parse(fixture(executable_record("maybe")));
  });
  require_rejected("unknown reference path", [] {
    (void)CapabilityCatalog::parse(
        fixture(executable_record("probe_required", "magic")));
  });
  require_rejected("unsupported without reason", [] {
    (void)CapabilityCatalog::parse(fixture(unsupported_record("")));
  });
  require_rejected("unknown reason", [] {
    (void)CapabilityCatalog::parse(fixture(unsupported_record("unknown")));
  });
  require_rejected("supported with skip reason", [] {
    (void)CapabilityCatalog::parse(
        fixture(executable_record("supported", "stable_primitive",
                                  "real_device_qualification_pending")));
  });
  require_rejected("probe without reason", [] {
    (void)CapabilityCatalog::parse(
        fixture(executable_record("probe_required", "stable_primitive", "")));
  });
  require_rejected("executable without constraints", [] {
    (void)CapabilityCatalog::parse(
        fixture(executable_record("probe_required", "stable_primitive",
                                  "real_device_qualification_pending",
                                  "null")));
  });
  require_rejected("duplicate case name", [] {
    std::string value = fixture(executable_record());
    const std::string needle = "[\"candidate\",\"skipped\"]";
    value.replace(value.find(needle), needle.size(),
                  "[\"candidate\",\"candidate\"]");
    (void)CapabilityCatalog::parse(value);
  });
  require_rejected("override without owned case", [] {
    std::string value = fixture(executable_record());
    const std::string needle = "\"candidate\":";
    value.replace(value.rfind(needle), needle.size(), "\"unowned\":");
    (void)CapabilityCatalog::parse(value);
  });
  require_rejected("empty dtype constraints", [] {
    std::string constraints(kConstraints);
    constraints.replace(constraints.find("[\"fp32\"]"), 8, "[]");
    (void)CapabilityCatalog::parse(fixture(executable_record(
        "probe_required", "stable_primitive",
        "real_device_qualification_pending", constraints)));
  });
  require_rejected("empty attribute values", [] {
    std::string constraints(kConstraints);
    constraints.replace(constraints.find("[\"1\"]"), 5, "[]");
    (void)CapabilityCatalog::parse(fixture(executable_record(
        "probe_required", "stable_primitive",
        "real_device_qualification_pending", constraints)));
  });
}

void run_numeric_type_contract() {
  namespace tv = flagdnn::validation::thead;
  if (tv::element_size(FLAGDNN_DATA_FLOAT32) != 4 ||
      tv::element_size(FLAGDNN_DATA_FLOAT16) != 2 ||
      tv::element_size(FLAGDNN_DATA_BFLOAT16) != 2 ||
      tv::element_size(FLAGDNN_DATA_BOOLEAN) != 1) {
    throw std::runtime_error("THead validation element sizes are invalid");
  }
  if (tv::element_size(FLAGDNN_DATA_FP8_E4M3) != 1 ||
      tv::element_size(FLAGDNN_DATA_FP8_E5M2) != 1) {
    throw std::runtime_error("FP8 raw storage must occupy one byte");
  }
  require_rejected("FP8 host numeric conversion", [] {
    const std::array<float, 1> value{1.0F};
    (void)tv::encode_floating(FLAGDNN_DATA_FP8_E4M3, value);
  });

  const std::array<float, 10> values = {
      0.0F,
      -0.0F,
      1.0F,
      -2.0F,
      65504.0F,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      std::bit_cast<float>(std::uint32_t{0x3f801000U}),
      std::bit_cast<float>(std::uint32_t{0x3f803000U}),
  };
  const std::vector<std::byte> fp16 =
      tv::encode_floating(FLAGDNN_DATA_FLOAT16, values);
  if (fp16.size() != values.size() * 2) {
    throw std::runtime_error("FP16 validation encoding has wrong size");
  }
  const auto half_bits = [&fp16](std::size_t index) {
    return static_cast<std::uint16_t>(
        std::to_integer<std::uint8_t>(fp16.at(index * 2))) |
           static_cast<std::uint16_t>(
               std::to_integer<std::uint8_t>(fp16.at(index * 2 + 1)))
               << 8U;
  };
  if (half_bits(0) != 0x0000U || half_bits(1) != 0x8000U ||
      half_bits(2) != 0x3c00U || half_bits(3) != 0xc000U ||
      half_bits(4) != 0x7bffU || half_bits(5) != 0x7c00U ||
      half_bits(6) != 0xfc00U || (half_bits(7) & 0x7c00U) != 0x7c00U ||
      (half_bits(7) & 0x03ffU) == 0 || half_bits(8) != 0x3c00U ||
      half_bits(9) != 0x3c02U) {
    throw std::runtime_error("FP16 validation encoding is not IEEE RNE");
  }
  const std::vector<float> fp16_round_trip =
      tv::decode_floating(FLAGDNN_DATA_FLOAT16, fp16);
  if (fp16_round_trip.size() != values.size() ||
      fp16_round_trip[0] != 0.0F || !std::signbit(fp16_round_trip[1]) ||
      fp16_round_trip[2] != 1.0F || fp16_round_trip[3] != -2.0F ||
      fp16_round_trip[4] != 65504.0F ||
      !std::isinf(fp16_round_trip[5]) ||
      !std::isinf(fp16_round_trip[6]) || !std::isnan(fp16_round_trip[7])) {
    throw std::runtime_error("FP16 validation round trip is invalid");
  }

  const std::array<float, 6> bf16_values = {
      0.0F,
      -0.0F,
      1.0F,
      std::numeric_limits<float>::infinity(),
      std::bit_cast<float>(std::uint32_t{0x3f808000U}),
      std::bit_cast<float>(std::uint32_t{0x3f818000U}),
  };
  const std::vector<std::byte> bf16 =
      tv::encode_floating(FLAGDNN_DATA_BFLOAT16, bf16_values);
  const auto bfloat_bits = [&bf16](std::size_t index) {
    return static_cast<std::uint16_t>(
        std::to_integer<std::uint8_t>(bf16.at(index * 2))) |
           static_cast<std::uint16_t>(
               std::to_integer<std::uint8_t>(bf16.at(index * 2 + 1)))
               << 8U;
  };
  if (bfloat_bits(0) != 0x0000U || bfloat_bits(1) != 0x8000U ||
      bfloat_bits(2) != 0x3f80U || bfloat_bits(3) != 0x7f80U ||
      bfloat_bits(4) != 0x3f80U || bfloat_bits(5) != 0x3f82U) {
    throw std::runtime_error("BF16 validation encoding is not IEEE RNE");
  }
  const std::vector<float> bf16_round_trip =
      tv::decode_floating(FLAGDNN_DATA_BFLOAT16, bf16);
  if (bf16_round_trip.size() != bf16_values.size() ||
      bf16_round_trip[0] != 0.0F || !std::signbit(bf16_round_trip[1]) ||
      bf16_round_trip[2] != 1.0F || !std::isinf(bf16_round_trip[3])) {
    throw std::runtime_error("BF16 validation round trip is invalid");
  }

  const std::array<float, 3> fp32_values = {1.25F, -3.5F, 0.0F};
  const std::vector<std::byte> fp32 =
      tv::encode_floating(FLAGDNN_DATA_FLOAT32, fp32_values);
  if (tv::decode_floating(FLAGDNN_DATA_FLOAT32, fp32) !=
      std::vector<float>(fp32_values.begin(), fp32_values.end())) {
    throw std::runtime_error("FP32 validation round trip is invalid");
  }
  require_rejected("mis-sized FP16 validation bytes", [] {
    const std::array<std::byte, 1> bytes{};
    (void)tv::decode_floating(FLAGDNN_DATA_FLOAT16, bytes);
  });
}

void run_api_contract() {
  run_numeric_type_contract();
  run_capability_negative_fixtures();
  const CapabilityCatalog catalog =
      CapabilityCatalog::parse(fixture(executable_record()));
  catalog.require_exact_cases({{"add", {"candidate", "skipped"}}});
  catalog.validate_versions("2.0.0-715aa1", 1400, 1400);
  require_rejected("SDK below catalog range", [&catalog] {
    catalog.validate_versions("1.9.9", 1400, 1400);
  });
  require_rejected("acDNN header above catalog range", [&catalog] {
    catalog.validate_versions("2.0.0-715aa1", 1401, 1400);
  });
  require_rejected("case owned by wrong operator", [&catalog] {
    catalog.require_exact_cases({{"other", {"candidate", "skipped"}}});
  });
  require_rejected("missing case", [&catalog] {
    catalog.require_exact_cases({{"add", {"candidate"}}});
  });

  const auto &candidate = catalog.lookup("add", "candidate");
  if (candidate.status != CapabilityStatus::kProbeRequired ||
      candidate.path != ReferencePath::kStablePrimitive ||
      !candidate.constraints.has_value() ||
      candidate.reference_plan !=
          std::vector<std::string>{"acdnnOpTensor(ADD)"}) {
    throw std::runtime_error("candidate capability record was not preserved");
  }
  const auto selected =
      flagdnn::validation::thead::select_reference(candidate);
  if (!std::holds_alternative<ReferencePlan>(selected) ||
      std::get<ReferencePlan>(selected).qualification !=
          CapabilityStatus::kProbeRequired) {
    throw std::runtime_error("candidate did not produce a typed plan");
  }
  const auto skipped = flagdnn::validation::thead::select_reference(
      catalog.lookup("add", "skipped"));
  if (!std::holds_alternative<UnsupportedCapability>(skipped) ||
      std::get<UnsupportedCapability>(skipped).reason_code !=
          "production_kernel_unavailable") {
    throw std::runtime_error("unsupported case lost its stable reason");
  }

  flagdnn::validation::thead::check_acdnn(ACDNN_STATUS_SUCCESS,
                                           "success fixture");
  require_rejected("unexpected acDNN status", [] {
    flagdnn::validation::thead::check_acdnn(ACDNN_STATUS_BAD_PARAM,
                                             "bad status fixture");
  });
  try {
    flagdnn::validation::thead::check_acdnn(ACDNN_STATUS_BAD_PARAM,
                                             "typed status fixture");
  } catch (const AcdnnStatusError &error) {
    if (error.status() != ACDNN_STATUS_BAD_PARAM) {
      throw std::runtime_error("acDNN status error lost its status value");
    }
  }
  require_rejected("NOT_SUPPORTED on executable capability", [&candidate] {
    flagdnn::validation::thead::require_reference_status(
        candidate, ACDNN_STATUS_NOT_SUPPORTED, "supported execution fixture");
  });

  flagdnn::validation::thead::validate_transfer_bounds(32, 8, 16);
  require_rejected("transfer exceeds allocation", [] {
    flagdnn::validation::thead::validate_transfer_bounds(16, 12, 8);
  });
  require_rejected("typed transfer byte offset overflow", [] {
    DeviceBuffer empty;
    const std::span<const std::uint64_t> no_values;
    flagdnn::validation::thead::copy_to_device_async(
        empty, no_values,
        std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t) + 1,
        nullptr);
  });
}

void run_catalog_contract(const std::string &catalog_path,
                          const std::string &operator_path,
                          std::string_view sdk_version) {
  const CaseMap cases = neutral_cases();
  require_operator_manifest(cases, operator_path);
  const CapabilityCatalog catalog = CapabilityCatalog::load(catalog_path);
  catalog.require_exact_cases(cases);
  catalog.validate_versions(sdk_version, ACDNN_VERSION,
                            static_cast<int>(acdnnGetVersion()));

  std::size_t total = 0;
  std::size_t supported = 0;
  std::size_t probes = 0;
  std::size_t skipped = 0;
  for (const auto &[operation, operation_cases] : catalog.records()) {
    (void)operation;
    total += operation_cases.size();
    for (const auto &[case_name, record] : operation_cases) {
      (void)case_name;
      supported += record.status == CapabilityStatus::kSupported;
      probes += record.status == CapabilityStatus::kProbeRequired;
      skipped += record.status == CapabilityStatus::kUnsupported;
    }
  }
  const auto executable = [](const CapabilityRecord &record) {
    return record.status == CapabilityStatus::kSupported ||
           record.status == CapabilityStatus::kProbeRequired;
  };
  const CapabilityRecord &sub = catalog.lookup("sub", "sub_fp32_1x1x16");
  if (!executable(sub) ||
      sub.path != ReferencePath::kStablePrimitive ||
      sub.reference_plan != std::vector<std::string>{
                                "acdnnOpTensor(ADD,alpha_right=-alpha)"}) {
    throw std::runtime_error("Sub capability slice contract mismatch");
  }
  const CapabilityRecord &minimum =
      catalog.lookup("min", "min_fp32_1x1x16");
  if (!executable(minimum) ||
      minimum.path != ReferencePath::kStablePrimitive ||
      minimum.reference_plan !=
          std::vector<std::string>{"acdnnOpTensor(MIN)"}) {
    throw std::runtime_error("Min capability slice contract mismatch");
  }
  const CapabilityRecord &maximum =
      catalog.lookup("max", "max_fp32_1x1x16");
  if (!executable(maximum) ||
      maximum.path != ReferencePath::kStablePrimitive ||
      maximum.reference_plan !=
          std::vector<std::string>{"acdnnOpTensor(MAX)"}) {
    throw std::runtime_error("Max capability slice contract mismatch");
  }
  const std::array<std::tuple<std::string_view, std::string_view,
                              std::string_view, bool>,
                   3>
      converted_bfloat16_slices = {{
          {"min", "min_bfloat16_1x1x16", "acdnnOpTensor(MIN)", true},
          {"max", "max_bfloat16_1x1x16", "acdnnOpTensor(MAX)", true},
          {"relu", "relu_bfloat16_1x1x16",
           "acdnnActivationForward(RELU,NOT_PROPAGATE_NAN)", false},
      }};
  for (const auto &[operation, case_name, primitive, binary] :
       converted_bfloat16_slices) {
    const CapabilityRecord &record = catalog.lookup(operation, case_name);
    std::vector<std::string> expected = {
        "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-input0-fp32)"};
    if (binary) {
      expected.emplace_back(
          "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-input1-fp32)");
    }
    expected.emplace_back(primitive);
    expected.emplace_back(
        "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-output-bfloat16)");
    if (!executable(record) ||
        record.path != ReferencePath::kBackendDescriptor ||
        record.reference_plan != expected || !record.constraints.has_value() ||
        record.constraints->dtypes != std::vector<std::string>{"bf16"}) {
      throw std::runtime_error(std::string(operation) +
                               " converted BF16 DAG contract mismatch");
    }
  }
  const CapabilityRecord &scale =
      catalog.lookup("scale", "scale_fp32_1x1x16");
  if (!executable(scale) ||
      scale.path != ReferencePath::kStablePrimitive ||
      scale.reference_plan !=
          std::vector<std::string>{"acdnnOpTensor(MUL)"}) {
    throw std::runtime_error("Scale capability slice contract mismatch");
  }
  const CapabilityRecord &relu =
      catalog.lookup("relu", "relu_fp32_1x1x16");
  if (!executable(relu) ||
      relu.path != ReferencePath::kStablePrimitive ||
      relu.reference_plan != std::vector<std::string>{
                                 "acdnnActivationForward("
                                 "RELU,NOT_PROPAGATE_NAN)"}) {
    throw std::runtime_error("Relu capability slice contract mismatch");
  }
  const CapabilityRecord &sigmoid =
      catalog.lookup("sigmoid", "sigmoid_fp32_1x1x16");
  if (!executable(sigmoid) ||
      sigmoid.path != ReferencePath::kStablePrimitive ||
      sigmoid.reference_plan != std::vector<std::string>{
                                    "acdnnActivationForward("
                                    "SIGMOID,NOT_PROPAGATE_NAN)"}) {
    throw std::runtime_error("Sigmoid capability slice contract mismatch");
  }
  const CapabilityRecord &tanh =
      catalog.lookup("tanh", "tanh_fp32_1x1x16");
  if (!executable(tanh) ||
      tanh.path != ReferencePath::kStablePrimitive ||
      tanh.reference_plan != std::vector<std::string>{
                                 "acdnnActivationForward("
                                 "TANH,NOT_PROPAGATE_NAN)"}) {
    throw std::runtime_error("Tanh capability slice contract mismatch");
  }
  const CapabilityRecord &elu =
      catalog.lookup("elu", "elu_fp32_1x1x16");
  if (!executable(elu) ||
      elu.path != ReferencePath::kStablePrimitive ||
      elu.reference_plan != std::vector<std::string>{
                                "acdnnActivationForward("
                                "ELU,NOT_PROPAGATE_NAN,alpha=1)"}) {
    throw std::runtime_error("Elu capability slice contract mismatch");
  }
  const CapabilityRecord &identity =
      catalog.lookup("identity", "identity_fp32_2x3x4");
  if (!executable(identity) ||
      identity.path != ReferencePath::kStablePrimitive ||
      identity.reference_plan != std::vector<std::string>{
                                     "acdnnTransformTensor("
                                     "alpha=1,beta=0)"}) {
    throw std::runtime_error("Identity capability slice contract mismatch");
  }
  const CapabilityRecord &gelu =
      catalog.lookup("gelu", "gelu_fp32_1x1x16");
  if (!executable(gelu) ||
      gelu.path != ReferencePath::kStablePrimitive ||
      gelu.reference_plan != std::vector<std::string>{
                                 "acdnnActivationForward("
                                 "GELU,NOT_PROPAGATE_NAN)"}) {
    throw std::runtime_error("Gelu capability slice contract mismatch");
  }
  const CapabilityRecord &leaky_relu =
      catalog.lookup("leaky_relu", "leaky_relu_fp32_1x1x16");
  if (!executable(leaky_relu) ||
      leaky_relu.path != ReferencePath::kBackendDescriptor ||
      leaky_relu.reference_plan != std::vector<std::string>{
          "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,leaky-input-fp32)",
          "acdnnTransformTensor(alpha=-1,beta=0)",
          "acdnnActivationForward(RELU,NOT_PROPAGATE_NAN,positive)",
          "acdnnActivationForward(RELU,NOT_PROPAGATE_NAN,negative)",
          "acdnnOpTensor(ADD,alpha_right=-0.2)",
          "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,leaky-output-data-type)"}) {
    throw std::runtime_error("LeakyRelu acDNN DAG contract mismatch");
  }
  const CapabilityRecord &sqrt =
      catalog.lookup("sqrt", "sqrt_fp32_1x1x16");
  if (!executable(sqrt) ||
      sqrt.path != ReferencePath::kStablePrimitive ||
      sqrt.reference_plan !=
          std::vector<std::string>{"acdnnOpTensor(SQRT)"}) {
    throw std::runtime_error("Sqrt capability slice contract mismatch");
  }
  const CapabilityRecord &neg =
      catalog.lookup("neg", "neg_fp32_1x1x16");
  if (!executable(neg) ||
      neg.path != ReferencePath::kStablePrimitive ||
      neg.reference_plan !=
          std::vector<std::string>{
              "acdnnTransformTensor(alpha=-1,beta=0)"}) {
    throw std::runtime_error("Neg capability slice contract mismatch");
  }
  const CapabilityRecord &abs =
      catalog.lookup("abs", "abs_fp32_1x1x16");
  if (!executable(abs) ||
      abs.path != ReferencePath::kBackendDescriptor ||
      abs.reference_plan !=
          std::vector<std::string>{
              "acdnnBackendExecute(POINTWISE_ABS)"}) {
    throw std::runtime_error("Abs capability slice contract mismatch");
  }
  const CapabilityRecord &ceil =
      catalog.lookup("ceil", "ceil_fp32_1x1x16");
  if (!executable(ceil) ||
      ceil.path != ReferencePath::kBackendDescriptor ||
      ceil.reference_plan !=
          std::vector<std::string>{
              "acdnnBackendExecute(POINTWISE_CEIL)"}) {
    throw std::runtime_error("Ceil capability slice contract mismatch");
  }
  const CapabilityRecord &floor =
      catalog.lookup("floor", "floor_fp32_1x1x16");
  if (!executable(floor) ||
      floor.path != ReferencePath::kBackendDescriptor ||
      floor.reference_plan !=
          std::vector<std::string>{
              "acdnnBackendExecute(POINTWISE_FLOOR)"}) {
    throw std::runtime_error("Floor capability slice contract mismatch");
  }
  const CapabilityRecord &exp =
      catalog.lookup("exp", "exp_fp32_1x1x16");
  if (!executable(exp) ||
      exp.path != ReferencePath::kBackendDescriptor ||
      exp.reference_plan !=
          std::vector<std::string>{
              "acdnnBackendExecute(POINTWISE_EXP)"}) {
    throw std::runtime_error("Exp capability slice contract mismatch");
  }
  const std::array<std::pair<std::string_view, std::string_view>, 9>
      backend_unary_slices = {{
          {"log", "acdnnBackendExecute(POINTWISE_LOG)"},
          {"cos", "acdnnBackendExecute(POINTWISE_COS)"},
          {"rsqrt", "acdnnBackendExecute(POINTWISE_RSQRT)"},
          {"sin", "acdnnBackendExecute(POINTWISE_SIN)"},
          {"tan", "acdnnBackendExecute(POINTWISE_TAN)"},
          {"softplus",
           "acdnnBackendExecute(POINTWISE_SOFTPLUS_FWD,beta=1)"},
          {"swish",
           "acdnnBackendExecute(POINTWISE_SWISH_FWD,beta=1.25)"},
          {"gelu_approx_tanh",
           "acdnnBackendExecute(POINTWISE_GELU_APPROX_TANH_FWD)"},
          {"reciprocal",
           "acdnnBackendExecute(POINTWISE_DIV,numerator=1)"},
      }};
  for (const auto &[operation, primitive] : backend_unary_slices) {
    const CapabilityRecord &record = catalog.lookup(
        operation, std::string(operation) + "_fp32_1x1x16");
    if (!executable(record) ||
        record.path != ReferencePath::kBackendDescriptor ||
        record.reference_plan !=
            std::vector<std::string>{std::string(primitive)}) {
      throw std::runtime_error(std::string(operation) +
                               " capability slice contract mismatch");
    }
  }
  const std::array<std::pair<std::string_view, std::string_view>, 3>
      backend_binary_slices = {{
          {"div", "acdnnBackendExecute(POINTWISE_DIV)"},
          {"pow", "acdnnBackendExecute(POINTWISE_POW)"},
          {"sigmoid_backward",
           "acdnnBackendExecute(POINTWISE_SIGMOID_BWD)"},
      }};
  for (const auto &[operation, primitive] : backend_binary_slices) {
    const CapabilityRecord &record = catalog.lookup(
        operation, std::string(operation) + "_fp32_1x1x16");
    if (!executable(record) ||
        record.path != ReferencePath::kBackendDescriptor ||
        record.reference_plan !=
            std::vector<std::string>{std::string(primitive)}) {
      throw std::runtime_error(std::string(operation) +
                               " capability slice contract mismatch");
    }
  }
  const std::array<std::string_view, 4> segmented_tail_operations = {
      "div", "pow", "reciprocal", "sigmoid_backward"};
  for (std::string_view operation : segmented_tail_operations) {
    const CapabilityRecord &record = catalog.lookup(
        operation, std::string(operation) + "_fp32_2x3x5x7");
    if (!executable(record) ||
        record.path != ReferencePath::kBackendDescriptor ||
        !record.constraints.has_value() ||
        record.constraints->dtypes != std::vector<std::string>{"fp32"}) {
      throw std::runtime_error(std::string(operation) +
                               " segmented tail contract mismatch");
    }
  }
  const CapabilityRecord &add_square =
      catalog.lookup("add_square", "add_square_fp32_1x1x16");
  if (!executable(add_square) ||
      add_square.path != ReferencePath::kStablePrimitive ||
      add_square.reference_plan !=
          std::vector<std::string>{"acdnnOpTensor(MUL)",
                                   "acdnnOpTensor(ADD)"}) {
    throw std::runtime_error("AddSquare acDNN DAG contract mismatch");
  }

  const std::array<std::pair<std::string_view, std::string_view>, 6>
      comparison_slices = {{
          {"cmp_eq", "acdnnBackendExecute(POINTWISE_CMP_EQ)"},
          {"cmp_neq", "acdnnBackendExecute(POINTWISE_CMP_NEQ)"},
          {"cmp_gt", "acdnnBackendExecute(POINTWISE_CMP_GT)"},
          {"cmp_ge", "acdnnBackendExecute(POINTWISE_CMP_GE)"},
          {"cmp_lt", "acdnnBackendExecute(POINTWISE_CMP_LT)"},
          {"cmp_le", "acdnnBackendExecute(POINTWISE_CMP_LE)"},
      }};
  for (const auto &[operation, primitive] : comparison_slices) {
    const CapabilityRecord &record = catalog.lookup(
        operation, std::string(operation) + "_fp32_1x1x16");
    if (!executable(record) ||
        record.path != ReferencePath::kBackendDescriptor ||
        record.reference_plan !=
            std::vector<std::string>{std::string(primitive)} ||
        !record.constraints.has_value() ||
        record.constraints->compute_type != "bool") {
      throw std::runtime_error(std::string(operation) +
                               " comparison capability slice mismatch");
    }
  }
  for (const auto &[operation, mode] :
       std::array<std::pair<std::string_view, flagdnnPointwiseMode_t>, 6>{{
           {"logical_not", FLAGDNN_POINTWISE_LOGICAL_NOT},
           {"logical_and", FLAGDNN_POINTWISE_LOGICAL_AND},
           {"logical_or", FLAGDNN_POINTWISE_LOGICAL_OR},
           {"erf", FLAGDNN_POINTWISE_ERF},
           {"mod", FLAGDNN_POINTWISE_MOD},
           {"binary_select", FLAGDNN_POINTWISE_BINARY_SELECT}}}) {
    const auto &records = catalog.records().at(std::string(operation));
    if (records.empty() || std::ranges::any_of(records, [&](const auto &entry) {
          return !executable(entry.second) ||
                 entry.second.path != ReferencePath::kBackendDescriptor ||
                 entry.second.reference_plan != flagdnn::validation::thead::acdnn_pointwise_dag_plan(mode);
        })) {
      throw std::runtime_error(std::string(operation) + " acDNN composite reference contract mismatch");
    }
  }

  const std::array<std::tuple<std::string_view, std::string_view,
                              std::string_view>,
                   3>
      layout_slices = {{
          {"reshape", "reshape_fp32_2x3x4_to_6x4",
           "acdnnTransformTensor(flattened,alpha=1,beta=0)"},
          {"transpose", "transpose_fp32_2x3x4",
           "acdnnTransformTensor(permuted-stride,alpha=1,beta=0)"},
          {"slice", "slice_fp32_case0_2x4x5",
           "acdnnTransformTensor(slice-segments,alpha=1,beta=0)"},
      }};
  for (const auto &[operation, case_name, primitive] : layout_slices) {
    const CapabilityRecord &record = catalog.lookup(operation, case_name);
    if (!executable(record) ||
        record.path != ReferencePath::kStablePrimitive ||
        record.reference_plan !=
            std::vector<std::string>{std::string(primitive)} ||
        !record.constraints.has_value() ||
        record.constraints->dtypes != std::vector<std::string>{"fp32"}) {
      throw std::runtime_error(std::string(operation) +
                               " Layout capability slice mismatch");
    }
  }
  const CapabilityRecord &converted_transpose =
      catalog.lookup("transpose", "transpose_bfloat16_2x3x4");
  if (!executable(converted_transpose) ||
      converted_transpose.path != ReferencePath::kBackendDescriptor ||
      converted_transpose.reference_plan != std::vector<std::string>{
          "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-input0-fp32)",
          "acdnnTransformTensor(permuted-stride,alpha=1,beta=0)",
          "acdnnBackendExecute(POINTWISE_IDENTITY_FWD,convert-output-bfloat16)"} ||
      !converted_transpose.constraints.has_value() ||
      converted_transpose.constraints->dtypes !=
          std::vector<std::string>{"bf16"}) {
    throw std::runtime_error("converted BF16 Transpose DAG contract mismatch");
  }
  const std::array<std::int64_t, 4> canonical_transpose_dimensions = {
      4, 64, 128, 32};
  const std::array<std::int64_t, 4> canonical_transpose_permutation = {
      0, 2, 3, 1};
  const std::array<std::int64_t, 4> inverse_transpose_dimensions = {
      2, 128, 128, 64};
  const std::array<std::int64_t, 4> inverse_transpose_permutation = {
      0, 3, 1, 2};
  const std::array<std::int64_t, 4> general_transpose_dimensions = {
      2, 3, 4, 5};
  const std::array<std::int64_t, 4> general_transpose_permutation = {
      0, 1, 3, 2};
  if (!flagdnn::validation::thead::
          legacy_acdnn_transpose_descriptor_compatible(
              canonical_transpose_dimensions,
              canonical_transpose_permutation) ||
      !flagdnn::validation::thead::
          legacy_acdnn_transpose_descriptor_compatible(
              inverse_transpose_dimensions,
              inverse_transpose_permutation) ||
      flagdnn::validation::thead::
          legacy_acdnn_transpose_descriptor_compatible(
              general_transpose_dimensions,
              general_transpose_permutation)) {
    throw std::runtime_error(
        "legacy acDNN Transpose descriptor compatibility mismatch");
  }
  const std::array<std::pair<std::string_view, std::string_view>, 3>
      reduction_slices = {{
          {"reduction_sum_fp32_axis1_keepdim_2x4x8x8",
           "acdnnReduceTensor(ADD,alpha=1,beta=0)"},
          {"reduction_avg_fp32_axis1_keepdim_2x4x8x8",
           "acdnnReduceTensor(AVG,alpha=1,beta=0)"},
          {"reduction_mul_fp32_axis1_keepdim_2x4x8x8",
           "acdnnReduceTensor(MUL,alpha=1,beta=0)"},
      }};
  for (const auto &[case_name, primitive] : reduction_slices) {
    const CapabilityRecord &record = catalog.lookup("reduction", case_name);
    if (!executable(record) ||
        record.path != ReferencePath::kStablePrimitive ||
        record.reference_plan !=
            std::vector<std::string>{std::string(primitive)}) {
      throw std::runtime_error("Reduction capability slice mismatch");
    }
  }

  const CapabilityRecord &batchnorm = catalog.lookup(
      "batchnorm", "batchnorm_fp32_2x8x8x8_contiguous");
  if (!executable(batchnorm) ||
      batchnorm.path != ReferencePath::kStablePrimitive ||
      batchnorm.reference_plan != std::vector<std::string>{
          "acdnnTransformTensor(previous_running_mean)",
          "acdnnTransformTensor(previous_running_variance)",
          "acdnnBatchNormalizationForwardTraining(SPATIAL)"}) {
    throw std::runtime_error("BatchNorm capability slice mismatch");
  }
  const CapabilityRecord &batchnorm_inference = catalog.lookup(
      "batchnorm_inference",
      "batchnorm_inference_fp32_2x8x16x16_contiguous");
  if (!executable(batchnorm_inference) ||
      batchnorm_inference.path != ReferencePath::kStablePrimitive ||
      batchnorm_inference.reference_plan != std::vector<std::string>{
          "acdnnOpTensor(ADD[-mean])->MUL(inv_variance)->MUL(scale)->ADD(bias)"}) {
    throw std::runtime_error(
        "BatchNorm-inference capability slice mismatch");
  }

  const CapabilityRecord &matmul = catalog.lookup(
      "matmul", "matmul_fp32_4x16x32_by_4x32x24");
  if (!executable(matmul) ||
      matmul.path != ReferencePath::kBackendDescriptor ||
      matmul.reference_plan !=
          std::vector<std::string>{"acdnnBackendExecute(MATMUL)"}) {
    throw std::runtime_error("MatMul capability slice mismatch");
  }
  const CapabilityRecord &segmented_matmul = catalog.lookup(
      "matmul", "matmul_fp32_2x1x17x30_by_3x30x23");
  if (!executable(segmented_matmul) ||
      segmented_matmul.path != ReferencePath::kBackendDescriptor ||
      segmented_matmul.reference_plan != std::vector<std::string>{
          "acdnnBackendExecute(MATMUL,batch-segments)"}) {
    throw std::runtime_error("segmented batch MatMul contract mismatch");
  }

  const std::array<std::tuple<std::string_view, std::string_view,
                              std::string_view>,
                   3>
      convolution_slices = {{
          {"conv_fprop",
           "conv2d_fprop_fp32_nchw_smoke_1x2x5x5_by_2x2x3x3",
           "acdnnConvolutionForward(IMPLICIT_GEMM)"},
          {"conv_dgrad",
           "conv2d_dgrad_fp32_nchw_symmetric_2x8x16x16_by_16x8x3x3",
           "acdnnConvolutionBackwardData(ALGO_0)"},
          {"conv_wgrad",
           "conv2d_wgrad_fp32_nchw_symmetric_2x8x16x16_by_16x8x3x3",
           "acdnnConvolutionBackwardFilter(ALGO_0)"},
      }};
  for (const auto &[operation, case_name, primitive] : convolution_slices) {
    const CapabilityRecord &record = catalog.lookup(operation, case_name);
    if (!executable(record) ||
        record.path != ReferencePath::kStablePrimitive ||
        record.reference_plan !=
            std::vector<std::string>{std::string(primitive)} ||
        !record.constraints.has_value() ||
        record.constraints->dtypes != std::vector<std::string>{"fp32"}) {
      throw std::runtime_error(std::string(operation) +
                               " convolution capability slice mismatch");
    }
  }

  const std::array<std::tuple<std::string_view, std::string_view,
                              std::vector<std::string>>,
                   3>
      asymmetric_convolution_slices = {{
          {"conv_fprop",
           "conv2d_fprop_fp32_nhwc_asymmetric_dilation_1x5x19x21_by_9x5x3x3",
           {"acdnnConvolutionForward(IMPLICIT_GEMM,symmetric-superset)",
            "acdnnTransformTensor(alpha=1,beta=0,asymmetric-output-slice)"}},
          {"conv_dgrad",
           "conv2d_dgrad_fp32_nchw_asymmetric_padding_2x4x12x13_by_7x4x3x3",
           {"acdnnTransformTensor(alpha=1,beta=0,asymmetric-gradient-zero)",
            "acdnnTransformTensor(alpha=1,beta=0,asymmetric-gradient-pad)",
            "acdnnConvolutionBackwardData(ALGO_0,symmetric-superset)"}},
          {"conv_wgrad",
           "conv2d_wgrad_fp32_nchw_asymmetric_padding_2x4x12x13_by_7x4x3x3",
           {"acdnnTransformTensor(alpha=1,beta=0,asymmetric-gradient-zero)",
            "acdnnTransformTensor(alpha=1,beta=0,asymmetric-gradient-pad)",
            "acdnnConvolutionBackwardFilter(ALGO_0,symmetric-superset)"}},
      }};
  for (const auto &[operation, case_name, plan] :
       asymmetric_convolution_slices) {
    const CapabilityRecord &record = catalog.lookup(operation, case_name);
    if (!executable(record) ||
        record.path != ReferencePath::kStablePrimitive ||
        record.reference_plan != plan) {
      throw std::runtime_error(std::string(operation) +
                               " asymmetric convolution DAG mismatch");
    }
  }

  const CapabilityRecord &conv_bias_relu = catalog.lookup(
      "conv_bias_relu",
      "conv_bias_relu_fp32_x2x8x16x16_w16x8x3x3_s1x1_p1x1_d1x1");
  if (!executable(conv_bias_relu) ||
      conv_bias_relu.path != ReferencePath::kStablePrimitive ||
      conv_bias_relu.reference_plan != std::vector<std::string>{
          "acdnnConvolutionBiasActivationForward(RELU)"} ||
      !conv_bias_relu.constraints.has_value() ||
      conv_bias_relu.constraints->dtypes !=
          std::vector<std::string>{"fp32"}) {
    throw std::runtime_error("ConvBiasRelu capability slice mismatch");
  }

  const std::vector<std::string> layernorm_plan = {
      "acdnnReduceTensor(AVG,mean)",
      "acdnnOpTensor(ADD[-mean])",
      "acdnnOpTensor(MUL,square)",
      "acdnnReduceTensor(AVG,variance)",
      "acdnnOpTensor(ADD,epsilon)",
      "acdnnBackendExecute(POINTWISE_RSQRT)",
      "acdnnOpTensor(MUL,normalize)",
      "acdnnOpTensor(MUL,scale)",
      "acdnnOpTensor(ADD,bias)"};
  const std::vector<std::string> rmsnorm_plan = {
      "acdnnOpTensor(MUL,square)",
      "acdnnReduceTensor(AVG,mean_square)",
      "acdnnOpTensor(ADD,epsilon)",
      "acdnnBackendExecute(POINTWISE_RSQRT)",
      "acdnnOpTensor(MUL,normalize)",
      "acdnnOpTensor(MUL,scale)",
      "acdnnOpTensor(ADD,bias)"};
  for (const auto &[operation, case_name, expected_plan] :
       std::array<std::tuple<std::string_view, std::string_view,
                             const std::vector<std::string> *>,
                  2>{{
           {"layernorm", "layernorm_fp32_2x5x17_suffix1",
            &layernorm_plan},
           {"rmsnorm", "rmsnorm_fp32_2x5x17_suffix1", &rmsnorm_plan},
       }}) {
    const CapabilityRecord &record = catalog.lookup(operation, case_name);
    if (!executable(record) ||
        record.path != ReferencePath::kBackendDescriptor ||
        record.reference_plan != *expected_plan ||
        !record.constraints.has_value() ||
        record.constraints->dtypes != std::vector<std::string>{"fp32"}) {
      throw std::runtime_error(std::string(operation) +
                               " normalization capability slice mismatch");
    }
  }

  for (const auto operation : {"sdpa", "sdpa_backward"}) {
    const auto &records = catalog.records().at(operation);
    const auto plan = flagdnn::validation::thead::acdnn_attention_plan(std::string_view(operation) == "sdpa_backward");
    if (records.size() != 4 ||
        std::ranges::any_of(records, [&](const auto &entry) {
          return !executable(entry.second) ||
                 entry.second.path != ReferencePath::kBackendDescriptor ||
                 entry.second.reference_plan != plan;
        })) {
      throw std::runtime_error(std::string(operation) +
                               " acDNN attention DAG contract mismatch");
    }
  }

  for (const auto operation : {"sdpa_fp8", "sdpa_fp8_backward"}) {
    const bool backward = std::string_view(operation) == "sdpa_fp8_backward";
    const auto &records = catalog.records().at(operation);
    const auto plan = flagdnn::validation::thead::acdnn_fp8_attention_plan(backward);
    if (records.size() != (backward ? 2U : 4U) ||
        std::ranges::any_of(records, [&](const auto &entry) {
          return !executable(entry.second) ||
                 entry.second.path != ReferencePath::kBackendDescriptor ||
                 entry.second.reference_plan != plan;
        })) {
      throw std::runtime_error(std::string(operation) +
                               " acDNN FP8 attention DAG contract mismatch");
    }
  }

  if (catalog.records().size() != 61 || total != 1142 ||
      supported != total || probes != 0 || skipped != 0) {
    throw std::runtime_error("capability catalog accounting mismatch");
  }
  std::cout << "PASS THead acDNN capability catalog: operators="
            << catalog.records().size() << " cases=" << total
            << " supported=" << supported << " probes=" << probes
            << " skipped=" << skipped << '\n';
}

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--api-contract") {
      run_api_contract();
      std::cout << "PASS THead validation API contract\n";
      return EXIT_SUCCESS;
    }
    if (argc == 5 && std::string_view(argv[1]) == "--catalog-contract") {
      run_catalog_contract(argv[2], argv[3], argv[4]);
      return EXIT_SUCCESS;
    }
    std::cerr << "usage: validation_contract --api-contract | "
                 "--catalog-contract <catalog> <Operators.cmake> <sdk>\n";
    return EXIT_FAILURE;
  } catch (const std::exception &error) {
    std::cerr << "FAIL THead validation contract: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
