/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/artifact.hpp"

#include "backends/ascend/extended_artifact.hpp"

#include "backends/ascend/error.hpp"
#include "backends/ascend/target_policy.hpp"
#include "runtime/json.hpp"
#include "runtime/sha256.hpp"

#include <flagdnn/version.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace flagdnn::ascend {
namespace {

using flagdnn::native::json::Value;

constexpr std::int64_t kArtifactSchemaVersion = 5;
constexpr std::int64_t kExecutionProgramVersion = 3;
constexpr std::int64_t kLaunchPayloadVersion = 1;
constexpr std::size_t kMaximumManifestSize = 16U << 20;
constexpr std::size_t kMaximumSourceSize = 1U << 20;
constexpr std::size_t kMaximumCandidates = 1024;
constexpr std::size_t kMaximumRank = 8;
constexpr std::size_t kMaximumMatmulBatchRank = 6;
constexpr std::size_t kMaximumGraphNodes = 1024;
constexpr std::size_t kGraphWorkspaceAlignment = 256;
constexpr std::uint32_t kMaximumAiCoreCount = 65535;
constexpr std::uint32_t kWorkersPerAiCore = 2;
constexpr std::string_view kTuningSourceSha256 =
    FLAGDNN_ASCEND_TUNING_SOURCE_SHA256;
constexpr std::string_view kBinarySourceSha256 =
    FLAGDNN_ASCEND_BINARY_SOURCE_SHA256;
constexpr std::string_view kUnarySourceSha256 =
    FLAGDNN_ASCEND_UNARY_SOURCE_SHA256;
constexpr std::string_view kTernarySourceSha256 =
    FLAGDNN_ASCEND_TERNARY_SOURCE_SHA256;
constexpr std::string_view kLayoutSourceSha256 =
    FLAGDNN_ASCEND_LAYOUT_SOURCE_SHA256;
constexpr std::string_view kReductionSourceSha256 =
    FLAGDNN_ASCEND_REDUCTION_SOURCE_SHA256;
constexpr std::string_view kMatmulSourceSha256 =
    FLAGDNN_ASCEND_MATMUL_SOURCE_SHA256;
constexpr std::string_view kConvolutionSourceSha256 =
    FLAGDNN_ASCEND_CONVOLUTION_SOURCE_SHA256;
constexpr std::string_view kNormalizationSourceSha256 =
    FLAGDNN_ASCEND_NORMALIZATION_SOURCE_SHA256;
constexpr unsigned int kAutotuneWarmup = 5;
constexpr unsigned int kAutotuneRepetitions = 10;

struct AllowedTuningConfiguration {
  unsigned int block_size;
  unsigned int num_warps;
  unsigned int num_stages;
};

constexpr std::array<AllowedTuningConfiguration, 2>
    kAllowedTuningConfigurations = {{{256, 4, 1}, {128, 4, 1}}};
constexpr std::array<AllowedTuningConfiguration, 2>
    kAllowedPointwiseTuningConfigurations = {{{1024, 4, 1}, {256, 4, 1}}};
constexpr std::array<AllowedTuningConfiguration, 2>
    kAllowedMatmulTuningConfigurations = {{{128, 4, 1}, {64, 4, 1}}};
constexpr std::array<AllowedTuningConfiguration, 2>
    kAllowedBatchNormTuningConfigurations = {{{1024, 4, 1}, {256, 4, 1}}};
constexpr std::array<AllowedTuningConfiguration, 2>
    kAllowedBatchNormInferenceTuningConfigurations =
        {{{4096, 4, 1}, {1024, 4, 1}}};
constexpr std::array<AllowedTuningConfiguration, 3>
    kAllowedNormalizationTuningConfigurations =
        {{{4096, 4, 1}, {2048, 4, 1}, {1024, 4, 1}}};

[[nodiscard]] constexpr bool uses_pointwise_tuning(
    KernelFamily family) noexcept {
  return family == KernelFamily::kBinary || family == KernelFamily::kUnary ||
         family == KernelFamily::kTernary || family == KernelFamily::kLayout;
}

[[nodiscard]] constexpr bool uses_wide_normalization_tuning(
    KernelFamily family) noexcept {
  return family == KernelFamily::kRmsNorm ||
         family == KernelFamily::kLayerNorm;
}

[[nodiscard]] constexpr std::span<const AllowedTuningConfiguration>
allowed_tuning_configurations(KernelFamily family) noexcept {
  if (family == KernelFamily::kMatMul) {
    return kAllowedMatmulTuningConfigurations;
  }
  if (family == KernelFamily::kBatchNorm) {
    return kAllowedBatchNormTuningConfigurations;
  }
  if (family == KernelFamily::kBatchNormInference) {
    return kAllowedBatchNormInferenceTuningConfigurations;
  }
  if (uses_pointwise_tuning(family)) {
    return kAllowedPointwiseTuningConfigurations;
  }
  if (uses_wide_normalization_tuning(family)) {
    return kAllowedNormalizationTuningConfigurations;
  }
  return kAllowedTuningConfigurations;
}

[[nodiscard]] constexpr std::size_t tuning_configuration_count(
    KernelFamily family) noexcept {
  return allowed_tuning_configurations(family).size();
}

struct BinaryPointwiseContract {
  std::string_view operation;
  std::int64_t pointwise_mode;
  bool supports_alpha;
};

constexpr std::array<BinaryPointwiseContract, 17>
    kBinaryPointwiseContracts = {{{"add", 1, true},
                                  {"sub", 17, true},
                                  {"mul", 18, false},
                                  {"div", 19, false},
                                  {"min", 20, false},
                                  {"max", 21, false},
                                  {"mod", 22, false},
                                  {"pow", 23, false},
                                  {"cmp_eq", 25, false},
                                  {"cmp_neq", 26, false},
                                  {"cmp_gt", 27, false},
                                  {"cmp_ge", 28, false},
                                  {"cmp_lt", 29, false},
                                  {"cmp_le", 30, false},
                                  {"logical_and", 31, false},
                                  {"logical_or", 32, false},
                                  {"sigmoid_backward", 40, false}}};

struct UnaryPointwiseContract {
  std::string_view operation;
  std::int64_t pointwise_mode;
};

constexpr std::array<UnaryPointwiseContract, 23>
    kUnaryPointwiseContracts = {{{"relu", 2},
                                 {"sqrt", 3},
                                 {"erf", 4},
                                 {"identity", 5},
                                 {"exp", 6},
                                 {"log", 7},
                                 {"neg", 8},
                                 {"abs", 9},
                                 {"ceil", 10},
                                 {"cos", 11},
                                 {"floor", 12},
                                 {"rsqrt", 13},
                                 {"sin", 14},
                                 {"tan", 15},
                                 {"reciprocal", 16},
                                 {"logical_not", 24},
                                 {"sigmoid", 33},
                                 {"tanh", 34},
                                 {"elu", 35},
                                 {"gelu", 36},
                                 {"softplus", 37},
                                 {"swish", 38},
                                 {"gelu_approx_tanh", 39}}};

[[noreturn]] void artifact_failure(std::string message) {
  throw AscendError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                    std::move(message));
}

const BinaryPointwiseContract& binary_pointwise_contract(
    std::string_view operation) {
  const auto contract = std::find_if(
      kBinaryPointwiseContracts.begin(), kBinaryPointwiseContracts.end(),
      [operation](const BinaryPointwiseContract& candidate) {
        return candidate.operation == operation;
      });
  if (contract == kBinaryPointwiseContracts.end()) {
    artifact_failure("Ascend binary pointwise operation is unsupported");
  }
  return *contract;
}

const UnaryPointwiseContract* unary_pointwise_contract(
    std::string_view operation) {
  const auto contract = std::find_if(
      kUnaryPointwiseContracts.begin(), kUnaryPointwiseContracts.end(),
      [operation](const UnaryPointwiseContract& candidate) {
        return candidate.operation == operation;
      });
  return contract == kUnaryPointwiseContracts.end() ? nullptr : &*contract;
}

bool is_unary_operation(std::string_view operation) {
  return unary_pointwise_contract(operation) != nullptr;
}

bool is_ternary_operation(std::string_view operation) noexcept {
  return operation == "binary_select";
}

bool is_layout_operation(std::string_view operation) noexcept {
  return operation == "reshape" || operation == "transpose" ||
         operation == "slice";
}

bool is_reduction_operation(std::string_view operation) noexcept {
  return operation == "reduction_sum" || operation == "reduction_avg" ||
         operation == "reduction_mul";
}

std::int64_t reduction_mode(std::string_view operation) {
  if (operation == "reduction_sum") {
    return 0;
  }
  if (operation == "reduction_avg") {
    return 1;
  }
  if (operation == "reduction_mul") {
    return 2;
  }
  artifact_failure("Ascend reduction operation is unsupported");
}

KernelFamily kernel_family_for_operation(std::string_view operation) {
  if (operation == "convolution_fprop") {
    return KernelFamily::kConvolutionFprop;
  }
  if (operation == "matmul") {
    return KernelFamily::kMatMul;
  }
  if (operation == "batchnorm") {
    return KernelFamily::kBatchNorm;
  }
  if (operation == "layernorm") {
    return KernelFamily::kLayerNorm;
  }
  if (operation == "rmsnorm") {
    return KernelFamily::kRmsNorm;
  }
  if (operation == "batchnorm_inference") {
    return KernelFamily::kBatchNormInference;
  }
  if (is_reduction_operation(operation)) {
    return KernelFamily::kReduction;
  }
  if (is_layout_operation(operation)) {
    return KernelFamily::kLayout;
  }
  if (is_unary_operation(operation)) {
    return KernelFamily::kUnary;
  }
  if (is_ternary_operation(operation)) {
    return KernelFamily::kTernary;
  }
  (void)binary_pointwise_contract(operation);
  return KernelFamily::kBinary;
}

std::string_view kernel_family_name(KernelFamily family) noexcept {
  switch (family) {
    case KernelFamily::kBinary:
      return "binary";
    case KernelFamily::kUnary:
      return "unary";
    case KernelFamily::kTernary:
      return "ternary";
    case KernelFamily::kLayout:
      return "layout";
    case KernelFamily::kReduction:
      return "reduction";
    case KernelFamily::kMatMul:
      return "matmul";
    case KernelFamily::kConvolutionFprop:
      return "convolution_fprop";
    case KernelFamily::kBatchNorm:
      return "batchnorm";
    case KernelFamily::kBatchNormInference:
      return "batchnorm_inference";
    case KernelFamily::kRmsNorm:
      return "rmsnorm";
    case KernelFamily::kLayerNorm:
      return "layernorm";
  }
  return "";
}

bool is_logical_operation(std::string_view operation) noexcept {
  return operation == "logical_not" || operation == "logical_and" ||
         operation == "logical_or";
}

bool is_comparison_operation(std::string_view operation) noexcept {
  return operation == "cmp_eq" || operation == "cmp_neq" ||
         operation == "cmp_gt" || operation == "cmp_ge" ||
         operation == "cmp_lt" || operation == "cmp_le";
}

std::string read_text_file(const std::filesystem::path& path,
                           std::size_t maximum_size) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  if (error || !std::filesystem::is_regular_file(status) ||
      std::filesystem::is_symlink(status)) {
    artifact_failure("artifact metadata is missing or is not a regular file");
  }
  const std::uintmax_t file_size = std::filesystem::file_size(path, error);
  if (error || file_size == 0 || file_size > maximum_size) {
    artifact_failure("artifact metadata exceeds its size limit");
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    artifact_failure("cannot open artifact metadata");
  }
  std::string result(static_cast<std::size_t>(file_size), '\0');
  input.read(result.data(), static_cast<std::streamsize>(result.size()));
  if (!input) {
    artifact_failure("cannot read artifact metadata");
  }
  return result;
}

bool is_sha256(std::string_view value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](const char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

std::uint32_t parse_target_ai_core_count(std::string_view target) {
  constexpr std::string_view kPrefix = "ascend_";
  constexpr std::string_view kCannMarker = "_cann_";
  constexpr std::string_view kAiCoreMarker = "_aic_";
  const std::size_t cann = target.find(kCannMarker);
  const std::size_t ai_core = target.rfind(kAiCoreMarker);
  if (!target.starts_with(kPrefix) || cann == std::string_view::npos ||
      ai_core == std::string_view::npos || cann <= kPrefix.size() ||
      ai_core <= cann + kCannMarker.size() ||
      target.find(kAiCoreMarker, cann + kCannMarker.size()) != ai_core ||
      ai_core + kAiCoreMarker.size() == target.size()) {
    artifact_failure("Ascend target fingerprint capability is invalid");
  }

  const std::string_view arch =
      target.substr(kPrefix.size(), cann - kPrefix.size());
  if (!target_policy::supported_target_soc(arch)) {
    artifact_failure("Ascend target SoC capability is unsupported");
  }

  const std::string_view cann_version = target.substr(
      cann + kCannMarker.size(),
      ai_core - (cann + kCannMarker.size()));
  if (cann_version.empty() || cann_version.front() < '0' ||
      cann_version.front() > '9' ||
      !std::all_of(cann_version.begin(), cann_version.end(), [](char value) {
        return (value >= '0' && value <= '9') ||
               (value >= 'A' && value <= 'Z') ||
               (value >= 'a' && value <= 'z') || value == '_' ||
               value == '.' || value == '-';
      })) {
    artifact_failure("Ascend target CANN capability is invalid");
  }

  const std::string_view encoded_count =
      target.substr(ai_core + kAiCoreMarker.size());
  std::uint32_t result = 0;
  const auto [end, error] = std::from_chars(
      encoded_count.data(), encoded_count.data() + encoded_count.size(), result);
  if (error != std::errc{} ||
      end != encoded_count.data() + encoded_count.size() || result == 0 ||
      result > kMaximumAiCoreCount || std::to_string(result) != encoded_count) {
    artifact_failure("Ascend target AI core capability is invalid");
  }
  return result;
}

bool is_identifier(std::string_view value) {
  if (value.empty() || value.size() > 128 ||
      (!std::isalpha(static_cast<unsigned char>(value.front())) &&
       value.front() != '_')) {
    return false;
  }
  return std::all_of(
      value.begin() + 1, value.end(), [](const unsigned char character) {
        return std::isalnum(character) != 0 || character == '_';
      });
}

bool is_safe_id(std::string_view value) {
  return !value.empty() && value.size() <= 128 &&
         std::all_of(value.begin(), value.end(), [](const unsigned char c) {
           return std::isalnum(c) != 0 || c == '_' || c == '-';
         });
}

std::string tuning_configuration_id(std::string_view operation,
                                    unsigned int block_size,
                                    unsigned int num_warps,
                                    unsigned int num_stages) {
  const std::string canonical =
      "{\"configuration\":{\"META\":{\"BLOCK_SIZE\":" +
      std::to_string(block_size) + "},\"num_stages\":" +
      std::to_string(num_stages) + ",\"num_warps\":" +
      std::to_string(num_warps) + "},\"operation\":\"" +
      std::string(operation) + "\"}";
  return "config_" + flagdnn::native::sha256(canonical);
}

std::size_t tuning_configuration_index(KernelFamily family,
                                       unsigned int block_size,
                                       unsigned int num_warps,
                                       unsigned int num_stages) {
  const auto configurations = allowed_tuning_configurations(family);
  for (std::size_t index = 0; index < configurations.size(); ++index) {
    const AllowedTuningConfiguration& allowed =
        configurations[index];
    if (allowed.block_size == block_size &&
        allowed.num_warps == num_warps &&
        allowed.num_stages == num_stages) {
      return index;
    }
  }
  artifact_failure(
      "Ascend pointwise candidate is outside its tuning whitelist");
}

std::string f64_identity(double value) {
  if (!std::isfinite(value)) {
    artifact_failure("pointwise float cannot be identified canonically");
  }
  constexpr char kHex[] = "0123456789abcdef";
  const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
  std::string result(16, '0');
  for (std::size_t index = 0; index < result.size(); ++index) {
    const unsigned int shift =
        static_cast<unsigned int>((result.size() - 1 - index) * 4);
    result[index] = kHex[(bits >> shift) & 0xFU];
  }
  return result;
}

void require_exact_keys(const Value& value,
                        std::initializer_list<std::string_view> keys,
                        const char* description) {
  const auto& object = value.as_object();
  if (object.size() != keys.size()) {
    artifact_failure(std::string(description) +
                     " contains missing or unknown fields");
  }
  for (const std::string_view key : keys) {
    if (object.find(key) == object.end()) {
      artifact_failure(std::string(description) + " is missing field " +
                       std::string(key));
    }
  }
}

void require_default_precision_keys(const Value& attributes,
                                    std::initializer_list<std::string_view> keys,
                                    const char* description) {
  auto normalized = attributes.as_object();
  const auto precision = normalized.find("input_precision");
  if (precision != normalized.end()) {
    if (precision->second.as_int() != 0) {
      artifact_failure("persistent Ascend plan requires default input precision");
    }
    normalized.erase(precision);
  }
  require_exact_keys(Value(std::move(normalized)), keys, description);
}

void require_normalization_attributes(const Value& attributes,
                                      const char* description) {
  const auto& object = attributes.as_object();
  if (object.find("forward_phase") == object.end()) {
    require_exact_keys(attributes,
                       {"rows", "normalized_elements", "epsilon"},
                       description);
    return;
  }
  require_exact_keys(attributes,
                     {"rows", "normalized_elements", "epsilon",
                      "forward_phase"},
                     description);
  if (attributes.at("forward_phase").as_int() != 2) {
    artifact_failure(std::string(description) +
                     " forward phase must be TRAINING");
  }
}

std::size_t checked_size(std::int64_t value, const char* field) {
  if (value < 0 || static_cast<std::uint64_t>(value) >
                       std::numeric_limits<std::size_t>::max()) {
    artifact_failure(std::string("invalid artifact size: ") + field);
  }
  return static_cast<std::size_t>(value);
}

unsigned int checked_positive_unsigned(std::int64_t value,
                                       const char* field) {
  if (value <= 0 || static_cast<std::uint64_t>(value) >
                        std::numeric_limits<unsigned int>::max()) {
    artifact_failure(std::string("invalid positive artifact field: ") +
                     field);
  }
  return static_cast<unsigned int>(value);
}

bool valid_alignment(std::size_t value) {
  return value != 0 && value <= 256 && (value & (value - 1)) == 0;
}

std::filesystem::path validate_materialized_source(
    const std::filesystem::path& directory,
    const Value& descriptor,
    std::string_view compiler_identity) {
  require_exact_keys(descriptor, {"file", "size", "sha256"},
                     "materialized source descriptor");
  const std::string file = descriptor.at("file").as_string();
  const std::string hash = descriptor.at("sha256").as_string();
  const std::size_t size =
      checked_size(descriptor.at("size").as_int(), "source.size");
  if (!is_sha256(hash) || !is_sha256(compiler_identity) || size == 0 ||
      size > kMaximumSourceSize) {
    artifact_failure("materialized source identity is invalid");
  }
  const std::string expected = "source-" + std::string(compiler_identity) +
                               "-" + hash + ".py";
  const std::filesystem::path relative(file);
  if (file != expected || relative.is_absolute() || relative.has_parent_path() ||
      relative.filename() != relative) {
    artifact_failure("materialized source path is not content-addressed");
  }
  const std::filesystem::path path = directory / relative;
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  if (error || !std::filesystem::is_regular_file(status) ||
      std::filesystem::is_symlink(status) ||
      std::filesystem::file_size(path, error) != size || error) {
    artifact_failure("materialized source file metadata differs");
  }
  if (flagdnn::native::sha256_file(path) != hash) {
    artifact_failure("materialized source SHA-256 differs");
  }
  return path;
}

RawArgumentType parse_raw_type(std::string_view value) {
  if (value == "pointer") {
    return RawArgumentType::kPointer;
  }
  if (value == "i32") {
    return RawArgumentType::kI32;
  }
  if (value == "i64") {
    return RawArgumentType::kI64;
  }
  if (value == "f32") {
    return RawArgumentType::kF32;
  }
  if (value == "f64") {
    return RawArgumentType::kF64;
  }
  artifact_failure("ltj_npu_raw_v1 contains an unsupported argument type");
}

std::vector<std::string> split_signature(std::string_view signature) {
  if (signature.empty() || signature.size() > (64U << 10) ||
      std::any_of(signature.begin(), signature.end(), [](unsigned char c) {
        return c <= 0x20U || c > 0x7eU || c == '(' || c == ')';
      })) {
    artifact_failure("ltj_npu_raw_v1 full signature is invalid");
  }
  std::vector<std::string> tokens;
  std::size_t offset = 0;
  while (offset <= signature.size()) {
    const std::size_t separator = signature.find(',', offset);
    const std::size_t end = separator == std::string_view::npos
                                ? signature.size()
                                : separator;
    if (end == offset) {
      artifact_failure("ltj_npu_raw_v1 signature contains an empty token");
    }
    tokens.emplace_back(signature.substr(offset, end - offset));
    if (separator == std::string_view::npos) {
      break;
    }
    offset = separator + 1;
  }
  return tokens;
}

struct SignatureRuntimeToken {
  RawArgumentType type = RawArgumentType::kI64;
  std::size_t pointer_alignment = 1;
};

SignatureRuntimeToken parse_runtime_token(std::string_view token) {
  if (token.starts_with('*')) {
    const std::size_t colon = token.find(':');
    const std::string_view base = token.substr(
        0, colon == std::string_view::npos ? token.size() : colon);
    if (base != "*fp32" && base != "*fp16" && base != "*bf16" &&
        base != "*i8" && base != "*i16" && base != "*i32") {
      artifact_failure("ltj_npu_raw_v1 pointer token is unsupported");
    }
    std::size_t alignment = 1;
    if (colon != std::string_view::npos) {
      const std::string_view encoded = token.substr(colon + 1);
      unsigned long long parsed = 0;
      const auto [end, error] = std::from_chars(
          encoded.data(), encoded.data() + encoded.size(), parsed);
      if (error != std::errc{} || end != encoded.data() + encoded.size() ||
          parsed > std::numeric_limits<std::size_t>::max() ||
          !valid_alignment(static_cast<std::size_t>(parsed))) {
        artifact_failure("ltj_npu_raw_v1 pointer alignment is invalid");
      }
      alignment = static_cast<std::size_t>(parsed);
    }
    return {RawArgumentType::kPointer, alignment};
  }
  if (token == "i32") {
    return {RawArgumentType::kI32, 1};
  }
  if (token == "i64") {
    return {RawArgumentType::kI64, 1};
  }
  if (token == "f32") {
    return {RawArgumentType::kF32, 1};
  }
  if (token == "f64") {
    return {RawArgumentType::kF64, 1};
  }
  artifact_failure("ltj_npu_raw_v1 runtime signature token is unsupported");
}

std::int64_t parse_canonical_integer(std::string_view token) {
  std::int64_t value = 0;
  const auto [end, error] =
      std::from_chars(token.data(), token.data() + token.size(), value);
  if (error != std::errc{} || end != token.data() + token.size() ||
      std::to_string(value) != token) {
    artifact_failure("constexpr integer signature token is not canonical");
  }
  return value;
}

double parse_finite_number(std::string_view token) {
  double value = 0.0;
  const auto [end, error] =
      std::from_chars(token.data(), token.data() + token.size(), value);
  if (error != std::errc{} || end != token.data() + token.size() ||
      !std::isfinite(value)) {
    artifact_failure("constexpr floating signature token is invalid");
  }
  return value;
}

std::vector<std::string> expected_meta_names(std::string_view entry_point) {
  std::vector<std::string> result;
  if (entry_point == "binary_contiguous_kernel") {
    return {"OP_KIND", "ALPHA", "BLOCK_SIZE", "WORKER_COUNT"};
  }
  if (entry_point == "unary_pointwise_contiguous_kernel") {
    return {"OPERATION", "negative_slope", "lower_clip", "upper_clip",
            "HAS_UPPER_CLIP", "SWISH_BETA", "ELU_ALPHA",
            "SOFTPLUS_BETA", "BLOCK_SIZE", "WORKER_COUNT"};
  }
  if (entry_point == "binary_select_contiguous_kernel") {
    return {"BLOCK_SIZE", "WORKER_COUNT"};
  }
  if (entry_point == "layout_copy_kernel") {
    result.push_back("INPUT_BASE");
    result.push_back("ELEMENT_SIZE_BYTES");
    result.push_back("LAYOUT_MODE");
    for (const std::string_view prefix :
         {"INPUT_DIM_", "INPUT_STRIDE_", "OUTPUT_DIM_",
          "OUTPUT_STRIDE_"}) {
      for (int axis = 0; axis < 8; ++axis) {
        result.push_back(std::string(prefix) + std::to_string(axis));
      }
    }
    result.insert(result.end(), {"BLOCK_SIZE", "WORKER_COUNT"});
    return result;
  }
  if (entry_point == "reduction_3d_persistent_kernel") {
    return {"RANK", "OUTPUT_RANK", "AXIS", "KEEP_DIMENSIONS", "OUTER",
            "REDUCTION_SIZE", "INNER", "OUTPUT_ELEMENTS", "REDUCTION_MODE",
            "BLOCK_SIZE", "WORKER_COUNT"};
  }
  if (entry_point == "reduction_strided_persistent_kernel") {
    result = {"RANK", "OUTPUT_RANK", "AXIS", "KEEP_DIMENSIONS", "OUTER",
              "REDUCTION_SIZE", "INNER", "OUTPUT_ELEMENTS",
              "REDUCTION_MODE"};
    for (const std::string_view prefix :
         {"INPUT_DIM_", "INPUT_STRIDE_", "OUTPUT_DIM_", "OUTPUT_STRIDE_"}) {
      for (int axis = 0; axis < 8; ++axis) {
        result.push_back(std::string(prefix) + std::to_string(axis));
      }
    }
    result.insert(result.end(), {"BLOCK_SIZE", "WORKER_COUNT"});
    return result;
  }
  if (entry_point == "matmul_strided_kernel") {
    result = {"BATCH", "M", "N", "K"};
    for (int axis = 0; axis < 6; ++axis) {
      result.push_back("DIM_" + std::to_string(axis));
    }
    for (const std::string_view prefix :
         {"A_BATCH_STRIDE_", "B_BATCH_STRIDE_", "C_BATCH_STRIDE_"}) {
      for (int axis = 0; axis < 6; ++axis) {
        result.push_back(std::string(prefix) + std::to_string(axis));
      }
    }
    result.insert(result.end(),
                  {"A_STRIDE_M", "A_STRIDE_K", "B_STRIDE_K", "B_STRIDE_N",
                   "C_STRIDE_M", "C_STRIDE_N", "INPUT_IS_FLOAT32", "GROUP_M",
                   "BLOCK_SIZE", "WORKER_COUNT"});
    return result;
  }
  if (entry_point == "convolution_fprop_persistent_kernel" ||
      entry_point == "convolution_fprop_im2col_kernel") {
    result = {"SPATIAL_RANK", "GROUPS", "INPUT_CHANNELS",
              "OUTPUT_CHANNELS", "CHANNELS_PER_GROUP"};
    for (const std::string_view prefix :
         {"INPUT_DIM_", "INPUT_STRIDE_", "FILTER_DIM_", "FILTER_STRIDE_",
          "OUTPUT_DIM_", "OUTPUT_STRIDE_"}) {
      for (int axis = 0; axis < 5; ++axis) {
        result.push_back(std::string(prefix) + std::to_string(axis));
      }
    }
    for (const std::string_view prefix :
         {"PRE_PADDING_", "POST_PADDING_", "CONV_STRIDE_", "DILATION_"}) {
      for (int axis = 0; axis < 3; ++axis) {
        result.push_back(std::string(prefix) + std::to_string(axis));
      }
    }
    result.insert(result.end(), {"BLOCK_SIZE", "WORKER_COUNT"});
    return result;
  }
  if (entry_point == "batchnorm_inference_nchw_persistent_kernel") {
    return {"RANK", "CHANNELS", "SPATIAL", "BLOCK_SIZE", "WORKER_COUNT"};
  }
  if (entry_point == "batchnorm_inference_strided_persistent_kernel") {
    result = {"RANK", "CHANNELS", "SPATIAL"};
    for (const std::string_view prefix :
         {"DIM_", "X_STRIDE_", "Y_STRIDE_"}) {
      for (int axis = 0; axis < 8; ++axis) {
        result.push_back(std::string(prefix) + std::to_string(axis));
      }
    }
    result.insert(result.end(), {"BLOCK_SIZE", "WORKER_COUNT"});
    return result;
  }
  if (entry_point == "batchnorm_training_persistent_kernel") {
    result = {"RANK", "BATCH", "CHANNELS", "SPATIAL",
              "REDUCTION_ELEMENTS", "EPSILON", "MOMENTUM"};
    for (const std::string_view prefix :
         {"DIM_", "X_STRIDE_", "Y_STRIDE_"}) {
      for (int axis = 0; axis < 8; ++axis) {
        result.push_back(std::string(prefix) + std::to_string(axis));
      }
    }
    result.insert(result.end(), {"BLOCK_SIZE", "WORKER_COUNT"});
    return result;
  }
  if (entry_point == "rmsnorm_persistent_kernel" ||
      entry_point == "layernorm_persistent_kernel") {
    return {"ROWS", "NORMALIZED_ELEMENTS", "EPSILON", "BLOCK_SIZE",
            "WORKER_COUNT"};
  }
  const bool binary = entry_point == "binary_strided_kernel";
  const bool unary = entry_point == "unary_pointwise_strided_kernel";
  const bool ternary = entry_point == "binary_select_strided_kernel";
  if (!binary && !unary && !ternary) {
    artifact_failure("Ascend pointwise artifact entry point is unsupported");
  }
  const std::vector<std::string_view> prefixes =
      binary ? std::vector<std::string_view>{
                   "DIM_", "LEFT_STRIDE_", "RIGHT_STRIDE_", "OUTPUT_STRIDE_"}
             : unary ? std::vector<std::string_view>{
                           "DIM_", "INPUT_STRIDE_", "OUTPUT_STRIDE_"}
                     : std::vector<std::string_view>{
                           "DIM_", "LEFT_STRIDE_", "RIGHT_STRIDE_",
                           "MASK_STRIDE_", "OUTPUT_STRIDE_"};
  for (const std::string_view prefix : prefixes) {
    for (int axis = 0; axis < 8; ++axis) {
      result.push_back(std::string(prefix) + std::to_string(axis));
    }
  }
  if (binary) {
    result.insert(
        result.end(), {"OP_KIND", "ALPHA", "BLOCK_SIZE", "WORKER_COUNT"});
  } else if (unary) {
    result.insert(result.end(),
                  {"OPERATION", "negative_slope", "lower_clip", "upper_clip",
                   "HAS_UPPER_CLIP", "SWISH_BETA", "ELU_ALPHA",
                   "SOFTPLUS_BETA", "BLOCK_SIZE", "WORKER_COUNT"});
  } else {
    result.insert(result.end(), {"BLOCK_SIZE", "WORKER_COUNT"});
  }
  return result;
}

void validate_meta_and_signature(const Value& meta,
                                 std::string_view entry_point,
                                 std::string_view full_signature,
                                 const std::vector<ArgumentSource>& sources,
                                 const std::vector<RawArgumentType>& abi,
                                 std::int64_t expected_pointwise_mode,
                                 std::uint32_t expected_worker_count) {
  const auto& meta_object = meta.as_object();
  const std::vector<std::string> meta_names =
      expected_meta_names(entry_point);
  if (meta_object.size() != meta_names.size()) {
    artifact_failure("ltj_npu_raw_v1 meta parameter set is invalid");
  }
  for (const std::string& name : meta_names) {
    if (meta_object.find(name) == meta_object.end()) {
      artifact_failure("ltj_npu_raw_v1 is missing a meta parameter");
    }
  }

  const std::vector<std::string> tokens = split_signature(full_signature);
  if (tokens.size() != sources.size() + meta_names.size() ||
      abi.size() != sources.size()) {
    artifact_failure("ltj_npu_raw_v1 signature/ABI argument count differs");
  }
  for (std::size_t index = 0; index < sources.size(); ++index) {
    const SignatureRuntimeToken token = parse_runtime_token(tokens[index]);
    if (token.type != abi[index] || abi[index] != sources[index].type) {
      artifact_failure("ltj_npu_raw_v1 signature and argument ABI differ");
    }
    if (token.type == RawArgumentType::kPointer) {
      const std::size_t expected_alignment =
          sources[index].alignment >= 16 ? 16 : 1;
      if (token.pointer_alignment != expected_alignment) {
        artifact_failure(
            "signature pointer alignment differs from source alignment");
      }
    }
  }
  for (std::size_t index = 0; index < meta_names.size(); ++index) {
    const std::string& name = meta_names[index];
    const std::string& token = tokens[sources.size() + index];
    const Value& value = meta_object.at(name);
    if (name == "ALPHA" || name == "EPSILON" || name == "MOMENTUM" ||
        name == "negative_slope" ||
        name == "lower_clip" || name == "upper_clip" ||
        name == "SWISH_BETA" || name == "ELU_ALPHA" ||
        name == "SOFTPLUS_BETA") {
      const double expected = value.as_double();
      const double parsed = parse_finite_number(token);
      if (!std::isfinite(expected) || (name == "EPSILON" && expected <= 0.0) ||
          (name == "MOMENTUM" && (expected < 0.0 || expected > 1.0)) ||
          std::abs(expected) > std::numeric_limits<float>::max() ||
          parsed != expected) {
        artifact_failure("floating signature token differs from meta");
      }
    } else {
      const std::int64_t expected = value.as_int();
      if (parse_canonical_integer(token) != expected) {
        artifact_failure("integer signature token differs from meta");
      }
      if (((name == "OP_KIND" || name == "OPERATION") &&
           expected != expected_pointwise_mode) ||
          (name == "REDUCTION_MODE" &&
           expected != expected_pointwise_mode) ||
          (name == "RANK" && (expected <= 0 || expected > 8)) ||
          ((name == "BATCH" || name == "CHANNELS" || name == "SPATIAL" ||
            name == "REDUCTION_ELEMENTS" || name == "SPATIAL_RANK" ||
            name == "GROUPS" || name == "INPUT_CHANNELS" ||
            name == "OUTPUT_CHANNELS" ||
            name == "CHANNELS_PER_GROUP") &&
           expected <= 0) ||
          (name == "OUTPUT_RANK" && (expected < 0 || expected > 8)) ||
          (name == "AXIS" && (expected < 0 || expected >= 8)) ||
          (name == "KEEP_DIMENSIONS" && expected != 0 && expected != 1) ||
          ((name == "OUTER" || name == "REDUCTION_SIZE" ||
            name == "INNER" || name == "OUTPUT_ELEMENTS" ||
            name == "ROWS" || name == "NORMALIZED_ELEMENTS") &&
           expected <= 0) ||
          (name == "HAS_UPPER_CLIP" && expected != 0 && expected != 1) ||
          (name == "BLOCK_SIZE" &&
           (entry_point == "matmul_strided_kernel"
                ? expected != 16 && expected != 32 && expected != 64 &&
                      expected != 128
                : expected != 128 && expected != 256 && expected != 1024 &&
                      expected != 2048 && expected != 4096)) ||
          (name == "WORKER_COUNT" &&
           expected != static_cast<std::int64_t>(expected_worker_count)) ||
          ((name.starts_with("DIM_") ||
            name.starts_with("INPUT_DIM_") ||
            name.starts_with("OUTPUT_DIM_")) &&
           expected <= 0) ||
          ((name == "M" || name == "N" || name == "K") && expected <= 0) ||
          (name == "INPUT_IS_FLOAT32" && expected != 0 && expected != 1) ||
          (name == "GROUP_M" && expected != 1 && expected != 8) ||
          (name == "INPUT_BASE" && expected < 0) ||
          (name == "ELEMENT_SIZE_BYTES" && expected != 2 && expected != 4) ||
          (name == "LAYOUT_MODE" && (expected < 0 || expected > 2)) ||
          (name.find("STRIDE_") != std::string::npos && expected < 0)) {
        artifact_failure(
            "Ascend pointwise meta parameter is outside its envelope");
      }
    }
  }
}

struct ExpectedTensor {
  std::int64_t uid = 0;
  std::string data_type;
  std::vector<std::int64_t> dimensions;
  std::vector<std::int64_t> strides;
  std::size_t alignment = 1;
  bool is_virtual = false;
  std::size_t storage_size = 0;
  std::size_t workspace_offset = 0;
};

struct ExpectedArgument {
  ArgumentSourceKind source = ArgumentSourceKind::kBinding;
  std::int64_t uid = 0;
  std::size_t size = 0;
  std::size_t alignment = 1;
  std::size_t workspace_offset = 0;
};

struct ExpectedStage {
  std::size_t stage_id = 0;
  std::size_t source_node_id = 0;
  std::vector<std::size_t> dependencies;
  std::string operation;
  KernelFamily kernel_family = KernelFamily::kBinary;
  std::int64_t pointwise_mode = 0;
  std::size_t input_count = 0;
  std::string entry_point;
  std::vector<ExpectedArgument> arguments;
  std::vector<std::string> pointer_tokens;
  std::vector<std::string> tensor_data_types;
  std::vector<ExpectedTensor> tensors;
  std::int32_t n_elements = 0;
  double alpha = 1.0;
  double negative_slope = 0.0;
  double lower_clip = 0.0;
  double upper_clip = 0.0;
  bool has_upper_clip = false;
  double swish_beta = 1.0;
  double elu_alpha = 1.0;
  double softplus_beta = 1.0;
  std::array<std::int64_t, kMaximumRank> dimensions{};
  std::array<std::int64_t, kMaximumRank> left_strides{};
  std::array<std::int64_t, kMaximumRank> right_strides{};
  std::array<std::int64_t, kMaximumRank> mask_strides{};
  std::array<std::int64_t, kMaximumRank> output_strides{};
  std::int64_t matmul_batch = 0;
  std::int64_t matmul_m = 0;
  std::int64_t matmul_n = 0;
  std::int64_t matmul_k = 0;
  std::array<std::int64_t, kMaximumMatmulBatchRank>
      matmul_batch_dimensions{};
  std::array<std::int64_t, kMaximumMatmulBatchRank>
      matmul_a_batch_strides{};
  std::array<std::int64_t, kMaximumMatmulBatchRank>
      matmul_b_batch_strides{};
  std::array<std::int64_t, kMaximumMatmulBatchRank>
      matmul_output_batch_strides{};
  std::int64_t matmul_a_stride_m = 0;
  std::int64_t matmul_a_stride_k = 0;
  std::int64_t matmul_b_stride_k = 0;
  std::int64_t matmul_b_stride_n = 0;
  std::int64_t matmul_output_stride_m = 0;
  std::int64_t matmul_output_stride_n = 0;
  std::int64_t convolution_spatial_rank = 0;
  std::int64_t convolution_groups = 0;
  std::int64_t convolution_input_channels = 0;
  std::int64_t convolution_output_channels = 0;
  std::int64_t convolution_channels_per_group = 0;
  std::array<std::int64_t, 5> convolution_input_dimensions{};
  std::array<std::int64_t, 5> convolution_input_strides{};
  std::array<std::int64_t, 5> convolution_filter_dimensions{};
  std::array<std::int64_t, 5> convolution_filter_strides{};
  std::array<std::int64_t, 5> convolution_output_dimensions{};
  std::array<std::int64_t, 5> convolution_output_strides{};
  std::array<std::int64_t, 3> convolution_pre_padding{};
  std::array<std::int64_t, 3> convolution_post_padding{};
  std::array<std::int64_t, 3> convolution_stride{};
  std::array<std::int64_t, 3> convolution_dilation{};
  std::int64_t batchnorm_rank = 0;
  std::int64_t batchnorm_batch = 0;
  std::int64_t batchnorm_channels = 0;
  std::int64_t batchnorm_spatial = 0;
  std::int64_t batchnorm_reduction_elements = 0;
  double batchnorm_epsilon = 0.0;
  double batchnorm_momentum = 0.0;
  std::array<std::int64_t, kMaximumRank> batchnorm_dimensions{};
  std::array<std::int64_t, kMaximumRank> batchnorm_x_strides{};
  std::array<std::int64_t, kMaximumRank> batchnorm_y_strides{};
  std::int64_t rmsnorm_rows = 0;
  std::int64_t rmsnorm_normalized_elements = 0;
  double rmsnorm_epsilon = 0.0;
  std::int64_t layernorm_rows = 0;
  std::int64_t layernorm_normalized_elements = 0;
  double layernorm_epsilon = 0.0;
  std::int64_t layout_input_base = 0;
  std::int64_t layout_mode = 0;
  std::array<std::int64_t, kMaximumRank> layout_input_dimensions{};
  std::array<std::int64_t, kMaximumRank> layout_input_strides{};
  std::array<std::int64_t, kMaximumRank> layout_output_dimensions{};
  std::int64_t reduction_axis = 0;
  std::int64_t reduction_rank = 0;
  std::int64_t reduction_output_rank = 0;
  std::int64_t reduction_keep_dimensions = 0;
  std::int64_t reduction_outer = 0;
  std::int64_t reduction_size = 0;
  std::int64_t reduction_inner = 0;
  std::array<std::int64_t, kMaximumRank> reduction_input_dimensions{};
  std::array<std::int64_t, kMaximumRank> reduction_input_strides{};
  std::array<std::int64_t, kMaximumRank> reduction_output_dimensions{};
  std::size_t workspace_offset = 0;
  std::size_t workspace_size = 0;
  std::size_t workspace_alignment = 1;
};

struct ExpectedGraph {
  std::vector<ExpectedStage> stages;
  std::size_t workspace_size = 0;
  std::size_t node_count = 0;
  bool autotune_requested = false;
};

const Value* find_field(const Value& object, std::string_view name) {
  const auto& fields = object.as_object();
  const auto iterator = fields.find(name);
  return iterator == fields.end() ? nullptr : &iterator->second;
}

double graph_f32(const Value& attributes,
                 std::string_view name,
                 double default_value) {
  const Value* value = find_field(attributes, name);
  const double result = value == nullptr ? default_value : value->as_double();
  if (!std::isfinite(result) ||
      std::abs(result) > std::numeric_limits<float>::max()) {
    artifact_failure("Graph pointwise floating attribute is outside float32");
  }
  return result;
}

void validate_optional_mode(const Value& attributes,
                            std::string_view name,
                            std::int64_t expected) {
  const Value* value = find_field(attributes, name);
  if (value != nullptr && value->as_int() != expected) {
    artifact_failure("Graph pointwise mode metadata disagrees");
  }
}

void validate_optional_relu_float(const Value& attributes,
                                  std::string_view name,
                                  double expected) {
  const Value* value = find_field(attributes, name);
  if (value != nullptr && graph_f32(attributes, name, expected) != expected) {
    artifact_failure("Graph raw and normalized ReLU metadata disagree");
  }
}

std::size_t graph_positive_size(const Value& value,
                                std::size_t maximum,
                                const char* description) {
  const std::int64_t integer = value.as_int();
  if (integer <= 0 || static_cast<std::uint64_t>(integer) > maximum) {
    artifact_failure(std::string(description) + " is outside its envelope");
  }
  return static_cast<std::size_t>(integer);
}

std::size_t graph_nonnegative_size(const Value& value,
                                   std::size_t maximum,
                                   const char* description) {
  const std::int64_t integer = value.as_int();
  if (integer < 0 || static_cast<std::uint64_t>(integer) > maximum) {
    artifact_failure(std::string(description) + " is outside its envelope");
  }
  return static_cast<std::size_t>(integer);
}

std::size_t element_size(std::string_view data_type) {
  if (data_type == "float32") {
    return 4;
  }
  if (data_type == "float16" || data_type == "bfloat16") {
    return 2;
  }
  if (data_type == "boolean") {
    return 1;
  }
  artifact_failure("Graph pointwise tensor data type is unsupported");
}

StorageDataType storage_data_type(std::string_view data_type) {
  if (data_type == "float32") {
    return StorageDataType::kFloat32;
  }
  if (data_type == "float16") {
    return StorageDataType::kFloat16;
  }
  if (data_type == "bfloat16") {
    return StorageDataType::kBFloat16;
  }
  if (data_type == "boolean") {
    return StorageDataType::kBoolean;
  }
  artifact_failure("Graph pointwise tensor data type is unsupported");
}

std::string pointer_token(const ExpectedTensor& tensor,
                          bool raw_layout = false) {
  std::string result;
  if (raw_layout && element_size(tensor.data_type) == 4) {
    result = "*i32";
  } else if (raw_layout && element_size(tensor.data_type) == 2) {
    result = "*i16";
  } else if (tensor.data_type == "float32") {
    result = "*fp32";
  } else if (tensor.data_type == "float16") {
    result = "*fp16";
  } else if (tensor.data_type == "bfloat16") {
    result = "*bf16";
  } else if (tensor.data_type == "boolean") {
    result = "*i8";
  } else {
    artifact_failure("Graph pointwise tensor data type is unsupported");
  }
  const std::size_t alignment =
      tensor.is_virtual ? kGraphWorkspaceAlignment : tensor.alignment;
  if (alignment >= 16) {
    result += ":16";
  }
  return result;
}

bool has_non_overlapping_strides(
    const std::vector<std::int64_t>& dimensions,
    const std::vector<std::int64_t>& strides) {
  std::vector<std::pair<std::int64_t, std::int64_t>> axes;
  for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
    if (dimensions[axis] > 1) {
      axes.emplace_back(strides[axis], dimensions[axis]);
    }
  }
  std::sort(axes.begin(), axes.end());
  std::uint64_t required_span = 1;
  for (const auto& [stride, dimension] : axes) {
    if (static_cast<std::uint64_t>(stride) < required_span) {
      return false;
    }
    const std::uint64_t width =
        static_cast<std::uint64_t>(dimension - 1);
    const std::uint64_t encoded_stride =
        static_cast<std::uint64_t>(stride);
    if (encoded_stride != 0 &&
        width > std::numeric_limits<std::uint64_t>::max() /
                    encoded_stride) {
      artifact_failure("Graph tensor stride extent overflows");
    }
    const std::uint64_t extent = width * encoded_stride;
    if (extent > static_cast<std::uint64_t>(
                     std::numeric_limits<std::int64_t>::max()) -
                     required_span) {
      artifact_failure("Graph tensor stride span exceeds int64");
    }
    required_span += extent;
  }
  return true;
}

std::size_t tensor_storage_size(
    const std::vector<std::int64_t>& dimensions,
    const std::vector<std::int64_t>& strides,
    std::size_t bytes_per_element) {
  std::uint64_t elements = 1;
  for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
    const std::uint64_t width =
        static_cast<std::uint64_t>(dimensions[axis] - 1);
    const std::uint64_t stride =
        static_cast<std::uint64_t>(strides[axis]);
    if (stride != 0 &&
        width > std::numeric_limits<std::uint64_t>::max() / stride) {
      artifact_failure("Graph tensor storage extent overflows");
    }
    const std::uint64_t extent = width * stride;
    if (extent > static_cast<std::uint64_t>(
                     std::numeric_limits<std::int64_t>::max()) -
                     elements) {
      artifact_failure("Graph tensor storage span exceeds int64");
    }
    elements += extent;
  }
  if (elements > static_cast<std::uint64_t>(
                     std::numeric_limits<std::int64_t>::max()) /
                     bytes_per_element ||
      elements > std::numeric_limits<std::size_t>::max() /
                     bytes_per_element) {
    artifact_failure("Graph tensor storage size exceeds its envelope");
  }
  return static_cast<std::size_t>(elements * bytes_per_element);
}

std::size_t dense_element_count(const ExpectedTensor& tensor) {
  std::uint64_t result = 1;
  for (const std::int64_t dimension : tensor.dimensions) {
    if (result > static_cast<std::uint64_t>(
                     std::numeric_limits<std::int32_t>::max()) /
                     static_cast<std::uint64_t>(dimension)) {
      artifact_failure("Graph pointwise output element count exceeds int32");
    }
    result *= static_cast<std::uint64_t>(dimension);
  }
  return static_cast<std::size_t>(result);
}

ExpectedTensor parse_graph_tensor(const Value& value) {
  ExpectedTensor result;
  result.uid = value.at("uid").as_int();
  if (result.uid <= 0) {
    artifact_failure("Graph tensor UID is invalid");
  }
  result.data_type = value.at("data_type").as_string();
  const std::size_t bytes_per_element = element_size(result.data_type);
  result.alignment = graph_positive_size(
      value.at("alignment"), kGraphWorkspaceAlignment,
      "Graph tensor alignment");
  if (!valid_alignment(result.alignment)) {
    artifact_failure("Graph tensor alignment is invalid");
  }
  result.is_virtual = value.at("virtual").as_bool();

  const auto& dimensions = value.at("dimensions").as_array();
  const auto& strides = value.at("strides").as_array();
  if (dimensions.size() > kMaximumRank ||
      dimensions.size() != strides.size()) {
    artifact_failure("Graph tensor rank is invalid");
  }
  result.dimensions.reserve(dimensions.size());
  result.strides.reserve(strides.size());
  for (const Value& dimension : dimensions) {
    result.dimensions.push_back(static_cast<std::int64_t>(graph_positive_size(
        dimension, std::numeric_limits<std::int32_t>::max(),
        "Graph tensor dimension")));
  }
  for (const Value& stride : strides) {
    const std::int64_t encoded = stride.as_int();
    if (encoded < 0) {
      artifact_failure("Graph tensor stride is negative");
    }
    result.strides.push_back(encoded);
  }
  if (!has_non_overlapping_strides(result.dimensions, result.strides)) {
    artifact_failure("Graph tensor strides overlap");
  }
  result.storage_size = tensor_storage_size(
      result.dimensions, result.strides, bytes_per_element);
  return result;
}

const ExpectedTensor& parse_graph_port(
    const Value& value,
    std::string_view expected_name,
    const std::map<std::int64_t, ExpectedTensor>& tensors) {
  if (value.at("name").as_string() != expected_name) {
    artifact_failure("Graph pointwise port name is invalid");
  }
  if (const Value* optional = find_field(value, "optional");
      optional != nullptr && optional->as_bool()) {
    artifact_failure("Graph pointwise does not support absent optional ports");
  }
  const std::int64_t uid = value.at("uid").as_int();
  const auto iterator = tensors.find(uid);
  if (uid <= 0 || iterator == tensors.end()) {
    artifact_failure("Graph pointwise port references an unknown tensor");
  }
  return iterator->second;
}

std::vector<std::int64_t> broadcast_dimensions(
    const ExpectedTensor& left,
    const ExpectedTensor& right) {
  const std::size_t rank =
      std::max(left.dimensions.size(), right.dimensions.size());
  if (rank == 0 || rank > kMaximumRank) {
    artifact_failure("Graph Add broadcast rank is invalid");
  }
  std::vector<std::int64_t> result(rank, 1);
  for (std::size_t trailing = 0; trailing < rank; ++trailing) {
    const std::int64_t left_dimension =
        trailing < left.dimensions.size()
            ? left.dimensions[left.dimensions.size() - 1 - trailing]
            : 1;
    const std::int64_t right_dimension =
        trailing < right.dimensions.size()
            ? right.dimensions[right.dimensions.size() - 1 - trailing]
            : 1;
    if (left_dimension != right_dimension && left_dimension != 1 &&
        right_dimension != 1) {
      artifact_failure("Graph Add input shapes cannot broadcast");
    }
    result[rank - 1 - trailing] =
        std::max(left_dimension, right_dimension);
  }
  return result;
}

bool is_physically_dense(const ExpectedTensor& tensor) {
  std::uint64_t elements = 1;
  for (const std::int64_t dimension : tensor.dimensions) {
    if (elements > std::numeric_limits<std::size_t>::max() /
                       static_cast<std::uint64_t>(dimension)) {
      return false;
    }
    elements *= static_cast<std::uint64_t>(dimension);
  }
  const std::size_t bytes = element_size(tensor.data_type);
  return elements <= std::numeric_limits<std::size_t>::max() / bytes &&
         static_cast<std::size_t>(elements) * bytes == tensor.storage_size;
}

bool is_row_major_tensor(const ExpectedTensor& tensor) {
  std::int64_t expected_stride = 1;
  for (std::size_t trailing = 0; trailing < tensor.dimensions.size();
       ++trailing) {
    const std::size_t axis = tensor.dimensions.size() - 1 - trailing;
    if (tensor.dimensions[axis] > 1 &&
        tensor.strides[axis] != expected_stride) {
      return false;
    }
    if (expected_stride > std::numeric_limits<std::int64_t>::max() /
                              tensor.dimensions[axis]) {
      return false;
    }
    expected_stride *= tensor.dimensions[axis];
  }
  return is_physically_dense(tensor);
}

std::array<std::int64_t, kMaximumRank> padded_dimensions(
    const ExpectedTensor& tensor) {
  std::array<std::int64_t, kMaximumRank> result{};
  result.fill(1);
  const std::size_t leading = kMaximumRank - tensor.dimensions.size();
  std::copy(tensor.dimensions.begin(), tensor.dimensions.end(),
            result.begin() + static_cast<std::ptrdiff_t>(leading));
  return result;
}

std::array<std::int64_t, kMaximumRank> effective_strides(
    const ExpectedTensor& tensor,
    std::size_t output_rank,
    bool zero_singleton_dimensions) {
  std::array<std::int64_t, kMaximumRank> result{};
  const std::size_t graph_leading = kMaximumRank - output_rank;
  const std::size_t tensor_leading = output_rank - tensor.dimensions.size();
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    result[graph_leading + tensor_leading + axis] =
        zero_singleton_dimensions && tensor.dimensions[axis] == 1
            ? 0
            : tensor.strides[axis];
  }
  return result;
}

std::vector<std::int64_t> graph_integer_array(
    const Value& attributes,
    std::string_view name,
    std::size_t expected_size,
    std::int64_t minimum,
    std::int64_t maximum) {
  const auto& values = attributes.at(name).as_array();
  if (values.size() != expected_size) {
    artifact_failure("Graph layout integer array rank is invalid");
  }
  std::vector<std::int64_t> result;
  result.reserve(values.size());
  for (const Value& value : values) {
    const std::int64_t encoded = value.as_int();
    if (encoded < minimum || encoded > maximum) {
      artifact_failure("Graph layout integer array value is invalid");
    }
    result.push_back(encoded);
  }
  return result;
}

void validate_layout_tensor_metadata(
    const Value& attributes,
    std::string_view name,
    const std::vector<std::int64_t>& expected,
    std::int64_t minimum) {
  if (graph_integer_array(attributes,
                          name,
                          expected.size(),
                          minimum,
                          std::numeric_limits<std::int64_t>::max()) !=
      expected) {
    artifact_failure("Graph layout attributes differ from tensor metadata");
  }
}

std::array<std::int64_t, kMaximumRank> padded_layout_values(
    const std::vector<std::int64_t>& values,
    std::int64_t leading_value) {
  std::array<std::int64_t, kMaximumRank> result{};
  result.fill(leading_value);
  std::copy(values.begin(), values.end(),
            result.begin() + static_cast<std::ptrdiff_t>(
                                 kMaximumRank - values.size()));
  return result;
}

struct LayoutMapping {
  std::int64_t input_base = 0;
  std::int64_t mode = 0;
  std::array<std::int64_t, kMaximumRank> input_dimensions{};
  std::array<std::int64_t, kMaximumRank> input_strides{};
  std::array<std::int64_t, kMaximumRank> output_dimensions{};
  std::array<std::int64_t, kMaximumRank> output_strides{};
};

bool is_row_major_layout(
    const std::array<std::int64_t, kMaximumRank>& dimensions,
    const std::array<std::int64_t, kMaximumRank>& strides) {
  std::uint64_t expected_stride = 1;
  for (std::size_t trailing = 0; trailing < kMaximumRank; ++trailing) {
    const std::size_t axis = kMaximumRank - 1 - trailing;
    const std::int64_t dimension = dimensions[axis];
    if (dimension == 1) {
      continue;
    }
    if (strides[axis] < 0 ||
        static_cast<std::uint64_t>(strides[axis]) != expected_stride) {
      return false;
    }
    const auto encoded_dimension = static_cast<std::uint64_t>(dimension);
    if (expected_stride >
        std::numeric_limits<std::uint64_t>::max() / encoded_dimension) {
      return false;
    }
    expected_stride *= encoded_dimension;
  }
  return true;
}

bool is_dense_permutation_layout(
    const std::array<std::int64_t, kMaximumRank>& dimensions,
    const std::array<std::int64_t, kMaximumRank>& strides) {
  std::vector<std::pair<std::int64_t, std::int64_t>> axes;
  axes.reserve(kMaximumRank);
  for (std::size_t axis = 0; axis < kMaximumRank; ++axis) {
    if (dimensions[axis] != 1) {
      axes.emplace_back(strides[axis], dimensions[axis]);
    }
  }
  std::sort(axes.begin(), axes.end());
  std::uint64_t expected_stride = 1;
  for (const auto& [stride, dimension] : axes) {
    if (stride < 0 || static_cast<std::uint64_t>(stride) != expected_stride) {
      return false;
    }
    const auto encoded_dimension = static_cast<std::uint64_t>(dimension);
    if (expected_stride >
        std::numeric_limits<std::uint64_t>::max() / encoded_dimension) {
      return false;
    }
    expected_stride *= encoded_dimension;
  }
  return true;
}

std::int64_t layout_mapping_mode(std::string_view operation,
                                 const LayoutMapping& mapping) {
  const bool shared_mapping =
      mapping.input_dimensions == mapping.output_dimensions &&
      mapping.input_strides == mapping.output_strides;
  if (operation == "reshape" &&
      is_row_major_layout(mapping.input_dimensions, mapping.input_strides) &&
      is_row_major_layout(mapping.output_dimensions,
                          mapping.output_strides)) {
    return 1;
  }
  if (operation == "transpose" && shared_mapping &&
      is_dense_permutation_layout(mapping.output_dimensions,
                                  mapping.output_strides)) {
    return 1;
  }
  return shared_mapping ? 2 : 0;
}

LayoutMapping parse_layout_mapping(std::string_view operation,
                                   const Value& attributes,
                                   const ExpectedTensor& input,
                                   const ExpectedTensor& output,
                                   std::size_t n_elements) {
  validate_layout_tensor_metadata(
      attributes, "input_dimensions", input.dimensions, 1);
  validate_layout_tensor_metadata(
      attributes, "input_strides", input.strides, 0);
  validate_layout_tensor_metadata(
      attributes, "output_dimensions", output.dimensions, 1);
  validate_layout_tensor_metadata(
      attributes, "output_strides", output.strides, 0);

  std::vector<std::int64_t> logical_input_dimensions = input.dimensions;
  std::vector<std::int64_t> logical_input_strides = input.strides;
  std::int64_t input_base = 0;
  if (operation == "reshape") {
    if (graph_nonnegative_size(attributes.at("input_rank"),
                               kMaximumRank,
                               "Graph reshape input rank") !=
            input.dimensions.size() ||
        graph_nonnegative_size(attributes.at("output_rank"),
                               kMaximumRank,
                               "Graph reshape output rank") !=
            output.dimensions.size() ||
        attributes.at("reshape_mode").as_int() != 2) {
      artifact_failure("Graph reshape rank or LOGICAL mode is invalid");
    }
    if (dense_element_count(input) != n_elements) {
      artifact_failure("Graph reshape element counts differ");
    }
  } else if (operation == "transpose") {
    const std::size_t rank = input.dimensions.size();
    if (rank == 0 || rank != output.dimensions.size() ||
        graph_positive_size(attributes.at("rank"),
                            kMaximumRank,
                            "Graph transpose rank") != rank) {
      artifact_failure("Graph transpose rank is invalid");
    }
    const std::vector<std::int64_t> permutation = graph_integer_array(
        attributes,
        "permutation",
        rank,
        0,
        static_cast<std::int64_t>(rank - 1));
    std::vector<bool> seen(rank, false);
    logical_input_dimensions = output.dimensions;
    logical_input_strides.clear();
    logical_input_strides.reserve(rank);
    for (std::size_t axis = 0; axis < rank; ++axis) {
      const std::size_t source_axis =
          static_cast<std::size_t>(permutation[axis]);
      if (seen[source_axis] ||
          output.dimensions[axis] != input.dimensions[source_axis]) {
        artifact_failure("Graph transpose permutation is invalid");
      }
      seen[source_axis] = true;
      logical_input_strides.push_back(input.strides[source_axis]);
    }
  } else if (operation == "slice") {
    const std::size_t rank = input.dimensions.size();
    if (rank == 0 || rank != output.dimensions.size() ||
        graph_positive_size(attributes.at("rank"),
                            kMaximumRank,
                            "Graph slice rank") != rank) {
      artifact_failure("Graph slice rank is invalid");
    }
    const std::vector<std::int64_t> starts = graph_integer_array(
        attributes, "starts", rank, 0,
        std::numeric_limits<std::int64_t>::max());
    const std::vector<std::int64_t> limits = graph_integer_array(
        attributes, "limits", rank, 1,
        std::numeric_limits<std::int64_t>::max());
    const std::vector<std::int64_t> steps = graph_integer_array(
        attributes, "slice_strides", rank, 1,
        std::numeric_limits<std::int64_t>::max());
    logical_input_dimensions = output.dimensions;
    logical_input_strides.clear();
    logical_input_strides.reserve(rank);
    for (std::size_t axis = 0; axis < rank; ++axis) {
      const std::int64_t span = limits[axis] - starts[axis];
      const std::int64_t output_dimension =
          1 + (span - 1) / steps[axis];
      if (starts[axis] >= limits[axis] ||
          limits[axis] > input.dimensions[axis] ||
          output.dimensions[axis] != output_dimension) {
        artifact_failure("Graph slice range or output shape is invalid");
      }
      if (input.strides[axis] != 0 &&
          steps[axis] > std::numeric_limits<std::int64_t>::max() /
                            input.strides[axis]) {
        artifact_failure("Graph slice logical stride overflows");
      }
      logical_input_strides.push_back(input.strides[axis] * steps[axis]);
      if (input.strides[axis] != 0 &&
          starts[axis] > std::numeric_limits<std::int64_t>::max() /
                             input.strides[axis]) {
        artifact_failure("Graph slice input base overflows");
      }
      const std::int64_t term = starts[axis] * input.strides[axis];
      if (term > std::numeric_limits<std::int64_t>::max() - input_base) {
        artifact_failure("Graph slice input base overflows");
      }
      input_base += term;
    }
  } else {
    artifact_failure("Graph layout operation is unsupported");
  }

  std::uint64_t maximum_input_offset =
      static_cast<std::uint64_t>(input_base);
  for (std::size_t axis = 0; axis < logical_input_dimensions.size(); ++axis) {
    const std::uint64_t width = static_cast<std::uint64_t>(
        logical_input_dimensions[axis] - 1);
    const std::uint64_t stride =
        static_cast<std::uint64_t>(logical_input_strides[axis]);
    if (stride != 0 &&
        width > std::numeric_limits<std::uint64_t>::max() / stride) {
      artifact_failure("Graph layout input mapping overflows");
    }
    const std::uint64_t extent = width * stride;
    if (extent > std::numeric_limits<std::uint64_t>::max() -
                     maximum_input_offset) {
      artifact_failure("Graph layout input mapping overflows");
    }
    maximum_input_offset += extent;
  }
  const std::size_t input_elements =
      input.storage_size / element_size(input.data_type);
  if (maximum_input_offset >= input_elements) {
    artifact_failure("Graph layout input mapping exceeds tensor storage");
  }

  LayoutMapping result;
  result.input_base = input_base;
  result.input_dimensions =
      padded_layout_values(logical_input_dimensions, 1);
  result.input_strides = padded_layout_values(logical_input_strides, 0);
  result.output_dimensions = padded_layout_values(output.dimensions, 1);
  result.output_strides = padded_layout_values(output.strides, 0);
  result.mode = layout_mapping_mode(operation, result);
  return result;
}

std::size_t align_graph_workspace(std::size_t value) {
  if (value > std::numeric_limits<std::size_t>::max() -
                  (kGraphWorkspaceAlignment - 1)) {
    artifact_failure("Graph workspace alignment overflows");
  }
  return (value + kGraphWorkspaceAlignment - 1) /
         kGraphWorkspaceAlignment * kGraphWorkspaceAlignment;
}

ExpectedArgument expected_argument(const ExpectedTensor& tensor) {
  ExpectedArgument result;
  result.source = tensor.is_virtual ? ArgumentSourceKind::kGraphWorkspace
                                    : ArgumentSourceKind::kBinding;
  result.uid = tensor.uid;
  result.size = tensor.storage_size;
  result.alignment = tensor.is_virtual ? kGraphWorkspaceAlignment
                                       : tensor.alignment;
  result.workspace_offset = tensor.is_virtual ? tensor.workspace_offset : 0;
  return result;
}

ExpectedGraph parse_expected_graph(const Value& request) {
  const Value& options = request.at("build_options");
  require_exact_keys(options, {"heuristic_modes", "autotune"},
                     "Ascend build options");
  const auto& heuristic_modes = options.at("heuristic_modes").as_array();
  std::set<std::string> unique_modes;
  if (heuristic_modes.empty()) {
    artifact_failure("Ascend build request has no heuristic mode");
  }
  for (const Value& mode : heuristic_modes) {
    const std::string name = mode.as_string();
    if ((name != "A" && name != "FALLBACK") ||
        !unique_modes.insert(name).second) {
      artifact_failure("Ascend build request heuristic mode is invalid");
    }
  }

  ExpectedGraph result;
  result.autotune_requested = options.at("autotune").as_bool();
  const Value& graph = request.at("graph");
  const auto& raw_tensors = graph.at("tensors").as_array();
  if (graph_nonnegative_size(graph.at("tensor_count"),
                             std::numeric_limits<std::size_t>::max(),
                             "Graph tensor count") != raw_tensors.size() ||
      raw_tensors.empty()) {
    artifact_failure("Graph tensor count is invalid");
  }
  std::map<std::int64_t, ExpectedTensor> tensors;
  for (const Value& value : raw_tensors) {
    ExpectedTensor tensor = parse_graph_tensor(value);
    if (!tensors.emplace(tensor.uid, std::move(tensor)).second) {
      artifact_failure("Graph tensor UID is duplicated");
    }
  }

  bool has_external_binding = false;
  for (const auto& [uid, tensor] : tensors) {
    (void)uid;
    if (!tensor.is_virtual) {
      has_external_binding = true;
    }
  }

  const auto& raw_nodes = graph.at("nodes").as_array();
  const std::size_t node_count = graph_positive_size(
      graph.at("node_count"), kMaximumGraphNodes, "Graph node count");
  if (node_count != raw_nodes.size()) {
    artifact_failure("Graph node count differs from its node array");
  }
  result.node_count = node_count;
  struct ParsedNode {
    std::size_t node_id = 0;
    std::string operation;
    KernelFamily kernel_family = KernelFamily::kBinary;
    std::int64_t pointwise_mode = 0;
    std::vector<const ExpectedTensor*> inputs;
    const ExpectedTensor* output = nullptr;
    const ExpectedTensor* second_output = nullptr;
    const ExpectedTensor* third_output = nullptr;
    const ExpectedTensor* fourth_output = nullptr;
    const ExpectedTensor* fifth_output = nullptr;
    std::int32_t n_elements = 0;
    double alpha = 1.0;
    double negative_slope = 0.0;
    double lower_clip = 0.0;
    double upper_clip = 0.0;
    bool has_upper_clip = false;
    double swish_beta = 1.0;
    double elu_alpha = 1.0;
    double softplus_beta = 1.0;
    LayoutMapping layout;
    std::int64_t reduction_axis = 0;
    std::int64_t reduction_keep_dimensions = 0;
    std::int64_t reduction_outer = 0;
    std::int64_t reduction_size = 0;
    std::int64_t reduction_inner = 0;
    std::int64_t matmul_batch = 0;
    std::int64_t matmul_m = 0;
    std::int64_t matmul_n = 0;
    std::int64_t matmul_k = 0;
    std::array<std::int64_t, kMaximumMatmulBatchRank>
        matmul_batch_dimensions{};
    std::array<std::int64_t, kMaximumMatmulBatchRank>
        matmul_a_batch_strides{};
    std::array<std::int64_t, kMaximumMatmulBatchRank>
        matmul_b_batch_strides{};
    std::array<std::int64_t, kMaximumMatmulBatchRank>
        matmul_output_batch_strides{};
    std::int64_t matmul_a_stride_m = 0;
    std::int64_t matmul_a_stride_k = 0;
    std::int64_t matmul_b_stride_k = 0;
    std::int64_t matmul_b_stride_n = 0;
    std::int64_t matmul_output_stride_m = 0;
    std::int64_t matmul_output_stride_n = 0;
    std::int64_t convolution_spatial_rank = 0;
    std::int64_t convolution_groups = 0;
    std::int64_t convolution_input_channels = 0;
    std::int64_t convolution_output_channels = 0;
    std::int64_t convolution_channels_per_group = 0;
    std::array<std::int64_t, 5> convolution_input_dimensions{};
    std::array<std::int64_t, 5> convolution_input_strides{};
    std::array<std::int64_t, 5> convolution_filter_dimensions{};
    std::array<std::int64_t, 5> convolution_filter_strides{};
    std::array<std::int64_t, 5> convolution_output_dimensions{};
    std::array<std::int64_t, 5> convolution_output_strides{};
    std::array<std::int64_t, 3> convolution_pre_padding{};
    std::array<std::int64_t, 3> convolution_post_padding{};
    std::array<std::int64_t, 3> convolution_stride{};
    std::array<std::int64_t, 3> convolution_dilation{};
    std::int64_t batchnorm_rank = 0;
    std::int64_t batchnorm_batch = 0;
    std::int64_t batchnorm_channels = 0;
    std::int64_t batchnorm_spatial = 0;
    std::int64_t batchnorm_reduction_elements = 0;
    double batchnorm_epsilon = 0.0;
    double batchnorm_momentum = 0.0;
    std::array<std::int64_t, kMaximumRank> batchnorm_dimensions{};
    std::array<std::int64_t, kMaximumRank> batchnorm_x_strides{};
    std::array<std::int64_t, kMaximumRank> batchnorm_y_strides{};
    std::int64_t rmsnorm_rows = 0;
    std::int64_t rmsnorm_normalized_elements = 0;
    double rmsnorm_epsilon = 0.0;
    std::int64_t layernorm_rows = 0;
    std::int64_t layernorm_normalized_elements = 0;
    double layernorm_epsilon = 0.0;
    bool convolution_im2col = false;
  };
  std::vector<ParsedNode> parsed_nodes;
  parsed_nodes.reserve(node_count);
  std::set<std::size_t> node_ids;
  std::map<std::int64_t, std::size_t> producer_positions;
  bool has_external_output = false;
  for (std::size_t position = 0; position < raw_nodes.size(); ++position) {
    const Value& node = raw_nodes[position];
    const std::size_t node_id = graph_nonnegative_size(
        node.at("id"), node_count - 1, "Graph node ID");
    if (!node_ids.insert(node_id).second) {
      artifact_failure("Graph pointwise node identity is invalid");
    }
    const std::string operation = node.at("type").as_string();
    const KernelFamily kernel_family = kernel_family_for_operation(operation);
    const UnaryPointwiseContract* unary_contract =
        unary_pointwise_contract(operation);
    const bool unary = kernel_family == KernelFamily::kUnary;
    const bool ternary = kernel_family == KernelFamily::kTernary;
    const bool layout = kernel_family == KernelFamily::kLayout;
    const bool reduction = kernel_family == KernelFamily::kReduction;
    const bool matmul = kernel_family == KernelFamily::kMatMul;
    const bool convolution_fprop =
        kernel_family == KernelFamily::kConvolutionFprop;
    const bool batchnorm_training =
        kernel_family == KernelFamily::kBatchNorm;
    const bool batchnorm =
        kernel_family == KernelFamily::kBatchNormInference;
    const bool rmsnorm = kernel_family == KernelFamily::kRmsNorm;
    const bool layernorm = kernel_family == KernelFamily::kLayerNorm;
    const BinaryPointwiseContract* binary_contract = nullptr;
    if (kernel_family == KernelFamily::kBinary) {
      binary_contract = &binary_pointwise_contract(operation);
    }
    const std::int64_t pointwise_mode =
        unary ? unary_contract->pointwise_mode
              : ternary ? 41
                        : layout || matmul || convolution_fprop ||
                                  batchnorm_training || batchnorm || rmsnorm ||
                                  layernorm
                              ? 0
                              : reduction ? reduction_mode(operation)
                                          : binary_contract->pointwise_mode;
    const auto& inputs = node.at("inputs").as_array();
    const auto& outputs = node.at("outputs").as_array();
    const std::size_t expected_input_count =
        batchnorm_training || batchnorm ? 5U
                  : rmsnorm || layernorm ? 3U
                  : unary || layout || reduction ? 1U : ternary ? 3U : 2U;
    const std::size_t expected_output_count =
        batchnorm_training ? 5U : layernorm ? 3U : rmsnorm ? 2U : 1U;
    if (inputs.size() != expected_input_count ||
        outputs.size() != expected_output_count) {
      artifact_failure("Graph pointwise arity is invalid");
    }
    std::vector<const ExpectedTensor*> input_tensors;
    input_tensors.push_back(&parse_graph_port(
        inputs[0], batchnorm_training || batchnorm || rmsnorm || layernorm ? "x"
                             : convolution_fprop ? "input"
                             : matmul ? "a"
                             : unary || layout || reduction ? "input"
                                                            : ternary ? "a"
                                                                      : "left",
        tensors));
    if (batchnorm_training) {
      input_tensors.push_back(&parse_graph_port(inputs[1], "scale", tensors));
      input_tensors.push_back(&parse_graph_port(inputs[2], "bias", tensors));
      input_tensors.push_back(&parse_graph_port(
          inputs[3], "previous_running_mean", tensors));
      input_tensors.push_back(&parse_graph_port(
          inputs[4], "previous_running_variance", tensors));
    } else if (batchnorm) {
      input_tensors.push_back(&parse_graph_port(inputs[1], "mean", tensors));
      input_tensors.push_back(
          &parse_graph_port(inputs[2], "inv_variance", tensors));
      input_tensors.push_back(&parse_graph_port(inputs[3], "scale", tensors));
      input_tensors.push_back(&parse_graph_port(inputs[4], "bias", tensors));
    } else if (rmsnorm || layernorm) {
      input_tensors.push_back(&parse_graph_port(inputs[1], "scale", tensors));
      input_tensors.push_back(&parse_graph_port(inputs[2], "bias", tensors));
    } else if (ternary) {
      input_tensors.push_back(&parse_graph_port(inputs[1], "b", tensors));
      input_tensors.push_back(&parse_graph_port(inputs[2], "t", tensors));
    } else if (!unary && !layout && !reduction) {
      input_tensors.push_back(&parse_graph_port(
          inputs[1], convolution_fprop ? "filter" : matmul ? "b" : "right",
          tensors));
    }
    const ExpectedTensor& output = parse_graph_port(
        outputs[0], batchnorm_training || batchnorm || rmsnorm || layernorm
                        ? "y"
                        : "output",
        tensors);
    const ExpectedTensor* second_output =
        batchnorm_training
            ? &parse_graph_port(outputs[1], "mean", tensors)
            : layernorm ? &parse_graph_port(outputs[1], "mean", tensors)
                  : rmsnorm ? &parse_graph_port(outputs[1], "inv_variance",
                                                tensors)
                            : nullptr;
    const ExpectedTensor* third_output =
        batchnorm_training
            ? &parse_graph_port(outputs[2], "inv_variance", tensors)
            : layernorm
            ? &parse_graph_port(outputs[2], "inv_variance", tensors)
            : nullptr;
    const ExpectedTensor* fourth_output =
        batchnorm_training
            ? &parse_graph_port(outputs[3], "next_running_mean", tensors)
            : nullptr;
    const ExpectedTensor* fifth_output =
        batchnorm_training
            ? &parse_graph_port(outputs[4], "next_running_variance", tensors)
            : nullptr;
    const bool logical = is_logical_operation(operation);
    const bool comparison = is_comparison_operation(operation);
    if (batchnorm_training) {
      std::set<std::int64_t> port_uids;
      for (const ExpectedTensor* tensor : input_tensors) {
        port_uids.insert(tensor->uid);
      }
      for (const ExpectedTensor* tensor :
           std::array<const ExpectedTensor*, 5>{&output, second_output,
                                                 third_output, fourth_output,
                                                 fifth_output}) {
        port_uids.insert(tensor->uid);
      }
      if (port_uids.size() != 10) {
        artifact_failure("Graph batchnorm training tensor UIDs must be distinct");
      }
      if (input_tensors[0]->data_type != output.data_type ||
          input_tensors[1]->data_type != output.data_type ||
          input_tensors[2]->data_type != output.data_type ||
          output.data_type == "boolean") {
        artifact_failure(
            "Graph batchnorm training X/scale/bias/Y require matching "
            "floating storage data types");
      }
      for (const ExpectedTensor* statistic :
           std::array<const ExpectedTensor*, 6>{
               input_tensors[3], input_tensors[4], second_output,
               third_output, fourth_output, fifth_output}) {
        if (statistic->data_type != "float32") {
          artifact_failure(
              "Graph batchnorm training statistics must use float32 storage");
        }
      }
      if (node.at("compute_data_type").as_string() != "float32") {
        artifact_failure("Graph batchnorm training requires float32 compute");
      }
    } else if (layernorm) {
      std::set<std::int64_t> port_uids;
      for (const ExpectedTensor* tensor : input_tensors) {
        port_uids.insert(tensor->uid);
      }
      port_uids.insert(output.uid);
      port_uids.insert(second_output->uid);
      port_uids.insert(third_output->uid);
      if (port_uids.size() != 6) {
        artifact_failure("Graph layernorm tensor UIDs must be distinct");
      }
      if (input_tensors[0]->data_type != output.data_type ||
          input_tensors[1]->data_type != output.data_type ||
          input_tensors[2]->data_type != output.data_type ||
          output.data_type == "boolean") {
        artifact_failure(
            "Graph layernorm X/scale/bias/Y require matching floating storage "
            "data types");
      }
      if (second_output->data_type != "float32" ||
          third_output->data_type != "float32") {
        artifact_failure("Graph layernorm statistics must use float32 storage");
      }
      if (node.at("compute_data_type").as_string() != "float32") {
        artifact_failure("Graph layernorm requires float32 compute");
      }
    } else if (rmsnorm) {
      std::set<std::int64_t> port_uids;
      for (const ExpectedTensor* tensor : input_tensors) {
        port_uids.insert(tensor->uid);
      }
      port_uids.insert(output.uid);
      port_uids.insert(second_output->uid);
      if (port_uids.size() != 5) {
        artifact_failure("Graph rmsnorm tensor UIDs must be distinct");
      }
      if (input_tensors[0]->data_type != output.data_type ||
          input_tensors[1]->data_type != output.data_type ||
          input_tensors[2]->data_type != output.data_type ||
          output.data_type == "boolean") {
        artifact_failure(
            "Graph rmsnorm X/scale/bias/Y require matching floating storage "
            "data types");
      }
      if (second_output->data_type != "float32") {
        artifact_failure(
            "Graph rmsnorm inverse variance must use float32 storage");
      }
      if (node.at("compute_data_type").as_string() != "float32") {
        artifact_failure("Graph rmsnorm requires float32 compute");
      }
    } else if (batchnorm) {
      std::set<std::int64_t> port_uids;
      for (const ExpectedTensor* tensor : input_tensors) {
        port_uids.insert(tensor->uid);
      }
      port_uids.insert(output.uid);
      if (port_uids.size() != 6) {
        artifact_failure("Graph batchnorm tensor UIDs must be distinct");
      }
      if (input_tensors[0]->data_type != output.data_type ||
          output.data_type == "boolean") {
        artifact_failure(
            "Graph batchnorm X/Y require matching floating storage data types");
      }
      for (std::size_t index = 1; index < input_tensors.size(); ++index) {
        if (input_tensors[index]->data_type != "float32" ||
            input_tensors[index]->is_virtual) {
          artifact_failure(
              "Graph batchnorm parameters must be external float32 tensors");
        }
      }
      if (node.at("compute_data_type").as_string() != "float32") {
        artifact_failure("Graph batchnorm requires float32 compute");
      }
    } else if (convolution_fprop) {
      if (input_tensors[0]->uid == input_tensors[1]->uid ||
          input_tensors[0]->uid == output.uid ||
          input_tensors[1]->uid == output.uid) {
        artifact_failure("Graph convolution tensor UIDs must be distinct");
      }
      if (input_tensors[0]->data_type != input_tensors[1]->data_type ||
          input_tensors[0]->data_type != output.data_type ||
          output.data_type == "boolean") {
        artifact_failure(
            "Graph convolution requires matching floating storage data types");
      }
      if (node.at("compute_data_type").as_string() != "float32") {
        artifact_failure("Graph convolution_fprop requires float32 compute");
      }
    } else if (matmul) {
      if (input_tensors[0]->uid == input_tensors[1]->uid ||
          input_tensors[0]->uid == output.uid ||
          input_tensors[1]->uid == output.uid) {
        artifact_failure("Graph matmul tensor UIDs must be distinct");
      }
      if (input_tensors[0]->data_type != input_tensors[1]->data_type ||
          input_tensors[0]->data_type != output.data_type ||
          output.data_type == "boolean") {
        artifact_failure(
            "Graph matmul requires matching floating storage data types");
      }
      if (node.at("compute_data_type").as_string() != "float32") {
        artifact_failure("Graph matmul requires float32 compute");
      }
    } else if (reduction) {
      if (input_tensors[0]->data_type != output.data_type ||
          output.data_type == "boolean") {
        artifact_failure(
            "Graph reduction requires matching floating storage data types");
      }
      if (node.at("compute_data_type").as_string() != "float32") {
        artifact_failure("Graph reduction requires float32 compute");
      }
    } else if (layout) {
      if (input_tensors[0]->data_type != output.data_type) {
        artifact_failure(
            "Graph layout input/output storage data types differ");
      }
      const std::string compute_data_type =
          node.at("compute_data_type").as_string();
      if (compute_data_type != "float32" && compute_data_type != "float16" &&
          compute_data_type != "bfloat16" &&
          compute_data_type != "boolean") {
        artifact_failure("Graph layout compute data type is unsupported");
      }
    } else if (ternary) {
      if (input_tensors[0]->data_type == "boolean" ||
          input_tensors[1]->data_type != input_tensors[0]->data_type ||
          output.data_type != input_tensors[0]->data_type) {
        artifact_failure(
            "Graph binary_select requires matching floating A/B/output "
            "storage data types");
      }
      if (input_tensors[2]->data_type != "boolean") {
        artifact_failure(
            "Graph binary_select requires a BOOLEAN predicate storage data "
            "type");
      }
      if (node.at("compute_data_type").as_string() != "float32") {
        artifact_failure("Graph binary_select requires float32 compute");
      }
    } else if (comparison) {
      if (input_tensors[0]->data_type != input_tensors[1]->data_type ||
          input_tensors[0]->data_type == "boolean") {
        artifact_failure(
            "Graph comparison pointwise requires same floating input "
            "storage data types");
      }
      if (output.data_type != "boolean") {
        artifact_failure(
            "Graph comparison pointwise requires BOOLEAN output storage");
      }
      if (node.at("compute_data_type").as_string() != "boolean") {
        artifact_failure(
            "Graph comparison pointwise requires BOOLEAN compute data type");
      }
    } else if (logical) {
      if (std::any_of(input_tensors.begin(), input_tensors.end(),
                      [&output](const ExpectedTensor* input_tensor) {
                        return input_tensor->data_type != output.data_type;
                      })) {
        artifact_failure("Graph pointwise storage data types do not match");
      }
      if (output.data_type != "boolean") {
        artifact_failure(
            "Graph logical pointwise requires BOOLEAN storage data types");
      }
      if (node.at("compute_data_type").as_string() != "boolean") {
        artifact_failure(
            "Graph logical pointwise requires BOOLEAN compute data type");
      }
    } else {
      if (std::any_of(input_tensors.begin(), input_tensors.end(),
                      [&output](const ExpectedTensor* input_tensor) {
                        return input_tensor->data_type != output.data_type;
                      })) {
        artifact_failure("Graph pointwise storage data types do not match");
      }
      if (output.data_type == "boolean") {
        artifact_failure(
            "Graph numeric pointwise storage data types must be floating");
      }
      if (node.at("compute_data_type").as_string() != "float32") {
        artifact_failure("Graph pointwise compute data type must be float32");
      }
    }
    const Value& attributes = node.at("attributes");
    std::int64_t reduction_axis = 0;
    std::int64_t reduction_keep_dimensions = 0;
    std::int64_t reduction_outer = 0;
    std::int64_t reduction_size = 0;
    std::int64_t reduction_inner = 0;
    std::int64_t matmul_batch = 0;
    std::int64_t matmul_m = 0;
    std::int64_t matmul_n = 0;
    std::int64_t matmul_k = 0;
    std::array<std::int64_t, kMaximumMatmulBatchRank>
        matmul_batch_dimensions{};
    std::array<std::int64_t, kMaximumMatmulBatchRank>
        matmul_a_batch_strides{};
    std::array<std::int64_t, kMaximumMatmulBatchRank>
        matmul_b_batch_strides{};
    std::array<std::int64_t, kMaximumMatmulBatchRank>
        matmul_output_batch_strides{};
    std::int64_t matmul_a_stride_m = 0;
    std::int64_t matmul_a_stride_k = 0;
    std::int64_t matmul_b_stride_k = 0;
    std::int64_t matmul_b_stride_n = 0;
    std::int64_t matmul_output_stride_m = 0;
    std::int64_t matmul_output_stride_n = 0;
    std::int64_t convolution_spatial_rank = 0;
    std::int64_t convolution_groups = 0;
    std::int64_t convolution_input_channels = 0;
    std::int64_t convolution_output_channels = 0;
    std::int64_t convolution_channels_per_group = 0;
    std::array<std::int64_t, 5> convolution_input_dimensions{};
    std::array<std::int64_t, 5> convolution_input_strides{};
    std::array<std::int64_t, 5> convolution_filter_dimensions{};
    std::array<std::int64_t, 5> convolution_filter_strides{};
    std::array<std::int64_t, 5> convolution_output_dimensions{};
    std::array<std::int64_t, 5> convolution_output_strides{};
    std::array<std::int64_t, 3> convolution_pre_padding{};
    std::array<std::int64_t, 3> convolution_post_padding{};
    std::array<std::int64_t, 3> convolution_stride{};
    std::array<std::int64_t, 3> convolution_dilation{};
    std::int64_t batchnorm_rank = 0;
    std::int64_t batchnorm_batch = 0;
    std::int64_t batchnorm_channels = 0;
    std::int64_t batchnorm_spatial = 0;
    std::int64_t batchnorm_reduction_elements = 0;
    double batchnorm_epsilon = 0.0;
    double batchnorm_momentum = 0.0;
    std::array<std::int64_t, kMaximumRank> batchnorm_dimensions{};
    std::array<std::int64_t, kMaximumRank> batchnorm_x_strides{};
    std::array<std::int64_t, kMaximumRank> batchnorm_y_strides{};
    std::int64_t rmsnorm_rows = 0;
    std::int64_t rmsnorm_normalized_elements = 0;
    double rmsnorm_epsilon = 0.0;
    std::int64_t layernorm_rows = 0;
    std::int64_t layernorm_normalized_elements = 0;
    double layernorm_epsilon = 0.0;
    std::vector<std::int64_t> expected_dimensions;
    if (convolution_fprop) {
      require_default_precision_keys(attributes,
                         {"spatial_rank", "groups", "n_outputs",
                          "pre_padding", "post_padding", "stride",
                          "dilation"},
                         "Graph convolution_fprop attributes");
      convolution_spatial_rank = static_cast<std::int64_t>(graph_positive_size(
          attributes.at("spatial_rank"), 3, "Graph convolution spatial rank"));
      convolution_groups = static_cast<std::int64_t>(graph_positive_size(
          attributes.at("groups"),
          std::numeric_limits<std::int32_t>::max(),
          "Graph convolution groups"));
      const std::size_t spatial_rank =
          static_cast<std::size_t>(convolution_spatial_rank);
      const ExpectedTensor& input = *input_tensors[0];
      const ExpectedTensor& filter = *input_tensors[1];
      if (input.dimensions.size() != spatial_rank + 2 ||
          filter.dimensions.size() != spatial_rank + 2 ||
          output.dimensions.size() != spatial_rank + 2) {
        artifact_failure(
            "Graph convolution tensor ranks must equal spatial_rank + 2");
      }
      const std::vector<std::int64_t> pre_padding = graph_integer_array(
          attributes, "pre_padding", spatial_rank, 0,
          std::numeric_limits<std::int32_t>::max());
      const std::vector<std::int64_t> post_padding = graph_integer_array(
          attributes, "post_padding", spatial_rank, 0,
          std::numeric_limits<std::int32_t>::max());
      const std::vector<std::int64_t> stride = graph_integer_array(
          attributes, "stride", spatial_rank, 1,
          std::numeric_limits<std::int32_t>::max());
      const std::vector<std::int64_t> dilation = graph_integer_array(
          attributes, "dilation", spatial_rank, 1,
          std::numeric_limits<std::int32_t>::max());
      convolution_input_channels = input.dimensions[1];
      convolution_output_channels = filter.dimensions[0];
      if (convolution_input_channels % convolution_groups != 0 ||
          convolution_output_channels % convolution_groups != 0) {
        artifact_failure("Graph convolution group metadata is invalid");
      }
      convolution_channels_per_group =
          convolution_input_channels / convolution_groups;
      if (filter.dimensions[1] != convolution_channels_per_group) {
        artifact_failure("Graph convolution filter channels are invalid");
      }
      std::vector<std::int64_t> recomputed_output = {
          input.dimensions[0], convolution_output_channels};
      for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
        const std::int64_t effective_filter =
            dilation[axis] * (filter.dimensions[axis + 2] - 1) + 1;
        const std::int64_t numerator =
            input.dimensions[axis + 2] + pre_padding[axis] +
            post_padding[axis] - effective_filter;
        if (numerator < 0) {
          artifact_failure(
              "Graph convolution filter is larger than padded input");
        }
        recomputed_output.push_back(numerator / stride[axis] + 1);
      }
      if (output.dimensions != recomputed_output) {
        artifact_failure("Graph convolution output shape is inconsistent");
      }
      const std::size_t n_outputs = dense_element_count(output);
      if (graph_positive_size(attributes.at("n_outputs"),
                              std::numeric_limits<std::int32_t>::max(),
                              "Graph convolution output elements") !=
          n_outputs) {
        artifact_failure(
            "Graph convolution n_outputs disagrees with output tensor");
      }
      const std::size_t spatial_offset = 3 - spatial_rank;
      const auto canonicalize_tensor =
          [spatial_rank, spatial_offset](
              const ExpectedTensor& tensor,
              std::array<std::int64_t, 5>& dimensions,
              std::array<std::int64_t, 5>& strides) {
            dimensions.fill(1);
            strides.fill(0);
            dimensions[0] = tensor.dimensions[0];
            dimensions[1] = tensor.dimensions[1];
            strides[0] = tensor.strides[0];
            strides[1] = tensor.strides[1];
            for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
              dimensions[2 + spatial_offset + axis] =
                  tensor.dimensions[2 + axis];
              strides[2 + spatial_offset + axis] = tensor.strides[2 + axis];
            }
          };
      canonicalize_tensor(input,
                          convolution_input_dimensions,
                          convolution_input_strides);
      canonicalize_tensor(filter,
                          convolution_filter_dimensions,
                          convolution_filter_strides);
      canonicalize_tensor(output,
                          convolution_output_dimensions,
                          convolution_output_strides);
      convolution_pre_padding.fill(0);
      convolution_post_padding.fill(0);
      convolution_stride.fill(1);
      convolution_dilation.fill(1);
      for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
        convolution_pre_padding[spatial_offset + axis] = pre_padding[axis];
        convolution_post_padding[spatial_offset + axis] = post_padding[axis];
        convolution_stride[spatial_offset + axis] = stride[axis];
        convolution_dilation[spatial_offset + axis] = dilation[axis];
      }
      expected_dimensions = output.dimensions;
    } else if (matmul) {
      require_default_precision_keys(attributes, {"batch", "m", "n", "k"},
                         "Graph matmul attributes");
      const ExpectedTensor& a = *input_tensors[0];
      const ExpectedTensor& b = *input_tensors[1];
      if (a.dimensions.size() < 2 || a.dimensions.size() > kMaximumRank ||
          b.dimensions.size() < 2 || b.dimensions.size() > kMaximumRank ||
          output.dimensions.size() < 2 ||
          output.dimensions.size() > kMaximumRank ||
          a.dimensions.back() != b.dimensions[b.dimensions.size() - 2]) {
        artifact_failure("Graph matmul ranks or contraction are invalid");
      }
      matmul_m = a.dimensions[a.dimensions.size() - 2];
      matmul_k = a.dimensions.back();
      matmul_n = b.dimensions.back();
      const std::size_t a_batch_rank = a.dimensions.size() - 2;
      const std::size_t b_batch_rank = b.dimensions.size() - 2;
      const std::size_t batch_rank = std::max(a_batch_rank, b_batch_rank);
      if (batch_rank > kMaximumMatmulBatchRank) {
        artifact_failure("Graph matmul batch rank is invalid");
      }
      std::vector<std::int64_t> batch_dimensions(batch_rank, 1);
      matmul_batch = 1;
      const std::size_t leading = kMaximumMatmulBatchRank - batch_rank;
      for (std::size_t axis = 0; axis < leading; ++axis) {
        matmul_batch_dimensions[axis] = 1;
      }
      for (std::size_t axis = 0; axis < batch_rank; ++axis) {
        const bool a_has_axis = axis + a_batch_rank >= batch_rank;
        const bool b_has_axis = axis + b_batch_rank >= batch_rank;
        const std::size_t a_axis =
            a_has_axis ? axis + a_batch_rank - batch_rank : 0;
        const std::size_t b_axis =
            b_has_axis ? axis + b_batch_rank - batch_rank : 0;
        const std::int64_t a_dimension =
            a_has_axis ? a.dimensions[a_axis] : 1;
        const std::int64_t b_dimension =
            b_has_axis ? b.dimensions[b_axis] : 1;
        if (a_dimension != b_dimension && a_dimension != 1 &&
            b_dimension != 1) {
          artifact_failure("Graph matmul batch dimensions cannot broadcast");
        }
        const std::int64_t dimension = std::max(a_dimension, b_dimension);
        batch_dimensions[axis] = dimension;
        matmul_batch_dimensions[leading + axis] = dimension;
        matmul_a_batch_strides[leading + axis] =
            !a_has_axis || a_dimension == 1 ? 0 : a.strides[a_axis];
        matmul_b_batch_strides[leading + axis] =
            !b_has_axis || b_dimension == 1 ? 0 : b.strides[b_axis];
        if (matmul_batch >
            std::numeric_limits<std::int64_t>::max() / dimension) {
          artifact_failure("Graph matmul batch extent overflows");
        }
        matmul_batch *= dimension;
      }
      expected_dimensions = batch_dimensions;
      expected_dimensions.push_back(matmul_m);
      expected_dimensions.push_back(matmul_n);
      if (output.dimensions != expected_dimensions) {
        artifact_failure("Graph matmul output shape is inconsistent");
      }
      for (std::size_t axis = 0; axis < batch_rank; ++axis) {
        matmul_output_batch_strides[leading + axis] = output.strides[axis];
      }
      matmul_a_stride_m = a.strides[a.strides.size() - 2];
      matmul_a_stride_k = a.strides.back();
      matmul_b_stride_k = b.strides[b.strides.size() - 2];
      matmul_b_stride_n = b.strides.back();
      matmul_output_stride_m = output.strides[output.strides.size() - 2];
      matmul_output_stride_n = output.strides.back();
      if (attributes.at("batch").as_int() != matmul_batch ||
          attributes.at("m").as_int() != matmul_m ||
          attributes.at("n").as_int() != matmul_n ||
          attributes.at("k").as_int() != matmul_k) {
        artifact_failure("Graph matmul decomposition differs from tensors");
      }
    } else if (batchnorm_training) {
      require_exact_keys(
          attributes,
          {"n_elements", "batch", "channels", "spatial", "rank",
           "epsilon", "momentum", "dimensions", "x_strides",
           "y_strides"},
          "Graph batchnorm training attributes");
      const ExpectedTensor& x = *input_tensors[0];
      batchnorm_rank = static_cast<std::int64_t>(x.dimensions.size());
      if (batchnorm_rank < 2 ||
          batchnorm_rank > static_cast<std::int64_t>(kMaximumRank) ||
          output.dimensions != x.dimensions) {
        artifact_failure("Graph batchnorm training X/Y shape is invalid");
      }
      batchnorm_batch = x.dimensions[0];
      batchnorm_channels = x.dimensions[1];
      batchnorm_spatial = 1;
      for (std::size_t axis = 2; axis < x.dimensions.size(); ++axis) {
        if (batchnorm_spatial >
            std::numeric_limits<std::int64_t>::max() / x.dimensions[axis]) {
          artifact_failure("Graph batchnorm training spatial extent overflows");
        }
        batchnorm_spatial *= x.dimensions[axis];
      }
      if (batchnorm_batch >
          std::numeric_limits<std::int64_t>::max() / batchnorm_spatial) {
        artifact_failure("Graph batchnorm training reduction extent overflows");
      }
      batchnorm_reduction_elements = batchnorm_batch * batchnorm_spatial;
      for (std::size_t index = 1; index < input_tensors.size(); ++index) {
        if (dense_element_count(*input_tensors[index]) !=
                static_cast<std::size_t>(batchnorm_channels) ||
            !is_row_major_tensor(*input_tensors[index])) {
          artifact_failure(
              "Graph batchnorm training parameters/statistics must be "
              "contiguous with exactly channels elements");
        }
      }
      for (const ExpectedTensor* statistic :
           std::array<const ExpectedTensor*, 4>{second_output, third_output,
                                                 fourth_output, fifth_output}) {
        if (dense_element_count(*statistic) !=
                static_cast<std::size_t>(batchnorm_channels) ||
            !is_row_major_tensor(*statistic)) {
          artifact_failure(
              "Graph batchnorm training output statistics must be "
              "contiguous with exactly channels elements");
        }
      }
      const auto exact_attribute_array =
          [&attributes](std::string_view name,
                        const std::vector<std::int64_t>& expected) {
            const auto& raw = attributes.at(name).as_array();
            if (raw.size() != expected.size()) {
              artifact_failure(
                  "Graph batchnorm training tensor metadata rank differs");
            }
            for (std::size_t index = 0; index < raw.size(); ++index) {
              if (raw[index].as_int() != expected[index]) {
                artifact_failure(
                    "Graph batchnorm training tensor metadata differs");
              }
            }
          };
      exact_attribute_array("dimensions", x.dimensions);
      exact_attribute_array("x_strides", x.strides);
      exact_attribute_array("y_strides", output.strides);
      const std::size_t dense_elements = dense_element_count(x);
      if (attributes.at("n_elements").as_int() !=
              static_cast<std::int64_t>(dense_elements) ||
          attributes.at("batch").as_int() != batchnorm_batch ||
          attributes.at("channels").as_int() != batchnorm_channels ||
          attributes.at("spatial").as_int() != batchnorm_spatial ||
          attributes.at("rank").as_int() != batchnorm_rank) {
        artifact_failure(
            "Graph batchnorm training decomposition differs from shape");
      }
      batchnorm_epsilon = attributes.at("epsilon").as_double();
      batchnorm_momentum = attributes.at("momentum").as_double();
      if (!std::isfinite(batchnorm_epsilon) || batchnorm_epsilon <= 0.0 ||
          std::abs(batchnorm_epsilon) > std::numeric_limits<float>::max() ||
          !std::isfinite(batchnorm_momentum) || batchnorm_momentum < 0.0 ||
          batchnorm_momentum > 1.0 ||
          std::abs(batchnorm_momentum) > std::numeric_limits<float>::max()) {
        artifact_failure(
            "Graph batchnorm training epsilon or momentum is invalid");
      }
      batchnorm_dimensions = padded_dimensions(x);
      batchnorm_x_strides =
          effective_strides(x, x.dimensions.size(), false);
      batchnorm_y_strides =
          effective_strides(output, output.dimensions.size(), false);
      expected_dimensions = x.dimensions;
    } else if (layernorm) {
      require_normalization_attributes(attributes,
                                       "Graph layernorm attributes");
      const ExpectedTensor& x = *input_tensors[0];
      const ExpectedTensor& scale = *input_tensors[1];
      const ExpectedTensor& bias = *input_tensors[2];
      if (x.dimensions.empty() || x.dimensions.size() > kMaximumRank ||
          output.dimensions != x.dimensions || !is_row_major_tensor(x) ||
          !is_row_major_tensor(output)) {
        artifact_failure("Graph layernorm X/Y shape or layout is invalid");
      }
      if (scale.dimensions.empty() ||
          scale.dimensions.size() > x.dimensions.size() ||
          scale.dimensions != bias.dimensions || !is_row_major_tensor(scale) ||
          !is_row_major_tensor(bias)) {
        artifact_failure(
            "Graph layernorm scale/bias suffix metadata is invalid");
      }
      const std::size_t leading =
          x.dimensions.size() - scale.dimensions.size();
      bool normalized_suffix = false;
      std::int64_t normalized_elements = 1;
      std::vector<std::int64_t> statistic_dimensions = x.dimensions;
      for (std::size_t axis = 0; axis < x.dimensions.size(); ++axis) {
        const std::int64_t scale_dimension =
            axis < leading ? 1 : scale.dimensions[axis - leading];
        if (scale_dimension != 1) {
          if (scale_dimension != x.dimensions[axis]) {
            artifact_failure("Graph layernorm scale shape differs from X");
          }
          normalized_suffix = true;
        } else if (normalized_suffix && x.dimensions[axis] != 1) {
          artifact_failure(
              "Graph layernorm scale does not describe a contiguous suffix");
        }
        if (normalized_suffix) {
          if (normalized_elements >
              std::numeric_limits<std::int64_t>::max() /
                  x.dimensions[axis]) {
            artifact_failure("Graph layernorm normalized extent overflows");
          }
          normalized_elements *= x.dimensions[axis];
          statistic_dimensions[axis] = 1;
        }
      }
      if (!normalized_suffix ||
          dense_element_count(scale) !=
              static_cast<std::size_t>(normalized_elements) ||
          dense_element_count(bias) !=
              static_cast<std::size_t>(normalized_elements) ||
          second_output->dimensions != statistic_dimensions ||
          third_output->dimensions != statistic_dimensions ||
          !is_row_major_tensor(*second_output) ||
          !is_row_major_tensor(*third_output)) {
        artifact_failure("Graph layernorm parameter/statistic metadata is invalid");
      }
      const std::int64_t rows = static_cast<std::int64_t>(
          dense_element_count(x) /
          static_cast<std::size_t>(normalized_elements));
      const double epsilon = attributes.at("epsilon").as_double();
      if (attributes.at("rows").as_int() != rows ||
          attributes.at("normalized_elements").as_int() !=
              normalized_elements ||
          !std::isfinite(epsilon) || epsilon <= 0.0 ||
          std::abs(epsilon) > std::numeric_limits<float>::max()) {
        artifact_failure("Graph layernorm decomposition is invalid");
      }
      layernorm_rows = rows;
      layernorm_normalized_elements = normalized_elements;
      layernorm_epsilon = epsilon;
      expected_dimensions = x.dimensions;
    } else if (rmsnorm) {
      require_normalization_attributes(attributes,
                                       "Graph rmsnorm attributes");
      const ExpectedTensor& x = *input_tensors[0];
      const ExpectedTensor& scale = *input_tensors[1];
      const ExpectedTensor& bias = *input_tensors[2];
      if (x.dimensions.empty() || x.dimensions.size() > kMaximumRank ||
          output.dimensions != x.dimensions || !is_row_major_tensor(x) ||
          !is_row_major_tensor(output)) {
        artifact_failure("Graph rmsnorm X/Y shape or layout is invalid");
      }
      if (scale.dimensions.empty() ||
          scale.dimensions.size() > x.dimensions.size() ||
          scale.dimensions != bias.dimensions || !is_row_major_tensor(scale) ||
          !is_row_major_tensor(bias)) {
        artifact_failure(
            "Graph rmsnorm scale/bias suffix metadata is invalid");
      }
      const std::size_t leading =
          x.dimensions.size() - scale.dimensions.size();
      bool normalized_suffix = false;
      std::int64_t normalized_elements = 1;
      std::vector<std::int64_t> statistic_dimensions = x.dimensions;
      for (std::size_t axis = 0; axis < x.dimensions.size(); ++axis) {
        const std::int64_t scale_dimension =
            axis < leading ? 1 : scale.dimensions[axis - leading];
        if (scale_dimension != 1) {
          if (scale_dimension != x.dimensions[axis]) {
            artifact_failure("Graph rmsnorm scale shape differs from X");
          }
          normalized_suffix = true;
        } else if (normalized_suffix && x.dimensions[axis] != 1) {
          artifact_failure(
              "Graph rmsnorm scale does not describe a contiguous suffix");
        }
        if (normalized_suffix) {
          if (normalized_elements >
              std::numeric_limits<std::int64_t>::max() /
                  x.dimensions[axis]) {
            artifact_failure("Graph rmsnorm normalized extent overflows");
          }
          normalized_elements *= x.dimensions[axis];
          statistic_dimensions[axis] = 1;
        }
      }
      if (!normalized_suffix ||
          dense_element_count(scale) !=
              static_cast<std::size_t>(normalized_elements) ||
          dense_element_count(bias) !=
              static_cast<std::size_t>(normalized_elements) ||
          second_output->dimensions != statistic_dimensions ||
          !is_row_major_tensor(*second_output)) {
        artifact_failure("Graph rmsnorm parameter/statistic metadata is invalid");
      }
      const std::int64_t rows = static_cast<std::int64_t>(
          dense_element_count(x) /
          static_cast<std::size_t>(normalized_elements));
      const double epsilon = attributes.at("epsilon").as_double();
      if (attributes.at("rows").as_int() != rows ||
          attributes.at("normalized_elements").as_int() !=
              normalized_elements ||
          !std::isfinite(epsilon) || epsilon <= 0.0 ||
          std::abs(epsilon) > std::numeric_limits<float>::max()) {
        artifact_failure("Graph rmsnorm decomposition is invalid");
      }
      rmsnorm_rows = rows;
      rmsnorm_normalized_elements = normalized_elements;
      rmsnorm_epsilon = epsilon;
      expected_dimensions = x.dimensions;
    } else if (batchnorm) {
      require_exact_keys(attributes,
                         {"n_elements", "channels", "spatial", "rank",
                          "dimensions", "x_strides", "y_strides"},
                         "Graph batchnorm attributes");
      const ExpectedTensor& x = *input_tensors[0];
      batchnorm_rank = static_cast<std::int64_t>(x.dimensions.size());
      if (batchnorm_rank < 2 || batchnorm_rank >
                                    static_cast<std::int64_t>(kMaximumRank) ||
          output.dimensions != x.dimensions) {
        artifact_failure("Graph batchnorm X/Y shape is invalid");
      }
      batchnorm_channels = x.dimensions[1];
      batchnorm_spatial = 1;
      for (std::size_t axis = 2; axis < x.dimensions.size(); ++axis) {
        batchnorm_spatial *= x.dimensions[axis];
      }
      for (std::size_t index = 1; index < input_tensors.size(); ++index) {
        if (dense_element_count(*input_tensors[index]) !=
            static_cast<std::size_t>(batchnorm_channels)) {
          artifact_failure(
              "Graph batchnorm parameters must contain exactly channels "
              "elements");
        }
        if (!is_row_major_tensor(*input_tensors[index])) {
          artifact_failure("Graph batchnorm parameters must be contiguous");
        }
      }
      const auto exact_attribute_array =
          [&attributes](std::string_view name,
                        const std::vector<std::int64_t>& expected) {
            const auto& raw = attributes.at(name).as_array();
            if (raw.size() != expected.size()) {
              artifact_failure("Graph batchnorm tensor metadata rank differs");
            }
            for (std::size_t index = 0; index < raw.size(); ++index) {
              if (raw[index].as_int() != expected[index]) {
                artifact_failure(
                    "Graph batchnorm tensor metadata differs from tensor");
              }
            }
          };
      exact_attribute_array("dimensions", x.dimensions);
      exact_attribute_array("x_strides", x.strides);
      exact_attribute_array("y_strides", output.strides);
      if (attributes.at("rank").as_int() != batchnorm_rank ||
          attributes.at("channels").as_int() != batchnorm_channels ||
          attributes.at("spatial").as_int() != batchnorm_spatial) {
        artifact_failure("Graph batchnorm decomposition differs from shape");
      }
      batchnorm_dimensions = padded_dimensions(x);
      batchnorm_x_strides =
          effective_strides(x, x.dimensions.size(), false);
      batchnorm_y_strides =
          effective_strides(output, output.dimensions.size(), false);
      expected_dimensions = x.dimensions;
    } else if (reduction) {
      require_exact_keys(attributes,
                         {"mode", "axis", "keep_dimensions", "outer",
                          "reduction", "inner", "output_elements"},
                         "Graph reduction attributes");
      if (attributes.at("mode").as_int() != pointwise_mode) {
        artifact_failure("Graph reduction mode disagrees with operation");
      }
      const std::size_t rank = input_tensors[0]->dimensions.size();
      if (rank == 0) {
        artifact_failure("Graph reduction input rank is invalid");
      }
      reduction_axis = attributes.at("axis").as_int();
      if (reduction_axis < -static_cast<std::int64_t>(rank) ||
          reduction_axis >= static_cast<std::int64_t>(rank)) {
        artifact_failure("Graph reduction axis is invalid");
      }
      if (reduction_axis < 0) {
        reduction_axis += static_cast<std::int64_t>(rank);
      }
      reduction_keep_dimensions = attributes.at("keep_dimensions").as_int();
      if (reduction_keep_dimensions != 0 && reduction_keep_dimensions != 1) {
        artifact_failure("Graph reduction keep_dimensions is invalid");
      }
      reduction_outer = 1;
      for (std::int64_t axis = 0; axis < reduction_axis; ++axis) {
        reduction_outer *= input_tensors[0]->dimensions[axis];
      }
      reduction_size = input_tensors[0]->dimensions[reduction_axis];
      reduction_inner = 1;
      for (std::size_t axis = static_cast<std::size_t>(reduction_axis) + 1;
           axis < rank; ++axis) {
        reduction_inner *= input_tensors[0]->dimensions[axis];
      }
      if (attributes.at("outer").as_int() != reduction_outer ||
          attributes.at("reduction").as_int() != reduction_size ||
          attributes.at("inner").as_int() != reduction_inner ||
          attributes.at("output_elements").as_int() !=
              reduction_outer * reduction_inner) {
        artifact_failure("Graph reduction decomposition differs from shape");
      }
      expected_dimensions = input_tensors[0]->dimensions;
      if (reduction_keep_dimensions != 0) {
        expected_dimensions[reduction_axis] = 1;
      } else {
        expected_dimensions.erase(expected_dimensions.begin() + reduction_axis);
      }
    } else if (layout) {
      expected_dimensions = output.dimensions;
    } else if (unary) {
      expected_dimensions = input_tensors[0]->dimensions;
    } else if (ternary) {
      ExpectedTensor partial;
      partial.dimensions =
          broadcast_dimensions(*input_tensors[0], *input_tensors[1]);
      expected_dimensions =
          broadcast_dimensions(partial, *input_tensors[2]);
    } else if (operation == "sigmoid_backward") {
      if (input_tensors[0]->dimensions != input_tensors[1]->dimensions ||
          input_tensors[0]->dimensions != output.dimensions) {
        artifact_failure(
            "Graph sigmoid_backward requires left, right, and output exactly "
            "equal dimensions");
      }
      expected_dimensions = output.dimensions;
    } else {
      expected_dimensions =
          broadcast_dimensions(*input_tensors[0], *input_tensors[1]);
    }
    if (output.dimensions != expected_dimensions) {
      artifact_failure("Graph pointwise output shape is inconsistent");
    }
    const std::size_t n_elements =
        rmsnorm || layernorm || matmul || convolution_fprop
            ? dense_element_count(output)
            : graph_positive_size(
                  attributes.at(reduction ? "output_elements" : "n_elements"),
                  std::numeric_limits<std::int32_t>::max(),
                  "Graph element count");
    if (n_elements >
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
      artifact_failure("Graph element count is outside int32");
    }
    if (n_elements != dense_element_count(output)) {
      artifact_failure("Graph pointwise element count differs from output");
    }
    double alpha = 1.0;
    double negative_slope = 0.0;
    double lower_clip = 0.0;
    double upper_clip = 0.0;
    bool has_upper_clip = false;
    double swish_beta = 1.0;
    double elu_alpha = 1.0;
    double softplus_beta = 1.0;
    LayoutMapping layout_mapping;
    if (reduction || matmul || convolution_fprop || rmsnorm || layernorm) {
      // All reduction semantics were derived above directly from Graph shape.
    } else if (layout) {
      layout_mapping = parse_layout_mapping(
          operation, attributes, *input_tensors[0], output, n_elements);
    } else if (kernel_family == KernelFamily::kBinary) {
      if (attributes.at("pointwise_mode").as_int() != pointwise_mode) {
        artifact_failure(
            "Graph binary pointwise mode disagrees with operation");
      }
      validate_optional_mode(attributes, "mode", pointwise_mode);
      alpha = graph_f32(attributes, "alpha", 1.0);
      if (!binary_contract->supports_alpha && alpha != 1.0) {
        artifact_failure(
            "Graph binary pointwise operation does not support alpha");
      }
    } else if (unary) {
      validate_optional_mode(attributes, "mode", pointwise_mode);
      validate_optional_mode(attributes, "pointwise_mode", pointwise_mode);
      negative_slope = graph_f32(attributes, "negative_slope", 0.0);
      lower_clip = graph_f32(attributes, "lower_clip", 0.0);
      upper_clip = graph_f32(attributes, "upper_clip", 0.0);
      const Value* has_upper = find_field(attributes, "has_upper_clip");
      const std::int64_t encoded_has_upper =
          has_upper == nullptr ? 0 : has_upper->as_int();
      if (encoded_has_upper != 0 && encoded_has_upper != 1) {
        artifact_failure("Graph unary has_upper_clip is invalid");
      }
      has_upper_clip = encoded_has_upper != 0;
      if (has_upper_clip && upper_clip < lower_clip) {
        artifact_failure("Graph ReLU clip interval is invalid");
      }
      validate_optional_relu_float(
          attributes, "relu_lower_clip_slope", negative_slope);
      validate_optional_relu_float(attributes, "relu_lower_clip", lower_clip);
      validate_optional_relu_float(attributes, "relu_upper_clip", upper_clip);
      if (const Value* raw = find_field(attributes, "relu_upper_clip_set");
          raw != nullptr && raw->as_bool() != has_upper_clip) {
        artifact_failure("Graph raw and normalized ReLU flags disagree");
      }
      swish_beta = graph_f32(attributes, "swish_beta", 1.0);
      elu_alpha = graph_f32(attributes, "elu_alpha", 1.0);
      softplus_beta = graph_f32(attributes, "softplus_beta", 1.0);
      if (softplus_beta <= 0.0) {
        artifact_failure("Graph softplus_beta must be positive");
      }
      if ((operation != "swish" && swish_beta != 1.0) ||
          (operation != "elu" && elu_alpha != 1.0) ||
          (operation != "softplus" && softplus_beta != 1.0)) {
        artifact_failure("Graph unary contains another mode's attrs");
      }
      if (operation != "relu" &&
          (negative_slope != 0.0 || lower_clip != 0.0 ||
           upper_clip != 0.0 || has_upper_clip)) {
        artifact_failure("Graph non-ReLU unary contains ReLU attributes");
      }
    } else {
      validate_optional_mode(attributes, "mode", pointwise_mode);
      validate_optional_mode(attributes, "pointwise_mode", pointwise_mode);
    }
    for (const ExpectedTensor* produced :
         std::array<const ExpectedTensor*, 5>{&output, second_output,
                                               third_output, fourth_output,
                                               fifth_output}) {
      if (produced == nullptr) {
        continue;
      }
      if (!producer_positions.emplace(produced->uid, position).second) {
        artifact_failure("Graph tensor has more than one producer");
      }
      has_external_output = has_external_output || !produced->is_virtual;
    }
    parsed_nodes.push_back({node_id,
                            operation,
                            kernel_family,
                            pointwise_mode,
                            std::move(input_tensors),
                            &output,
                            second_output,
                            third_output,
                            fourth_output,
                            fifth_output,
                            static_cast<std::int32_t>(n_elements),
                            alpha,
                            negative_slope,
                            lower_clip,
                            upper_clip,
                            has_upper_clip,
                            swish_beta,
                            elu_alpha,
                            softplus_beta,
                            layout_mapping,
                            reduction_axis,
                            reduction_keep_dimensions,
                            reduction_outer,
                            reduction_size,
                            reduction_inner,
                            matmul_batch,
                            matmul_m,
                            matmul_n,
                            matmul_k,
                            matmul_batch_dimensions,
                            matmul_a_batch_strides,
                            matmul_b_batch_strides,
                            matmul_output_batch_strides,
                            matmul_a_stride_m,
                            matmul_a_stride_k,
                            matmul_b_stride_k,
                            matmul_b_stride_n,
                            matmul_output_stride_m,
                            matmul_output_stride_n,
                            convolution_spatial_rank,
                            convolution_groups,
                            convolution_input_channels,
                            convolution_output_channels,
                            convolution_channels_per_group,
                            convolution_input_dimensions,
                            convolution_input_strides,
                            convolution_filter_dimensions,
                            convolution_filter_strides,
                            convolution_output_dimensions,
                            convolution_output_strides,
                            convolution_pre_padding,
                            convolution_post_padding,
                            convolution_stride,
                            convolution_dilation,
                            batchnorm_rank,
                            batchnorm_batch,
                            batchnorm_channels,
                            batchnorm_spatial,
                            batchnorm_reduction_elements,
                            batchnorm_epsilon,
                            batchnorm_momentum,
                            batchnorm_dimensions,
                            batchnorm_x_strides,
                            batchnorm_y_strides,
                            rmsnorm_rows,
                            rmsnorm_normalized_elements,
                            rmsnorm_epsilon,
                            layernorm_rows,
                            layernorm_normalized_elements,
                            layernorm_epsilon,
                            false});
  }
  if (!has_external_binding || !has_external_output) {
    artifact_failure("Graph pointwise has no external binding or output");
  }
  for (std::size_t position = 0; position < parsed_nodes.size(); ++position) {
    for (const ExpectedTensor* input : parsed_nodes[position].inputs) {
      const auto producer = producer_positions.find(input->uid);
      if ((input->is_virtual && producer == producer_positions.end()) ||
          (producer != producer_positions.end() &&
           producer->second >= position)) {
        artifact_failure("Graph pointwise nodes are not in topological order");
      }
    }
  }

  std::uint64_t next_internal_uid =
      static_cast<std::uint64_t>(tensors.rbegin()->first) + 1U;
  const auto make_dense_virtual_tensor = [](
                                             std::int64_t uid,
                                             const std::string& data_type,
                                             std::vector<std::int64_t> dimensions) {
    ExpectedTensor tensor;
    tensor.uid = uid;
    tensor.data_type = data_type;
    tensor.dimensions = std::move(dimensions);
    tensor.strides.resize(tensor.dimensions.size(), 1);
    std::int64_t stride = 1;
    for (std::size_t axis = tensor.dimensions.size(); axis-- > 0;) {
      tensor.strides[axis] = stride;
      if (axis != 0) {
        if (stride > std::numeric_limits<std::int64_t>::max() /
                         tensor.dimensions[axis]) {
          artifact_failure("internal convolution tensor stride overflows");
        }
        stride *= tensor.dimensions[axis];
      }
    }
    tensor.alignment = 16;
    tensor.is_virtual = true;
    tensor.storage_size =
        dense_element_count(tensor) * element_size(tensor.data_type);
    return tensor;
  };

  std::vector<ParsedNode> execution_nodes;
  execution_nodes.reserve(parsed_nodes.size() * 2);
  for (const ParsedNode& node : parsed_nodes) {
    const bool im2col_fits_i32 = [&node]() {
      std::uint64_t elements = 1U;
      constexpr std::uint64_t kLimit =
          static_cast<std::uint64_t>(
              std::numeric_limits<std::int32_t>::max());
      const auto include = [&elements](std::int64_t factor) {
        constexpr std::uint64_t kInnerLimit =
            static_cast<std::uint64_t>(
                std::numeric_limits<std::int32_t>::max());
        if (factor <= 0 ||
            static_cast<std::uint64_t>(factor) > kInnerLimit / elements) {
          return false;
        }
        elements *= static_cast<std::uint64_t>(factor);
        return true;
      };
      if (node.kernel_family != KernelFamily::kConvolutionFprop ||
          !include(node.inputs[0]->dimensions[0])) {
        return false;
      }
      for (std::size_t axis = 2; axis < node.output->dimensions.size(); ++axis) {
        if (!include(node.output->dimensions[axis])) {
          return false;
        }
      }
      return include(node.convolution_channels_per_group) &&
             include(node.convolution_filter_dimensions[2]) &&
             include(node.convolution_filter_dimensions[3]) &&
             include(node.convolution_filter_dimensions[4]) &&
             elements <= kLimit;
    }();
    const bool im2col_block_offsets_fit_i32 = [&node]() {
      if (node.kernel_family != KernelFamily::kConvolutionFprop) {
        return false;
      }
      const auto as_u64 = [](std::int64_t value) {
        return static_cast<std::uint64_t>(value);
      };
      const std::uint64_t output_width =
          as_u64(node.convolution_output_dimensions[4]);
      const std::uint64_t filter_width =
          as_u64(node.convolution_filter_dimensions[4]);
      const std::uint64_t width_extent =
          (output_width - 1U) * as_u64(node.convolution_stride[2]) +
          (filter_width - 1U) * as_u64(node.convolution_dilation[2]) + 255U;
      const std::uint64_t maximum_offset =
          static_cast<std::uint64_t>(
              std::numeric_limits<std::int32_t>::max()) +
          as_u64(node.convolution_pre_padding[2]);
      return width_extent <= maximum_offset;
    }();
    const bool use_1d_im2col =
        node.kernel_family == KernelFamily::kConvolutionFprop &&
        im2col_fits_i32 && im2col_block_offsets_fit_i32 &&
        next_internal_uid <= static_cast<std::uint64_t>(
                                 std::numeric_limits<std::int64_t>::max()) &&
        node.convolution_spatial_rank == 1 && node.convolution_groups == 1 &&
        is_row_major_tensor(*node.inputs[0]) &&
        is_row_major_tensor(*node.inputs[1]) &&
        node.output->dimensions.size() == 3 &&
        is_physically_dense(*node.output) &&
        node.output->strides ==
            std::vector<std::int64_t>{
                node.output->dimensions[1] * node.output->dimensions[2],
                1,
                node.output->dimensions[1]};
    const bool use_2d_im2col =
        node.kernel_family == KernelFamily::kConvolutionFprop &&
        im2col_fits_i32 && im2col_block_offsets_fit_i32 &&
        next_internal_uid <= static_cast<std::uint64_t>(
                                 std::numeric_limits<std::int64_t>::max()) &&
        node.convolution_spatial_rank == 2 && node.convolution_groups == 1 &&
        (node.output->dimensions.back() % 32 == 0 ||
         node.n_elements >= 65536) &&
        is_row_major_tensor(*node.inputs[0]) &&
        is_row_major_tensor(*node.inputs[1]) &&
        is_row_major_tensor(*node.output);
    const bool use_3d_im2col =
        node.kernel_family == KernelFamily::kConvolutionFprop &&
        im2col_fits_i32 && im2col_block_offsets_fit_i32 &&
        next_internal_uid <= static_cast<std::uint64_t>(
                                 std::numeric_limits<std::int64_t>::max()) &&
        node.convolution_spatial_rank == 3 && node.convolution_groups == 1 &&
        is_row_major_tensor(*node.inputs[0]) &&
        is_row_major_tensor(*node.inputs[1]) &&
        is_row_major_tensor(*node.output);
    if (!use_1d_im2col && !use_2d_im2col && !use_3d_im2col) {
      execution_nodes.push_back(node);
      continue;
    }

    const std::int64_t batch = node.inputs[0]->dimensions[0];
    std::int64_t output_spatial = 1;
    for (std::size_t axis = 2; axis < node.output->dimensions.size(); ++axis) {
      if (output_spatial > std::numeric_limits<std::int64_t>::max() /
                               node.output->dimensions[axis]) {
        artifact_failure("internal convolution output spatial size overflows");
      }
      output_spatial *= node.output->dimensions[axis];
    }
    const std::int64_t output_channels = node.output->dimensions[1];

    // Dense group-one 1x1 convolution is already [O, C] x [N, C, HW].
    // Mirror add_plan.py exactly and bypass the otherwise identical im2col
    // workspace for shapes selected by the 2D im2col policy.
    const bool use_2d_pointwise_matmul =
        use_2d_im2col && node.convolution_filter_dimensions[3] == 1 &&
        node.convolution_filter_dimensions[4] == 1 &&
        node.convolution_pre_padding[1] == 0 &&
        node.convolution_pre_padding[2] == 0 &&
        node.convolution_post_padding[1] == 0 &&
        node.convolution_post_padding[2] == 0 &&
        node.convolution_stride[1] == 1 &&
        node.convolution_stride[2] == 1 &&
        node.convolution_dilation[1] == 1 &&
        node.convolution_dilation[2] == 1;
    if (use_2d_pointwise_matmul) {
      ParsedNode matmul_node = node;
      matmul_node.operation = "matmul";
      matmul_node.kernel_family = KernelFamily::kMatMul;
      matmul_node.pointwise_mode = 0;
      matmul_node.inputs = {node.inputs[1], node.inputs[0]};
      matmul_node.output = node.output;
      matmul_node.second_output = nullptr;
      matmul_node.third_output = nullptr;
      matmul_node.fourth_output = nullptr;
      matmul_node.fifth_output = nullptr;
      matmul_node.n_elements = node.n_elements;
      matmul_node.matmul_batch = batch;
      matmul_node.matmul_m = output_channels;
      matmul_node.matmul_n = output_spatial;
      matmul_node.matmul_k = node.inputs[0]->dimensions[1];
      matmul_node.matmul_batch_dimensions.fill(1);
      matmul_node.matmul_batch_dimensions.back() = batch;
      matmul_node.matmul_a_batch_strides.fill(0);
      matmul_node.matmul_b_batch_strides.fill(0);
      if (batch != 1) {
        matmul_node.matmul_b_batch_strides.back() =
            node.inputs[0]->strides[0];
      }
      matmul_node.matmul_output_batch_strides.fill(0);
      matmul_node.matmul_output_batch_strides.back() =
          node.output->strides[0];
      matmul_node.matmul_a_stride_m = node.inputs[1]->strides[0];
      matmul_node.matmul_a_stride_k = node.inputs[1]->strides[1];
      matmul_node.matmul_b_stride_k = node.inputs[0]->strides[1];
      matmul_node.matmul_b_stride_n = 1;
      matmul_node.matmul_output_stride_m = node.output->strides[1];
      matmul_node.matmul_output_stride_n = 1;
      matmul_node.convolution_im2col = false;
      execution_nodes.push_back(std::move(matmul_node));
      continue;
    }

    if (node.convolution_channels_per_group >
        std::numeric_limits<std::int64_t>::max() /
            node.convolution_filter_dimensions[2]) {
      artifact_failure("internal convolution reduction extent overflows");
    }
    const std::int64_t channel_filter_depth =
        node.convolution_channels_per_group *
        node.convolution_filter_dimensions[2];
    if (channel_filter_depth >
        std::numeric_limits<std::int64_t>::max() /
            node.convolution_filter_dimensions[3]) {
      artifact_failure("internal convolution reduction extent overflows");
    }
    const std::int64_t channel_filter_height =
        channel_filter_depth *
        node.convolution_filter_dimensions[3];
    if (channel_filter_height >
        std::numeric_limits<std::int64_t>::max() /
            node.convolution_filter_dimensions[4]) {
      artifact_failure("internal convolution reduction extent overflows");
    }
    const std::int64_t reduction_extent = channel_filter_height *
        node.convolution_filter_dimensions[4];
    ExpectedTensor columns = make_dense_virtual_tensor(
        static_cast<std::int64_t>(next_internal_uid++),
        node.inputs[0]->data_type,
        {batch, output_spatial, reduction_extent});
    columns.strides = {
        output_spatial * reduction_extent, 1, output_spatial};
    const auto columns_inserted =
        tensors.emplace(columns.uid, std::move(columns));
    if (!columns_inserted.second) {
      artifact_failure("internal convolution columns UID is duplicated");
    }
    const ExpectedTensor* columns_tensor = &columns_inserted.first->second;

    ParsedNode im2col_node = node;
    im2col_node.output = columns_tensor;
    im2col_node.second_output = nullptr;
    im2col_node.third_output = nullptr;
    im2col_node.fourth_output = nullptr;
    im2col_node.fifth_output = nullptr;
    im2col_node.n_elements = static_cast<std::int32_t>(
        dense_element_count(*columns_tensor));
    im2col_node.convolution_im2col = true;
    execution_nodes.push_back(std::move(im2col_node));

    ParsedNode matmul_node = node;
    matmul_node.operation = "matmul";
    matmul_node.kernel_family = KernelFamily::kMatMul;
    matmul_node.pointwise_mode = 0;
    matmul_node.output = node.output;
    matmul_node.second_output = nullptr;
    matmul_node.third_output = nullptr;
    matmul_node.fourth_output = nullptr;
    matmul_node.fifth_output = nullptr;
    matmul_node.n_elements = node.n_elements;
    matmul_node.matmul_batch = batch;
    matmul_node.matmul_k = reduction_extent;
    matmul_node.matmul_batch_dimensions.fill(1);
    matmul_node.matmul_batch_dimensions.back() = batch;
    matmul_node.matmul_a_batch_strides.fill(0);
    matmul_node.matmul_b_batch_strides.fill(0);
    matmul_node.matmul_output_batch_strides.fill(0);
    matmul_node.matmul_output_batch_strides.back() = node.output->strides[0];
    if (use_2d_im2col || use_3d_im2col) {
      matmul_node.inputs = {node.inputs[1], columns_tensor};
      matmul_node.matmul_m = output_channels;
      matmul_node.matmul_n = output_spatial;
      if (batch != 1) {
        matmul_node.matmul_b_batch_strides.back() =
            columns_tensor->strides[0];
      }
      matmul_node.matmul_a_stride_m = reduction_extent;
      matmul_node.matmul_a_stride_k = 1;
      matmul_node.matmul_b_stride_k = output_spatial;
      matmul_node.matmul_b_stride_n = 1;
      matmul_node.matmul_output_stride_m = node.output->strides[1];
      matmul_node.matmul_output_stride_n = node.output->strides.back();
    } else {
      matmul_node.inputs = {columns_tensor, node.inputs[1]};
      matmul_node.matmul_m = output_spatial;
      matmul_node.matmul_n = output_channels;
      if (batch != 1) {
        matmul_node.matmul_a_batch_strides.back() =
            columns_tensor->strides[0];
      }
      matmul_node.matmul_a_stride_m = columns_tensor->strides[1];
      matmul_node.matmul_a_stride_k = columns_tensor->strides[2];
      matmul_node.matmul_b_stride_k = 1;
      matmul_node.matmul_b_stride_n = reduction_extent;
      matmul_node.matmul_output_stride_m = node.output->strides.back();
      matmul_node.matmul_output_stride_n = node.output->strides[1];
    }
    matmul_node.convolution_im2col = false;
    execution_nodes.push_back(std::move(matmul_node));
  }

  std::size_t workspace_end = 0;
  for (auto& [uid, tensor] : tensors) {
    (void)uid;
    if (!tensor.is_virtual) {
      continue;
    }
    workspace_end = align_graph_workspace(workspace_end);
    tensor.workspace_offset = workspace_end;
    if (tensor.storage_size >
        std::numeric_limits<std::size_t>::max() - workspace_end) {
      artifact_failure("Graph workspace size overflows");
    }
    workspace_end += tensor.storage_size;
  }
  result.workspace_size =
      workspace_end == 0 ? 0 : align_graph_workspace(workspace_end);

  std::map<std::int64_t, std::size_t> tensor_to_stage;
  result.stages.reserve(execution_nodes.size());
  for (std::size_t stage_id = 0; stage_id < execution_nodes.size();
       ++stage_id) {
    const ParsedNode& node = execution_nodes[stage_id];
    ExpectedStage stage;
    stage.stage_id = stage_id;
    stage.source_node_id = node.node_id;
    stage.operation = node.operation;
    stage.kernel_family = node.kernel_family;
    stage.pointwise_mode = node.pointwise_mode;
    stage.input_count = node.inputs.size();
    for (const ExpectedTensor* input : node.inputs) {
      const auto producer = tensor_to_stage.find(input->uid);
      if (producer != tensor_to_stage.end() &&
          std::find(stage.dependencies.begin(), stage.dependencies.end(),
                    producer->second) == stage.dependencies.end()) {
        stage.dependencies.push_back(producer->second);
      }
    }
    std::sort(stage.dependencies.begin(), stage.dependencies.end());
    bool contiguous = is_physically_dense(*node.output);
    for (const ExpectedTensor* input : node.inputs) {
      contiguous = contiguous && input->dimensions == node.output->dimensions &&
                   input->strides == node.output->strides &&
                   is_physically_dense(*input);
    }
    switch (node.kernel_family) {
      case KernelFamily::kConvolutionFprop:
        stage.entry_point =
            node.convolution_im2col
                ? "convolution_fprop_im2col_kernel"
                : "convolution_fprop_persistent_kernel";
        break;
      case KernelFamily::kMatMul:
        stage.entry_point = "matmul_strided_kernel";
        break;
      case KernelFamily::kBatchNorm:
        stage.entry_point = "batchnorm_training_persistent_kernel";
        break;
      case KernelFamily::kLayerNorm:
        stage.entry_point = "layernorm_persistent_kernel";
        break;
      case KernelFamily::kRmsNorm:
        stage.entry_point = "rmsnorm_persistent_kernel";
        break;
      case KernelFamily::kBatchNormInference:
        stage.entry_point =
            is_row_major_tensor(*node.inputs[0]) &&
                    is_row_major_tensor(*node.output)
                ? "batchnorm_inference_nchw_persistent_kernel"
                : "batchnorm_inference_strided_persistent_kernel";
        break;
      case KernelFamily::kReduction:
        stage.entry_point =
            is_row_major_tensor(*node.inputs[0]) &&
                    is_row_major_tensor(*node.output)
                ? "reduction_3d_persistent_kernel"
                : "reduction_strided_persistent_kernel";
        break;
      case KernelFamily::kLayout:
        stage.entry_point = "layout_copy_kernel";
        break;
      case KernelFamily::kUnary:
        stage.entry_point = contiguous
                                ? "unary_pointwise_contiguous_kernel"
                                : "unary_pointwise_strided_kernel";
        break;
      case KernelFamily::kTernary:
        stage.entry_point = contiguous
                                ? "binary_select_contiguous_kernel"
                                : "binary_select_strided_kernel";
        break;
      case KernelFamily::kBinary:
        stage.entry_point = contiguous ? "binary_contiguous_kernel"
                                       : "binary_strided_kernel";
        break;
    }
    std::vector<const ExpectedTensor*> stage_tensors = node.inputs;
    stage_tensors.push_back(node.output);
    if (node.second_output != nullptr) {
      stage_tensors.push_back(node.second_output);
    }
    if (node.third_output != nullptr) {
      stage_tensors.push_back(node.third_output);
    }
    if (node.fourth_output != nullptr) {
      stage_tensors.push_back(node.fourth_output);
    }
    if (node.fifth_output != nullptr) {
      stage_tensors.push_back(node.fifth_output);
    }
    stage.arguments.reserve(stage_tensors.size());
    stage.pointer_tokens.reserve(stage_tensors.size());
    stage.tensor_data_types.reserve(stage_tensors.size());
    stage.tensors.reserve(stage_tensors.size());
    std::size_t range_start = std::numeric_limits<std::size_t>::max();
    std::size_t range_end = 0;
    bool has_workspace = false;
    for (std::size_t index = 0; index < stage_tensors.size(); ++index) {
      const ExpectedTensor& tensor = *stage_tensors[index];
      stage.arguments.push_back(expected_argument(tensor));
      stage.pointer_tokens.push_back(
          pointer_token(tensor, node.kernel_family == KernelFamily::kLayout));
      stage.tensor_data_types.push_back(tensor.data_type);
      stage.tensors.push_back(tensor);
      if (tensor.is_virtual) {
        has_workspace = true;
        range_start = std::min(range_start, tensor.workspace_offset);
        range_end = std::max(range_end,
                             tensor.workspace_offset + tensor.storage_size);
      }
    }
    if (has_workspace) {
      stage.workspace_offset = range_start;
      stage.workspace_size = range_end - range_start;
      stage.workspace_alignment = kGraphWorkspaceAlignment;
    }
    stage.n_elements = node.n_elements;
    stage.alpha = node.alpha;
    stage.negative_slope = node.negative_slope;
    stage.lower_clip = node.lower_clip;
    stage.upper_clip = node.upper_clip;
    stage.has_upper_clip = node.has_upper_clip;
    stage.swish_beta = node.swish_beta;
    stage.elu_alpha = node.elu_alpha;
    stage.softplus_beta = node.softplus_beta;
    stage.matmul_batch = node.matmul_batch;
    stage.matmul_m = node.matmul_m;
    stage.matmul_n = node.matmul_n;
    stage.matmul_k = node.matmul_k;
    stage.matmul_batch_dimensions = node.matmul_batch_dimensions;
    stage.matmul_a_batch_strides = node.matmul_a_batch_strides;
    stage.matmul_b_batch_strides = node.matmul_b_batch_strides;
    stage.matmul_output_batch_strides = node.matmul_output_batch_strides;
    stage.matmul_a_stride_m = node.matmul_a_stride_m;
    stage.matmul_a_stride_k = node.matmul_a_stride_k;
    stage.matmul_b_stride_k = node.matmul_b_stride_k;
    stage.matmul_b_stride_n = node.matmul_b_stride_n;
    stage.matmul_output_stride_m = node.matmul_output_stride_m;
    stage.matmul_output_stride_n = node.matmul_output_stride_n;
    stage.convolution_spatial_rank = node.convolution_spatial_rank;
    stage.convolution_groups = node.convolution_groups;
    stage.convolution_input_channels = node.convolution_input_channels;
    stage.convolution_output_channels = node.convolution_output_channels;
    stage.convolution_channels_per_group =
        node.convolution_channels_per_group;
    stage.convolution_input_dimensions = node.convolution_input_dimensions;
    stage.convolution_input_strides = node.convolution_input_strides;
    stage.convolution_filter_dimensions = node.convolution_filter_dimensions;
    stage.convolution_filter_strides = node.convolution_filter_strides;
    stage.convolution_output_dimensions = node.convolution_output_dimensions;
    stage.convolution_output_strides = node.convolution_output_strides;
    stage.convolution_pre_padding = node.convolution_pre_padding;
    stage.convolution_post_padding = node.convolution_post_padding;
    stage.convolution_stride = node.convolution_stride;
    stage.convolution_dilation = node.convolution_dilation;
    stage.batchnorm_rank = node.batchnorm_rank;
    stage.batchnorm_batch = node.batchnorm_batch;
    stage.batchnorm_channels = node.batchnorm_channels;
    stage.batchnorm_spatial = node.batchnorm_spatial;
    stage.batchnorm_reduction_elements =
        node.batchnorm_reduction_elements;
    stage.batchnorm_epsilon = node.batchnorm_epsilon;
    stage.batchnorm_momentum = node.batchnorm_momentum;
    stage.batchnorm_dimensions = node.batchnorm_dimensions;
    stage.batchnorm_x_strides = node.batchnorm_x_strides;
    stage.batchnorm_y_strides = node.batchnorm_y_strides;
    stage.rmsnorm_rows = node.rmsnorm_rows;
    stage.rmsnorm_normalized_elements = node.rmsnorm_normalized_elements;
    stage.rmsnorm_epsilon = node.rmsnorm_epsilon;
    stage.layernorm_rows = node.layernorm_rows;
    stage.layernorm_normalized_elements = node.layernorm_normalized_elements;
    stage.layernorm_epsilon = node.layernorm_epsilon;
    stage.reduction_axis = node.reduction_axis;
    stage.reduction_rank = static_cast<std::int64_t>(node.inputs[0]->dimensions.size());
    stage.reduction_output_rank =
        static_cast<std::int64_t>(node.output->dimensions.size());
    stage.reduction_keep_dimensions = node.reduction_keep_dimensions;
    stage.reduction_outer = node.reduction_outer;
    stage.reduction_size = node.reduction_size;
    stage.reduction_inner = node.reduction_inner;
    if (node.kernel_family == KernelFamily::kMatMul ||
        node.kernel_family == KernelFamily::kConvolutionFprop) {
      // These families use independently recomputed operation metadata.
    } else if (node.kernel_family == KernelFamily::kReduction) {
      stage.reduction_input_dimensions = padded_dimensions(*node.inputs[0]);
      stage.reduction_input_strides = effective_strides(
          *node.inputs[0], node.inputs[0]->dimensions.size(), false);
      stage.reduction_output_dimensions = padded_dimensions(*node.output);
      stage.output_strides = effective_strides(
          *node.output, node.output->dimensions.size(), false);
    } else if (node.kernel_family == KernelFamily::kLayout) {
      stage.layout_input_base = node.layout.input_base;
      stage.layout_mode = node.layout.mode;
      stage.layout_input_dimensions = node.layout.input_dimensions;
      stage.layout_input_strides = node.layout.input_strides;
      stage.layout_output_dimensions = node.layout.output_dimensions;
      stage.output_strides = node.layout.output_strides;
    } else if (node.kernel_family != KernelFamily::kBatchNorm &&
               node.kernel_family != KernelFamily::kMatMul &&
               node.kernel_family != KernelFamily::kConvolutionFprop &&
               node.kernel_family != KernelFamily::kRmsNorm &&
               node.kernel_family != KernelFamily::kLayerNorm) {
      stage.dimensions = padded_dimensions(*node.output);
      const std::size_t rank = node.output->dimensions.size();
      stage.left_strides = effective_strides(*node.inputs[0], rank, true);
      if (node.kernel_family != KernelFamily::kUnary) {
        stage.right_strides = effective_strides(*node.inputs[1], rank, true);
      }
      if (node.kernel_family == KernelFamily::kTernary) {
        stage.mask_strides = effective_strides(*node.inputs[2], rank, true);
      }
      stage.output_strides = effective_strides(*node.output, rank, false);
    }
    result.stages.push_back(std::move(stage));
    tensor_to_stage[node.output->uid] = stage_id;
    if (node.second_output != nullptr) {
      tensor_to_stage[node.second_output->uid] = stage_id;
    }
    if (node.third_output != nullptr) {
      tensor_to_stage[node.third_output->uid] = stage_id;
    }
    if (node.fourth_output != nullptr) {
      tensor_to_stage[node.fourth_output->uid] = stage_id;
    }
    if (node.fifth_output != nullptr) {
      tensor_to_stage[node.fifth_output->uid] = stage_id;
    }
  }
  return result;
}

void validate_argument_graph_contract(
    const std::vector<ArgumentSource>& arguments,
    const ExpectedStage& expected) {
  if (arguments.size() != expected.arguments.size() + 1) {
    artifact_failure("Ascend pointwise argument count differs from Graph");
  }
  for (std::size_t index = 0; index < expected.arguments.size(); ++index) {
    const ArgumentSource& actual = arguments[index];
    const ExpectedArgument& wanted = expected.arguments[index];
    if (actual.source != wanted.source || actual.uid != wanted.uid ||
        actual.size != wanted.size ||
        actual.alignment != wanted.alignment ||
        actual.workspace_offset != wanted.workspace_offset) {
      artifact_failure("Ascend pointwise pointer source differs from Graph");
    }
  }
  const auto* n_elements =
      std::get_if<std::int32_t>(
          &arguments[expected.arguments.size()].scalar);
  if (n_elements == nullptr || *n_elements != expected.n_elements) {
    artifact_failure("Ascend pointwise element scalar differs from Graph");
  }
}

void validate_payload_graph_contract(const Value& payload,
                                     const ExpectedStage& expected,
                                     std::uint32_t expected_worker_count) {
  const std::vector<std::string> signature =
      split_signature(payload.at("full_signature").as_string());
  if (signature.size() <= expected.pointer_tokens.size()) {
    artifact_failure("Ascend pointwise signature is incomplete");
  }
  for (std::size_t index = 0; index < expected.pointer_tokens.size(); ++index) {
    if (signature[index] != expected.pointer_tokens[index]) {
      artifact_failure(
          "Ascend pointwise signature data type differs from Graph");
    }
  }
  if (signature[expected.pointer_tokens.size()] != "i32") {
    artifact_failure("Ascend pointwise scalar signature differs from Graph");
  }
  const auto& grid = payload.at("grid").as_array();
  const std::int64_t block_size = payload.at("meta").at("BLOCK_SIZE").as_int();
  if (expected.kernel_family == KernelFamily::kMatMul
          ? block_size != 16 && block_size != 32 && block_size != 64 &&
                block_size != 128
          : uses_pointwise_tuning(expected.kernel_family)
                ? block_size != 256 && block_size != 1024
                : expected.kernel_family == KernelFamily::kBatchNorm
                      ? block_size != 256 && block_size != 1024
                : expected.kernel_family == KernelFamily::kBatchNormInference
                      ? block_size != 1024 && block_size != 4096
                : uses_wide_normalization_tuning(expected.kernel_family)
                      ? block_size != 1024 && block_size != 2048 &&
                            block_size != 4096
                : block_size != 128 && block_size != 256) {
    artifact_failure(
        "Ascend pointwise BLOCK_SIZE is outside its tuning whitelist");
  }
  std::uint64_t work_items =
      (static_cast<std::uint64_t>(expected.n_elements) +
       static_cast<std::uint64_t>(block_size) - 1U) /
      static_cast<std::uint64_t>(block_size);
  if (expected.kernel_family == KernelFamily::kReduction) {
    work_items = static_cast<std::uint64_t>(expected.n_elements);
  }
  if (expected.kernel_family == KernelFamily::kBatchNorm) {
    work_items = static_cast<std::uint64_t>(expected.batchnorm_channels);
  }
  if (expected.kernel_family == KernelFamily::kRmsNorm) {
    work_items = static_cast<std::uint64_t>(expected.rmsnorm_rows);
  }
  if (expected.kernel_family == KernelFamily::kLayerNorm) {
    work_items = static_cast<std::uint64_t>(expected.layernorm_rows);
  }
  if (expected.kernel_family == KernelFamily::kBatchNormInference &&
      expected.entry_point ==
          "batchnorm_inference_strided_persistent_kernel" &&
      expected.batchnorm_rank == 5) {
    const std::uint64_t channels =
        static_cast<std::uint64_t>(expected.batchnorm_channels);
    const std::uint64_t spatial =
        static_cast<std::uint64_t>(expected.batchnorm_spatial);
    const std::uint64_t batch_elements = channels * spatial;
    if (batch_elements == 0U ||
        static_cast<std::uint64_t>(expected.n_elements) % batch_elements !=
            0U) {
      artifact_failure("Ascend BatchNorm inference dimensions are inconsistent");
    }
    const std::uint64_t batches =
        static_cast<std::uint64_t>(expected.n_elements) / batch_elements;
    const std::uint64_t block_spatial =
        static_cast<std::uint64_t>(block_size) / 128U;
    if (block_spatial == 0U) {
      artifact_failure("Ascend BatchNorm inference block size is invalid");
    }
    work_items = batches * ((channels + 1U) / 2U) *
                 ((spatial + block_spatial - 1U) / block_spatial);
  }
  if (expected.kernel_family == KernelFamily::kLayout &&
      expected.layout_mode == 2) {
    const std::int64_t inner_elements =
        expected.layout_output_dimensions[kMaximumRank - 1];
    if (inner_elements <= 0 ||
        expected.n_elements % inner_elements != 0) {
      artifact_failure("Ascend shared-row layout dimensions are inconsistent");
    }
    work_items = static_cast<std::uint64_t>(expected.n_elements /
                                            inner_elements);
  }
  std::uint64_t expected_grid_x =
      std::min(work_items, static_cast<std::uint64_t>(expected_worker_count));
  if (expected.kernel_family == KernelFamily::kMatMul) {
    const std::uint64_t tiles =
        ((static_cast<std::uint64_t>(expected.matmul_m) + block_size - 1U) /
         block_size) *
        ((static_cast<std::uint64_t>(expected.matmul_n) + block_size - 1U) /
         block_size);
    expected_grid_x = std::min(
        tiles * static_cast<std::uint64_t>(expected.matmul_batch),
        static_cast<std::uint64_t>(expected_worker_count));
  }
  if (grid.size() != 3 || grid[0].as_int() !=
                              static_cast<std::int64_t>(expected_grid_x) ||
      grid[1].as_int() != 1 ||
      grid[2].as_int() != 1) {
    artifact_failure("Ascend pointwise grid differs from Graph");
  }
  const Value& meta = payload.at("meta");
  if (meta.at("WORKER_COUNT").as_int() !=
      static_cast<std::int64_t>(expected_worker_count)) {
    artifact_failure("Ascend constexpr worker metadata differs from Graph");
  }
  if (expected.kernel_family == KernelFamily::kLayout) {
    if (meta.at("INPUT_BASE").as_int() != expected.layout_input_base) {
      artifact_failure("Ascend layout input base differs from Graph");
    }
    if (meta.at("LAYOUT_MODE").as_int() != expected.layout_mode) {
      artifact_failure("Ascend layout addressing mode differs from Graph");
    }
    if (expected.tensor_data_types.empty() ||
        meta.at("ELEMENT_SIZE_BYTES").as_int() !=
            static_cast<std::int64_t>(
                element_size(expected.tensor_data_types.front()))) {
      artifact_failure("Ascend layout element size differs from Graph");
    }
    for (const auto& [prefix, values] :
         std::array<std::pair<
                        std::string_view,
                        const std::array<std::int64_t, kMaximumRank>*>,
                    4>{{{"INPUT_DIM", &expected.layout_input_dimensions},
                        {"INPUT_STRIDE", &expected.layout_input_strides},
                        {"OUTPUT_DIM", &expected.layout_output_dimensions},
                        {"OUTPUT_STRIDE", &expected.output_strides}}}) {
      for (std::size_t axis = 0; axis < kMaximumRank; ++axis) {
        const std::string name =
            std::string(prefix) + "_" + std::to_string(axis);
        if (meta.at(name).as_int() != (*values)[axis]) {
          artifact_failure("Ascend layout metadata differs from Graph");
        }
      }
    }
    return;
  }
  if (expected.kernel_family == KernelFamily::kConvolutionFprop) {
    if (meta.at("SPATIAL_RANK").as_int() !=
            expected.convolution_spatial_rank ||
        meta.at("GROUPS").as_int() != expected.convolution_groups ||
        meta.at("INPUT_CHANNELS").as_int() !=
            expected.convolution_input_channels ||
        meta.at("OUTPUT_CHANNELS").as_int() !=
            expected.convolution_output_channels ||
        meta.at("CHANNELS_PER_GROUP").as_int() !=
            expected.convolution_channels_per_group) {
      artifact_failure("Ascend convolution metadata differs from Graph");
    }
    for (const auto& [prefix, values] :
         std::array<std::pair<std::string_view,
                              const std::array<std::int64_t, 5>*>,
                    6>{{{"INPUT_DIM", &expected.convolution_input_dimensions},
                        {"INPUT_STRIDE", &expected.convolution_input_strides},
                        {"FILTER_DIM", &expected.convolution_filter_dimensions},
                        {"FILTER_STRIDE", &expected.convolution_filter_strides},
                        {"OUTPUT_DIM", &expected.convolution_output_dimensions},
                        {"OUTPUT_STRIDE", &expected.convolution_output_strides}}}) {
      for (std::size_t axis = 0; axis < values->size(); ++axis) {
        const std::string name =
            std::string(prefix) + "_" + std::to_string(axis);
        if (meta.at(name).as_int() != (*values)[axis]) {
          artifact_failure("Ascend convolution tensor metadata differs from Graph");
        }
      }
    }
    for (const auto& [prefix, values] :
         std::array<std::pair<std::string_view,
                              const std::array<std::int64_t, 3>*>,
                    4>{{{"PRE_PADDING", &expected.convolution_pre_padding},
                        {"POST_PADDING", &expected.convolution_post_padding},
                        {"CONV_STRIDE", &expected.convolution_stride},
                        {"DILATION", &expected.convolution_dilation}}}) {
      for (std::size_t axis = 0; axis < values->size(); ++axis) {
        const std::string name =
            std::string(prefix) + "_" + std::to_string(axis);
        if (meta.at(name).as_int() != (*values)[axis]) {
          artifact_failure("Ascend convolution spatial metadata differs from Graph");
        }
      }
    }
    return;
  }
  if (expected.kernel_family == KernelFamily::kMatMul) {
    if (meta.at("BATCH").as_int() != expected.matmul_batch ||
        meta.at("M").as_int() != expected.matmul_m ||
        meta.at("N").as_int() != expected.matmul_n ||
        meta.at("K").as_int() != expected.matmul_k ||
        meta.at("A_STRIDE_M").as_int() != expected.matmul_a_stride_m ||
        meta.at("A_STRIDE_K").as_int() != expected.matmul_a_stride_k ||
        meta.at("B_STRIDE_K").as_int() != expected.matmul_b_stride_k ||
        meta.at("B_STRIDE_N").as_int() != expected.matmul_b_stride_n ||
        meta.at("C_STRIDE_M").as_int() != expected.matmul_output_stride_m ||
        meta.at("C_STRIDE_N").as_int() != expected.matmul_output_stride_n ||
        meta.at("INPUT_IS_FLOAT32").as_int() !=
            static_cast<std::int64_t>(expected.tensor_data_types.front() ==
                                      "float32") ||
        meta.at("GROUP_M").as_int() !=
            (expected.matmul_m >= 2048 ? 8 : 1)) {
      artifact_failure("Ascend matmul metadata differs from Graph");
    }
    for (const auto& [prefix, values] :
         std::array<std::pair<
                        std::string_view,
                        const std::array<std::int64_t,
                                         kMaximumMatmulBatchRank>*>,
                    4>{{{"DIM", &expected.matmul_batch_dimensions},
                        {"A_BATCH_STRIDE", &expected.matmul_a_batch_strides},
                        {"B_BATCH_STRIDE", &expected.matmul_b_batch_strides},
                        {"C_BATCH_STRIDE",
                         &expected.matmul_output_batch_strides}}}) {
      for (std::size_t axis = 0; axis < kMaximumMatmulBatchRank; ++axis) {
        const std::string name =
            std::string(prefix) + "_" + std::to_string(axis);
        if (meta.at(name).as_int() != (*values)[axis]) {
          artifact_failure("Ascend matmul batch metadata differs from Graph");
        }
      }
    }
    return;
  }
  if (expected.kernel_family == KernelFamily::kReduction) {
    if (meta.at("RANK").as_int() != expected.reduction_rank ||
        meta.at("OUTPUT_RANK").as_int() != expected.reduction_output_rank ||
        meta.at("AXIS").as_int() != expected.reduction_axis ||
        meta.at("KEEP_DIMENSIONS").as_int() !=
            expected.reduction_keep_dimensions ||
        meta.at("OUTER").as_int() != expected.reduction_outer ||
        meta.at("REDUCTION_SIZE").as_int() != expected.reduction_size ||
        meta.at("INNER").as_int() != expected.reduction_inner ||
        meta.at("OUTPUT_ELEMENTS").as_int() != expected.n_elements ||
        meta.at("REDUCTION_MODE").as_int() != expected.pointwise_mode) {
      artifact_failure("Ascend reduction metadata differs from Graph");
    }
    if (expected.entry_point == "reduction_strided_persistent_kernel") {
      for (const auto& [prefix, values] :
           std::array<std::pair<
                          std::string_view,
                          const std::array<std::int64_t, kMaximumRank>*>,
                      4>{{{"INPUT_DIM", &expected.reduction_input_dimensions},
                          {"INPUT_STRIDE", &expected.reduction_input_strides},
                          {"OUTPUT_DIM", &expected.reduction_output_dimensions},
                          {"OUTPUT_STRIDE", &expected.output_strides}}}) {
        for (std::size_t axis = 0; axis < kMaximumRank; ++axis) {
          const std::string name =
              std::string(prefix) + "_" + std::to_string(axis);
          if (meta.at(name).as_int() != (*values)[axis]) {
            artifact_failure(
                "Ascend reduction stride metadata differs from Graph");
          }
        }
      }
    }
    return;
  }
  if (expected.kernel_family == KernelFamily::kBatchNorm) {
    if (meta.at("RANK").as_int() != expected.batchnorm_rank ||
        meta.at("BATCH").as_int() != expected.batchnorm_batch ||
        meta.at("CHANNELS").as_int() != expected.batchnorm_channels ||
        meta.at("SPATIAL").as_int() != expected.batchnorm_spatial ||
        meta.at("REDUCTION_ELEMENTS").as_int() !=
            expected.batchnorm_reduction_elements ||
        meta.at("EPSILON").as_double() != expected.batchnorm_epsilon ||
        meta.at("MOMENTUM").as_double() != expected.batchnorm_momentum) {
      artifact_failure(
          "Ascend batchnorm training metadata differs from Graph");
    }
    for (const auto& [prefix, values] :
         std::array<std::pair<
                        std::string_view,
                        const std::array<std::int64_t, kMaximumRank>*>,
                    3>{{{"DIM", &expected.batchnorm_dimensions},
                        {"X_STRIDE", &expected.batchnorm_x_strides},
                        {"Y_STRIDE", &expected.batchnorm_y_strides}}}) {
      for (std::size_t axis = 0; axis < kMaximumRank; ++axis) {
        const std::string name =
            std::string(prefix) + "_" + std::to_string(axis);
        if (meta.at(name).as_int() != (*values)[axis]) {
          artifact_failure(
              "Ascend batchnorm training stride metadata differs from Graph");
        }
      }
    }
    return;
  }
  if (expected.kernel_family == KernelFamily::kBatchNormInference) {
    if (meta.at("RANK").as_int() != expected.batchnorm_rank ||
        meta.at("CHANNELS").as_int() != expected.batchnorm_channels ||
        meta.at("SPATIAL").as_int() != expected.batchnorm_spatial) {
      artifact_failure("Ascend batchnorm metadata differs from Graph");
    }
    if (expected.entry_point ==
        "batchnorm_inference_strided_persistent_kernel") {
      for (const auto& [prefix, values] :
           std::array<std::pair<
                          std::string_view,
                          const std::array<std::int64_t, kMaximumRank>*>,
                      3>{{{"DIM", &expected.batchnorm_dimensions},
                          {"X_STRIDE", &expected.batchnorm_x_strides},
                          {"Y_STRIDE", &expected.batchnorm_y_strides}}}) {
        for (std::size_t axis = 0; axis < kMaximumRank; ++axis) {
          const std::string name =
              std::string(prefix) + "_" + std::to_string(axis);
          if (meta.at(name).as_int() != (*values)[axis]) {
            artifact_failure(
                "Ascend batchnorm stride metadata differs from Graph");
          }
        }
      }
    }
    return;
  }
  if (expected.kernel_family == KernelFamily::kRmsNorm) {
    if (meta.at("ROWS").as_int() != expected.rmsnorm_rows ||
        meta.at("NORMALIZED_ELEMENTS").as_int() !=
            expected.rmsnorm_normalized_elements ||
        meta.at("EPSILON").as_double() != expected.rmsnorm_epsilon) {
      artifact_failure("Ascend rmsnorm metadata differs from Graph");
    }
    return;
  }
  if (expected.kernel_family == KernelFamily::kLayerNorm) {
    if (meta.at("ROWS").as_int() != expected.layernorm_rows ||
        meta.at("NORMALIZED_ELEMENTS").as_int() !=
            expected.layernorm_normalized_elements ||
        meta.at("EPSILON").as_double() != expected.layernorm_epsilon) {
      artifact_failure("Ascend layernorm metadata differs from Graph");
    }
    return;
  }
  const bool unary = expected.kernel_family == KernelFamily::kUnary;
  const bool ternary = expected.kernel_family == KernelFamily::kTernary;
  if ((!ternary &&
       meta.at(unary ? "OPERATION" : "OP_KIND").as_int() !=
           expected.pointwise_mode)) {
    artifact_failure("Ascend pointwise constexpr metadata differs from Graph");
  }
  if ((!unary && !ternary &&
       meta.at("ALPHA").as_double() != expected.alpha) ||
      (unary &&
       (meta.at("negative_slope").as_double() != expected.negative_slope ||
        meta.at("lower_clip").as_double() != expected.lower_clip ||
        meta.at("upper_clip").as_double() != expected.upper_clip ||
        meta.at("HAS_UPPER_CLIP").as_int() !=
            static_cast<std::int64_t>(expected.has_upper_clip) ||
        meta.at("SWISH_BETA").as_double() != expected.swish_beta ||
        meta.at("ELU_ALPHA").as_double() != expected.elu_alpha ||
        meta.at("SOFTPLUS_BETA").as_double() != expected.softplus_beta))) {
    artifact_failure("Ascend pointwise attributes differ from Graph");
  }
  if (expected.entry_point == "binary_strided_kernel" ||
      expected.entry_point == "unary_pointwise_strided_kernel" ||
      expected.entry_point == "binary_select_strided_kernel") {
    std::vector<std::pair<std::string_view,
                          const std::array<std::int64_t, kMaximumRank>*>>
        groups = {{"DIM", &expected.dimensions},
                  {unary ? "INPUT_STRIDE" : "LEFT_STRIDE",
                   &expected.left_strides}};
    if (!unary) {
      groups.emplace_back("RIGHT_STRIDE", &expected.right_strides);
    }
    if (ternary) {
      groups.emplace_back("MASK_STRIDE", &expected.mask_strides);
    }
    groups.emplace_back("OUTPUT_STRIDE", &expected.output_strides);
    for (const auto& [prefix, values] : groups) {
      for (std::size_t axis = 0; axis < kMaximumRank; ++axis) {
        const std::string name =
            std::string(prefix) + "_" + std::to_string(axis);
        if (meta.at(name).as_int() != (*values)[axis]) {
          artifact_failure(
              "Ascend pointwise stride metadata differs from Graph");
        }
      }
    }
  }
}

void validate_stage_graph_contract(const AscendStageArtifact& stage,
                                   const ExpectedStage& expected) {
  if (stage.stage_id != expected.stage_id ||
      stage.source_node_ids !=
          std::vector<std::size_t>{expected.source_node_id} ||
      stage.dependencies != expected.dependencies ||
      stage.operation != expected.operation ||
      stage.kernel_family != expected.kernel_family ||
      stage.entry_point != expected.entry_point ||
      stage.workspace_offset != expected.workspace_offset ||
      stage.workspace_size != expected.workspace_size ||
      stage.workspace_alignment != expected.workspace_alignment) {
    artifact_failure("Ascend execution stage differs from Graph plan");
  }
  validate_argument_graph_contract(stage.arguments, expected);
}

ArgumentSource parse_argument_source(const Value& value,
                                     std::size_t expected_index,
                                     std::size_t graph_workspace_size,
                                     std::vector<std::int64_t>& binding_uids) {
  const auto& object = value.as_object();
  const std::string source = value.at("source").as_string();
  ArgumentSource result;
  result.index = checked_size(value.at("index").as_int(), "argument.index");
  result.name = value.at("name").as_string();
  if (result.index != expected_index || !is_identifier(result.name)) {
    artifact_failure("argument source index or name is invalid");
  }

  if (source == "binding") {
    require_exact_keys(value,
                       {"index", "name", "source", "uid", "size",
                        "alignment"},
                       "binding argument source");
    result.source = ArgumentSourceKind::kBinding;
    result.type = RawArgumentType::kPointer;
    result.uid = value.at("uid").as_int();
    result.size = checked_size(value.at("size").as_int(), "binding.size");
    result.alignment = checked_size(value.at("alignment").as_int(),
                                    "binding.alignment");
    if (result.uid <= 0 || result.size == 0 ||
        !valid_alignment(result.alignment)) {
      artifact_failure("binding argument source metadata is invalid");
    }
    if (std::find(binding_uids.begin(), binding_uids.end(), result.uid) ==
        binding_uids.end()) {
      binding_uids.push_back(result.uid);
    }
    return result;
  }
  if (source == "graph_workspace") {
    require_exact_keys(value,
                       {"index", "name", "source", "uid", "offset",
                        "size", "alignment"},
                       "workspace argument source");
    result.source = ArgumentSourceKind::kGraphWorkspace;
    result.type = RawArgumentType::kPointer;
    result.uid = value.at("uid").as_int();
    result.workspace_offset =
        checked_size(value.at("offset").as_int(), "workspace.offset");
    result.size =
        checked_size(value.at("size").as_int(), "workspace.size");
    result.alignment = checked_size(value.at("alignment").as_int(),
                                    "workspace.alignment");
    if (result.uid <= 0 || result.size == 0 ||
        !valid_alignment(result.alignment) ||
        result.workspace_offset % result.alignment != 0 ||
        result.workspace_offset > graph_workspace_size ||
        result.size > graph_workspace_size - result.workspace_offset) {
      artifact_failure("workspace argument source range is invalid");
    }
    return result;
  }
  if (source == "scalar") {
    require_exact_keys(value, {"index", "name", "source", "type", "value"},
                       "scalar argument source");
    result.source = ArgumentSourceKind::kScalar;
    result.type = parse_raw_type(value.at("type").as_string());
    if (result.type == RawArgumentType::kPointer) {
      artifact_failure("scalar argument cannot have pointer type");
    }
    switch (result.type) {
      case RawArgumentType::kI32: {
        const std::int64_t integer = value.at("value").as_int();
        if (integer < std::numeric_limits<std::int32_t>::min() ||
            integer > std::numeric_limits<std::int32_t>::max()) {
          artifact_failure("i32 scalar is out of range");
        }
        result.scalar = static_cast<std::int32_t>(integer);
        break;
      }
      case RawArgumentType::kI64:
        result.scalar = value.at("value").as_int();
        break;
      case RawArgumentType::kF32: {
        const double number = value.at("value").as_double();
        if (!std::isfinite(number) ||
            std::abs(number) > std::numeric_limits<float>::max()) {
          artifact_failure("f32 scalar is out of range");
        }
        result.scalar = static_cast<float>(number);
        break;
      }
      case RawArgumentType::kF64: {
        const double number = value.at("value").as_double();
        if (!std::isfinite(number)) {
          artifact_failure("f64 scalar is not finite");
        }
        result.scalar = number;
        break;
      }
      case RawArgumentType::kPointer:
        artifact_failure("pointer scalar is invalid");
    }
    return result;
  }
  (void)object;
  artifact_failure("argument source discriminator is unsupported");
}

LtjNpuRawCandidate parse_candidate(
    const Value& value,
    const AscendStageArtifact& stage,
    const std::vector<ArgumentSource>& sources,
    const ExpectedStage& expected,
    std::uint32_t expected_worker_count) {
  require_exact_keys(value, {"candidate_id", "launch_abi", "payload"},
                     "Ascend launch candidate");
  LtjNpuRawCandidate result;
  result.candidate_id = value.at("candidate_id").as_string();
  if (!is_safe_id(result.candidate_id) ||
      value.at("launch_abi").as_string() != "ltj_npu_raw_v1") {
    artifact_failure("Ascend candidate ID or launch ABI is invalid");
  }

  const Value& payload = value.at("payload");
  require_exact_keys(payload,
                     {"schema_version", "source_path", "source_sha256",
                      "entry_point", "full_signature", "grid",
                      "compile_options", "meta", "argument_abi"},
                     "ltj_npu_raw_v1 payload");
  if (payload.at("schema_version").as_int() != kLaunchPayloadVersion) {
    artifact_failure("ltj_npu_raw_v1 payload version is unsupported");
  }
  const std::string source_path = payload.at("source_path").as_string();
  result.source = stage.source;
  result.source_sha256 = payload.at("source_sha256").as_string();
  result.entry_point = payload.at("entry_point").as_string();
  result.full_signature = payload.at("full_signature").as_string();
  if (source_path != stage.source.filename().string() ||
      result.source_sha256 != stage.source_sha256 ||
      result.entry_point != stage.entry_point) {
    artifact_failure("candidate source identity differs from stage kernel");
  }

  const auto& grid = payload.at("grid").as_array();
  if (grid.size() != 3) {
    artifact_failure("ltj_npu_raw_v1 grid must have three dimensions");
  }
  std::uint64_t grid_product = 1;
  for (std::size_t axis = 0; axis < grid.size(); ++axis) {
    result.grid[axis] =
        checked_positive_unsigned(grid[axis].as_int(), "payload.grid");
    if (grid_product > std::numeric_limits<std::uint32_t>::max() /
                           result.grid[axis]) {
      artifact_failure("ltj_npu_raw_v1 grid product overflows uint32_t");
    }
    grid_product *= result.grid[axis];
  }

  const Value& options = payload.at("compile_options");
  require_exact_keys(options, {"num_warps", "num_stages"},
                     "ltj_npu_raw_v1 compile options");
  result.num_warps = checked_positive_unsigned(
      options.at("num_warps").as_int(), "compile_options.num_warps");
  result.num_stages = checked_positive_unsigned(
      options.at("num_stages").as_int(), "compile_options.num_stages");
  if (result.num_warps != 4 || result.num_stages != 1 ||
      result.num_warps > static_cast<unsigned int>(
                             std::numeric_limits<int>::max()) ||
      result.num_stages > static_cast<unsigned int>(
                              std::numeric_limits<int>::max())) {
    artifact_failure("ltj_npu_raw_v1 compile options are outside policy");
  }

  const auto& argument_abi = payload.at("argument_abi").as_array();
  if (argument_abi.empty() ||
      argument_abi.size() > FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS ||
      argument_abi.size() != sources.size()) {
    artifact_failure("ltj_npu_raw_v1 argument ABI count is invalid");
  }
  result.argument_types.reserve(argument_abi.size());
  for (std::size_t index = 0; index < argument_abi.size(); ++index) {
    require_exact_keys(argument_abi[index], {"index", "name", "type"},
                       "ltj_npu_raw_v1 argument ABI entry");
    const std::size_t encoded_index = checked_size(
        argument_abi[index].at("index").as_int(), "argument_abi.index");
    const std::string name =
        argument_abi[index].at("name").as_string();
    const RawArgumentType type =
        parse_raw_type(argument_abi[index].at("type").as_string());
    if (encoded_index != index || name != sources[index].name ||
        type != sources[index].type) {
      artifact_failure("argument ABI does not match argument sources");
    }
    result.argument_types.push_back(type);
  }
  validate_meta_and_signature(payload.at("meta"),
                              result.entry_point,
                              result.full_signature,
                              sources,
                              result.argument_types,
                              expected.pointwise_mode,
                              expected_worker_count);
  validate_payload_graph_contract(payload, expected, expected_worker_count);
  result.block_size = checked_positive_unsigned(
      payload.at("meta").at("BLOCK_SIZE").as_int(), "meta.BLOCK_SIZE");
  const std::size_t configuration_index = tuning_configuration_index(
      expected.kernel_family, result.block_size, result.num_warps,
      result.num_stages);
  const std::string expected_candidate_id =
      stage.autotune
          ? tuning_configuration_id(expected.operation,
                                    result.block_size,
                                    result.num_warps,
                                    result.num_stages)
          : "default";
  std::size_t expected_fixed_configuration_index = 0;
  if (expected.kernel_family == KernelFamily::kUnary &&
      (expected.operation == "log" || expected.operation == "rsqrt") &&
      expected.entry_point == "unary_pointwise_contiguous_kernel" &&
      expected.n_elements < 1024) {
    for (std::size_t index = 0;
         index < kAllowedPointwiseTuningConfigurations.size(); ++index) {
      if (kAllowedPointwiseTuningConfigurations[index].block_size == 256) {
        expected_fixed_configuration_index = index;
        break;
      }
    }
  } else if (uses_wide_normalization_tuning(expected.kernel_family)) {
    const std::int64_t normalized_elements =
        expected.kernel_family == KernelFamily::kLayerNorm
            ? expected.layernorm_normalized_elements
            : expected.rmsnorm_normalized_elements;
    std::uint32_t preferred_block_size = 0;
    for (std::size_t index = 0;
         index < kAllowedNormalizationTuningConfigurations.size(); ++index) {
      const std::uint32_t candidate_block_size =
          kAllowedNormalizationTuningConfigurations[index].block_size;
      if (candidate_block_size >= normalized_elements &&
          (preferred_block_size == 0 ||
           candidate_block_size < preferred_block_size)) {
        preferred_block_size = candidate_block_size;
        expected_fixed_configuration_index = index;
      }
    }
    if (preferred_block_size == 0) {
      for (std::size_t index = 0;
           index < kAllowedNormalizationTuningConfigurations.size();
           ++index) {
        const std::uint32_t candidate_block_size =
            kAllowedNormalizationTuningConfigurations[index].block_size;
        if (candidate_block_size > preferred_block_size) {
          preferred_block_size = candidate_block_size;
          expected_fixed_configuration_index = index;
        }
      }
    }
  }
  if (result.candidate_id != expected_candidate_id ||
      (!stage.autotune &&
       configuration_index != expected_fixed_configuration_index)) {
    artifact_failure(
        "Ascend pointwise candidate identity differs from tuning data");
  }
  return result;
}

void parse_autotune(const Value& value, AscendStageArtifact& stage) {
  const auto& object = value.as_object();
  const bool enabled = value.at("enabled").as_bool();
  const std::size_t expected_size = enabled ? 11 : 3;
  if (object.size() != expected_size ||
      value.at("schema_version").as_int() != 1) {
    artifact_failure("Ascend autotune policy schema is invalid");
  }
  const Value& selection = value.at("selection");
  require_exact_keys(selection, {"state", "candidate_id"},
                     "autotune selection");
  const std::string state = selection.at("state").as_string();
  stage.selected_candidate = selection.at("candidate_id").as_string();
  stage.autotune = enabled;
  if (!enabled) {
    if (state != "fixed" || !is_safe_id(stage.selected_candidate)) {
      artifact_failure("disabled autotune must select a fixed candidate");
    }
    return;
  }

  require_exact_keys(value,
                     {"schema_version", "enabled", "selection", "source",
                      "source_sha256", "table", "key", "strategy",
                      "warmup", "repetitions", "candidate_identity"},
                     "enabled Ascend autotune policy");
  if (state != "pending" || !stage.selected_candidate.empty()) {
    artifact_failure("enabled autotune must have a pending selection");
  }
  if (!is_sha256(value.at("source_sha256").as_string()) ||
      !is_sha256(value.at("candidate_identity").as_string()) ||
      value.at("source_sha256").as_string() != kTuningSourceSha256 ||
      value.at("table").as_string() !=
          (stage.kernel_family == KernelFamily::kReduction
               ? "reduction"
               : stage.kernel_family == KernelFamily::kConvolutionFprop
                     ? "convolution_fprop"
               : stage.kernel_family == KernelFamily::kMatMul
                     ? "matmul"
               : stage.kernel_family == KernelFamily::kBatchNorm
                     ? "batchnorm"
               : stage.kernel_family == KernelFamily::kUnary
                     ? "unary"
                     : stage.kernel_family == KernelFamily::kRmsNorm
                           ? "rmsnorm"
                     : stage.kernel_family == KernelFamily::kLayerNorm
                           ? "layernorm"
                     : stage.kernel_family ==
                               KernelFamily::kBatchNormInference
                           ? "batchnorm_inference"
                           : "binary") ||
      value.at("key").as_string() != "n_elements" ||
      value.at("strategy").as_string() != "align32" ||
      value.at("source").as_string() != "common.yaml") {
    artifact_failure("enabled Ascend autotune identity is invalid");
  }
  stage.warmup = checked_positive_unsigned(value.at("warmup").as_int(),
                                           "autotune.warmup");
  stage.repetitions = checked_positive_unsigned(
      value.at("repetitions").as_int(), "autotune.repetitions");
  if (stage.warmup != kAutotuneWarmup ||
      stage.repetitions != kAutotuneRepetitions) {
    artifact_failure("enabled Ascend autotune sampling policy is invalid");
  }
  stage.candidate_identity =
      value.at("candidate_identity").as_string();
}

std::string expected_autotune_candidate_identity(
    const AscendStageArtifact& stage, const ExpectedStage& expected) {
  std::string attributes;
  if (expected.kernel_family == KernelFamily::kUnary) {
    attributes =
        "{\"elu_alpha\":\"" + f64_identity(expected.elu_alpha) +
        "\",\"has_upper_clip\":" +
        std::to_string(static_cast<int>(expected.has_upper_clip)) +
        ",\"lower_clip\":\"" + f64_identity(expected.lower_clip) +
        "\",\"negative_slope\":\"" + f64_identity(expected.negative_slope) +
        "\",\"softplus_beta\":\"" +
        f64_identity(expected.softplus_beta) +
        "\",\"swish_beta\":\"" + f64_identity(expected.swish_beta) +
        "\",\"upper_clip\":\"" + f64_identity(expected.upper_clip) + "\"" +
        "}";
  } else if (expected.kernel_family == KernelFamily::kBinary) {
    attributes = "{\"alpha\":\"" + f64_identity(expected.alpha) +
                 "\"}";
  } else if (expected.kernel_family == KernelFamily::kReduction) {
    attributes =
        "{\"axis\":" + std::to_string(expected.reduction_axis) +
        ",\"inner\":" + std::to_string(expected.reduction_inner) +
        ",\"keep_dimensions\":" +
        std::to_string(expected.reduction_keep_dimensions) +
        ",\"outer\":" + std::to_string(expected.reduction_outer) +
        ",\"output_elements\":" + std::to_string(expected.n_elements) +
        ",\"reduction_mode\":" +
        std::to_string(expected.pointwise_mode) +
        ",\"reduction_size\":" +
        std::to_string(expected.reduction_size) + "}";
  } else if (expected.kernel_family == KernelFamily::kMatMul) {
    if (expected.tensors.size() != 3 || expected.arguments.size() != 3) {
      artifact_failure("MatMul autotune identity tensor roles are invalid");
    }
    const auto integer_array = [](const std::vector<std::int64_t>& values) {
      std::string result = "[";
      for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
          result += ',';
        }
        result += std::to_string(values[index]);
      }
      result += ']';
      return result;
    };
    static constexpr std::array<std::string_view, 3> kArgumentNames = {
        "a_ptr", "b_ptr", "output_ptr"};
    static constexpr std::array<std::string_view, 3> kTensorRoles = {
        "a", "b", "output"};
    std::string arguments = "[";
    for (std::size_t index = 0; index < expected.arguments.size(); ++index) {
      if (index != 0) {
        arguments += ',';
      }
      const ExpectedArgument& argument = expected.arguments[index];
      arguments += "{\"alignment\":" + std::to_string(argument.alignment) +
                   ",\"index\":" + std::to_string(index) +
                   ",\"name\":\"" + std::string(kArgumentNames[index]) +
                   "\"";
      if (argument.source == ArgumentSourceKind::kGraphWorkspace) {
        arguments += ",\"offset\":" +
                     std::to_string(argument.workspace_offset);
      }
      arguments += ",\"size\":" + std::to_string(argument.size) +
                   ",\"source\":\"" +
                   std::string(argument.source ==
                                       ArgumentSourceKind::kGraphWorkspace
                                   ? "graph_workspace"
                                   : "binding") +
                   "\",\"uid\":" + std::to_string(argument.uid) + "}";
    }
    arguments +=
        ",{\"index\":3,\"name\":\"n_elements\",\"source\":\"scalar\","
        "\"type\":\"i32\",\"value\":" +
        std::to_string(expected.n_elements) + "}]";
    std::string tensors = "[";
    for (std::size_t index = 0; index < expected.tensors.size(); ++index) {
      if (index != 0) {
        tensors += ',';
      }
      const ExpectedTensor& tensor = expected.tensors[index];
      tensors += "{\"alignment\":" + std::to_string(tensor.alignment) +
                 ",\"data_type\":\"" + tensor.data_type +
                 "\",\"dimensions\":" + integer_array(tensor.dimensions) +
                 ",\"role\":\"" + std::string(kTensorRoles[index]) +
                 "\",\"storage_size\":" +
                 std::to_string(tensor.storage_size) +
                 ",\"strides\":" + integer_array(tensor.strides) +
                 ",\"virtual\":" +
                 std::string(tensor.is_virtual ? "true" : "false") + "}";
    }
    tensors += ']';
    attributes =
        "{\"arguments\":" + arguments +
        ",\"batch\":" + std::to_string(expected.matmul_batch) +
        ",\"k\":" + std::to_string(expected.matmul_k) +
        ",\"m\":" + std::to_string(expected.matmul_m) +
        ",\"n\":" + std::to_string(expected.matmul_n) +
        ",\"tensors\":" + tensors + "}";
  } else if (expected.kernel_family == KernelFamily::kConvolutionFprop) {
    if (expected.tensors.size() != 3 || expected.arguments.size() != 3) {
      artifact_failure("Convolution autotune identity tensor roles are invalid");
    }
    const auto integer_array = [](const std::vector<std::int64_t>& values) {
      std::string result = "[";
      for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
          result += ',';
        }
        result += std::to_string(values[index]);
      }
      result += ']';
      return result;
    };
    static constexpr std::array<std::string_view, 3> kArgumentNames = {
        "input_ptr", "filter_ptr", "output_ptr"};
    static constexpr std::array<std::string_view, 3> kTensorRoles = {
        "input", "filter", "output"};
    std::string arguments = "[";
    for (std::size_t index = 0; index < expected.arguments.size(); ++index) {
      if (index != 0) {
        arguments += ',';
      }
      const ExpectedArgument& argument = expected.arguments[index];
      arguments += "{\"alignment\":" + std::to_string(argument.alignment) +
                   ",\"index\":" + std::to_string(index) +
                   ",\"name\":\"" + std::string(kArgumentNames[index]) +
                   "\"";
      if (argument.source == ArgumentSourceKind::kGraphWorkspace) {
        arguments += ",\"offset\":" +
                     std::to_string(argument.workspace_offset);
      }
      arguments += ",\"size\":" + std::to_string(argument.size) +
                   ",\"source\":\"" +
                   std::string(argument.source ==
                                       ArgumentSourceKind::kGraphWorkspace
                                   ? "graph_workspace"
                                   : "binding") +
                   "\",\"uid\":" + std::to_string(argument.uid) + "}";
    }
    arguments +=
        ",{\"index\":3,\"name\":\"n_elements\",\"source\":\"scalar\","
        "\"type\":\"i32\",\"value\":" +
        std::to_string(expected.n_elements) + "}]";
    std::vector<std::pair<std::string, std::int64_t>> meta_values = {
        {"channels_per_group", expected.convolution_channels_per_group},
        {"groups", expected.convolution_groups},
        {"input_channels", expected.convolution_input_channels},
        {"output_channels", expected.convolution_output_channels},
        {"spatial_rank", expected.convolution_spatial_rank}};
    const auto append_meta = [&meta_values](
                                 std::string_view prefix,
                                 const auto& values) {
      for (std::size_t axis = 0; axis < values.size(); ++axis) {
        meta_values.emplace_back(std::string(prefix) + std::to_string(axis),
                                 values[axis]);
      }
    };
    append_meta("input_dim_", expected.convolution_input_dimensions);
    append_meta("input_stride_", expected.convolution_input_strides);
    append_meta("filter_dim_", expected.convolution_filter_dimensions);
    append_meta("filter_stride_", expected.convolution_filter_strides);
    append_meta("output_dim_", expected.convolution_output_dimensions);
    append_meta("output_stride_", expected.convolution_output_strides);
    append_meta("pre_padding_", expected.convolution_pre_padding);
    append_meta("post_padding_", expected.convolution_post_padding);
    append_meta("conv_stride_", expected.convolution_stride);
    append_meta("dilation_", expected.convolution_dilation);
    std::sort(meta_values.begin(), meta_values.end(),
              [](const auto& left, const auto& right) {
                return left.first < right.first;
              });
    std::string meta = "{";
    for (std::size_t index = 0; index < meta_values.size(); ++index) {
      if (index != 0) {
        meta += ',';
      }
      meta += "\"" + meta_values[index].first + "\":" +
              std::to_string(meta_values[index].second);
    }
    meta += '}';
    std::string tensors = "[";
    for (std::size_t index = 0; index < expected.tensors.size(); ++index) {
      if (index != 0) {
        tensors += ',';
      }
      const ExpectedTensor& tensor = expected.tensors[index];
      tensors += "{\"alignment\":" + std::to_string(tensor.alignment) +
                 ",\"data_type\":\"" + tensor.data_type +
                 "\",\"dimensions\":" + integer_array(tensor.dimensions) +
                 ",\"role\":\"" + std::string(kTensorRoles[index]) +
                 "\",\"storage_size\":" +
                 std::to_string(tensor.storage_size) +
                 ",\"strides\":" + integer_array(tensor.strides) +
                 ",\"virtual\":" +
                 std::string(tensor.is_virtual ? "true" : "false") + "}";
    }
    tensors += ']';
    attributes = "{\"arguments\":" + arguments +
                 ",\"meta\":" + meta +
                 ",\"n_elements\":" + std::to_string(expected.n_elements) +
                 ",\"tensors\":" + tensors + "}";
  } else if (expected.kernel_family == KernelFamily::kBatchNormInference) {
    if (expected.tensors.size() != 6 || expected.arguments.size() != 6) {
      artifact_failure("BatchNorm autotune identity tensor roles are invalid");
    }
    const auto integer_array = [](const std::vector<std::int64_t>& values) {
      std::string result = "[";
      for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
          result += ',';
        }
        result += std::to_string(values[index]);
      }
      result += ']';
      return result;
    };
    static constexpr std::array<std::string_view, 6> kArgumentNames = {
        "x_ptr", "mean_ptr", "inv_variance_ptr", "scale_ptr", "bias_ptr",
        "y_ptr"};
    static constexpr std::array<std::string_view, 6> kTensorRoles = {
        "x", "mean", "inv_variance", "scale", "bias", "y"};
    std::string arguments = "[";
    for (std::size_t index = 0; index < expected.arguments.size(); ++index) {
      if (index != 0) {
        arguments += ',';
      }
      const ExpectedArgument& argument = expected.arguments[index];
      arguments += "{\"alignment\":" + std::to_string(argument.alignment) +
                   ",\"index\":" + std::to_string(index) +
                   ",\"name\":\"" + std::string(kArgumentNames[index]) +
                   "\"";
      if (argument.source == ArgumentSourceKind::kGraphWorkspace) {
        arguments += ",\"offset\":" +
                     std::to_string(argument.workspace_offset);
      }
      arguments += ",\"size\":" + std::to_string(argument.size) +
                   ",\"source\":\"" +
                   std::string(argument.source ==
                                       ArgumentSourceKind::kGraphWorkspace
                                   ? "graph_workspace"
                                   : "binding") +
                   "\",\"uid\":" + std::to_string(argument.uid) + "}";
    }
    arguments +=
        ",{\"index\":6,\"name\":\"n_elements\",\"source\":\"scalar\","
        "\"type\":\"i32\",\"value\":" +
        std::to_string(expected.n_elements) + "}]";

    std::string tensors = "[";
    for (std::size_t index = 0; index < expected.tensors.size(); ++index) {
      if (index != 0) {
        tensors += ',';
      }
      const ExpectedTensor& tensor = expected.tensors[index];
      tensors += "{\"alignment\":" + std::to_string(tensor.alignment) +
                 ",\"data_type\":\"" + tensor.data_type +
                 "\",\"dimensions\":" + integer_array(tensor.dimensions) +
                 ",\"role\":\"" + std::string(kTensorRoles[index]) +
                 "\",\"storage_size\":" +
                 std::to_string(tensor.storage_size) +
                 ",\"strides\":" + integer_array(tensor.strides) +
                 ",\"virtual\":" +
                 std::string(tensor.is_virtual ? "true" : "false") + "}";
    }
    tensors += ']';
    attributes =
        "{\"arguments\":" + arguments +
        ",\"channels\":" + std::to_string(expected.batchnorm_channels) +
        ",\"n_elements\":" + std::to_string(expected.n_elements) +
        ",\"rank\":" + std::to_string(expected.batchnorm_rank) +
        ",\"spatial\":" + std::to_string(expected.batchnorm_spatial) +
        ",\"tensors\":" + tensors + "}";
  } else if (expected.kernel_family == KernelFamily::kBatchNorm) {
    if (expected.tensors.size() != 10 || expected.arguments.size() != 10) {
      artifact_failure(
          "BatchNorm training autotune identity tensor roles are invalid");
    }
    const auto integer_array = [](const std::vector<std::int64_t>& values) {
      std::string result = "[";
      for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
          result += ',';
        }
        result += std::to_string(values[index]);
      }
      result += ']';
      return result;
    };
    static constexpr std::array<std::string_view, 10> kArgumentNames = {
        "x_ptr", "scale_ptr", "bias_ptr", "previous_running_mean_ptr",
        "previous_running_variance_ptr", "y_ptr", "mean_ptr",
        "inv_variance_ptr", "next_running_mean_ptr",
        "next_running_variance_ptr"};
    static constexpr std::array<std::string_view, 10> kTensorRoles = {
        "x", "scale", "bias", "previous_running_mean",
        "previous_running_variance", "y", "mean", "inv_variance",
        "next_running_mean", "next_running_variance"};
    std::string arguments = "[";
    for (std::size_t index = 0; index < expected.arguments.size(); ++index) {
      if (index != 0) {
        arguments += ',';
      }
      const ExpectedArgument& argument = expected.arguments[index];
      arguments += "{\"alignment\":" + std::to_string(argument.alignment) +
                   ",\"index\":" + std::to_string(index) +
                   ",\"name\":\"" + std::string(kArgumentNames[index]) +
                   "\"";
      if (argument.source == ArgumentSourceKind::kGraphWorkspace) {
        arguments += ",\"offset\":" +
                     std::to_string(argument.workspace_offset);
      }
      arguments += ",\"size\":" + std::to_string(argument.size) +
                   ",\"source\":\"" +
                   std::string(argument.source ==
                                       ArgumentSourceKind::kGraphWorkspace
                                   ? "graph_workspace"
                                   : "binding") +
                   "\",\"uid\":" + std::to_string(argument.uid) + "}";
    }
    arguments +=
        ",{\"index\":10,\"name\":\"n_elements\",\"source\":\"scalar\","
        "\"type\":\"i32\",\"value\":" +
        std::to_string(expected.n_elements) + "}]";
    std::string tensors = "[";
    for (std::size_t index = 0; index < expected.tensors.size(); ++index) {
      if (index != 0) {
        tensors += ',';
      }
      const ExpectedTensor& tensor = expected.tensors[index];
      tensors += "{\"alignment\":" + std::to_string(tensor.alignment) +
                 ",\"data_type\":\"" + tensor.data_type +
                 "\",\"dimensions\":" + integer_array(tensor.dimensions) +
                 ",\"role\":\"" + std::string(kTensorRoles[index]) +
                 "\",\"storage_size\":" +
                 std::to_string(tensor.storage_size) +
                 ",\"strides\":" + integer_array(tensor.strides) +
                 ",\"virtual\":" +
                 std::string(tensor.is_virtual ? "true" : "false") + "}";
    }
    tensors += ']';
    attributes =
        "{\"arguments\":" + arguments +
        ",\"batch\":" + std::to_string(expected.batchnorm_batch) +
        ",\"channels\":" + std::to_string(expected.batchnorm_channels) +
        ",\"epsilon\":\"" + f64_identity(expected.batchnorm_epsilon) +
        "\",\"momentum\":\"" + f64_identity(expected.batchnorm_momentum) +
        "\",\"n_elements\":" + std::to_string(expected.n_elements) +
        ",\"rank\":" + std::to_string(expected.batchnorm_rank) +
        ",\"reduction_elements\":" +
        std::to_string(expected.batchnorm_reduction_elements) +
        ",\"spatial\":" + std::to_string(expected.batchnorm_spatial) +
        ",\"tensors\":" + tensors + "}";
  } else if (expected.kernel_family == KernelFamily::kRmsNorm ||
             expected.kernel_family == KernelFamily::kLayerNorm) {
    const bool layernorm =
        expected.kernel_family == KernelFamily::kLayerNorm;
    const std::size_t expected_tensor_count = layernorm ? 6U : 5U;
    if (expected.tensors.size() != expected_tensor_count ||
        expected.arguments.size() != expected_tensor_count) {
      artifact_failure(
          "normalization autotune identity tensor roles are invalid");
    }
    const auto integer_array = [](const std::vector<std::int64_t>& values) {
      std::string result = "[";
      for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
          result += ',';
        }
        result += std::to_string(values[index]);
      }
      result += ']';
      return result;
    };
    const std::vector<std::string_view> argument_names =
        layernorm
            ? std::vector<std::string_view>{"x_ptr", "scale_ptr", "bias_ptr",
                                            "y_ptr", "mean_ptr",
                                            "inv_variance_ptr"}
            : std::vector<std::string_view>{"x_ptr", "scale_ptr", "bias_ptr",
                                            "y_ptr", "inv_variance_ptr"};
    const std::vector<std::string_view> tensor_roles =
        layernorm
            ? std::vector<std::string_view>{"x", "scale", "bias", "y",
                                            "mean", "inv_variance"}
            : std::vector<std::string_view>{"x", "scale", "bias", "y",
                                            "inv_variance"};
    std::string arguments = "[";
    for (std::size_t index = 0; index < expected.arguments.size(); ++index) {
      if (index != 0) {
        arguments += ',';
      }
      const ExpectedArgument& argument = expected.arguments[index];
      arguments += "{\"alignment\":" + std::to_string(argument.alignment) +
                   ",\"index\":" + std::to_string(index) +
                   ",\"name\":\"" + std::string(argument_names[index]) +
                   "\"";
      if (argument.source == ArgumentSourceKind::kGraphWorkspace) {
        arguments += ",\"offset\":" +
                     std::to_string(argument.workspace_offset);
      }
      arguments += ",\"size\":" + std::to_string(argument.size) +
                   ",\"source\":\"" +
                   std::string(argument.source ==
                                       ArgumentSourceKind::kGraphWorkspace
                                   ? "graph_workspace"
                                   : "binding") +
                   "\",\"uid\":" + std::to_string(argument.uid) + "}";
    }
    arguments +=
        ",{\"index\":" + std::to_string(expected_tensor_count) +
        ",\"name\":\"n_elements\",\"source\":\"scalar\","
        "\"type\":\"i32\",\"value\":" +
        std::to_string(expected.n_elements) + "}]";

    std::string tensors = "[";
    for (std::size_t index = 0; index < expected.tensors.size(); ++index) {
      if (index != 0) {
        tensors += ',';
      }
      const ExpectedTensor& tensor = expected.tensors[index];
      tensors += "{\"alignment\":" + std::to_string(tensor.alignment) +
                 ",\"data_type\":\"" + tensor.data_type +
                 "\",\"dimensions\":" + integer_array(tensor.dimensions) +
                 ",\"role\":\"" + std::string(tensor_roles[index]) +
                 "\",\"storage_size\":" +
                 std::to_string(tensor.storage_size) +
                 ",\"strides\":" + integer_array(tensor.strides) +
                 ",\"virtual\":" +
                 std::string(tensor.is_virtual ? "true" : "false") + "}";
    }
    tensors += ']';
    attributes =
        "{\"arguments\":" + arguments +
        ",\"epsilon\":\"" +
        f64_identity(layernorm ? expected.layernorm_epsilon
                               : expected.rmsnorm_epsilon) +
        "\",\"normalized_elements\":" +
        std::to_string(layernorm ? expected.layernorm_normalized_elements
                                 : expected.rmsnorm_normalized_elements) +
        ",\"rows\":" +
        std::to_string(layernorm ? expected.layernorm_rows
                                 : expected.rmsnorm_rows) +
        ",\"tensors\":" + tensors + "}";
  } else {
    attributes = "{}";
  }
  std::vector<const LtjNpuRawCandidate*> identity_candidates;
  identity_candidates.reserve(stage.candidates.size());
  for (const LtjNpuRawCandidate& candidate : stage.candidates) {
    identity_candidates.push_back(&candidate);
  }
  if (expected.kernel_family == KernelFamily::kMatMul ||
      expected.kernel_family == KernelFamily::kConvolutionFprop ||
      expected.kernel_family == KernelFamily::kBatchNorm ||
      expected.kernel_family == KernelFamily::kBatchNormInference ||
      expected.kernel_family == KernelFamily::kRmsNorm ||
      expected.kernel_family == KernelFamily::kLayerNorm) {
    std::sort(identity_candidates.begin(),
              identity_candidates.end(),
              [](const LtjNpuRawCandidate* left,
                 const LtjNpuRawCandidate* right) {
                return left->candidate_id < right->candidate_id;
              });
  }
  std::string canonical = "{\"attributes\":" + attributes +
                          ",\"backend\":\"ascend\",\"candidates\":[";
  for (std::size_t index = 0; index < identity_candidates.size(); ++index) {
    if (index != 0) {
      canonical += ',';
    }
    const LtjNpuRawCandidate& candidate = *identity_candidates[index];
    canonical += "{\"candidate_id\":\"" + candidate.candidate_id +
                 "\"";
    if (expected.kernel_family == KernelFamily::kMatMul ||
        expected.kernel_family == KernelFamily::kConvolutionFprop ||
        expected.kernel_family == KernelFamily::kBatchNorm ||
        expected.kernel_family == KernelFamily::kBatchNormInference ||
        expected.kernel_family == KernelFamily::kRmsNorm ||
        expected.kernel_family == KernelFamily::kLayerNorm) {
      canonical += ",\"grid\":[" + std::to_string(candidate.grid[0]) + ',' +
                   std::to_string(candidate.grid[1]) + ',' +
                   std::to_string(candidate.grid[2]) + ']';
    }
    canonical += ",\"meta\":{\"BLOCK_SIZE\":" +
                 std::to_string(candidate.block_size) +
                 "},\"num_stages\":" +
                 std::to_string(candidate.num_stages) +
                 ",\"num_warps\":" +
                 std::to_string(candidate.num_warps) + "}";
  }
  canonical += "],\"function\":\"" + stage.entry_point +
               "\"";
  if (expected.kernel_family == KernelFamily::kMatMul ||
      expected.kernel_family == KernelFamily::kConvolutionFprop ||
      expected.kernel_family == KernelFamily::kBatchNorm ||
      expected.kernel_family == KernelFamily::kBatchNormInference ||
      expected.kernel_family == KernelFamily::kRmsNorm ||
      expected.kernel_family == KernelFamily::kLayerNorm) {
    canonical += ",\"kernel_source_sha256\":\"" + stage.source_sha256 +
                 "\"";
  }
  canonical += ",\"key\":\"n_elements\",\"key_value\":" +
               std::to_string(expected.n_elements) +
               ",\"launch_abi\":\"ltj_npu_raw_v1\","
               "\"operation\":\"" + expected.operation +
               "\",\"schema_version\":1,"
               "\"source_sha256\":\"" +
               std::string(kTuningSourceSha256) +
               "\",\"strategy\":\"align32\"}";
  return flagdnn::native::sha256(canonical);
}

AscendStageArtifact parse_stage(
    const Value& value,
    std::size_t expected_stage_id,
    const ExpectedStage& expected,
    const std::filesystem::path& artifact_directory,
    std::string_view compiler_identity,
    std::size_t graph_workspace_size,
    std::vector<std::int64_t>& binding_uids,
    std::uint32_t expected_worker_count) {
  require_exact_keys(value,
                     {"stage_id", "kind", "source_node_ids",
                      "dependencies", "operation", "kernel_family", "kernel",
                      "workspace", "argument_sources", "candidates",
                      "autotune"},
                     "Ascend execution stage");
  AscendStageArtifact result;
  result.stage_id = checked_size(value.at("stage_id").as_int(), "stage_id");
  result.kernel_family = expected.kernel_family;
  if (result.stage_id != expected_stage_id ||
      value.at("kind").as_string() != "kernel" ||
      value.at("operation").as_string() != expected.operation ||
      value.at("kernel_family").as_string() !=
          kernel_family_name(expected.kernel_family)) {
    artifact_failure("Ascend stage identity is invalid");
  }
  result.operation = expected.operation;
  result.tensor_storage_data_types.reserve(expected.tensor_data_types.size());
  for (const std::string& data_type : expected.tensor_data_types) {
    result.tensor_storage_data_types.push_back(storage_data_type(data_type));
  }
  result.input_count = expected.input_count;
  result.n_elements = expected.n_elements;
  result.alpha = expected.alpha;
  result.negative_slope = expected.negative_slope;
  result.lower_clip = expected.lower_clip;
  result.upper_clip = expected.upper_clip;
  result.has_upper_clip = expected.has_upper_clip;
  result.swish_beta = expected.swish_beta;
  result.elu_alpha = expected.elu_alpha;
  result.softplus_beta = expected.softplus_beta;
  result.dimensions = expected.dimensions;
  result.left_strides = expected.left_strides;
  result.right_strides = expected.right_strides;
  result.mask_strides = expected.mask_strides;
  result.output_strides = expected.output_strides;
  result.matmul_batch = expected.matmul_batch;
  result.matmul_m = expected.matmul_m;
  result.matmul_n = expected.matmul_n;
  result.matmul_k = expected.matmul_k;
  result.matmul_batch_dimensions = expected.matmul_batch_dimensions;
  result.matmul_a_batch_strides = expected.matmul_a_batch_strides;
  result.matmul_b_batch_strides = expected.matmul_b_batch_strides;
  result.matmul_output_batch_strides = expected.matmul_output_batch_strides;
  result.matmul_a_stride_m = expected.matmul_a_stride_m;
  result.matmul_a_stride_k = expected.matmul_a_stride_k;
  result.matmul_b_stride_k = expected.matmul_b_stride_k;
  result.matmul_b_stride_n = expected.matmul_b_stride_n;
  result.matmul_output_stride_m = expected.matmul_output_stride_m;
  result.matmul_output_stride_n = expected.matmul_output_stride_n;
  result.convolution_spatial_rank = expected.convolution_spatial_rank;
  result.convolution_groups = expected.convolution_groups;
  result.convolution_input_channels = expected.convolution_input_channels;
  result.convolution_output_channels = expected.convolution_output_channels;
  result.convolution_channels_per_group =
      expected.convolution_channels_per_group;
  result.convolution_input_dimensions = expected.convolution_input_dimensions;
  result.convolution_input_strides = expected.convolution_input_strides;
  result.convolution_filter_dimensions = expected.convolution_filter_dimensions;
  result.convolution_filter_strides = expected.convolution_filter_strides;
  result.convolution_output_dimensions = expected.convolution_output_dimensions;
  result.convolution_output_strides = expected.convolution_output_strides;
  result.convolution_pre_padding = expected.convolution_pre_padding;
  result.convolution_post_padding = expected.convolution_post_padding;
  result.convolution_stride = expected.convolution_stride;
  result.convolution_dilation = expected.convolution_dilation;
  result.batchnorm_rank = expected.batchnorm_rank;
  result.batchnorm_batch = expected.batchnorm_batch;
  result.batchnorm_channels = expected.batchnorm_channels;
  result.batchnorm_spatial = expected.batchnorm_spatial;
  result.batchnorm_reduction_elements =
      expected.batchnorm_reduction_elements;
  result.batchnorm_epsilon = expected.batchnorm_epsilon;
  result.batchnorm_momentum = expected.batchnorm_momentum;
  result.batchnorm_dimensions = expected.batchnorm_dimensions;
  result.batchnorm_x_strides = expected.batchnorm_x_strides;
  result.batchnorm_y_strides = expected.batchnorm_y_strides;
  result.rmsnorm_rows = expected.rmsnorm_rows;
  result.rmsnorm_normalized_elements = expected.rmsnorm_normalized_elements;
  result.rmsnorm_epsilon = expected.rmsnorm_epsilon;
  result.layernorm_rows = expected.layernorm_rows;
  result.layernorm_normalized_elements = expected.layernorm_normalized_elements;
  result.layernorm_epsilon = expected.layernorm_epsilon;
  result.layout_input_base = expected.layout_input_base;
  result.layout_input_dimensions = expected.layout_input_dimensions;
  result.layout_input_strides = expected.layout_input_strides;
  result.layout_output_dimensions = expected.layout_output_dimensions;
  result.reduction_axis = expected.reduction_axis;
  result.reduction_rank = expected.reduction_rank;
  result.reduction_output_rank = expected.reduction_output_rank;
  result.reduction_keep_dimensions = expected.reduction_keep_dimensions;
  result.reduction_outer = expected.reduction_outer;
  result.reduction_size = expected.reduction_size;
  result.reduction_inner = expected.reduction_inner;
  result.reduction_input_dimensions = expected.reduction_input_dimensions;
  result.reduction_input_strides = expected.reduction_input_strides;
  result.reduction_output_dimensions = expected.reduction_output_dimensions;

  const auto& source_nodes = value.at("source_node_ids").as_array();
  if (source_nodes.empty()) {
    artifact_failure("Ascend stage has no source graph node");
  }
  for (const Value& source_node : source_nodes) {
    const std::size_t node =
        checked_size(source_node.as_int(), "source_node_id");
    if (std::find(result.source_node_ids.begin(),
                  result.source_node_ids.end(), node) !=
        result.source_node_ids.end()) {
      artifact_failure("Ascend stage repeats a source node ID");
    }
    result.source_node_ids.push_back(node);
  }
  for (const Value& dependency : value.at("dependencies").as_array()) {
    const std::size_t stage =
        checked_size(dependency.as_int(), "stage dependency");
    if (stage >= expected_stage_id ||
        std::find(result.dependencies.begin(), result.dependencies.end(),
                  stage) != result.dependencies.end()) {
      artifact_failure("Ascend stage dependency is cyclic or duplicated");
    }
    result.dependencies.push_back(stage);
  }

  const Value& workspace = value.at("workspace");
  require_exact_keys(workspace, {"offset", "size", "alignment"},
                     "stage workspace");
  result.workspace_offset =
      checked_size(workspace.at("offset").as_int(), "stage.workspace.offset");
  result.workspace_size =
      checked_size(workspace.at("size").as_int(), "stage.workspace.size");
  result.workspace_alignment = checked_size(
      workspace.at("alignment").as_int(), "stage.workspace.alignment");
  if (!valid_alignment(result.workspace_alignment) ||
      (result.workspace_size != 0 &&
       (result.workspace_offset % result.workspace_alignment != 0 ||
        result.workspace_offset > graph_workspace_size ||
        result.workspace_size > graph_workspace_size -
                                    result.workspace_offset))) {
    artifact_failure("stage Graph workspace range is invalid");
  }

  const Value& kernel = value.at("kernel");
  require_exact_keys(kernel,
                     {"provider", "ownership", "source", "source_sha256",
                      "entry_point", "materialized_source"},
                     "Ascend kernel envelope");
  std::string_view expected_source;
  std::string_view expected_source_sha256;
  switch (expected.kernel_family) {
    case KernelFamily::kBinary:
      expected_source = "binary.py";
      expected_source_sha256 = kBinarySourceSha256;
      break;
    case KernelFamily::kUnary:
      expected_source = "unary.py";
      expected_source_sha256 = kUnarySourceSha256;
      break;
    case KernelFamily::kTernary:
      expected_source = "ternary.py";
      expected_source_sha256 = kTernarySourceSha256;
      break;
    case KernelFamily::kLayout:
      expected_source = "layout.py";
      expected_source_sha256 = kLayoutSourceSha256;
      break;
    case KernelFamily::kReduction:
      expected_source = "reduction.py";
      expected_source_sha256 = kReductionSourceSha256;
      break;
    case KernelFamily::kMatMul:
      expected_source = "matmul.py";
      expected_source_sha256 = kMatmulSourceSha256;
      break;
    case KernelFamily::kConvolutionFprop:
      expected_source = "convolution.py";
      expected_source_sha256 = kConvolutionSourceSha256;
      break;
    case KernelFamily::kBatchNorm:
      expected_source = "normalization.py";
      expected_source_sha256 = kNormalizationSourceSha256;
      break;
    case KernelFamily::kBatchNormInference:
      expected_source = "normalization.py";
      expected_source_sha256 = kNormalizationSourceSha256;
      break;
    case KernelFamily::kRmsNorm:
      expected_source = "normalization.py";
      expected_source_sha256 = kNormalizationSourceSha256;
      break;
    case KernelFamily::kLayerNorm:
      expected_source = "normalization.py";
      expected_source_sha256 = kNormalizationSourceSha256;
      break;
  }
  if (kernel.at("provider").as_string() != "ascend_triton" ||
      kernel.at("ownership").as_string() != "platform" ||
      kernel.at("source").as_string() != expected_source) {
    artifact_failure(
        "Ascend stage must use its platform-owned kernel family");
  }
  result.source_sha256 = kernel.at("source_sha256").as_string();
  result.entry_point = kernel.at("entry_point").as_string();
  if (!is_sha256(result.source_sha256) ||
      result.source_sha256 != expected_source_sha256 ||
      result.entry_point != expected.entry_point) {
    artifact_failure("Ascend kernel source or entry identity is invalid");
  }
  result.source = validate_materialized_source(
      artifact_directory, kernel.at("materialized_source"), compiler_identity);
  if (kernel.at("materialized_source").at("sha256").as_string() !=
      result.source_sha256) {
    artifact_failure("canonical and materialized source hashes differ");
  }

  const auto& argument_sources = value.at("argument_sources").as_array();
  if (argument_sources.empty() ||
      argument_sources.size() > FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS) {
    artifact_failure("stage argument source count is invalid");
  }
  result.arguments.reserve(argument_sources.size());
  for (std::size_t index = 0; index < argument_sources.size(); ++index) {
    result.arguments.push_back(parse_argument_source(argument_sources[index],
                                                     index,
                                                     graph_workspace_size,
                                                     binding_uids));
  }
  std::vector<std::string_view> runtime_names;
  switch (expected.kernel_family) {
    case KernelFamily::kBinary:
      runtime_names = {"x_ptr", "y_ptr", "out_ptr", "n_elements"};
      break;
    case KernelFamily::kUnary:
      runtime_names = {"in_ptr", "out_ptr", "n_elements"};
      break;
    case KernelFamily::kTernary:
      runtime_names = {"x_ptr", "y_ptr", "t_ptr", "out_ptr", "n_elements"};
      break;
    case KernelFamily::kLayout:
      runtime_names = {"input_ptr", "output_ptr", "n_elements"};
      break;
    case KernelFamily::kReduction:
      runtime_names = {"input_ptr", "output_ptr", "n_elements"};
      break;
    case KernelFamily::kMatMul:
      runtime_names = {"a_ptr", "b_ptr", "output_ptr", "n_elements"};
      break;
    case KernelFamily::kConvolutionFprop:
      runtime_names =
          expected.entry_point == "convolution_fprop_im2col_kernel"
              ? std::vector<std::string_view>{
                    "input_ptr", "filter_ptr", "columns_ptr", "n_elements"}
              : std::vector<std::string_view>{
                    "input_ptr", "filter_ptr", "output_ptr", "n_elements"};
      break;
    case KernelFamily::kBatchNorm:
      runtime_names = {
          "x_ptr", "scale_ptr", "bias_ptr", "previous_running_mean_ptr",
          "previous_running_variance_ptr", "y_ptr", "mean_ptr",
          "inv_variance_ptr", "next_running_mean_ptr",
          "next_running_variance_ptr", "n_elements"};
      break;
    case KernelFamily::kBatchNormInference:
      runtime_names = {"x_ptr", "mean_ptr", "inv_variance_ptr", "scale_ptr",
                       "bias_ptr", "y_ptr", "n_elements"};
      break;
    case KernelFamily::kRmsNorm:
      runtime_names = {"x_ptr", "scale_ptr", "bias_ptr", "y_ptr",
                       "inv_variance_ptr", "n_elements"};
      break;
    case KernelFamily::kLayerNorm:
      runtime_names = {"x_ptr", "scale_ptr", "bias_ptr", "y_ptr",
                       "mean_ptr", "inv_variance_ptr", "n_elements"};
      break;
  }
  if (result.arguments.size() != runtime_names.size()) {
    artifact_failure("Ascend raw ABI slot count is invalid");
  }
  for (std::size_t index = 0; index < runtime_names.size(); ++index) {
    const ArgumentSource& argument = result.arguments[index];
    const std::size_t scalar_index = runtime_names.size() - 1;
    if (argument.name != runtime_names[index] ||
        (index < scalar_index && argument.type != RawArgumentType::kPointer) ||
        (index == scalar_index &&
         (argument.source != ArgumentSourceKind::kScalar ||
          argument.type != RawArgumentType::kI32))) {
      artifact_failure("Ascend raw ABI slot order is invalid");
    }
  }

  parse_autotune(value.at("autotune"), result);
  const auto& candidates = value.at("candidates").as_array();
  if (candidates.empty() || candidates.size() > kMaximumCandidates) {
    artifact_failure("Ascend candidate count is invalid");
  }
  result.candidates.reserve(candidates.size());
  std::set<std::string> candidate_ids;
  for (const Value& candidate : candidates) {
    LtjNpuRawCandidate parsed =
        parse_candidate(candidate,
                        result,
                        result.arguments,
                        expected,
                        expected_worker_count);
    if (!candidate_ids.insert(parsed.candidate_id).second) {
      artifact_failure("Ascend candidate ID is duplicated");
    }
    result.candidates.push_back(std::move(parsed));
  }
  if ((!result.autotune && result.candidates.size() != 1) ||
      (!result.autotune &&
       candidate_ids.find(result.selected_candidate) == candidate_ids.end()) ||
      (result.autotune &&
       result.candidates.size() !=
           tuning_configuration_count(result.kernel_family))) {
    artifact_failure("Ascend candidate set disagrees with autotune policy");
  }
  if (result.autotune) {
    const auto allowed_configurations =
        allowed_tuning_configurations(result.kernel_family);
    std::set<std::size_t> configuration_indices;
    for (const LtjNpuRawCandidate& actual : result.candidates) {
      const std::size_t configuration_index = tuning_configuration_index(
          result.kernel_family, actual.block_size, actual.num_warps,
          actual.num_stages);
      const AllowedTuningConfiguration& expected_configuration =
          allowed_configurations[configuration_index];
      if (actual.block_size != expected_configuration.block_size ||
          actual.num_warps != expected_configuration.num_warps ||
          actual.num_stages != expected_configuration.num_stages ||
          !configuration_indices.insert(configuration_index).second ||
          (&actual != &result.candidates.front() &&
           actual.argument_types != result.candidates.front().argument_types)) {
        artifact_failure("Ascend autotune candidate set differs from policy");
      }
    }
    if (configuration_indices.size() !=
        allowed_configurations.size()) {
      artifact_failure("Ascend autotune candidate set differs from policy");
    }
    if (result.candidate_identity !=
        expected_autotune_candidate_identity(result, expected)) {
      artifact_failure("Ascend autotune candidate identity differs");
    }
  }
  validate_stage_graph_contract(result, expected);
  return result;
}

std::string combined_source_hash(const std::vector<AscendStageArtifact>& stages) {
  std::string payload = "[";
  for (std::size_t index = 0; index < stages.size(); ++index) {
    if (index != 0) {
      payload += ',';
    }
    payload += '\"';
    payload += stages[index].source_sha256;
    payload += '\"';
  }
  payload += ']';
  return flagdnn::native::sha256(payload);
}

}  // namespace

AscendArtifact parse_ascend_artifact(
    const std::string& target_fingerprint,
    std::uint32_t ai_core_count,
    const flagdnnBackendBuildInputV2& input) {
  try {
    const std::uint32_t target_ai_core_count =
        parse_target_ai_core_count(target_fingerprint);
    if (target_ai_core_count != ai_core_count ||
        ai_core_count > kMaximumAiCoreCount) {
      artifact_failure(
          "Ascend engine AI core capability differs from target fingerprint");
    }
    const std::uint32_t expected_worker_count =
        ai_core_count * kWorkersPerAiCore;
    require(input.graph_ir != nullptr && input.graph_ir_size != 0,
            "graph IR is empty");
    require(input.artifact_directory != nullptr,
            "artifact directory is null");
    require(input.request_sha256 != nullptr, "request SHA-256 is null");
    const std::string_view graph_ir(
        static_cast<const char*>(input.graph_ir), input.graph_ir_size);
    const std::string_view request_hash(input.request_sha256);
    if (!is_sha256(request_hash) ||
        flagdnn::native::sha256(graph_ir) != request_hash) {
      artifact_failure("Graph IR SHA-256 does not match build input");
    }

    const Value request = flagdnn::native::json::parse(graph_ir);
    const std::string compiler_identity =
        request.at("compiler_identity").as_string();
    if (request.at("schema_version").as_int() != 3 ||
        request.at("flagdnn_version").as_string() != FLAGDNN_VERSION_STRING ||
        request.at("backend").as_string() != "ascend" ||
        request.at("target").as_string() != target_fingerprint ||
        !is_sha256(compiler_identity)) {
      artifact_failure("Ascend build request identity is invalid");
    }
    const std::filesystem::path artifact_directory(input.artifact_directory);
    const Value manifest = flagdnn::native::json::parse(read_text_file(
        artifact_directory / "manifest.json", kMaximumManifestSize));
    require_exact_keys(manifest,
                       {"schema_version", "artifact_kind", "flagdnn_version",
                        "backend", "target", "graph_node_count",
                        "request_sha256", "source_sha256", "compiler",
                        "workspace_size", "program"},
                       "Ascend artifact manifest");
    const std::size_t graph_node_count = checked_size(
        manifest.at("graph_node_count").as_int(), "graph_node_count");
    if (manifest.at("schema_version").as_int() != kArtifactSchemaVersion ||
        manifest.at("artifact_kind").as_string() !=
            "flagdnn_execution_program" ||
        manifest.at("flagdnn_version").as_string() !=
            FLAGDNN_VERSION_STRING ||
        manifest.at("backend").as_string() != "ascend" ||
        manifest.at("target").as_string() != target_fingerprint ||
        manifest.at("request_sha256").as_string() != request_hash ||
        manifest.at("compiler").at("identity_sha256").as_string() !=
            compiler_identity ||
        graph_node_count !=
            checked_size(request.at("graph").at("node_count").as_int(),
                         "node_count") ||
        !is_sha256(manifest.at("source_sha256").as_string())) {
      artifact_failure("Ascend artifact target or version differs");
    }

    if (manifest.at("program").at("schema_version").as_int() == 4) {
      return parse_extended_artifact(request, manifest, artifact_directory);
    }
    const ExpectedGraph expected_graph = parse_expected_graph(request);
    AscendArtifact result;
    result.workspace_size = checked_size(
        manifest.at("workspace_size").as_int(), "workspace_size");
    if (result.workspace_size != expected_graph.workspace_size) {
      artifact_failure("Ascend artifact workspace differs from Graph plan");
    }
    const Value& program = manifest.at("program");
    require_exact_keys(program, {"schema_version", "stage_count", "stages"},
                       "Ascend Execution Program");
    const auto& stages = program.at("stages").as_array();
    if (program.at("schema_version").as_int() !=
            kExecutionProgramVersion ||
        checked_size(program.at("stage_count").as_int(), "stage_count") !=
            stages.size() ||
        stages.empty() ||
        stages.size() != expected_graph.stages.size() ||
        stages.size() > FLAGDNN_BACKEND_MAX_EXECUTION_STAGES) {
      artifact_failure("Ascend Execution Program schema is invalid");
    }
    result.stages.reserve(stages.size());
    std::set<std::size_t> referenced_nodes;
    for (std::size_t index = 0; index < stages.size(); ++index) {
      AscendStageArtifact stage = parse_stage(stages[index],
                                             index,
                                             expected_graph.stages[index],
                                             artifact_directory,
                                             compiler_identity,
                                             result.workspace_size,
                                             result.binding_uids,
                                             expected_worker_count);
      referenced_nodes.insert(stage.source_node_ids.begin(),
                              stage.source_node_ids.end());
      result.stages.push_back(std::move(stage));
    }
    for (std::size_t index = 0; index < result.stages.size(); ++index) {
      const bool expected_autotune =
          expected_graph.autotune_requested &&
          expected_graph.stages[index].entry_point !=
              "convolution_fprop_im2col_kernel" &&
          tuning_configuration_count(
              expected_graph.stages[index].kernel_family) > 1U;
      if (result.stages[index].autotune != expected_autotune) {
        artifact_failure("Ascend artifact autotune mode differs from request");
      }
    }
    if (referenced_nodes.size() != graph_node_count ||
        combined_source_hash(result.stages) !=
            manifest.at("source_sha256").as_string()) {
      artifact_failure("Ascend stage graph/source identity differs");
    }
    for (std::size_t node = 0; node < graph_node_count; ++node) {
      if (referenced_nodes.find(node) == referenced_nodes.end()) {
        artifact_failure("Ascend stage references an unknown Graph node");
      }
    }
    std::sort(result.binding_uids.begin(), result.binding_uids.end());
    return result;
  } catch (const AscendError&) {
    throw;
  } catch (const std::exception& error) {
    throw AscendError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                      "cannot parse Ascend artifact: " +
                          std::string(error.what()));
  }
}

}  // namespace flagdnn::ascend
