/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/mthreads/artifact.hpp"

#include "backends/mthreads/error.hpp"
#include "runtime/json.hpp"
#include "runtime/sha256.hpp"

#include <flagdnn/version.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace flagdnn::mthreads {
namespace {

using JsonValue = flagdnn::native::json::Value;

constexpr std::size_t kMaximumMetadataSize = 16U << 20;
constexpr std::size_t kMaximumSourceSize = 16U << 20;
constexpr std::uintmax_t kMaximumArtifactSize = 128U << 20;
constexpr std::size_t kMaximumTensorRank = 8;
constexpr std::size_t kMaximumStages = 128;
constexpr std::size_t kMaximumVariants = 32;
constexpr std::size_t kMaximumArguments = 128;
constexpr std::size_t kMaximumTensorAlignment = 1U << 20;
constexpr std::size_t kMinimumWorkspaceAlignment = 256;
constexpr std::size_t kMaximumWorkspaceAlignment = 1ULL << 31;
constexpr std::size_t kPhaseOneWorkspaceSize = 4096;
constexpr std::size_t kPhaseOneWorkspaceAlignment = 256;
constexpr std::int64_t kMaximumLinearElements =
    static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) -
    ((1LL << 16) - 1);
constexpr std::string_view kBinarySourcePath = "kernels/binary.py";
constexpr std::string_view kUnarySourcePath = "kernels/unary.py";
constexpr std::string_view kIdentitySourcePath = "kernels/identity.py";
constexpr std::string_view kTernarySourcePath = "kernels/ternary.py";
constexpr std::string_view kLayoutSourcePath = "kernels/layout.py";
constexpr std::string_view kReductionSourcePath = "kernels/reduction.py";
constexpr std::string_view kMatmulSourcePath = "kernels/matmul.py";
constexpr std::string_view kConvolutionSourcePath =
    "kernels/convolution.py";
constexpr std::string_view kCompositeSourcePath = "kernels/composite.py";
constexpr std::string_view kConvBiasReluSourcePath =
    "kernels/conv_bias_relu.py";
constexpr std::string_view kNormalizationSourcePath =
    "kernels/normalization.py";
constexpr std::string_view kAttentionSourcePath = "kernels/attention.py";
constexpr std::string_view kSelectionCachePath = "tuning/stage-0.json";

[[noreturn]] void artifact_error(std::string message) {
  throw MthreadsError(
      FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED, std::move(message));
}

void require_exact_keys(
    const JsonValue& value,
    std::initializer_list<std::string_view> expected,
    std::string_view field) {
  const auto& object = value.as_object();
  const bool has_precision =
      object.contains("input_precision") &&
      (std::find(expected.begin(), expected.end(), "k") != expected.end() ||
       std::find(expected.begin(), expected.end(), "spatial_rank") !=
           expected.end());
  if (has_precision && (object.at("input_precision").as_int() < 0 ||
                        object.at("input_precision").as_int() > 2))
    artifact_error("input precision is invalid");
  if (object.size() != expected.size() + (has_precision ? 1U : 0U)) {
    artifact_error(
        "mthreads artifact/request object has invalid keys: " +
        std::string(field));
  }
  for (const std::string_view key : expected) {
    if (object.find(key) == object.end()) {
      artifact_error(
          "mthreads artifact/request object is missing key " +
          std::string(key) + ": " + std::string(field));
    }
  }
}

// Keep schema-3 requests from older clients compatible with the expanded
// pointwise attribute serialization. Non-default semantics are never dropped.
JsonValue normalize_pointwise_attributes(const JsonValue& request) {
  auto root = request.as_object();
  auto graph = root.at("graph").as_object();
  auto nodes = graph.at("nodes").as_array();
  for (auto& value : nodes) {
    auto node = value.as_object();
    auto attributes = node.at("attributes").as_object();
    const auto& operation = node.at("type").as_string();
    if (attributes.contains("pointwise_mode") ||
        operation == "binary_select") {
      for (const auto& [name, expected] :
           std::map<std::string, double>{{"relu_lower_clip", 0.0},
                                         {"relu_upper_clip", 0.0},
                                         {"relu_lower_clip_slope", 0.0},
                                         {"swish_beta", 1.0},
                                         {"elu_alpha", 1.0},
                                         {"softplus_beta", 1.0}}) {
        if (auto it = attributes.find(name); it != attributes.end()) {
          if (it->second.as_double() != expected)
            artifact_error("non-default pointwise attribute: " + name);
          attributes.erase(it);
        }
      }
      if (auto it = attributes.find("relu_upper_clip_set");
          it != attributes.end()) {
        if (it->second.as_bool())
          artifact_error("unexpected pointwise upper clip");
        attributes.erase(it);
      }
      node["attributes"] = JsonValue(std::move(attributes));
      value = JsonValue(std::move(node));
    }
  }
  graph["nodes"] = JsonValue(std::move(nodes));
  root["graph"] = JsonValue(std::move(graph));
  return JsonValue(std::move(root));
}

bool is_lower_sha256(std::string_view value) {
  return value.size() == 64 &&
         std::all_of(
             value.begin(), value.end(), [](unsigned char character) {
               return (character >= '0' && character <= '9') ||
                      (character >= 'a' && character <= 'f');
             });
}

bool is_power_of_two(std::size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

std::size_t checked_size(std::int64_t value, std::string_view field) {
  if (value < 0 ||
      static_cast<std::uint64_t>(value) >
          std::numeric_limits<std::size_t>::max()) {
    artifact_error(
        "mthreads field is outside size_t: " + std::string(field));
  }
  return static_cast<std::size_t>(value);
}

unsigned int checked_positive_unsigned(
    std::int64_t value, std::string_view field) {
  if (value <= 0 ||
      static_cast<std::uint64_t>(value) >
          std::numeric_limits<unsigned int>::max()) {
    artifact_error(
        "mthreads field is not a positive unsigned integer: " +
        std::string(field));
  }
  return static_cast<unsigned int>(value);
}

std::string read_file(
    const std::filesystem::path& path,
    std::size_t maximum_size,
    std::string_view description) {
  std::error_code error;
  const std::filesystem::file_status status =
      std::filesystem::symlink_status(path, error);
  if (error || std::filesystem::is_symlink(status) ||
      !std::filesystem::is_regular_file(status)) {
    artifact_error(
        "mthreads " + std::string(description) +
        " is missing, non-regular, or a symbolic link");
  }
  const std::uintmax_t file_size = std::filesystem::file_size(path, error);
  if (error || file_size > maximum_size) {
    artifact_error(
        "mthreads " + std::string(description) +
        " exceeds its size limit");
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    artifact_error(
        "cannot open mthreads " + std::string(description));
  }
  std::string result(static_cast<std::size_t>(file_size), '\0');
  input.read(result.data(), static_cast<std::streamsize>(result.size()));
  if (!input && !result.empty()) {
    artifact_error(
        "cannot read mthreads " + std::string(description));
  }
  return result;
}

bool path_is_below(
    const std::filesystem::path& root,
    const std::filesystem::path& child) {
  auto root_iterator = root.begin();
  auto child_iterator = child.begin();
  for (; root_iterator != root.end(); ++root_iterator, ++child_iterator) {
    if (child_iterator == child.end() ||
        *root_iterator != *child_iterator) {
      return false;
    }
  }
  return root != child;
}

bool printable_ascii_identifier(
    std::string_view value, std::size_t maximum_size) {
  if (value.empty() || value.size() > maximum_size) {
    return false;
  }
  const unsigned char first =
      static_cast<unsigned char>(value.front());
  if (!((first >= 'a' && first <= 'z') ||
        (first >= 'A' && first <= 'Z') || first == '_')) {
    return false;
  }
  return std::all_of(
      value.begin(), value.end(), [](unsigned char character) {
        return (character >= 'a' && character <= 'z') ||
               (character >= 'A' && character <= 'Z') ||
               (character >= '0' && character <= '9') ||
               character == '_';
      });
}

struct TensorSpec {
  std::int64_t uid = 0;
  std::string data_type;
  std::vector<std::int64_t> dimensions;
  std::vector<std::int64_t> strides;
  std::size_t alignment = 1;
  std::size_t storage_size = 0;
  bool is_virtual = false;
};

std::size_t element_size(std::string_view data_type) {
  if (data_type == "float32" || data_type == "int32") {
    return 4;
  }
  if (data_type == "float16" || data_type == "bfloat16") {
    return 2;
  }
  if (data_type == "boolean" || data_type == "fp8_e8m0") {
    return 1;
  }
  if (data_type == "fp8_e4m3" || data_type == "fp8_e5m2") {
    return 1;
  }
  artifact_error("mthreads pointwise data type is unsupported");
}

std::vector<std::int64_t> positive_i32_array(
    const JsonValue& value,
    std::string_view field,
    bool allow_empty = false) {
  const auto& array = value.as_array();
  if ((!allow_empty && array.empty()) ||
      array.size() > kMaximumTensorRank) {
    artifact_error(
        "mthreads tensor rank is invalid: " + std::string(field));
  }
  std::vector<std::int64_t> result;
  result.reserve(array.size());
  for (const JsonValue& entry : array) {
    const std::int64_t integer = entry.as_int();
    if (integer <= 0 ||
        integer > std::numeric_limits<std::int32_t>::max()) {
      artifact_error(
          "mthreads tensor dimension/stride is invalid: " +
          std::string(field));
    }
    result.push_back(integer);
  }
  return result;
}

TensorSpec parse_tensor(const JsonValue& value) {
  require_exact_keys(
      value,
      {"uid", "data_type", "dimensions", "strides", "alignment",
       "virtual"},
      "graph tensor");
  TensorSpec tensor;
  tensor.uid = value.at("uid").as_int();
  if (tensor.uid < 0) {
    artifact_error("mthreads tensor UID must be nonnegative");
  }
  tensor.data_type = value.at("data_type").as_string();
  const std::size_t bytes_per_element = element_size(tensor.data_type);
  tensor.dimensions =
      positive_i32_array(value.at("dimensions"), "dimensions", true);
  tensor.strides =
      positive_i32_array(value.at("strides"), "strides", true);
  if (tensor.dimensions.size() != tensor.strides.size()) {
    artifact_error("mthreads tensor dimensions and strides differ in rank");
  }
  tensor.alignment =
      checked_size(value.at("alignment").as_int(), "tensor.alignment");
  if (!is_power_of_two(tensor.alignment) ||
      tensor.alignment > kMaximumTensorAlignment) {
    artifact_error("mthreads tensor alignment is invalid");
  }
  tensor.is_virtual = value.at("virtual").as_bool();

  std::int64_t maximum_offset = 0;
  for (std::size_t index = 0; index < tensor.dimensions.size(); ++index) {
    const std::int64_t extent = tensor.dimensions[index] - 1;
    const std::int64_t stride = tensor.strides[index];
    if (extent != 0 &&
        stride > (std::numeric_limits<std::int64_t>::max() -
                  maximum_offset) /
                     extent) {
      artifact_error("mthreads tensor storage span overflows int64");
    }
    maximum_offset += extent * stride;
  }
  if (maximum_offset == std::numeric_limits<std::int64_t>::max() ||
      maximum_offset + 1 >
          std::numeric_limits<std::int64_t>::max() /
              static_cast<std::int64_t>(bytes_per_element)) {
    artifact_error("mthreads tensor storage size overflows int64");
  }
  tensor.storage_size = static_cast<std::size_t>(
      (maximum_offset + 1) *
      static_cast<std::int64_t>(bytes_per_element));

  std::vector<std::pair<std::int64_t, std::int64_t>> axes;
  for (std::size_t index = 0; index < tensor.dimensions.size(); ++index) {
    if (tensor.dimensions[index] > 1) {
      axes.emplace_back(tensor.strides[index], tensor.dimensions[index]);
    }
  }
  std::sort(axes.begin(), axes.end());
  std::int64_t required_span = 1;
  for (const auto& [stride, dimension] : axes) {
    if (stride < required_span ||
        stride > (std::numeric_limits<std::int64_t>::max() - required_span) /
                     (dimension - 1)) {
      artifact_error("mthreads tensor strides overlap");
    }
    required_span += stride * (dimension - 1);
  }
  return tensor;
}

bool is_physically_dense(const TensorSpec& tensor) {
  std::vector<std::pair<std::int64_t, std::int64_t>> axes;
  for (std::size_t index = 0; index < tensor.dimensions.size(); ++index) {
    if (tensor.dimensions[index] > 1) {
      axes.emplace_back(tensor.strides[index], tensor.dimensions[index]);
    }
  }
  std::sort(axes.begin(), axes.end());
  std::int64_t expected = 1;
  for (const auto& [stride, dimension] : axes) {
    if (stride != expected) {
      return false;
    }
    expected *= dimension;
  }
  return true;
}

bool is_row_major_contiguous(const TensorSpec& tensor) {
  std::int64_t expected = 1;
  for (std::size_t trailing = 0;
       trailing < tensor.dimensions.size(); ++trailing) {
    const std::size_t axis = tensor.dimensions.size() - 1 - trailing;
    if (tensor.strides[axis] != expected ||
        tensor.dimensions[axis] >
            std::numeric_limits<std::int64_t>::max() / expected) {
      return false;
    }
    expected *= tensor.dimensions[axis];
  }
  return true;
}

std::vector<std::int64_t> broadcast_dimensions(
    const TensorSpec& left, const TensorSpec& right) {
  const std::size_t rank =
      std::max(left.dimensions.size(), right.dimensions.size());
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
      artifact_error(
          "mthreads pointwise inputs are not broadcast-compatible");
    }
    result[rank - 1 - trailing] =
        std::max(left_dimension, right_dimension);
  }
  return result;
}

std::int32_t element_count(const TensorSpec& tensor) {
  std::int64_t result = 1;
  for (const std::int64_t dimension : tensor.dimensions) {
    if (dimension > kMaximumLinearElements / result) {
      artifact_error(
          "mthreads pointwise element count exceeds safe int32 linear "
          "index range");
    }
    result *= dimension;
  }
  return static_cast<std::int32_t>(result);
}

void validate_target(std::string_view target) {
  constexpr std::string_view prefix = "musa-";
  if (!target.starts_with(prefix) ||
      target.size() > FLAGDNN_BACKEND_MAX_TARGET_FINGERPRINT) {
    artifact_error("mthreads target fingerprint is invalid");
  }
  const std::size_t cc = target.rfind("-cc");
  const std::size_t warp = target.rfind("-w");
  if (cc == std::string_view::npos || warp == std::string_view::npos ||
      cc <= prefix.size() || warp <= cc + 3 || warp + 2 >= target.size()) {
    artifact_error("mthreads target fingerprint is invalid");
  }
  for (std::size_t index = prefix.size(); index < cc; ++index) {
    const unsigned char character =
        static_cast<unsigned char>(target[index]);
    if (!((character >= 'a' && character <= 'z') ||
          (character >= '0' && character <= '9') ||
          character == '_' || character == '.' || character == '+' ||
          character == '-')) {
      artifact_error("mthreads target fingerprint is invalid");
    }
  }
  int architecture = 0;
  int warp_size = 0;
  const auto architecture_result = std::from_chars(
      target.data() + cc + 3, target.data() + warp, architecture);
  const auto warp_result = std::from_chars(
      target.data() + warp + 2, target.data() + target.size(), warp_size);
  if (architecture_result.ec != std::errc{} ||
      architecture_result.ptr != target.data() + warp ||
      warp_result.ec != std::errc{} ||
      warp_result.ptr != target.data() + target.size() ||
      architecture <= 0 || architecture > 999 ||
      (warp_size != 32 && warp_size != 64)) {
    artifact_error("mthreads target attributes are unsupported");
  }
}

enum class PointwiseFamily {
  kBinary,
  kAddSquare,
  kConvBiasRelu,
  kUnary,
  kTernary,
  kLayout,
  kReduction,
  kMatmul,
  kConvolution,
  kNormalization,
  kBatchnorm,
  kBatchnormInference,
  kAttention,
};

struct PointwiseRequest {
  std::int64_t input_precision = 0;
  PointwiseFamily family = PointwiseFamily::kBinary;
  std::string flagdnn_version;
  std::string target;
  std::string compiler_identity;
  bool autotune = false;
  std::int64_t node_id = 0;
  std::string operation;
  std::int64_t pointwise_mode = 0;
  TensorSpec input;
  TensorSpec left;
  TensorSpec right;
  TensorSpec predicate;
  TensorSpec output;
  std::vector<std::int64_t> logical_input_dimensions;
  std::vector<std::int64_t> logical_input_strides;
  std::int64_t input_base = 0;
  std::int64_t reduction_mode = 0;
  std::int64_t reduction_axis = 0;
  bool keep_dimensions = false;
  std::int64_t outer = 0;
  std::int64_t extent = 0;
  std::int64_t inner = 0;
  std::int64_t matmul_batch = 0;
  std::int64_t matmul_m = 0;
  std::int64_t matmul_n = 0;
  std::int64_t matmul_k = 0;
  std::array<std::int64_t, 6> matmul_batch_dimensions{};
  std::array<std::int64_t, 6> matmul_a_batch_strides{};
  std::array<std::int64_t, 6> matmul_b_batch_strides{};
  std::array<std::int64_t, 6> matmul_output_batch_strides{};
  std::int64_t matmul_a_stride_m = 0;
  std::int64_t matmul_a_stride_k = 0;
  std::int64_t matmul_b_stride_k = 0;
  std::int64_t matmul_b_stride_n = 0;
  std::int64_t matmul_output_stride_m = 0;
  std::int64_t matmul_output_stride_n = 0;
  TensorSpec convolution_image;
  TensorSpec convolution_filter;
  TensorSpec convolution_bias;
  TensorSpec convolution_result;
  std::int64_t convolution_spatial_rank = 0;
  std::int64_t convolution_groups = 0;
  std::int64_t convolution_mode = 0;
  std::vector<std::int64_t> convolution_pre_padding;
  std::vector<std::int64_t> convolution_post_padding;
  std::vector<std::int64_t> convolution_stride;
  std::vector<std::int64_t> convolution_dilation;
  std::int64_t convolution_batch = 0;
  std::int64_t convolution_in_channels = 0;
  std::int64_t convolution_out_channels = 0;
  std::int64_t convolution_in_per_group = 0;
  std::int64_t convolution_out_per_group = 0;
  TensorSpec normalization_x;
  TensorSpec normalization_scale;
  TensorSpec normalization_bias;
  TensorSpec normalization_y;
  TensorSpec normalization_mean;
  TensorSpec normalization_inv_variance;
  TensorSpec normalization_previous_running_mean;
  TensorSpec normalization_previous_running_variance;
  TensorSpec normalization_next_running_mean;
  TensorSpec normalization_next_running_variance;
  std::int64_t normalization_rows = 0;
  std::int64_t normalization_elements = 0;
  std::int64_t normalization_batch = 0;
  std::int64_t normalization_channels = 0;
  std::int64_t normalization_spatial = 0;
  std::int64_t normalization_rank = 0;
  float normalization_epsilon = 0.0F;
  float normalization_momentum = 0.0F;
  std::map<std::string, TensorSpec> attention_tensors;
  std::int64_t attention_batch = 0;
  std::int64_t attention_heads = 0;
  std::int64_t attention_key_heads = 0;
  std::int64_t attention_value_heads = 0;
  std::int64_t attention_sequence_q = 0;
  std::int64_t attention_sequence_kv = 0;
  std::int64_t attention_head_dimension = 0;
  std::int64_t attention_value_dimension = 0;
  std::int64_t attention_q_per_k = 0;
  std::int64_t attention_q_per_v = 0;
  std::int64_t attention_min_diag = 0;
  std::int64_t attention_max_diag = 0;
  bool attention_has_bias = false;
  bool attention_has_dbias = false;
  bool attention_banded = false;
  bool attention_causal_top_left = false;
  bool attention_reverse_causal = false;
  bool attention_generate_stats = false;
  float attention_scale = 0.0F;
  float alpha = 0.0F;
  float negative_slope = 0.0F;
  float lower_clip = 0.0F;
  float upper_clip = 0.0F;
  std::int64_t has_upper_clip = 0;
  float swish_beta = 1.0F;
  float elu_alpha = 1.0F;
  float softplus_beta = 1.0F;
  std::int32_t n_elements = 0;
  std::vector<std::int64_t> external_binding_uids;
  bool dense = false;
};

bool is_comparison_mode(std::int64_t mode) {
  return mode >= 25 && mode <= 30;
}

bool is_logical_binary_mode(std::int64_t mode) {
  return mode == 31 || mode == 32;
}

bool is_floating_data_type(std::string_view data_type) {
  return data_type == "float32" || data_type == "float16" ||
         data_type == "bfloat16";
}

std::string binary_operation(std::int64_t mode) {
  switch (mode) {
    case 1:
      return "add";
    case 17:
      return "sub";
    case 18:
      return "mul";
    case 19:
      return "div";
    case 20:
      return "min";
    case 21:
      return "max";
    case 22:
      return "mod";
    case 23:
      return "pow";
    case 25:
      return "cmp_eq";
    case 26:
      return "cmp_neq";
    case 27:
      return "cmp_gt";
    case 28:
      return "cmp_ge";
    case 29:
      return "cmp_lt";
    case 30:
      return "cmp_le";
    case 31:
      return "logical_and";
    case 32:
      return "logical_or";
    case 40:
      return "sigmoid_backward";
    default:
      artifact_error("mthreads binary pointwise mode is unsupported");
  }
}

bool is_unary_mode(std::int64_t mode) {
  return (mode >= 2 && mode <= 16) || mode == 24 ||
         (mode >= 33 && mode <= 39);
}

std::string unary_operation(std::int64_t mode) {
  switch (mode) {
    case 2:
      return "relu";
    case 3:
      return "sqrt";
    case 4:
      return "erf";
    case 5:
      return "identity";
    case 6:
      return "exp";
    case 7:
      return "log";
    case 8:
      return "neg";
    case 9:
      return "abs";
    case 10:
      return "ceil";
    case 11:
      return "cos";
    case 12:
      return "floor";
    case 13:
      return "rsqrt";
    case 14:
      return "sin";
    case 15:
      return "tan";
    case 16:
      return "reciprocal";
    case 24:
      return "logical_not";
    case 33:
      return "sigmoid";
    case 34:
      return "tanh";
    case 35:
      return "elu";
    case 36:
      return "gelu";
    case 37:
      return "softplus";
    case 38:
      return "swish";
    case 39:
      return "gelu_approx_tanh";
    default:
      artifact_error("mthreads unary pointwise mode is unsupported");
  }
}

float checked_float32(const JsonValue& value, std::string_view field) {
  const double decoded = value.as_double();
  const float result = static_cast<float>(decoded);
  if (!std::isfinite(decoded) || !std::isfinite(result)) {
    artifact_error(
        "mthreads pointwise float is not representable as float32: " +
        std::string(field));
  }
  return result;
}

std::map<std::string, std::int64_t> parse_ports(
    const JsonValue& value,
    const std::map<std::int64_t, TensorSpec>& tensors,
    std::string_view field) {
  std::map<std::string, std::int64_t> result;
  const auto& array = value.as_array();
  if (array.size() > 32) {
    artifact_error("mthreads pointwise port count is invalid");
  }
  for (const JsonValue& entry : array) {
    require_exact_keys(entry, {"name", "uid"}, field);
    const std::string& name = entry.at("name").as_string();
    if (name.empty() || name.size() > 128 ||
        name.find('\0') != std::string::npos) {
      artifact_error("mthreads Graph port name is invalid");
    }
    const std::int64_t uid = entry.at("uid").as_int();
    if (uid < 0 || tensors.find(uid) == tensors.end() ||
        !result.emplace(name, uid).second) {
      artifact_error(
          "mthreads Graph port UID/role is invalid or duplicated");
    }
  }
  return result;
}

PointwiseRequest parse_binary_request(
    const JsonValue& request,
    std::string_view expected_target) {
  require_exact_keys(
      request,
      {"schema_version", "flagdnn_version", "backend", "target",
       "build_options", "graph", "compiler_identity"},
      "request");
  PointwiseRequest result;
  result.flagdnn_version = request.at("flagdnn_version").as_string();
  result.target = request.at("target").as_string();
  result.compiler_identity =
      request.at("compiler_identity").as_string();
  validate_target(result.target);
  if (request.at("schema_version").as_int() != 3 ||
      result.flagdnn_version != FLAGDNN_VERSION_STRING ||
      request.at("backend").as_string() != "mthreads" ||
      result.target != expected_target ||
      !is_lower_sha256(result.compiler_identity)) {
    artifact_error("mthreads Graph IR identity is invalid");
  }

  const JsonValue& build_options = request.at("build_options");
  require_exact_keys(
      build_options, {"heuristic_modes", "autotune"}, "build_options");
  const auto& heuristic_modes =
      build_options.at("heuristic_modes").as_array();
  if (heuristic_modes.empty()) {
    artifact_error("mthreads heuristic modes are empty");
  }
  std::set<std::string> unique_modes;
  for (const JsonValue& mode_value : heuristic_modes) {
    const std::string& mode = mode_value.as_string();
    if ((mode != "A" && mode != "FALLBACK") ||
        !unique_modes.insert(mode).second) {
      artifact_error("mthreads heuristic modes are invalid");
    }
  }
  result.autotune = build_options.at("autotune").as_bool();

  const JsonValue& graph = request.at("graph");
  require_exact_keys(
      graph, {"name", "tensor_count", "tensors", "node_count", "nodes"},
      "graph");
  const std::string& graph_name = graph.at("name").as_string();
  if (graph_name.size() > (1U << 20) ||
      graph_name.find('\0') != std::string::npos) {
    artifact_error("mthreads Graph name is invalid");
  }
  const auto& tensor_values = graph.at("tensors").as_array();
  if (graph.at("tensor_count").as_int() != 3 ||
      tensor_values.size() != 3) {
    artifact_error(
        "mthreads binary pointwise requires exactly three tensors");
  }
  std::map<std::int64_t, TensorSpec> tensors;
  for (const JsonValue& tensor_value : tensor_values) {
    TensorSpec tensor = parse_tensor(tensor_value);
    result.external_binding_uids.push_back(tensor.uid);
    if (!tensors.emplace(tensor.uid, std::move(tensor)).second) {
      artifact_error("mthreads Graph tensor UID is duplicated");
    }
  }

  const auto& nodes = graph.at("nodes").as_array();
  if (graph.at("node_count").as_int() != 1 || nodes.size() != 1) {
    artifact_error(
        "mthreads pointwise compiler requires exactly one node");
  }
  const JsonValue& node = nodes.front();
  require_exact_keys(
      node,
      {"id", "type", "name", "compute_data_type", "inputs", "outputs",
       "attributes"},
      "binary pointwise node");
  result.node_id = node.at("id").as_int();
  const std::string& node_name = node.at("name").as_string();
  if (result.node_id < 0 || node_name.size() > 4096 ||
      node_name.find('\0') != std::string::npos) {
    artifact_error("mthreads pointwise node identity is invalid");
  }
  const auto inputs =
      parse_ports(node.at("inputs"), tensors, "pointwise input");
  const auto outputs =
      parse_ports(node.at("outputs"), tensors, "pointwise output");
  if (inputs.size() != 2 || inputs.find("left") == inputs.end() ||
      inputs.find("right") == inputs.end() || outputs.size() != 1 ||
      outputs.find("output") == outputs.end()) {
    artifact_error("mthreads binary pointwise port roles are invalid");
  }
  const std::set<std::int64_t> referenced = {
      inputs.at("left"), inputs.at("right"), outputs.at("output")};
  if (referenced.size() != 3) {
    artifact_error("mthreads pointwise tensor roles must be distinct");
  }
  result.left = tensors.at(inputs.at("left"));
  result.right = tensors.at(inputs.at("right"));
  result.output = tensors.at(outputs.at("output"));
  if (result.left.is_virtual || result.right.is_virtual ||
      result.output.is_virtual) {
    artifact_error(
        "mthreads pointwise tensors must be externally bound");
  }
  if (result.left.data_type != result.right.data_type) {
    artifact_error("mthreads pointwise input data types do not match");
  }
  if (broadcast_dimensions(result.left, result.right) !=
      result.output.dimensions) {
    artifact_error(
        "mthreads pointwise output shape is not the broadcast result");
  }
  result.n_elements = element_count(result.output);

  const JsonValue& attributes = node.at("attributes");
  if (node.at("type").as_string() == "sigmoid_backward" &&
      attributes.as_object().contains("has_upper_clip")) {
    require_exact_keys(
        attributes,
        {"mode", "alpha", "n_elements", "pointwise_mode", "has_upper_clip"},
        "sigmoid backward attributes");
    if (attributes.at("has_upper_clip").as_int() != 0)
      artifact_error("mthreads sigmoid backward does not support clipping");
  } else {
    require_exact_keys(attributes,
                       {"mode", "alpha", "n_elements", "pointwise_mode"},
                       "binary pointwise attributes");
  }
  result.pointwise_mode = attributes.at("mode").as_int();
  result.operation = binary_operation(result.pointwise_mode);
  if (attributes.at("pointwise_mode").as_int() !=
          result.pointwise_mode ||
      node.at("type").as_string() != result.operation ||
      attributes.at("n_elements").as_int() != result.n_elements) {
    artifact_error(
        "mthreads pointwise attributes do not match the Graph");
  }
  const std::string& compute_type =
      node.at("compute_data_type").as_string();
  if (is_comparison_mode(result.pointwise_mode)) {
    if ((!is_floating_data_type(result.left.data_type) &&
         result.left.data_type != "int32") ||
        result.output.data_type != "boolean" || compute_type != "boolean") {
      artifact_error(
          "mthreads comparison pointwise storage/compute types differ");
    }
  } else if (is_logical_binary_mode(result.pointwise_mode)) {
    if (result.left.data_type != "boolean" ||
        result.output.data_type != "boolean" ||
        compute_type != "boolean") {
      artifact_error(
          "mthreads logical pointwise storage/compute types differ");
    }
  } else if ((!is_floating_data_type(result.left.data_type) &&
              result.left.data_type != "int32") ||
             result.output.data_type != result.left.data_type ||
             compute_type != "float32") {
    artifact_error(
        "mthreads numeric pointwise storage/compute types differ");
  }
  if (result.pointwise_mode == 40 &&
      (result.left.dimensions != result.right.dimensions ||
       result.output.dimensions != result.left.dimensions)) {
    artifact_error(
        "mthreads sigmoid backward tensors must have equal shapes");
  }
  const double alpha = attributes.at("alpha").as_double();
  result.alpha = static_cast<float>(alpha);
  if (!std::isfinite(alpha) || !std::isfinite(result.alpha)) {
    artifact_error(
        "mthreads pointwise alpha is not representable as float32");
  }
  if (result.pointwise_mode != 1 && result.pointwise_mode != 17 &&
      result.alpha != 1.0F) {
    artifact_error("mthreads pointwise alpha is invalid for its mode");
  }
  result.dense =
      result.left.dimensions == result.right.dimensions &&
      result.left.dimensions == result.output.dimensions &&
      result.left.strides == result.right.strides &&
      result.left.strides == result.output.strides &&
      is_physically_dense(result.left) &&
      is_physically_dense(result.right) &&
      is_physically_dense(result.output);
  return result;
}

PointwiseRequest parse_add_square_request(
    const JsonValue& request,
    std::string_view expected_target) {
  require_exact_keys(
      request,
      {"schema_version", "flagdnn_version", "backend", "target",
       "build_options", "graph", "compiler_identity"},
      "request");
  PointwiseRequest result;
  result.family = PointwiseFamily::kAddSquare;
  result.flagdnn_version = request.at("flagdnn_version").as_string();
  result.target = request.at("target").as_string();
  result.compiler_identity =
      request.at("compiler_identity").as_string();
  validate_target(result.target);
  if (request.at("schema_version").as_int() != 3 ||
      result.flagdnn_version != FLAGDNN_VERSION_STRING ||
      request.at("backend").as_string() != "mthreads" ||
      result.target != expected_target ||
      !is_lower_sha256(result.compiler_identity)) {
    artifact_error("mthreads AddSquare Graph IR identity is invalid");
  }

  const JsonValue& build_options = request.at("build_options");
  require_exact_keys(
      build_options, {"heuristic_modes", "autotune"}, "build_options");
  const auto& heuristic_modes =
      build_options.at("heuristic_modes").as_array();
  if (heuristic_modes.empty()) {
    artifact_error("mthreads AddSquare heuristic modes are empty");
  }
  std::set<std::string> unique_modes;
  for (const JsonValue& mode_value : heuristic_modes) {
    const std::string& mode = mode_value.as_string();
    if ((mode != "A" && mode != "FALLBACK") ||
        !unique_modes.insert(mode).second) {
      artifact_error("mthreads AddSquare heuristic modes are invalid");
    }
  }
  result.autotune = build_options.at("autotune").as_bool();

  const JsonValue& graph = request.at("graph");
  require_exact_keys(
      graph, {"name", "tensor_count", "tensors", "node_count", "nodes"},
      "graph");
  const std::string& graph_name = graph.at("name").as_string();
  if (graph_name.size() > (1U << 20) ||
      graph_name.find('\0') != std::string::npos) {
    artifact_error("mthreads AddSquare Graph name is invalid");
  }
  const auto& tensor_values = graph.at("tensors").as_array();
  if (graph.at("tensor_count").as_int() != 4 ||
      tensor_values.size() != 4) {
    artifact_error("mthreads AddSquare requires exactly four tensors");
  }
  std::map<std::int64_t, TensorSpec> tensors;
  for (const JsonValue& tensor_value : tensor_values) {
    TensorSpec tensor = parse_tensor(tensor_value);
    if (!tensor.is_virtual) {
      result.external_binding_uids.push_back(tensor.uid);
    }
    if (!tensors.emplace(tensor.uid, std::move(tensor)).second) {
      artifact_error("mthreads AddSquare tensor UID is duplicated");
    }
  }

  const auto& nodes = graph.at("nodes").as_array();
  if (graph.at("node_count").as_int() != 2 || nodes.size() != 2) {
    artifact_error("mthreads AddSquare requires exactly two nodes");
  }
  const JsonValue& square_node = nodes[0];
  const JsonValue& add_node = nodes[1];
  for (const JsonValue* node : {&square_node, &add_node}) {
    require_exact_keys(
        *node,
        {"id", "type", "name", "compute_data_type", "inputs", "outputs",
         "attributes"},
        "AddSquare node");
    const std::int64_t node_id = node->at("id").as_int();
    const std::string& node_name = node->at("name").as_string();
    if (node_id < 0 || node_name.size() > 4096 ||
        node_name.find('\0') != std::string::npos) {
      artifact_error("mthreads AddSquare node identity is invalid");
    }
  }
  if (square_node.at("id").as_int() == add_node.at("id").as_int()) {
    artifact_error("mthreads AddSquare node IDs must be distinct");
  }
  const auto square_inputs =
      parse_ports(square_node.at("inputs"), tensors, "AddSquare Mul input");
  const auto square_outputs =
      parse_ports(square_node.at("outputs"), tensors, "AddSquare Mul output");
  const auto add_inputs =
      parse_ports(add_node.at("inputs"), tensors, "AddSquare Add input");
  const auto add_outputs =
      parse_ports(add_node.at("outputs"), tensors, "AddSquare Add output");
  if (square_node.at("type").as_string() != "mul" ||
      square_inputs.size() != 2 ||
      square_inputs.find("left") == square_inputs.end() ||
      square_inputs.find("right") == square_inputs.end() ||
      square_inputs.at("left") != square_inputs.at("right") ||
      square_outputs.size() != 1 ||
      square_outputs.find("output") == square_outputs.end() ||
      add_node.at("type").as_string() != "add" ||
      add_inputs.size() != 2 ||
      add_inputs.find("left") == add_inputs.end() ||
      add_inputs.find("right") == add_inputs.end() ||
      add_outputs.size() != 1 ||
      add_outputs.find("output") == add_outputs.end() ||
      add_inputs.at("right") != square_outputs.at("output")) {
    artifact_error("mthreads AddSquare Graph topology is invalid");
  }

  const std::int64_t right_uid = square_inputs.at("left");
  const std::int64_t square_uid = square_outputs.at("output");
  const std::int64_t left_uid = add_inputs.at("left");
  const std::int64_t output_uid = add_outputs.at("output");
  const std::set<std::int64_t> referenced = {
      right_uid, square_uid, left_uid, output_uid};
  if (referenced.size() != 4 || referenced.size() != tensors.size()) {
    artifact_error("mthreads AddSquare tensor roles are invalid");
  }
  result.right = tensors.at(right_uid);
  const TensorSpec& square = tensors.at(square_uid);
  result.left = tensors.at(left_uid);
  result.output = tensors.at(output_uid);
  if (result.left.is_virtual || result.right.is_virtual ||
      !square.is_virtual || result.output.is_virtual ||
      result.external_binding_uids.size() != 3) {
    artifact_error(
        "mthreads AddSquare virtual/external tensor roles are invalid");
  }
  if ((!is_floating_data_type(result.output.data_type) &&
       result.output.data_type != "int32") ||
      result.left.data_type != result.output.data_type ||
      result.right.data_type != result.output.data_type ||
      square.data_type != result.output.data_type ||
      result.left.dimensions != result.output.dimensions ||
      result.right.dimensions != result.output.dimensions ||
      square.dimensions != result.output.dimensions ||
      result.left.strides != result.output.strides ||
      result.right.strides != result.output.strides ||
      square.strides != result.output.strides ||
      !is_physically_dense(result.left) ||
      !is_physically_dense(result.right) || !is_physically_dense(square) ||
      !is_physically_dense(result.output) ||
      square_node.at("compute_data_type").as_string() != "float32" ||
      add_node.at("compute_data_type").as_string() != "float32") {
    artifact_error("mthreads AddSquare tensor dtype/shape/layout is invalid");
  }
  result.n_elements = element_count(result.output);
  for (const auto& [node, expected_mode] :
       std::array<std::pair<const JsonValue*, std::int64_t>, 2>{
           std::pair{&square_node, std::int64_t{18}},
           std::pair{&add_node, std::int64_t{1}}}) {
    const JsonValue& attributes = node->at("attributes");
    require_exact_keys(
        attributes, {"mode", "alpha", "n_elements", "pointwise_mode"},
        "AddSquare attributes");
    const float alpha = checked_float32(attributes.at("alpha"), "alpha");
    if (attributes.at("mode").as_int() != expected_mode ||
        attributes.at("pointwise_mode").as_int() != expected_mode ||
        attributes.at("n_elements").as_int() != result.n_elements ||
        alpha != 1.0F) {
      artifact_error("mthreads AddSquare node attributes are invalid");
    }
  }
  result.node_id = add_node.at("id").as_int();
  result.operation = "add_square";
  result.dense = true;
  return result;
}

PointwiseRequest parse_conv_bias_relu_request(
    const JsonValue& request,
    std::string_view expected_target) {
  require_exact_keys(
      request,
      {"schema_version", "flagdnn_version", "backend", "target",
       "build_options", "graph", "compiler_identity"},
      "request");
  PointwiseRequest result;
  result.family = PointwiseFamily::kConvBiasRelu;
  result.flagdnn_version = request.at("flagdnn_version").as_string();
  result.target = request.at("target").as_string();
  result.compiler_identity =
      request.at("compiler_identity").as_string();
  validate_target(result.target);
  if (request.at("schema_version").as_int() != 3 ||
      result.flagdnn_version != FLAGDNN_VERSION_STRING ||
      request.at("backend").as_string() != "mthreads" ||
      result.target != expected_target ||
      !is_lower_sha256(result.compiler_identity)) {
    artifact_error("mthreads ConvBiasRelu Graph IR identity is invalid");
  }

  const JsonValue& build_options = request.at("build_options");
  require_exact_keys(
      build_options, {"heuristic_modes", "autotune"}, "build_options");
  const auto& heuristic_modes =
      build_options.at("heuristic_modes").as_array();
  if (heuristic_modes.empty()) {
    artifact_error("mthreads ConvBiasRelu heuristic modes are empty");
  }
  std::set<std::string> unique_modes;
  for (const JsonValue& mode_value : heuristic_modes) {
    const std::string& mode = mode_value.as_string();
    if ((mode != "A" && mode != "FALLBACK") ||
        !unique_modes.insert(mode).second) {
      artifact_error("mthreads ConvBiasRelu heuristic modes are invalid");
    }
  }
  result.autotune = build_options.at("autotune").as_bool();

  const JsonValue& graph = request.at("graph");
  require_exact_keys(
      graph, {"name", "tensor_count", "tensors", "node_count", "nodes"},
      "graph");
  const std::string& graph_name = graph.at("name").as_string();
  if (graph_name.size() > (1U << 20) ||
      graph_name.find('\0') != std::string::npos) {
    artifact_error("mthreads ConvBiasRelu Graph name is invalid");
  }
  const auto& tensor_values = graph.at("tensors").as_array();
  if (graph.at("tensor_count").as_int() != 6 ||
      tensor_values.size() != 6) {
    artifact_error("mthreads ConvBiasRelu requires exactly six tensors");
  }
  std::map<std::int64_t, TensorSpec> tensors;
  for (const JsonValue& tensor_value : tensor_values) {
    TensorSpec tensor = parse_tensor(tensor_value);
    if (!tensor.is_virtual) {
      result.external_binding_uids.push_back(tensor.uid);
    }
    if (!tensors.emplace(tensor.uid, std::move(tensor)).second) {
      artifact_error("mthreads ConvBiasRelu tensor UID is duplicated");
    }
  }

  const auto& nodes = graph.at("nodes").as_array();
  if (graph.at("node_count").as_int() != 3 || nodes.size() != 3) {
    artifact_error("mthreads ConvBiasRelu requires exactly three nodes");
  }
  const JsonValue& convolution_node = nodes[0];
  const JsonValue& bias_node = nodes[1];
  const JsonValue& relu_node = nodes[2];
  std::set<std::int64_t> node_ids;
  for (const JsonValue* node :
       {&convolution_node, &bias_node, &relu_node}) {
    require_exact_keys(
        *node,
        {"id", "type", "name", "compute_data_type", "inputs", "outputs",
         "attributes"},
        "ConvBiasRelu node");
    const std::int64_t node_id = node->at("id").as_int();
    const std::string& node_name = node->at("name").as_string();
    if (node_id < 0 || node_name.size() > 4096 ||
        node_name.find('\0') != std::string::npos ||
        !node_ids.insert(node_id).second) {
      artifact_error("mthreads ConvBiasRelu node identity is invalid");
    }
  }

  const auto convolution_inputs = parse_ports(
      convolution_node.at("inputs"), tensors, "ConvBiasRelu convolution input");
  const auto convolution_outputs = parse_ports(
      convolution_node.at("outputs"), tensors,
      "ConvBiasRelu convolution output");
  const auto bias_inputs =
      parse_ports(bias_node.at("inputs"), tensors, "ConvBiasRelu bias input");
  const auto bias_outputs = parse_ports(
      bias_node.at("outputs"), tensors, "ConvBiasRelu bias output");
  const auto relu_inputs =
      parse_ports(relu_node.at("inputs"), tensors, "ConvBiasRelu ReLU input");
  const auto relu_outputs = parse_ports(
      relu_node.at("outputs"), tensors, "ConvBiasRelu ReLU output");
  if (convolution_node.at("type").as_string() != "convolution_fprop" ||
      convolution_inputs.size() != 2 ||
      convolution_inputs.find("input") == convolution_inputs.end() ||
      convolution_inputs.find("filter") == convolution_inputs.end() ||
      convolution_outputs.size() != 1 ||
      convolution_outputs.find("output") == convolution_outputs.end() ||
      bias_node.at("type").as_string() != "add" ||
      bias_inputs.size() != 2 ||
      bias_inputs.find("left") == bias_inputs.end() ||
      bias_inputs.find("right") == bias_inputs.end() ||
      bias_outputs.size() != 1 ||
      bias_outputs.find("output") == bias_outputs.end() ||
      bias_inputs.at("left") != convolution_outputs.at("output") ||
      relu_node.at("type").as_string() != "relu" ||
      relu_inputs.size() != 1 ||
      relu_inputs.find("input") == relu_inputs.end() ||
      relu_outputs.size() != 1 ||
      relu_outputs.find("output") == relu_outputs.end() ||
      relu_inputs.at("input") != bias_outputs.at("output")) {
    artifact_error("mthreads ConvBiasRelu Graph topology is invalid");
  }

  const std::int64_t image_uid = convolution_inputs.at("input");
  const std::int64_t filter_uid = convolution_inputs.at("filter");
  const std::int64_t convolution_uid = convolution_outputs.at("output");
  const std::int64_t bias_uid = bias_inputs.at("right");
  const std::int64_t biased_uid = bias_outputs.at("output");
  const std::int64_t output_uid = relu_outputs.at("output");
  const std::set<std::int64_t> referenced = {
      image_uid, filter_uid, convolution_uid, bias_uid, biased_uid, output_uid};
  if (referenced.size() != 6 || referenced.size() != tensors.size()) {
    artifact_error("mthreads ConvBiasRelu tensor roles are invalid");
  }
  result.convolution_image = tensors.at(image_uid);
  result.convolution_filter = tensors.at(filter_uid);
  const TensorSpec& convolution = tensors.at(convolution_uid);
  result.convolution_bias = tensors.at(bias_uid);
  const TensorSpec& biased = tensors.at(biased_uid);
  result.convolution_result = tensors.at(output_uid);
  if (result.convolution_image.is_virtual ||
      result.convolution_filter.is_virtual || !convolution.is_virtual ||
      result.convolution_bias.is_virtual || !biased.is_virtual ||
      result.convolution_result.is_virtual ||
      result.external_binding_uids.size() != 4) {
    artifact_error(
        "mthreads ConvBiasRelu virtual/external tensor roles are invalid");
  }
  const std::array<const TensorSpec*, 6> graph_tensors = {
      &result.convolution_image,
      &result.convolution_filter,
      &convolution,
      &result.convolution_bias,
      &biased,
      &result.convolution_result};
  if (!is_floating_data_type(result.convolution_image.data_type) ||
      std::any_of(
          graph_tensors.begin(), graph_tensors.end(),
          [&](const TensorSpec* tensor) {
            return tensor->data_type !=
                       result.convolution_image.data_type ||
                   tensor->dimensions.size() != 4 ||
                   !is_physically_dense(*tensor);
          }) ||
      convolution_node.at("compute_data_type").as_string() != "float32" ||
      bias_node.at("compute_data_type").as_string() != "float32" ||
      relu_node.at("compute_data_type").as_string() != "float32") {
    artifact_error(
        "mthreads ConvBiasRelu tensor dtype/rank/layout is invalid");
  }

  const JsonValue& convolution_attributes =
      convolution_node.at("attributes");
  require_exact_keys(
      convolution_attributes,
      {"spatial_rank", "groups", "n_outputs", "pre_padding",
       "post_padding", "stride", "dilation"},
      "ConvBiasRelu convolution attributes");
  result.convolution_spatial_rank =
      convolution_attributes.at("spatial_rank").as_int();
  result.convolution_groups =
      convolution_attributes.at("groups").as_int();
  result.convolution_mode = 0;
  if (result.convolution_spatial_rank != 2 ||
      result.convolution_groups <= 0 ||
      result.convolution_groups >
          std::numeric_limits<std::int32_t>::max()) {
    artifact_error("mthreads ConvBiasRelu convolution metadata is invalid");
  }
  const auto spatial_array =
      [&](std::string_view name, std::int64_t minimum) {
        const auto& values = convolution_attributes.at(name).as_array();
        if (values.size() != 2) {
          artifact_error(
              "mthreads ConvBiasRelu spatial attribute length is invalid");
        }
        std::vector<std::int64_t> decoded;
        decoded.reserve(2);
        for (const JsonValue& value : values) {
          const std::int64_t entry = value.as_int();
          if (entry < minimum ||
              entry > std::numeric_limits<std::int32_t>::max()) {
            artifact_error(
                "mthreads ConvBiasRelu spatial attribute is invalid");
          }
          decoded.push_back(entry);
        }
        return decoded;
      };
  result.convolution_pre_padding = spatial_array("pre_padding", 0);
  result.convolution_post_padding = spatial_array("post_padding", 0);
  result.convolution_stride = spatial_array("stride", 1);
  result.convolution_dilation = spatial_array("dilation", 1);

  result.convolution_batch = result.convolution_image.dimensions[0];
  result.convolution_in_channels = result.convolution_image.dimensions[1];
  result.convolution_out_channels = result.convolution_filter.dimensions[0];
  if (result.convolution_in_channels % result.convolution_groups != 0 ||
      result.convolution_out_channels % result.convolution_groups != 0) {
    artifact_error("mthreads ConvBiasRelu grouped channels are invalid");
  }
  result.convolution_in_per_group =
      result.convolution_in_channels / result.convolution_groups;
  result.convolution_out_per_group =
      result.convolution_out_channels / result.convolution_groups;
  if (result.convolution_filter.dimensions[1] !=
      result.convolution_in_per_group) {
    artifact_error("mthreads ConvBiasRelu filter channels are invalid");
  }
  std::vector<std::int64_t> expected_output = {
      result.convolution_batch, result.convolution_out_channels};
  for (std::size_t axis = 0; axis < 2; ++axis) {
    const std::int64_t input_size =
        result.convolution_image.dimensions[axis + 2];
    const std::int64_t filter_size =
        result.convolution_filter.dimensions[axis + 2];
    const std::int64_t dilation = result.convolution_dilation[axis];
    const std::int64_t effective = (filter_size - 1) * dilation + 1;
    const std::int64_t padded =
        input_size + result.convolution_pre_padding[axis] +
        result.convolution_post_padding[axis];
    if (padded < effective) {
      artifact_error(
          "mthreads ConvBiasRelu filter exceeds the padded input");
    }
    expected_output.push_back(
        (padded - effective) / result.convolution_stride[axis] + 1);
  }
  if (convolution.dimensions != expected_output ||
      biased.dimensions != expected_output ||
      result.convolution_result.dimensions != expected_output ||
      convolution.strides != biased.strides ||
      convolution.strides != result.convolution_result.strides ||
      result.convolution_bias.dimensions !=
          std::vector<std::int64_t>{
              1, result.convolution_out_channels, 1, 1} ||
      result.convolution_bias.strides[1] != 1 ||
      broadcast_dimensions(convolution, result.convolution_bias) !=
          biased.dimensions) {
    artifact_error("mthreads ConvBiasRelu tensor geometry is invalid");
  }
  result.n_elements = element_count(result.convolution_result);
  if (convolution_attributes.at("n_outputs").as_int() !=
          result.n_elements ||
      element_count(convolution) != result.n_elements) {
    artifact_error("mthreads ConvBiasRelu output count is invalid");
  }

  const JsonValue& bias_attributes = bias_node.at("attributes");
  require_exact_keys(
      bias_attributes, {"mode", "alpha", "n_elements", "pointwise_mode"},
      "ConvBiasRelu bias attributes");
  if (bias_attributes.at("mode").as_int() != 1 ||
      bias_attributes.at("pointwise_mode").as_int() != 1 ||
      bias_attributes.at("n_elements").as_int() != result.n_elements ||
      checked_float32(bias_attributes.at("alpha"), "bias alpha") != 1.0F) {
    artifact_error("mthreads ConvBiasRelu bias attributes are invalid");
  }
  const JsonValue& relu_attributes = relu_node.at("attributes");
  require_exact_keys(
      relu_attributes,
      {"mode", "relu_lower_clip", "relu_upper_clip",
       "relu_lower_clip_slope", "relu_upper_clip_set", "swish_beta",
       "elu_alpha", "softplus_beta", "n_elements", "has_upper_clip",
       "negative_slope", "lower_clip", "upper_clip"},
      "ConvBiasRelu ReLU attributes");
  if (relu_attributes.at("mode").as_int() != 2 ||
      relu_attributes.at("n_elements").as_int() != result.n_elements ||
      relu_attributes.at("has_upper_clip").as_int() != 0 ||
      relu_attributes.at("relu_upper_clip_set").as_bool() ||
      checked_float32(
          relu_attributes.at("relu_lower_clip"), "relu_lower_clip") !=
          0.0F ||
      checked_float32(
          relu_attributes.at("relu_upper_clip"), "relu_upper_clip") !=
          0.0F ||
      checked_float32(
          relu_attributes.at("relu_lower_clip_slope"),
          "relu_lower_clip_slope") != 0.0F ||
      checked_float32(relu_attributes.at("swish_beta"), "swish_beta") !=
          1.0F ||
      checked_float32(relu_attributes.at("elu_alpha"), "elu_alpha") !=
          1.0F ||
      checked_float32(
          relu_attributes.at("softplus_beta"), "softplus_beta") != 1.0F ||
      checked_float32(
          relu_attributes.at("negative_slope"), "negative_slope") !=
          0.0F ||
      checked_float32(relu_attributes.at("lower_clip"), "lower_clip") !=
          0.0F ||
      checked_float32(relu_attributes.at("upper_clip"), "upper_clip") !=
          0.0F) {
    artifact_error("mthreads ConvBiasRelu ReLU attributes are invalid");
  }

  const std::int64_t output_spatial =
      result.convolution_result.dimensions[2] *
      result.convolution_result.dimensions[3];
  const std::int64_t reduction_extent =
      result.convolution_in_per_group *
      result.convolution_filter.dimensions[2] *
      result.convolution_filter.dimensions[3];
  if (result.convolution_batch >
          std::numeric_limits<std::int32_t>::max() /
              result.convolution_groups ||
      output_spatial >
          std::numeric_limits<std::int32_t>::max() /
              result.convolution_batch ||
      reduction_extent > std::numeric_limits<std::int32_t>::max()) {
    artifact_error("mthreads ConvBiasRelu launch extent exceeds int32");
  }
  result.node_id = relu_node.at("id").as_int();
  result.operation = "conv_bias_relu";
  result.dense = true;
  return result;
}

PointwiseRequest parse_unary_request(
    const JsonValue& request,
    std::string_view expected_target) {
  require_exact_keys(
      request,
      {"schema_version", "flagdnn_version", "backend", "target",
       "build_options", "graph", "compiler_identity"},
      "request");
  PointwiseRequest result;
  result.family = PointwiseFamily::kUnary;
  result.flagdnn_version = request.at("flagdnn_version").as_string();
  result.target = request.at("target").as_string();
  result.compiler_identity =
      request.at("compiler_identity").as_string();
  validate_target(result.target);
  if (request.at("schema_version").as_int() != 3 ||
      result.flagdnn_version != FLAGDNN_VERSION_STRING ||
      request.at("backend").as_string() != "mthreads" ||
      result.target != expected_target ||
      !is_lower_sha256(result.compiler_identity)) {
    artifact_error("mthreads Graph IR identity is invalid");
  }

  const JsonValue& build_options = request.at("build_options");
  require_exact_keys(
      build_options, {"heuristic_modes", "autotune"}, "build_options");
  const auto& heuristic_modes =
      build_options.at("heuristic_modes").as_array();
  if (heuristic_modes.empty()) {
    artifact_error("mthreads heuristic modes are empty");
  }
  std::set<std::string> unique_modes;
  for (const JsonValue& mode_value : heuristic_modes) {
    const std::string& mode = mode_value.as_string();
    if ((mode != "A" && mode != "FALLBACK") ||
        !unique_modes.insert(mode).second) {
      artifact_error("mthreads heuristic modes are invalid");
    }
  }
  result.autotune = build_options.at("autotune").as_bool();

  const JsonValue& graph = request.at("graph");
  require_exact_keys(
      graph, {"name", "tensor_count", "tensors", "node_count", "nodes"},
      "graph");
  const std::string& graph_name = graph.at("name").as_string();
  if (graph_name.size() > (1U << 20) ||
      graph_name.find('\0') != std::string::npos) {
    artifact_error("mthreads Graph name is invalid");
  }
  const auto& tensor_values = graph.at("tensors").as_array();
  if (graph.at("tensor_count").as_int() != 2 ||
      tensor_values.size() != 2) {
    artifact_error(
        "mthreads unary pointwise requires exactly two tensors");
  }
  std::map<std::int64_t, TensorSpec> tensors;
  for (const JsonValue& tensor_value : tensor_values) {
    TensorSpec tensor = parse_tensor(tensor_value);
    result.external_binding_uids.push_back(tensor.uid);
    if (!tensors.emplace(tensor.uid, std::move(tensor)).second) {
      artifact_error("mthreads Graph tensor UID is duplicated");
    }
  }

  const auto& nodes = graph.at("nodes").as_array();
  if (graph.at("node_count").as_int() != 1 || nodes.size() != 1) {
    artifact_error(
        "mthreads pointwise compiler requires exactly one node");
  }
  const JsonValue& node = nodes.front();
  require_exact_keys(
      node,
      {"id", "type", "name", "compute_data_type", "inputs", "outputs",
       "attributes"},
      "unary pointwise node");
  result.node_id = node.at("id").as_int();
  const std::string& node_name = node.at("name").as_string();
  if (result.node_id < 0 || node_name.size() > 4096 ||
      node_name.find('\0') != std::string::npos) {
    artifact_error("mthreads pointwise node identity is invalid");
  }
  const auto inputs =
      parse_ports(node.at("inputs"), tensors, "pointwise input");
  const auto outputs =
      parse_ports(node.at("outputs"), tensors, "pointwise output");
  if (inputs.size() != 1 || inputs.find("input") == inputs.end() ||
      outputs.size() != 1 || outputs.find("output") == outputs.end() ||
      inputs.at("input") == outputs.at("output")) {
    artifact_error("mthreads unary pointwise port roles are invalid");
  }
  result.input = tensors.at(inputs.at("input"));
  result.output = tensors.at(outputs.at("output"));
  if (result.input.is_virtual || result.output.is_virtual ||
      result.input.dimensions != result.output.dimensions) {
    artifact_error(
        "mthreads unary pointwise tensor bindings/shapes are invalid");
  }

  const JsonValue& attributes = node.at("attributes");
  require_exact_keys(
      attributes,
      {"mode", "relu_lower_clip", "relu_upper_clip",
       "relu_lower_clip_slope", "relu_upper_clip_set", "swish_beta",
       "elu_alpha", "softplus_beta", "n_elements", "has_upper_clip",
       "negative_slope", "lower_clip", "upper_clip"},
      "unary pointwise attributes");
  result.pointwise_mode = attributes.at("mode").as_int();
  result.operation = unary_operation(result.pointwise_mode);
  result.n_elements = element_count(result.output);
  if (node.at("type").as_string() != result.operation ||
      attributes.at("n_elements").as_int() != result.n_elements) {
    artifact_error(
        "mthreads unary pointwise attributes do not match the Graph");
  }
  const std::string& compute_type =
      node.at("compute_data_type").as_string();
  if (result.pointwise_mode == 24) {
    if (result.input.data_type != "boolean" ||
        result.output.data_type != "boolean" ||
        compute_type != "boolean") {
      artifact_error(
          "mthreads logical NOT storage/compute types differ");
    }
  } else if ((result.pointwise_mode != 5 &&
              !is_floating_data_type(result.input.data_type)) ||
             result.output.data_type != result.input.data_type ||
             compute_type != "float32") {
    artifact_error(
        "mthreads numeric unary storage/compute types differ");
  }

  const bool relu_upper_clip_set =
      attributes.at("relu_upper_clip_set").as_bool();
  result.has_upper_clip = attributes.at("has_upper_clip").as_int();
  if (result.has_upper_clip < 0 || result.has_upper_clip > 1 ||
      result.has_upper_clip !=
          static_cast<std::int64_t>(relu_upper_clip_set)) {
    artifact_error("mthreads unary upper-clip presence differs");
  }
  result.negative_slope =
      checked_float32(attributes.at("negative_slope"), "negative_slope");
  result.lower_clip =
      checked_float32(attributes.at("lower_clip"), "lower_clip");
  result.upper_clip =
      checked_float32(attributes.at("upper_clip"), "upper_clip");
  const float relu_negative_slope = checked_float32(
      attributes.at("relu_lower_clip_slope"), "relu_lower_clip_slope");
  const float relu_lower_clip = checked_float32(
      attributes.at("relu_lower_clip"), "relu_lower_clip");
  const float relu_upper_clip = checked_float32(
      attributes.at("relu_upper_clip"), "relu_upper_clip");
  result.swish_beta =
      checked_float32(attributes.at("swish_beta"), "swish_beta");
  result.elu_alpha =
      checked_float32(attributes.at("elu_alpha"), "elu_alpha");
  result.softplus_beta =
      checked_float32(attributes.at("softplus_beta"), "softplus_beta");
  if (result.negative_slope != relu_negative_slope ||
      result.lower_clip != relu_lower_clip ||
      result.upper_clip != relu_upper_clip) {
    artifact_error(
        "mthreads unary normalized ReLU attributes differ");
  }
  if (result.has_upper_clip != 0 &&
      result.upper_clip < result.lower_clip) {
    artifact_error("mthreads ReLU upper clip is below its lower clip");
  }
  if (result.softplus_beta <= 0.0F) {
    artifact_error("mthreads softplus beta is not positive");
  }
  const bool relu_defaults =
      result.negative_slope == 0.0F && result.lower_clip == 0.0F &&
      result.upper_clip == 0.0F && result.has_upper_clip == 0;
  if ((result.pointwise_mode != 2 && !relu_defaults) ||
      (result.pointwise_mode != 38 && result.swish_beta != 1.0F) ||
      (result.pointwise_mode != 35 && result.elu_alpha != 1.0F) ||
      (result.pointwise_mode != 37 && result.softplus_beta != 1.0F)) {
    artifact_error(
        "mthreads unary attributes are set for another mode");
  }
  result.dense = result.input.strides == result.output.strides &&
                 is_physically_dense(result.input) &&
                 is_physically_dense(result.output);
  return result;
}

PointwiseRequest parse_ternary_request(
    const JsonValue& request,
    std::string_view expected_target) {
  require_exact_keys(
      request,
      {"schema_version", "flagdnn_version", "backend", "target",
       "build_options", "graph", "compiler_identity"},
      "request");
  PointwiseRequest result;
  result.family = PointwiseFamily::kTernary;
  result.flagdnn_version = request.at("flagdnn_version").as_string();
  result.target = request.at("target").as_string();
  result.compiler_identity =
      request.at("compiler_identity").as_string();
  validate_target(result.target);
  if (request.at("schema_version").as_int() != 3 ||
      result.flagdnn_version != FLAGDNN_VERSION_STRING ||
      request.at("backend").as_string() != "mthreads" ||
      result.target != expected_target ||
      !is_lower_sha256(result.compiler_identity)) {
    artifact_error("mthreads ternary Graph IR identity is invalid");
  }

  const JsonValue& build_options = request.at("build_options");
  require_exact_keys(
      build_options, {"heuristic_modes", "autotune"}, "build_options");
  const auto& heuristic_modes =
      build_options.at("heuristic_modes").as_array();
  if (heuristic_modes.empty()) {
    artifact_error("mthreads heuristic modes are empty");
  }
  std::set<std::string> unique_modes;
  for (const JsonValue& mode_value : heuristic_modes) {
    const std::string& mode = mode_value.as_string();
    if ((mode != "A" && mode != "FALLBACK") ||
        !unique_modes.insert(mode).second) {
      artifact_error("mthreads heuristic modes are invalid");
    }
  }
  result.autotune = build_options.at("autotune").as_bool();

  const JsonValue& graph = request.at("graph");
  require_exact_keys(
      graph, {"name", "tensor_count", "tensors", "node_count", "nodes"},
      "graph");
  const std::string& graph_name = graph.at("name").as_string();
  if (graph_name.size() > (1U << 20) ||
      graph_name.find('\0') != std::string::npos) {
    artifact_error("mthreads ternary Graph name is invalid");
  }
  const auto& tensor_values = graph.at("tensors").as_array();
  if (graph.at("tensor_count").as_int() != 4 ||
      tensor_values.size() != 4) {
    artifact_error(
        "mthreads ternary pointwise requires exactly four tensors");
  }
  std::map<std::int64_t, TensorSpec> tensors;
  for (const JsonValue& tensor_value : tensor_values) {
    TensorSpec tensor = parse_tensor(tensor_value);
    result.external_binding_uids.push_back(tensor.uid);
    if (!tensors.emplace(tensor.uid, std::move(tensor)).second) {
      artifact_error("mthreads ternary Graph tensor UID is duplicated");
    }
  }

  const auto& nodes = graph.at("nodes").as_array();
  if (graph.at("node_count").as_int() != 1 || nodes.size() != 1) {
    artifact_error(
        "mthreads pointwise compiler requires exactly one node");
  }
  const JsonValue& node = nodes.front();
  require_exact_keys(
      node,
      {"id", "type", "name", "compute_data_type", "inputs", "outputs",
       "attributes"},
      "ternary pointwise node");
  result.node_id = node.at("id").as_int();
  const std::string& node_name = node.at("name").as_string();
  if (result.node_id < 0 || node_name.size() > 4096 ||
      node_name.find('\0') != std::string::npos) {
    artifact_error("mthreads ternary pointwise node identity is invalid");
  }
  const auto inputs =
      parse_ports(node.at("inputs"), tensors, "ternary pointwise input");
  const auto outputs =
      parse_ports(node.at("outputs"), tensors, "ternary pointwise output");
  if (inputs.size() != 3 || inputs.find("a") == inputs.end() ||
      inputs.find("b") == inputs.end() ||
      inputs.find("t") == inputs.end() || outputs.size() != 1 ||
      outputs.find("output") == outputs.end()) {
    artifact_error("mthreads ternary pointwise port roles are invalid");
  }
  const std::set<std::int64_t> referenced = {
      inputs.at("a"), inputs.at("b"), inputs.at("t"),
      outputs.at("output")};
  if (referenced.size() != 4) {
    artifact_error("mthreads ternary tensor roles must be distinct");
  }
  result.left = tensors.at(inputs.at("a"));
  result.right = tensors.at(inputs.at("b"));
  result.predicate = tensors.at(inputs.at("t"));
  result.output = tensors.at(outputs.at("output"));
  if (result.left.is_virtual || result.right.is_virtual ||
      result.predicate.is_virtual || result.output.is_virtual) {
    artifact_error(
        "mthreads ternary tensors must be externally bound");
  }
  TensorSpec partial;
  partial.dimensions = broadcast_dimensions(result.left, result.right);
  if (broadcast_dimensions(partial, result.predicate) !=
      result.output.dimensions) {
    artifact_error(
        "mthreads ternary output shape is not the broadcast result");
  }
  if ((!is_floating_data_type(result.left.data_type) &&
       result.left.data_type != "int32") ||
      result.right.data_type != result.left.data_type ||
      result.output.data_type != result.left.data_type ||
      result.predicate.data_type != "boolean" ||
      node.at("compute_data_type").as_string() != "float32") {
    artifact_error(
        "mthreads binary_select storage/compute types differ");
  }

  const JsonValue& attributes = node.at("attributes");
  require_exact_keys(
      attributes, {"mode", "n_elements"},
      "ternary pointwise attributes");
  result.pointwise_mode = attributes.at("mode").as_int();
  result.operation = "binary_select";
  result.n_elements = element_count(result.output);
  if (result.pointwise_mode != 41 ||
      node.at("type").as_string() != result.operation ||
      attributes.at("n_elements").as_int() != result.n_elements) {
    artifact_error(
        "mthreads ternary pointwise attributes do not match the Graph");
  }
  const auto dense_tensor = [&](const TensorSpec& tensor) {
    return tensor.dimensions == result.output.dimensions &&
           tensor.strides == result.output.strides &&
           is_physically_dense(tensor);
  };
  result.dense = dense_tensor(result.left) && dense_tensor(result.right) &&
                 dense_tensor(result.predicate) &&
                 dense_tensor(result.output);
  return result;
}

std::vector<std::int64_t> layout_integer_array(
    const JsonValue& value,
    std::size_t expected_size,
    std::int64_t minimum,
    std::string_view field) {
  const auto& array = value.as_array();
  if (array.size() != expected_size) {
    artifact_error(
        "mthreads layout attribute has the wrong length: " +
        std::string(field));
  }
  std::vector<std::int64_t> result;
  result.reserve(array.size());
  for (const JsonValue& entry : array) {
    const std::int64_t decoded = entry.as_int();
    if (decoded < minimum) {
      artifact_error(
          "mthreads layout attribute is below its minimum: " +
          std::string(field));
    }
    result.push_back(decoded);
  }
  return result;
}

PointwiseRequest parse_layout_request(
    const JsonValue& request,
    std::string_view expected_target) {
  require_exact_keys(
      request,
      {"schema_version", "flagdnn_version", "backend", "target",
       "build_options", "graph", "compiler_identity"},
      "request");
  PointwiseRequest result;
  result.family = PointwiseFamily::kLayout;
  result.flagdnn_version = request.at("flagdnn_version").as_string();
  result.target = request.at("target").as_string();
  result.compiler_identity =
      request.at("compiler_identity").as_string();
  validate_target(result.target);
  if (request.at("schema_version").as_int() != 3 ||
      result.flagdnn_version != FLAGDNN_VERSION_STRING ||
      request.at("backend").as_string() != "mthreads" ||
      result.target != expected_target ||
      !is_lower_sha256(result.compiler_identity)) {
    artifact_error("mthreads layout Graph IR identity is invalid");
  }

  const JsonValue& build_options = request.at("build_options");
  require_exact_keys(
      build_options, {"heuristic_modes", "autotune"}, "build_options");
  const auto& heuristic_modes =
      build_options.at("heuristic_modes").as_array();
  if (heuristic_modes.empty()) {
    artifact_error("mthreads heuristic modes are empty");
  }
  std::set<std::string> unique_modes;
  for (const JsonValue& mode_value : heuristic_modes) {
    const std::string& mode = mode_value.as_string();
    if ((mode != "A" && mode != "FALLBACK") ||
        !unique_modes.insert(mode).second) {
      artifact_error("mthreads heuristic modes are invalid");
    }
  }
  result.autotune = build_options.at("autotune").as_bool();

  const JsonValue& graph = request.at("graph");
  require_exact_keys(
      graph, {"name", "tensor_count", "tensors", "node_count", "nodes"},
      "graph");
  const std::string& graph_name = graph.at("name").as_string();
  if (graph_name.size() > (1U << 20) ||
      graph_name.find('\0') != std::string::npos) {
    artifact_error("mthreads layout Graph name is invalid");
  }
  const auto& tensor_values = graph.at("tensors").as_array();
  if (graph.at("tensor_count").as_int() != 2 ||
      tensor_values.size() != 2) {
    artifact_error("mthreads layout requires exactly two tensors");
  }
  std::map<std::int64_t, TensorSpec> tensors;
  for (const JsonValue& tensor_value : tensor_values) {
    TensorSpec tensor = parse_tensor(tensor_value);
    result.external_binding_uids.push_back(tensor.uid);
    if (!tensors.emplace(tensor.uid, std::move(tensor)).second) {
      artifact_error("mthreads layout Graph tensor UID is duplicated");
    }
  }

  const auto& nodes = graph.at("nodes").as_array();
  if (graph.at("node_count").as_int() != 1 || nodes.size() != 1) {
    artifact_error("mthreads layout compiler requires exactly one node");
  }
  const JsonValue& node = nodes.front();
  require_exact_keys(
      node,
      {"id", "type", "name", "compute_data_type", "inputs", "outputs",
       "attributes"},
      "layout node");
  result.node_id = node.at("id").as_int();
  const std::string& node_name = node.at("name").as_string();
  result.operation = node.at("type").as_string();
  if (result.node_id < 0 || node_name.size() > 4096 ||
      node_name.find('\0') != std::string::npos ||
      (result.operation != "reshape" && result.operation != "transpose" &&
       result.operation != "slice")) {
    artifact_error("mthreads layout node identity is invalid");
  }
  const auto inputs =
      parse_ports(node.at("inputs"), tensors, "layout input");
  const auto outputs =
      parse_ports(node.at("outputs"), tensors, "layout output");
  if (inputs.size() != 1 || inputs.find("input") == inputs.end() ||
      outputs.size() != 1 || outputs.find("output") == outputs.end() ||
      inputs.at("input") == outputs.at("output")) {
    artifact_error("mthreads layout port roles are invalid");
  }
  result.input = tensors.at(inputs.at("input"));
  result.output = tensors.at(outputs.at("output"));
  if (result.input.is_virtual || result.output.is_virtual ||
      result.input.data_type != result.output.data_type ||
      node.at("compute_data_type").as_string() != "float32") {
    artifact_error("mthreads layout tensor bindings/types are invalid");
  }

  const JsonValue& attributes = node.at("attributes");
  const std::size_t input_rank = result.input.dimensions.size();
  const std::size_t output_rank = result.output.dimensions.size();
  result.logical_input_dimensions = result.input.dimensions;
  result.logical_input_strides = result.input.strides;
  if (result.operation == "reshape") {
    require_exact_keys(
        attributes,
        {"n_elements", "input_rank", "output_rank", "reshape_mode",
         "input_dimensions", "input_strides", "output_dimensions",
         "output_strides"},
        "reshape attributes");
    if (attributes.at("input_rank").as_int() !=
            static_cast<std::int64_t>(input_rank) ||
        attributes.at("output_rank").as_int() !=
            static_cast<std::int64_t>(output_rank) ||
        attributes.at("reshape_mode").as_int() != 2 ||
        element_count(result.input) != element_count(result.output)) {
      artifact_error("mthreads reshape rank/mode/extent is invalid");
    }
  } else if (result.operation == "transpose") {
    require_exact_keys(
        attributes,
        {"n_elements", "rank", "permutation", "input_dimensions",
         "input_strides", "output_dimensions", "output_strides"},
        "transpose attributes");
    if (attributes.at("rank").as_int() !=
            static_cast<std::int64_t>(input_rank) ||
        output_rank != input_rank) {
      artifact_error("mthreads transpose ranks are invalid");
    }
    const auto permutation = layout_integer_array(
        attributes.at("permutation"), input_rank, 0, "permutation");
    std::set<std::int64_t> axes(permutation.begin(), permutation.end());
    if (axes.size() != input_rank ||
        (!axes.empty() &&
         *axes.rbegin() >= static_cast<std::int64_t>(input_rank))) {
      artifact_error("mthreads transpose permutation is invalid");
    }
    result.logical_input_dimensions = result.output.dimensions;
    result.logical_input_strides.clear();
    result.logical_input_strides.reserve(input_rank);
    for (std::size_t axis = 0; axis < input_rank; ++axis) {
      const std::size_t source =
          static_cast<std::size_t>(permutation[axis]);
      if (result.output.dimensions[axis] !=
          result.input.dimensions[source]) {
        artifact_error("mthreads transpose output shape is invalid");
      }
      result.logical_input_strides.push_back(
          result.input.strides[source]);
    }
  } else {
    require_exact_keys(
        attributes,
        {"n_elements", "rank", "starts", "limits", "slice_strides",
         "input_dimensions", "input_strides", "output_dimensions",
         "output_strides"},
        "slice attributes");
    if (attributes.at("rank").as_int() !=
            static_cast<std::int64_t>(input_rank) ||
        output_rank != input_rank) {
      artifact_error("mthreads slice ranks are invalid");
    }
    const auto starts = layout_integer_array(
        attributes.at("starts"), input_rank, 0, "starts");
    const auto limits = layout_integer_array(
        attributes.at("limits"), input_rank, 1, "limits");
    const auto slice_strides = layout_integer_array(
        attributes.at("slice_strides"), input_rank, 1, "slice_strides");
    result.logical_input_dimensions = result.output.dimensions;
    result.logical_input_strides.clear();
    result.logical_input_strides.reserve(input_rank);
    for (std::size_t axis = 0; axis < input_rank; ++axis) {
      const std::int64_t start = starts[axis];
      const std::int64_t limit = limits[axis];
      const std::int64_t step = slice_strides[axis];
      if (start >= limit || limit > result.input.dimensions[axis] ||
          result.output.dimensions[axis] !=
              1 + (limit - start - 1) / step) {
        artifact_error("mthreads slice range/output shape is invalid");
      }
      if (start != 0 &&
          result.input.strides[axis] >
              (std::numeric_limits<std::int64_t>::max() -
               result.input_base) /
                  start) {
        artifact_error("mthreads slice input base overflows int64");
      }
      result.input_base += start * result.input.strides[axis];
      if (step > std::numeric_limits<std::int64_t>::max() /
                     result.input.strides[axis]) {
        artifact_error("mthreads slice effective stride overflows int64");
      }
      result.logical_input_strides.push_back(
          step * result.input.strides[axis]);
    }
  }

  const auto input_dimensions = layout_integer_array(
      attributes.at("input_dimensions"), input_rank, 1,
      "input_dimensions");
  const auto input_strides = layout_integer_array(
      attributes.at("input_strides"), input_rank, 1, "input_strides");
  const auto output_dimensions = layout_integer_array(
      attributes.at("output_dimensions"), output_rank, 1,
      "output_dimensions");
  const auto output_strides = layout_integer_array(
      attributes.at("output_strides"), output_rank, 1,
      "output_strides");
  result.n_elements = element_count(result.output);
  if (input_dimensions != result.input.dimensions ||
      input_strides != result.input.strides ||
      output_dimensions != result.output.dimensions ||
      output_strides != result.output.strides ||
      attributes.at("n_elements").as_int() != result.n_elements) {
    artifact_error("mthreads layout attributes differ from tensor metadata");
  }
  result.dense = true;
  return result;
}

PointwiseRequest parse_reduction_request(
    const JsonValue& request,
    std::string_view expected_target) {
  require_exact_keys(
      request,
      {"schema_version", "flagdnn_version", "backend", "target",
       "build_options", "graph", "compiler_identity"},
      "request");
  PointwiseRequest result;
  result.family = PointwiseFamily::kReduction;
  result.flagdnn_version = request.at("flagdnn_version").as_string();
  result.target = request.at("target").as_string();
  result.compiler_identity =
      request.at("compiler_identity").as_string();
  validate_target(result.target);
  if (request.at("schema_version").as_int() != 3 ||
      result.flagdnn_version != FLAGDNN_VERSION_STRING ||
      request.at("backend").as_string() != "mthreads" ||
      result.target != expected_target ||
      !is_lower_sha256(result.compiler_identity)) {
    artifact_error("mthreads reduction Graph IR identity is invalid");
  }

  const JsonValue& build_options = request.at("build_options");
  require_exact_keys(
      build_options, {"heuristic_modes", "autotune"}, "build_options");
  const auto& heuristic_modes =
      build_options.at("heuristic_modes").as_array();
  if (heuristic_modes.empty()) {
    artifact_error("mthreads heuristic modes are empty");
  }
  std::set<std::string> unique_modes;
  for (const JsonValue& mode_value : heuristic_modes) {
    const std::string& mode = mode_value.as_string();
    if ((mode != "A" && mode != "FALLBACK") ||
        !unique_modes.insert(mode).second) {
      artifact_error("mthreads heuristic modes are invalid");
    }
  }
  result.autotune = build_options.at("autotune").as_bool();

  const JsonValue& graph = request.at("graph");
  require_exact_keys(
      graph, {"name", "tensor_count", "tensors", "node_count", "nodes"},
      "graph");
  const std::string& graph_name = graph.at("name").as_string();
  if (graph_name.size() > (1U << 20) ||
      graph_name.find('\0') != std::string::npos) {
    artifact_error("mthreads reduction Graph name is invalid");
  }
  const auto& tensor_values = graph.at("tensors").as_array();
  if (graph.at("tensor_count").as_int() != 2 ||
      tensor_values.size() != 2) {
    artifact_error("mthreads reduction requires exactly two tensors");
  }
  std::map<std::int64_t, TensorSpec> tensors;
  for (const JsonValue& tensor_value : tensor_values) {
    TensorSpec tensor = parse_tensor(tensor_value);
    result.external_binding_uids.push_back(tensor.uid);
    if (!tensors.emplace(tensor.uid, std::move(tensor)).second) {
      artifact_error("mthreads reduction tensor UID is duplicated");
    }
  }

  const auto& nodes = graph.at("nodes").as_array();
  if (graph.at("node_count").as_int() != 1 || nodes.size() != 1) {
    artifact_error("mthreads reduction compiler requires one node");
  }
  const JsonValue& node = nodes.front();
  require_exact_keys(
      node,
      {"id", "type", "name", "compute_data_type", "inputs", "outputs",
       "attributes"},
      "reduction node");
  result.node_id = node.at("id").as_int();
  result.operation = node.at("type").as_string();
  const std::string& node_name = node.at("name").as_string();
  if (result.node_id < 0 || node_name.size() > 4096 ||
      node_name.find('\0') != std::string::npos ||
      (result.operation != "reduction_sum" &&
       result.operation != "reduction_avg" &&
       result.operation != "reduction_mul")) {
    artifact_error("mthreads reduction node identity is invalid");
  }
  const auto inputs =
      parse_ports(node.at("inputs"), tensors, "reduction input");
  const auto outputs =
      parse_ports(node.at("outputs"), tensors, "reduction output");
  if (inputs.size() != 1 || inputs.find("input") == inputs.end() ||
      outputs.size() != 1 || outputs.find("output") == outputs.end() ||
      inputs.at("input") == outputs.at("output")) {
    artifact_error("mthreads reduction port roles are invalid");
  }
  result.input = tensors.at(inputs.at("input"));
  result.output = tensors.at(outputs.at("output"));
  if (result.input.is_virtual || result.output.is_virtual ||
      (!is_floating_data_type(result.input.data_type) &&
       result.input.data_type != "int32") ||
      (result.input.data_type != result.output.data_type &&
       result.output.data_type != "float32") ||
      node.at("compute_data_type").as_string() != "float32") {
    artifact_error("mthreads reduction tensor bindings/types are invalid");
  }
  const std::size_t rank = result.input.dimensions.size();
  if (rank == 0 || rank > kMaximumTensorRank) {
    artifact_error("mthreads reduction input rank is invalid");
  }

  const JsonValue& attributes = node.at("attributes");
  require_exact_keys(
      attributes,
      {"mode", "axis", "keep_dimensions", "outer", "reduction", "inner",
       "output_elements"},
      "reduction attributes");
  result.reduction_mode = attributes.at("mode").as_int();
  const std::string expected_operation =
      result.reduction_mode == 0
          ? "reduction_sum"
          : (result.reduction_mode == 1
                 ? "reduction_avg"
                 : (result.reduction_mode == 2 ? "reduction_mul" : ""));
  result.reduction_axis = attributes.at("axis").as_int();
  const std::int64_t keep_value =
      attributes.at("keep_dimensions").as_int();
  if (expected_operation.empty() || result.operation != expected_operation ||
      result.reduction_axis < 0 ||
      result.reduction_axis >= static_cast<std::int64_t>(rank) ||
      (keep_value != 0 && keep_value != 1)) {
    artifact_error("mthreads reduction mode/axis metadata is invalid");
  }
  result.keep_dimensions = keep_value == 1;
  std::vector<std::int64_t> expected_output = result.input.dimensions;
  if (result.keep_dimensions) {
    expected_output[static_cast<std::size_t>(result.reduction_axis)] = 1;
  } else {
    expected_output.erase(
        expected_output.begin() + result.reduction_axis);
  }
  if (result.output.dimensions != expected_output) {
    artifact_error("mthreads reduction output shape is invalid");
  }

  const auto checked_product = [&](std::size_t begin, std::size_t end) {
    std::int64_t product = 1;
    for (std::size_t axis = begin; axis < end; ++axis) {
      if (result.input.dimensions[axis] >
          std::numeric_limits<std::int64_t>::max() / product) {
        artifact_error("mthreads reduction extent overflows int64");
      }
      product *= result.input.dimensions[axis];
    }
    return product;
  };
  const std::size_t axis =
      static_cast<std::size_t>(result.reduction_axis);
  const std::int64_t expected_outer = checked_product(0, axis);
  const std::int64_t expected_extent = result.input.dimensions[axis];
  const std::int64_t expected_inner = checked_product(axis + 1, rank);
  if (expected_outer > std::numeric_limits<std::int32_t>::max() ||
      expected_extent > 65536 ||
      expected_inner > std::numeric_limits<std::int32_t>::max() ||
      expected_outer >
          std::numeric_limits<std::int32_t>::max() / expected_inner) {
    artifact_error("mthreads reduction launch extent exceeds int32");
  }
  result.outer = attributes.at("outer").as_int();
  result.extent = attributes.at("reduction").as_int();
  result.inner = attributes.at("inner").as_int();
  result.n_elements = element_count(result.output);
  if (result.outer != expected_outer || result.extent != expected_extent ||
      result.inner != expected_inner ||
      attributes.at("output_elements").as_int() != result.n_elements ||
      result.n_elements != expected_outer * expected_inner) {
    artifact_error(
        "mthreads reduction parameters differ from tensor metadata");
  }
  result.dense = is_row_major_contiguous(result.input) &&
                 is_row_major_contiguous(result.output);
  return result;
}

PointwiseRequest parse_matmul_request(
    const JsonValue& request,
    std::string_view expected_target) {
  require_exact_keys(
      request,
      {"schema_version", "flagdnn_version", "backend", "target",
       "build_options", "graph", "compiler_identity"},
      "request");
  PointwiseRequest result;
  result.family = PointwiseFamily::kMatmul;
  result.flagdnn_version = request.at("flagdnn_version").as_string();
  result.target = request.at("target").as_string();
  result.compiler_identity =
      request.at("compiler_identity").as_string();
  validate_target(result.target);
  if (request.at("schema_version").as_int() != 3 ||
      result.flagdnn_version != FLAGDNN_VERSION_STRING ||
      request.at("backend").as_string() != "mthreads" ||
      result.target != expected_target ||
      !is_lower_sha256(result.compiler_identity)) {
    artifact_error("mthreads Matmul Graph IR identity is invalid");
  }

  const JsonValue& build_options = request.at("build_options");
  require_exact_keys(
      build_options, {"heuristic_modes", "autotune"}, "build_options");
  const auto& heuristic_modes =
      build_options.at("heuristic_modes").as_array();
  if (heuristic_modes.empty()) {
    artifact_error("mthreads heuristic modes are empty");
  }
  std::set<std::string> unique_modes;
  for (const JsonValue& mode_value : heuristic_modes) {
    const std::string& mode = mode_value.as_string();
    if ((mode != "A" && mode != "FALLBACK") ||
        !unique_modes.insert(mode).second) {
      artifact_error("mthreads heuristic modes are invalid");
    }
  }
  result.autotune = build_options.at("autotune").as_bool();

  const JsonValue& graph = request.at("graph");
  require_exact_keys(
      graph, {"name", "tensor_count", "tensors", "node_count", "nodes"},
      "graph");
  const std::string& graph_name = graph.at("name").as_string();
  if (graph_name.size() > (1U << 20) ||
      graph_name.find('\0') != std::string::npos) {
    artifact_error("mthreads Matmul Graph name is invalid");
  }
  const auto& tensor_values = graph.at("tensors").as_array();
  if (graph.at("tensor_count").as_int() != 3 ||
      tensor_values.size() != 3) {
    artifact_error("mthreads Matmul requires exactly three tensors");
  }
  std::map<std::int64_t, TensorSpec> tensors;
  for (const JsonValue& tensor_value : tensor_values) {
    TensorSpec tensor = parse_tensor(tensor_value);
    result.external_binding_uids.push_back(tensor.uid);
    if (!tensors.emplace(tensor.uid, std::move(tensor)).second) {
      artifact_error("mthreads Matmul tensor UID is duplicated");
    }
  }

  const auto& nodes = graph.at("nodes").as_array();
  if (graph.at("node_count").as_int() != 1 || nodes.size() != 1) {
    artifact_error("mthreads Matmul compiler requires one node");
  }
  const JsonValue& node = nodes.front();
  require_exact_keys(
      node,
      {"id", "type", "name", "compute_data_type", "inputs", "outputs",
       "attributes"},
      "Matmul node");
  result.node_id = node.at("id").as_int();
  result.operation = node.at("type").as_string();
  const std::string& node_name = node.at("name").as_string();
  if (result.node_id < 0 || result.operation != "matmul" ||
      node_name.size() > 4096 ||
      node_name.find('\0') != std::string::npos) {
    artifact_error("mthreads Matmul node identity is invalid");
  }
  const auto inputs = parse_ports(node.at("inputs"), tensors, "Matmul input");
  const auto outputs =
      parse_ports(node.at("outputs"), tensors, "Matmul output");
  if (inputs.size() != 2 || inputs.find("a") == inputs.end() ||
      inputs.find("b") == inputs.end() || outputs.size() != 1 ||
      outputs.find("output") == outputs.end() ||
      inputs.at("a") == inputs.at("b") ||
      inputs.at("a") == outputs.at("output") ||
      inputs.at("b") == outputs.at("output")) {
    artifact_error("mthreads Matmul port roles are invalid");
  }
  result.left = tensors.at(inputs.at("a"));
  result.right = tensors.at(inputs.at("b"));
  result.output = tensors.at(outputs.at("output"));
  if (result.left.is_virtual || result.right.is_virtual ||
      result.output.is_virtual ||
      (!is_floating_data_type(result.left.data_type) &&
       result.left.data_type != "int32") ||
      result.left.data_type != result.right.data_type ||
      result.left.data_type != result.output.data_type ||
      node.at("compute_data_type").as_string() != "float32") {
    artifact_error("mthreads Matmul tensor bindings/types are invalid");
  }
  const std::size_t a_rank = result.left.dimensions.size();
  const std::size_t b_rank = result.right.dimensions.size();
  const std::size_t output_rank = result.output.dimensions.size();
  if (a_rank < 2 || a_rank > kMaximumTensorRank || b_rank < 2 ||
      b_rank > kMaximumTensorRank || output_rank < 2 ||
      output_rank > kMaximumTensorRank) {
    artifact_error("mthreads Matmul tensor ranks are invalid");
  }

  result.matmul_m = result.left.dimensions[a_rank - 2];
  result.matmul_k = result.left.dimensions[a_rank - 1];
  result.matmul_n = result.right.dimensions[b_rank - 1];
  if (result.right.dimensions[b_rank - 2] != result.matmul_k) {
    artifact_error("mthreads Matmul contraction dimensions differ");
  }
  const std::size_t a_batch_rank = a_rank - 2;
  const std::size_t b_batch_rank = b_rank - 2;
  const std::size_t batch_rank = std::max(a_batch_rank, b_batch_rank);
  if (batch_rank > 6 || output_rank != batch_rank + 2) {
    artifact_error("mthreads Matmul batch rank is invalid");
  }
  result.matmul_batch_dimensions.fill(1);
  result.matmul_a_batch_strides.fill(0);
  result.matmul_b_batch_strides.fill(0);
  result.matmul_output_batch_strides.fill(0);
  result.matmul_batch = 1;
  std::vector<std::int64_t> expected_output;
  expected_output.reserve(batch_rank + 2);
  const std::size_t a_leading = batch_rank - a_batch_rank;
  const std::size_t b_leading = batch_rank - b_batch_rank;
  const std::size_t padded_leading = 6 - batch_rank;
  for (std::size_t axis = 0; axis < batch_rank; ++axis) {
    const std::int64_t a_dimension =
        axis < a_leading ? 1 : result.left.dimensions[axis - a_leading];
    const std::int64_t b_dimension =
        axis < b_leading ? 1 : result.right.dimensions[axis - b_leading];
    if (a_dimension != b_dimension && a_dimension != 1 &&
        b_dimension != 1) {
      artifact_error("mthreads Matmul batch dimensions cannot broadcast");
    }
    const std::int64_t dimension = std::max(a_dimension, b_dimension);
    const std::size_t slot = padded_leading + axis;
    result.matmul_batch_dimensions[slot] = dimension;
    if (axis >= a_leading && a_dimension != 1) {
      result.matmul_a_batch_strides[slot] =
          result.left.strides[axis - a_leading];
    }
    if (axis >= b_leading && b_dimension != 1) {
      result.matmul_b_batch_strides[slot] =
          result.right.strides[axis - b_leading];
    }
    result.matmul_output_batch_strides[slot] = result.output.strides[axis];
    if (result.matmul_batch >
        std::numeric_limits<std::int32_t>::max() / dimension) {
      artifact_error("mthreads Matmul batch extent exceeds int32");
    }
    result.matmul_batch *= dimension;
    expected_output.push_back(dimension);
  }
  expected_output.push_back(result.matmul_m);
  expected_output.push_back(result.matmul_n);
  if (result.output.dimensions != expected_output) {
    artifact_error("mthreads Matmul output shape is invalid");
  }
  result.matmul_a_stride_m = result.left.strides[a_rank - 2];
  result.matmul_a_stride_k = result.left.strides[a_rank - 1];
  result.matmul_b_stride_k = result.right.strides[b_rank - 2];
  result.matmul_b_stride_n = result.right.strides[b_rank - 1];
  result.matmul_output_stride_m = result.output.strides[output_rank - 2];
  result.matmul_output_stride_n = result.output.strides[output_rank - 1];

  const JsonValue& attributes = node.at("attributes");
  if (attributes.as_object().contains("input_precision"))
    result.input_precision = attributes.at("input_precision").as_int();
  require_exact_keys(attributes, {"batch", "m", "n", "k"},
                     "Matmul attributes");
  if (attributes.at("batch").as_int() != result.matmul_batch ||
      attributes.at("m").as_int() != result.matmul_m ||
      attributes.at("n").as_int() != result.matmul_n ||
      attributes.at("k").as_int() != result.matmul_k) {
    artifact_error("mthreads Matmul parameters differ from tensor metadata");
  }
  result.n_elements = element_count(result.output);
  return result;
}

PointwiseRequest parse_convolution_request(
    const JsonValue& request,
    std::string_view expected_target) {
  require_exact_keys(
      request,
      {"schema_version", "flagdnn_version", "backend", "target",
       "build_options", "graph", "compiler_identity"},
      "request");
  PointwiseRequest result;
  result.family = PointwiseFamily::kConvolution;
  result.flagdnn_version = request.at("flagdnn_version").as_string();
  result.target = request.at("target").as_string();
  result.compiler_identity =
      request.at("compiler_identity").as_string();
  validate_target(result.target);
  if (request.at("schema_version").as_int() != 3 ||
      result.flagdnn_version != FLAGDNN_VERSION_STRING ||
      request.at("backend").as_string() != "mthreads" ||
      result.target != expected_target ||
      !is_lower_sha256(result.compiler_identity)) {
    artifact_error("mthreads convolution Graph IR identity is invalid");
  }

  const JsonValue& build_options = request.at("build_options");
  require_exact_keys(
      build_options, {"heuristic_modes", "autotune"}, "build_options");
  const auto& heuristic_modes =
      build_options.at("heuristic_modes").as_array();
  if (heuristic_modes.empty()) {
    artifact_error("mthreads convolution heuristic modes are empty");
  }
  std::set<std::string> unique_modes;
  for (const JsonValue& mode_value : heuristic_modes) {
    const std::string& mode = mode_value.as_string();
    if ((mode != "A" && mode != "FALLBACK") ||
        !unique_modes.insert(mode).second) {
      artifact_error("mthreads convolution heuristic modes are invalid");
    }
  }
  result.autotune = build_options.at("autotune").as_bool();

  const JsonValue& graph = request.at("graph");
  require_exact_keys(
      graph, {"name", "tensor_count", "tensors", "node_count", "nodes"},
      "graph");
  const std::string& graph_name = graph.at("name").as_string();
  if (graph_name.size() > (1U << 20) ||
      graph_name.find('\0') != std::string::npos) {
    artifact_error("mthreads convolution Graph name is invalid");
  }
  const auto& tensor_values = graph.at("tensors").as_array();
  if (graph.at("tensor_count").as_int() != 3 ||
      tensor_values.size() != 3) {
    artifact_error("mthreads convolution requires exactly three tensors");
  }
  std::map<std::int64_t, TensorSpec> tensors;
  for (const JsonValue& tensor_value : tensor_values) {
    TensorSpec tensor = parse_tensor(tensor_value);
    result.external_binding_uids.push_back(tensor.uid);
    if (!tensors.emplace(tensor.uid, std::move(tensor)).second) {
      artifact_error("mthreads convolution tensor UID is duplicated");
    }
  }

  const auto& nodes = graph.at("nodes").as_array();
  if (graph.at("node_count").as_int() != 1 || nodes.size() != 1) {
    artifact_error("mthreads convolution compiler requires one node");
  }
  const JsonValue& node = nodes.front();
  require_exact_keys(
      node,
      {"id", "type", "name", "compute_data_type", "inputs", "outputs",
       "attributes"},
      "convolution node");
  result.node_id = node.at("id").as_int();
  result.operation = node.at("type").as_string();
  const std::string& node_name = node.at("name").as_string();
  const bool fprop =
      result.operation == "conv2d_fprop" ||
      result.operation == "convolution_fprop";
  if (result.node_id < 0 || node_name.size() > 4096 ||
      node_name.find('\0') != std::string::npos ||
      (!fprop && result.operation != "convolution_dgrad" &&
       result.operation != "convolution_wgrad")) {
    artifact_error("mthreads convolution node identity is invalid");
  }
  const auto inputs =
      parse_ports(node.at("inputs"), tensors, "convolution input");
  const auto outputs =
      parse_ports(node.at("outputs"), tensors, "convolution output");
  std::set<std::int64_t> referenced;
  const TensorSpec* graph_output = nullptr;
  if (fprop) {
    if (inputs.size() != 2 || inputs.find("input") == inputs.end() ||
        inputs.find("filter") == inputs.end() || outputs.size() != 1 ||
        outputs.find("output") == outputs.end()) {
      artifact_error("mthreads convolution FProp port roles are invalid");
    }
    referenced = {
        inputs.at("input"), inputs.at("filter"), outputs.at("output")};
    result.convolution_image = tensors.at(inputs.at("input"));
    result.convolution_filter = tensors.at(inputs.at("filter"));
    result.convolution_result = tensors.at(outputs.at("output"));
    graph_output = &result.convolution_result;
  } else if (result.operation == "convolution_dgrad") {
    if (inputs.size() != 2 || inputs.find("dy") == inputs.end() ||
        inputs.find("w") == inputs.end() || outputs.size() != 1 ||
        outputs.find("dx") == outputs.end()) {
      artifact_error("mthreads convolution DGrad port roles are invalid");
    }
    referenced = {inputs.at("dy"), inputs.at("w"), outputs.at("dx")};
    result.convolution_result = tensors.at(inputs.at("dy"));
    result.convolution_filter = tensors.at(inputs.at("w"));
    result.convolution_image = tensors.at(outputs.at("dx"));
    graph_output = &result.convolution_image;
  } else {
    if (inputs.size() != 2 || inputs.find("dy") == inputs.end() ||
        inputs.find("x") == inputs.end() || outputs.size() != 1 ||
        outputs.find("dw") == outputs.end()) {
      artifact_error("mthreads convolution WGrad port roles are invalid");
    }
    referenced = {inputs.at("dy"), inputs.at("x"), outputs.at("dw")};
    result.convolution_result = tensors.at(inputs.at("dy"));
    result.convolution_image = tensors.at(inputs.at("x"));
    result.convolution_filter = tensors.at(outputs.at("dw"));
    graph_output = &result.convolution_filter;
  }
  if (referenced.size() != 3 || graph_output == nullptr ||
      result.convolution_image.is_virtual ||
      result.convolution_filter.is_virtual ||
      result.convolution_result.is_virtual ||
      !is_floating_data_type(result.convolution_image.data_type) ||
      result.convolution_filter.data_type !=
          result.convolution_image.data_type ||
      result.convolution_result.data_type !=
          result.convolution_image.data_type ||
      node.at("compute_data_type").as_string() != "float32") {
    artifact_error(
        "mthreads convolution tensor bindings/types are invalid");
  }

  const JsonValue& attributes = node.at("attributes");
  if (attributes.as_object().contains("input_precision"))
    result.input_precision = attributes.at("input_precision").as_int();
  if (fprop) {
    require_exact_keys(
        attributes,
        {"spatial_rank", "groups", "n_outputs", "pre_padding",
         "post_padding", "stride", "dilation"},
        "convolution FProp attributes");
    result.convolution_mode = 0;
  } else {
    require_exact_keys(
        attributes,
        {"spatial_rank", "groups", "convolution_mode", "n_outputs",
         "pre_padding", "post_padding", "stride", "dilation"},
        "convolution backward attributes");
    result.convolution_mode = attributes.at("convolution_mode").as_int();
    if (result.convolution_mode < 0 || result.convolution_mode > 1) {
      artifact_error("mthreads convolution mode is invalid");
    }
  }
  result.convolution_spatial_rank = attributes.at("spatial_rank").as_int();
  if (result.convolution_spatial_rank < 1 ||
      result.convolution_spatial_rank > 3) {
    artifact_error("mthreads convolution spatial rank is invalid");
  }
  const std::size_t spatial_rank =
      static_cast<std::size_t>(result.convolution_spatial_rank);
  const std::size_t rank = spatial_rank + 2;
  if (result.convolution_image.dimensions.size() != rank ||
      result.convolution_filter.dimensions.size() != rank ||
      result.convolution_result.dimensions.size() != rank) {
    artifact_error("mthreads convolution tensor ranks are invalid");
  }
  result.convolution_groups = attributes.at("groups").as_int();
  if (result.convolution_groups <= 0 ||
      result.convolution_groups >
          std::numeric_limits<std::int32_t>::max()) {
    artifact_error("mthreads convolution groups are invalid");
  }
  const auto attribute_array =
      [&](std::string_view name, std::int64_t minimum) {
        const auto& values = attributes.at(name).as_array();
        if (values.size() != spatial_rank) {
          artifact_error(
              "mthreads convolution spatial attribute length differs");
        }
        std::vector<std::int64_t> decoded;
        decoded.reserve(values.size());
        for (const JsonValue& value : values) {
          const std::int64_t entry = value.as_int();
          if (entry < minimum ||
              entry > std::numeric_limits<std::int32_t>::max()) {
            artifact_error(
                "mthreads convolution spatial attribute is invalid");
          }
          decoded.push_back(entry);
        }
        return decoded;
      };
  result.convolution_pre_padding = attribute_array("pre_padding", 0);
  result.convolution_post_padding = attribute_array("post_padding", 0);
  result.convolution_stride = attribute_array("stride", 1);
  result.convolution_dilation = attribute_array("dilation", 1);

  result.convolution_batch = result.convolution_image.dimensions[0];
  result.convolution_in_channels =
      result.convolution_image.dimensions[1];
  result.convolution_out_channels =
      result.convolution_filter.dimensions[0];
  if (result.convolution_result.dimensions[0] !=
          result.convolution_batch ||
      result.convolution_result.dimensions[1] !=
          result.convolution_out_channels ||
      result.convolution_in_channels % result.convolution_groups != 0 ||
      result.convolution_out_channels % result.convolution_groups != 0) {
    artifact_error("mthreads convolution batch/channel geometry is invalid");
  }
  result.convolution_in_per_group =
      result.convolution_in_channels / result.convolution_groups;
  result.convolution_out_per_group =
      result.convolution_out_channels / result.convolution_groups;
  if (result.convolution_filter.dimensions[1] !=
      result.convolution_in_per_group) {
    artifact_error("mthreads convolution filter channels are invalid");
  }

  std::vector<std::int64_t> expected_result = {
      result.convolution_batch, result.convolution_out_channels};
  expected_result.reserve(rank);
  for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
    const std::int64_t input_size =
        result.convolution_image.dimensions[axis + 2];
    const std::int64_t filter_size =
        result.convolution_filter.dimensions[axis + 2];
    const std::int64_t dilation = result.convolution_dilation[axis];
    if (filter_size - 1 >
        (std::numeric_limits<std::int64_t>::max() - 1) / dilation) {
      artifact_error("mthreads convolution effective filter overflows");
    }
    const std::int64_t effective = (filter_size - 1) * dilation + 1;
    const std::int64_t pre = result.convolution_pre_padding[axis];
    const std::int64_t post = result.convolution_post_padding[axis];
    if (input_size >
        std::numeric_limits<std::int64_t>::max() - pre - post) {
      artifact_error("mthreads convolution padded input overflows");
    }
    const std::int64_t padded = input_size + pre + post;
    if (padded < effective) {
      artifact_error(
          "mthreads convolution filter exceeds the padded input");
    }
    const std::int64_t output_size =
        (padded - effective) / result.convolution_stride[axis] + 1;
    if (output_size > std::numeric_limits<std::int32_t>::max()) {
      artifact_error("mthreads convolution output extent exceeds int32");
    }
    expected_result.push_back(output_size);
  }
  if (result.convolution_result.dimensions != expected_result) {
    artifact_error("mthreads convolution output/loss shape is invalid");
  }

  result.n_elements = element_count(*graph_output);
  if (attributes.at("n_outputs").as_int() != result.n_elements) {
    artifact_error(
        "mthreads convolution output count differs from tensor metadata");
  }
  const auto spatial_rows = [](const TensorSpec& tensor) {
    std::int64_t product = tensor.dimensions[0];
    for (std::size_t axis = 2; axis < tensor.dimensions.size(); ++axis) {
      if (tensor.dimensions[axis] >
          std::numeric_limits<std::int32_t>::max() / product) {
        artifact_error("mthreads convolution launch extent exceeds int32");
      }
      product *= tensor.dimensions[axis];
    }
    return product;
  };
  static_cast<void>(spatial_rows(result.convolution_image));
  static_cast<void>(spatial_rows(result.convolution_result));
  std::int64_t kernel_volume = 1;
  for (std::size_t axis = 2;
       axis < result.convolution_filter.dimensions.size(); ++axis) {
    if (result.convolution_filter.dimensions[axis] >
        std::numeric_limits<std::int32_t>::max() / kernel_volume) {
      artifact_error("mthreads convolution kernel volume exceeds int32");
    }
    kernel_volume *= result.convolution_filter.dimensions[axis];
  }
  if (result.convolution_groups >
      std::numeric_limits<std::int32_t>::max() /
          result.convolution_batch) {
    artifact_error("mthreads convolution batch-group grid exceeds int32");
  }
  return result;
}

std::string pointer_token(const TensorSpec& tensor) {
  std::string result;
  if (tensor.data_type == "int32") {
    result = "*i32";
  } else if (tensor.data_type == "fp8_e8m0") {
    result = "*i8";
  } else if (tensor.data_type == "float32") {
    result = "*fp32";
  } else if (tensor.data_type == "float16") {
    result = "*fp16";
  } else if (tensor.data_type == "bfloat16") {
    result = "*bf16";
  } else if (tensor.data_type == "boolean") {
    result = "*i8";
  } else if (tensor.data_type == "fp8_e4m3") {
    result = "*fp8e4nv";
  } else if (tensor.data_type == "fp8_e5m2") {
    result = "*fp8e5";
  } else {
    artifact_error("mthreads pointwise pointer type is invalid");
  }
  if (tensor.alignment >= 16) {
    result += ":16";
  }
  return result;
}

std::string python_float_token(float value) {
  const double promoted = static_cast<double>(value);
  std::array<char, 128> buffer{};
  const auto conversion = std::to_chars(
      buffer.data(), buffer.data() + buffer.size(), promoted,
      std::chars_format::general);
  if (conversion.ec != std::errc{}) {
    artifact_error("cannot encode mthreads float32 signature token");
  }
  std::string result(buffer.data(), conversion.ptr);
  if (result.find_first_of(".eE") == std::string::npos) {
    result += ".0";
  }
  return result;
}

std::vector<std::int64_t> binary_pointwise_constants(
    const PointwiseRequest& request) {
  const std::size_t rank = request.output.dimensions.size();
  std::vector<std::int64_t> dimensions(
      kMaximumTensorRank - rank, 1);
  dimensions.insert(
      dimensions.end(),
      request.output.dimensions.begin(),
      request.output.dimensions.end());

  const auto effective_strides =
      [rank](const TensorSpec& tensor) {
        std::vector<std::int64_t> result(
            rank - tensor.dimensions.size(), 0);
        for (std::size_t index = 0;
             index < tensor.dimensions.size(); ++index) {
          result.push_back(
              tensor.dimensions[index] == 1 ? 0 : tensor.strides[index]);
        }
        return result;
      };
  std::vector<std::int64_t> left(kMaximumTensorRank - rank, 0);
  const auto effective_left = effective_strides(request.left);
  left.insert(left.end(), effective_left.begin(), effective_left.end());
  std::vector<std::int64_t> right(kMaximumTensorRank - rank, 0);
  const auto effective_right = effective_strides(request.right);
  right.insert(right.end(), effective_right.begin(), effective_right.end());
  std::vector<std::int64_t> output(kMaximumTensorRank - rank, 0);
  output.insert(
      output.end(),
      request.output.strides.begin(),
      request.output.strides.end());

  std::vector<std::int64_t> result;
  result.reserve(32);
  result.insert(result.end(), dimensions.begin(), dimensions.end());
  result.insert(result.end(), left.begin(), left.end());
  result.insert(result.end(), right.begin(), right.end());
  result.insert(result.end(), output.begin(), output.end());
  return result;
}

std::vector<std::int64_t> unary_pointwise_constants(
    const PointwiseRequest& request) {
  const std::size_t rank = request.output.dimensions.size();
  std::vector<std::int64_t> result;
  result.reserve(24);
  result.insert(result.end(), kMaximumTensorRank - rank, 1);
  result.insert(
      result.end(),
      request.output.dimensions.begin(),
      request.output.dimensions.end());
  result.insert(result.end(), kMaximumTensorRank - rank, 0);
  result.insert(
      result.end(), request.input.strides.begin(), request.input.strides.end());
  result.insert(result.end(), kMaximumTensorRank - rank, 0);
  result.insert(
      result.end(),
      request.output.strides.begin(),
      request.output.strides.end());
  return result;
}

std::vector<std::int64_t> ternary_pointwise_constants(
    const PointwiseRequest& request) {
  const std::size_t rank = request.output.dimensions.size();
  const auto effective_strides =
      [rank](const TensorSpec& tensor) {
        if (tensor.dimensions.size() > rank) {
          artifact_error("mthreads ternary input rank exceeds output rank");
        }
        std::vector<std::int64_t> result(
            rank - tensor.dimensions.size(), 0);
        for (std::size_t index = 0;
             index < tensor.dimensions.size(); ++index) {
          result.push_back(
              tensor.dimensions[index] == 1 ? 0 : tensor.strides[index]);
        }
        return result;
      };

  std::vector<std::int64_t> result;
  result.reserve(40);
  result.insert(result.end(), kMaximumTensorRank - rank, 1);
  result.insert(
      result.end(),
      request.output.dimensions.begin(),
      request.output.dimensions.end());
  for (const TensorSpec* tensor :
       {&request.left, &request.right, &request.predicate}) {
    result.insert(result.end(), kMaximumTensorRank - rank, 0);
    const auto effective = effective_strides(*tensor);
    result.insert(result.end(), effective.begin(), effective.end());
  }
  result.insert(result.end(), kMaximumTensorRank - rank, 0);
  result.insert(
      result.end(),
      request.output.strides.begin(),
      request.output.strides.end());
  return result;
}

std::vector<std::int64_t> layout_constants(
    const PointwiseRequest& request) {
  const std::size_t input_rank =
      request.logical_input_dimensions.size();
  const std::size_t output_rank = request.output.dimensions.size();
  if (input_rank == 0 || input_rank > kMaximumTensorRank ||
      request.logical_input_strides.size() != input_rank ||
      output_rank == 0 || output_rank > kMaximumTensorRank ||
      request.input_base < 0) {
    artifact_error("mthreads layout kernel metadata is invalid");
  }
  std::vector<std::int64_t> result;
  result.reserve(33);
  result.push_back(request.input_base);
  result.insert(result.end(), kMaximumTensorRank - input_rank, 1);
  result.insert(
      result.end(), request.logical_input_dimensions.begin(),
      request.logical_input_dimensions.end());
  result.insert(result.end(), kMaximumTensorRank - input_rank, 0);
  result.insert(
      result.end(), request.logical_input_strides.begin(),
      request.logical_input_strides.end());
  result.insert(result.end(), kMaximumTensorRank - output_rank, 1);
  result.insert(
      result.end(), request.output.dimensions.begin(),
      request.output.dimensions.end());
  result.insert(result.end(), kMaximumTensorRank - output_rank, 0);
  result.insert(
      result.end(), request.output.strides.begin(),
      request.output.strides.end());
  return result;
}

std::vector<std::int64_t> reduction_strided_constants(
    const PointwiseRequest& request) {
  const std::size_t rank = request.input.dimensions.size();
  const std::size_t axis =
      static_cast<std::size_t>(request.reduction_axis);
  if (rank == 0 || rank > kMaximumTensorRank || axis >= rank) {
    artifact_error("mthreads reduction strided metadata is invalid");
  }

  std::vector<std::int64_t> logical_dimensions =
      request.input.dimensions;
  logical_dimensions[axis] = 1;
  std::vector<std::int64_t> input_strides = request.input.strides;
  const std::int64_t reduction_stride = input_strides[axis];
  input_strides[axis] = 0;

  std::vector<std::int64_t> output_strides;
  if (request.keep_dimensions) {
    if (request.output.dimensions.size() != rank) {
      artifact_error("mthreads reduction keep-dimension rank is invalid");
    }
    output_strides = request.output.strides;
  } else {
    if (request.output.dimensions.size() + 1 != rank) {
      artifact_error("mthreads reduction output rank is invalid");
    }
    output_strides.reserve(rank);
    std::size_t output_axis = 0;
    for (std::size_t input_axis = 0; input_axis < rank; ++input_axis) {
      if (input_axis == axis) {
        output_strides.push_back(0);
      } else {
        output_strides.push_back(request.output.strides[output_axis++]);
      }
    }
  }
  output_strides[axis] = 0;

  const std::size_t leading = kMaximumTensorRank - rank;
  std::vector<std::int64_t> result;
  result.reserve(25);
  result.push_back(reduction_stride);
  result.insert(result.end(), leading, 1);
  result.insert(
      result.end(), logical_dimensions.begin(), logical_dimensions.end());
  result.insert(result.end(), leading, 0);
  result.insert(result.end(), input_strides.begin(), input_strides.end());
  result.insert(result.end(), leading, 0);
  result.insert(result.end(), output_strides.begin(), output_strides.end());
  return result;
}

std::vector<std::int64_t> matmul_constants(
    const PointwiseRequest& request) {
  std::vector<std::int64_t> result = {
      request.matmul_m, request.matmul_n, request.matmul_k};
  result.reserve(35);
  result.insert(
      result.end(), request.matmul_batch_dimensions.begin(),
      request.matmul_batch_dimensions.end());
  result.insert(
      result.end(), request.matmul_a_batch_strides.begin(),
      request.matmul_a_batch_strides.end());
  result.insert(
      result.end(), request.matmul_b_batch_strides.begin(),
      request.matmul_b_batch_strides.end());
  result.insert(
      result.end(), request.matmul_output_batch_strides.begin(),
      request.matmul_output_batch_strides.end());
  result.insert(
      result.end(),
      {request.matmul_a_stride_m, request.matmul_a_stride_k,
       request.matmul_b_stride_k, request.matmul_b_stride_n,
       request.matmul_output_stride_m, request.matmul_output_stride_n,
       request.left.data_type == "float32" ? 1 : 0,
       request.left.data_type == "float32" &&
               (request.input_precision == 2 ||
                (request.input_precision == 0 &&
                 std::min({request.matmul_m, request.matmul_n,
                           request.matmul_k}) >= 512))
           ? 1
           : 0});
  return result;
}

bool uses_matmul_descriptor(const PointwiseRequest& request) {
  if (request.family != PointwiseFamily::kMatmul ||
      (request.left.data_type != "float16" &&
       request.left.data_type != "bfloat16") ||
      !is_row_major_contiguous(request.left) ||
      !is_row_major_contiguous(request.right) ||
      !is_row_major_contiguous(request.output) ||
      std::min(
          {request.left.alignment,
           request.right.alignment,
           request.output.alignment}) < 16 ||
      request.matmul_m < 128 || request.matmul_n < 128 ||
      request.matmul_k < 64 || request.matmul_m % 128 != 0 ||
      request.matmul_n % 128 != 0 || request.matmul_k % 64 != 0) {
    return false;
  }
  const std::size_t output_rank = request.output.dimensions.size();
  if (request.left.dimensions.size() != output_rank ||
      request.right.dimensions.size() != output_rank ||
      output_rank < 2 ||
      !std::equal(
          request.left.dimensions.begin(),
          request.left.dimensions.end() - 2,
          request.output.dimensions.begin()) ||
      !std::equal(
          request.right.dimensions.begin(),
          request.right.dimensions.end() - 2,
          request.output.dimensions.begin())) {
    return false;
  }
  constexpr std::int64_t maximum =
      std::numeric_limits<std::int32_t>::max();
  return request.matmul_batch <= maximum / request.matmul_m &&
         request.matmul_batch <= maximum / request.matmul_k;
}

struct MatmulTleConfig {
  unsigned int block_m;
  unsigned int block_n;
  unsigned int block_k;
  unsigned int pipeline_stages;
  unsigned int panel_width;
};

std::optional<MatmulTleConfig> matmul_tle_config(
    const PointwiseRequest& request) {
  if (!uses_matmul_descriptor(request)) {
    return std::nullopt;
  }
  const auto matches = [&](std::int64_t batch,
                           std::int64_t m,
                           std::int64_t n,
                           std::int64_t k) {
    return request.matmul_batch == batch && request.matmul_m == m &&
           request.matmul_n == n && request.matmul_k == k;
  };
  if (matches(32, 512, 512, 512)) {
    return MatmulTleConfig{128, 128, 32, 3, 2};
  }
  if (matches(16, 1024, 1024, 1024)) {
    return MatmulTleConfig{256, 256, 32, 3, 2};
  }
  if (matches(16, 2048, 2048, 512)) {
    return MatmulTleConfig{256, 256, 32, 3, 2};
  }
  if (matches(8, 2048, 2048, 2048)) {
    return MatmulTleConfig{256, 256, 64, 3, 2};
  }
  if (matches(32, 1024, 1024, 4096)) {
    return MatmulTleConfig{256, 256, 64, 3, 4};
  }
  if (matches(4, 4096, 4096, 4096)) {
    return MatmulTleConfig{256, 256, 64, 3, 2};
  }
  return std::nullopt;
}

bool uses_matmul_tle(const PointwiseRequest& request) {
  return matmul_tle_config(request).has_value();
}

struct Im2colFpropGeometry {
  std::int64_t output_area = 0;
  std::int64_t reduction_extent = 0;
  std::array<std::int64_t, 3> column_strides = {0, 0, 1};
  std::size_t workspace_size = 0;
};

Im2colFpropGeometry im2col_fprop_geometry(
    const PointwiseRequest& request) {
  const auto multiply = [](
                            std::uint64_t left,
                            std::uint64_t right,
                            std::string_view field) {
    if (right != 0 &&
        left > std::numeric_limits<std::uint64_t>::max() / right) {
      artifact_error(
          "mthreads im2col Fprop " + std::string(field) +
          " overflows uint64");
    }
    return left * right;
  };
  const auto align = [](
                         std::uint64_t value,
                         std::string_view field) {
    if (value >
        std::numeric_limits<std::uint64_t>::max() -
            (kPhaseOneWorkspaceAlignment - 1)) {
      artifact_error(
          "mthreads im2col Fprop " + std::string(field) +
          " alignment overflows uint64");
    }
    return (value + kPhaseOneWorkspaceAlignment - 1) /
        kPhaseOneWorkspaceAlignment * kPhaseOneWorkspaceAlignment;
  };
  if (request.convolution_image.dimensions.size() != 4 ||
      request.convolution_filter.dimensions.size() != 4 ||
      request.convolution_result.dimensions.size() != 4) {
    artifact_error("mthreads im2col Fprop tensor rank is invalid");
  }
  const std::uint64_t output_area = multiply(
      static_cast<std::uint64_t>(
          request.convolution_result.dimensions[2]),
      static_cast<std::uint64_t>(
          request.convolution_result.dimensions[3]),
      "output area");
  std::uint64_t reduction_extent = multiply(
      static_cast<std::uint64_t>(request.convolution_in_per_group),
      static_cast<std::uint64_t>(
          request.convolution_filter.dimensions[2]),
      "reduction height");
  reduction_extent = multiply(
      reduction_extent,
      static_cast<std::uint64_t>(
          request.convolution_filter.dimensions[3]),
      "reduction width");
  const std::uint64_t column_stride_n = multiply(
      reduction_extent, output_area, "batch stride");
  const std::uint64_t column_elements = multiply(
      static_cast<std::uint64_t>(request.convolution_batch),
      column_stride_n,
      "column elements");
  const std::uint64_t raw_workspace = multiply(
      column_elements,
      static_cast<std::uint64_t>(
          element_size(request.convolution_image.data_type)),
      "workspace bytes");
  const std::uint64_t workspace =
      align(raw_workspace, "workspace");
  constexpr std::uint64_t kMaximumInt64 =
      static_cast<std::uint64_t>(
          std::numeric_limits<std::int64_t>::max());
  if (output_area > kMaximumInt64 ||
      reduction_extent > kMaximumInt64 ||
      column_stride_n > kMaximumInt64 ||
      workspace > std::numeric_limits<std::size_t>::max()) {
    artifact_error("mthreads im2col Fprop geometry exceeds host limits");
  }
  return {
      static_cast<std::int64_t>(output_area),
      static_cast<std::int64_t>(reduction_extent),
      {
          static_cast<std::int64_t>(column_stride_n),
          static_cast<std::int64_t>(output_area),
          1,
      },
      std::max(
          kPhaseOneWorkspaceSize, static_cast<std::size_t>(workspace)),
  };
}

struct Im2colFpropMmBlocks {
  std::uint32_t output_channels = 64;
  std::uint32_t output_spatial = 64;
  std::uint32_t reduction = 32;
};

Im2colFpropMmBlocks im2col_fprop_mm_blocks(
    const PointwiseRequest& request,
    const Im2colFpropGeometry& geometry) {
  if (request.convolution_batch == 1 &&
      geometry.reduction_extent >= 1024) {
    return {64, 32, 64};
  }
  return {
      64,
      64,
      request.convolution_image.data_type == "float32" ? 64U : 32U,
  };
}

bool uses_im2col_fprop(const PointwiseRequest& request) {
  const TensorSpec& image = request.convolution_image;
  const TensorSpec& filter = request.convolution_filter;
  const TensorSpec& result = request.convolution_result;
  if (!(request.family == PointwiseFamily::kConvolution &&
        (request.operation == "conv2d_fprop" ||
         request.operation == "convolution_fprop") &&
        request.convolution_spatial_rank == 2 &&
        (image.data_type == "float32" ||
         image.data_type == "float16" ||
         image.data_type == "bfloat16") &&
        filter.data_type == image.data_type &&
        result.data_type == image.data_type &&
        image.dimensions.size() == 4 &&
        filter.dimensions.size() == 4 &&
        result.dimensions.size() == 4 &&
        request.convolution_groups == 1 &&
        request.convolution_mode == 0 &&
        is_row_major_contiguous(image) &&
        is_row_major_contiguous(filter) &&
        is_row_major_contiguous(result))) {
    return false;
  }
  const std::int64_t filter_height = filter.dimensions[2];
  const std::int64_t filter_width = filter.dimensions[3];
  const bool standard_stride2_3x3 =
      filter_height == 3 && filter_width == 3 &&
      request.convolution_stride ==
          std::vector<std::int64_t>{2, 2} &&
      request.convolution_pre_padding ==
          std::vector<std::int64_t>{1, 1} &&
      request.convolution_post_padding ==
          std::vector<std::int64_t>{1, 1} &&
      request.convolution_dilation ==
          std::vector<std::int64_t>{1, 1};
  const bool stride2_3x3 =
      standard_stride2_3x3 &&
      request.convolution_in_per_group >= 64;
  const bool fp32_stem =
      standard_stride2_3x3 &&
      image.data_type == "float32" &&
      image.dimensions ==
          std::vector<std::int64_t>{1, 3, 640, 640} &&
      filter.dimensions[1] == 3 &&
      (request.convolution_out_per_group == 16 ||
       request.convolution_out_per_group == 32 ||
       request.convolution_out_per_group == 64 ||
       request.convolution_out_per_group == 96);
  const bool bounded_filter =
      filter_height > 0 && filter_height <= 15 &&
      filter_width > 0 && filter_width <= 15;
  const std::int64_t filter_area =
      bounded_filter ? filter_height * filter_width : 0;
  const bool medium_batched =
      request.convolution_batch >= 4 &&
      request.convolution_in_per_group >= 32 &&
      filter_area >= 2 && filter_area <= 15;
  if (!(stride2_3x3 || fp32_stem || medium_batched)) {
    return false;
  }
  return im2col_fprop_geometry(request).workspace_size <=
      512ULL * 1024ULL * 1024ULL;
}

std::size_t expected_im2col_fprop_workspace_size(
    const PointwiseRequest& request) {
  return im2col_fprop_geometry(request).workspace_size;
}

bool uses_stride2_tile4_dgrad(const PointwiseRequest& request) {
  const TensorSpec& image = request.convolution_image;
  const TensorSpec& loss = request.convolution_result;
  const TensorSpec& filter = request.convolution_filter;
  return request.family == PointwiseFamily::kConvolution &&
         request.operation == "convolution_dgrad" &&
         request.convolution_spatial_rank == 2 &&
         (image.data_type == "float32" ||
          image.data_type == "float16" ||
          image.data_type == "bfloat16") &&
         loss.data_type == image.data_type &&
         filter.data_type == image.data_type &&
         filter.dimensions.size() == 4 &&
         filter.dimensions[2] == 3 &&
         filter.dimensions[3] == 3 &&
         request.convolution_stride ==
             std::vector<std::int64_t>{2, 2} &&
         request.convolution_pre_padding ==
             std::vector<std::int64_t>{1, 1} &&
         request.convolution_post_padding ==
             std::vector<std::int64_t>{1, 1} &&
         request.convolution_dilation ==
             std::vector<std::int64_t>{1, 1} &&
         request.convolution_mode == 0 &&
         is_row_major_contiguous(image) &&
         is_row_major_contiguous(loss) &&
         is_row_major_contiguous(filter);
}

bool uses_stride2_packed2_1d_dgrad(
    const PointwiseRequest& request) {
  const TensorSpec& image = request.convolution_image;
  const TensorSpec& loss = request.convolution_result;
  const TensorSpec& filter = request.convolution_filter;
  return request.family == PointwiseFamily::kConvolution &&
         request.operation == "convolution_dgrad" &&
         request.convolution_spatial_rank == 1 &&
         (image.data_type == "float32" ||
          image.data_type == "float16" ||
          image.data_type == "bfloat16") &&
         loss.data_type == image.data_type &&
         filter.data_type == image.data_type &&
         filter.dimensions.size() == 3 &&
         filter.dimensions[2] == 5 &&
         request.convolution_stride == std::vector<std::int64_t>{2} &&
         request.convolution_pre_padding ==
             std::vector<std::int64_t>{2} &&
         request.convolution_post_padding ==
             std::vector<std::int64_t>{1} &&
         request.convolution_dilation ==
             std::vector<std::int64_t>{1} &&
         request.convolution_mode == 0 &&
         is_row_major_contiguous(image) &&
         is_row_major_contiguous(loss) &&
         is_row_major_contiguous(filter);
}

bool uses_stride2_packed4_dgrad(const PointwiseRequest& request) {
  return uses_stride2_tile4_dgrad(request) &&
         request.convolution_in_per_group <= 4;
}

bool uses_dense_stride2_dgrad(const PointwiseRequest& request) {
  return uses_stride2_tile4_dgrad(request) &&
         request.convolution_groups == 1 &&
         request.convolution_in_per_group > 4;
}

struct DenseDgradGeometry {
  std::int64_t loss_rows = 0;
  std::int64_t loss_offset = 0;
  std::int64_t matrix_rows = 0;
  std::int64_t reduction_extent = 0;
  std::size_t workspace_size = 0;
};

DenseDgradGeometry dense_dgrad_geometry(
    const PointwiseRequest& request) {
  const auto multiply = [](
                            std::uint64_t left,
                            std::uint64_t right,
                            std::string_view field) {
    if (right != 0 &&
        left > std::numeric_limits<std::uint64_t>::max() / right) {
      artifact_error(
          "mthreads dense Dgrad " + std::string(field) +
          " overflows uint64");
    }
    return left * right;
  };
  const auto add = [](
                       std::uint64_t left,
                       std::uint64_t right,
                       std::string_view field) {
    if (left > std::numeric_limits<std::uint64_t>::max() - right) {
      artifact_error(
          "mthreads dense Dgrad " + std::string(field) +
          " overflows uint64");
    }
    return left + right;
  };
  const auto align = [&add](
                         std::uint64_t value,
                         std::string_view field) {
    const std::uint64_t padded = add(
        value, kPhaseOneWorkspaceAlignment - 1, field);
    return padded / kPhaseOneWorkspaceAlignment *
        kPhaseOneWorkspaceAlignment;
  };
  if (request.convolution_result.dimensions.size() != 4) {
    artifact_error("mthreads dense Dgrad loss rank is invalid");
  }
  const std::uint64_t cin = static_cast<std::uint64_t>(
      request.convolution_in_per_group);
  const std::uint64_t cout = static_cast<std::uint64_t>(
      request.convolution_out_per_group);
  const std::uint64_t matrix_rows = multiply(4, cin, "matrix rows");
  const std::uint64_t reduction_extent =
      multiply(4, cout, "reduction extent");
  std::uint64_t loss_rows = multiply(
      static_cast<std::uint64_t>(request.convolution_batch),
      static_cast<std::uint64_t>(
          request.convolution_result.dimensions[2]),
      "loss rows");
  loss_rows = multiply(
      loss_rows,
      static_cast<std::uint64_t>(
          request.convolution_result.dimensions[3]),
      "loss rows");
  const std::uint64_t packed_filter_elements = multiply(
      matrix_rows, reduction_extent, "packed filter elements");
  const std::uint64_t packed_filter_bytes = multiply(
      packed_filter_elements,
      element_size(request.convolution_image.data_type),
      "packed filter bytes");
  const std::uint64_t aligned_filter_bytes =
      align(packed_filter_bytes, "packed filter alignment");
  const std::uint64_t loss_offset = aligned_filter_bytes /
      element_size(request.convolution_image.data_type);
  const std::uint64_t packed_loss_bytes = multiply(
      multiply(reduction_extent, loss_rows, "packed loss elements"),
      element_size(request.convolution_image.data_type),
      "packed loss bytes");
  const std::uint64_t workspace = align(
      add(aligned_filter_bytes, packed_loss_bytes, "workspace size"),
      "workspace alignment");
  constexpr std::uint64_t kMaximumInt64 =
      static_cast<std::uint64_t>(
          std::numeric_limits<std::int64_t>::max());
  if (loss_rows > kMaximumInt64 || loss_offset > kMaximumInt64 ||
      matrix_rows > kMaximumInt64 || reduction_extent > kMaximumInt64 ||
      workspace > std::numeric_limits<std::size_t>::max()) {
    artifact_error("mthreads dense Dgrad geometry exceeds host limits");
  }
  return {
      static_cast<std::int64_t>(loss_rows),
      static_cast<std::int64_t>(loss_offset),
      static_cast<std::int64_t>(matrix_rows),
      static_cast<std::int64_t>(reduction_extent),
      std::max(
          kPhaseOneWorkspaceSize, static_cast<std::size_t>(workspace))};
}

std::size_t expected_dense_dgrad_workspace_size(
    const PointwiseRequest& request) {
  return dense_dgrad_geometry(request).workspace_size;
}

unsigned int stride2_packed4_dgrad_block_m(
    const PointwiseRequest& request, unsigned int block) {
  return uses_stride2_packed4_dgrad(request) ? block * 4 : block;
}


bool uses_standard_wgrad(const PointwiseRequest& request) {
  if (request.input_precision == 1) return false;
  const TensorSpec& image = request.convolution_image;
  const TensorSpec& loss = request.convolution_result;
  const TensorSpec& output = request.convolution_filter;
  const bool stride2 =
      image.dimensions == std::vector<std::int64_t>{8, 64, 56, 56} &&
      loss.dimensions == std::vector<std::int64_t>{8, 128, 28, 28} &&
      output.dimensions == std::vector<std::int64_t>{128, 64, 3, 3} &&
      request.convolution_stride == std::vector<std::int64_t>{2, 2} &&
      request.convolution_pre_padding ==
          std::vector<std::int64_t>{1, 1} &&
      request.convolution_post_padding ==
          std::vector<std::int64_t>{1, 1};
  const bool standard3x3 =
      image.dimensions == std::vector<std::int64_t>{8, 32, 32, 32} &&
      loss.dimensions == std::vector<std::int64_t>{8, 64, 32, 32} &&
      output.dimensions == std::vector<std::int64_t>{64, 32, 3, 3} &&
      request.convolution_stride == std::vector<std::int64_t>{1, 1} &&
      request.convolution_pre_padding ==
          std::vector<std::int64_t>{1, 1} &&
      request.convolution_post_padding ==
          std::vector<std::int64_t>{1, 1};
  const bool standard1x1 =
      image.dimensions == std::vector<std::int64_t>{8, 64, 28, 28} &&
      loss.dimensions == std::vector<std::int64_t>{8, 128, 28, 28} &&
      output.dimensions == std::vector<std::int64_t>{128, 64, 1, 1} &&
      request.convolution_stride == std::vector<std::int64_t>{1, 1} &&
      request.convolution_pre_padding ==
          std::vector<std::int64_t>{0, 0} &&
      request.convolution_post_padding ==
          std::vector<std::int64_t>{0, 0};
  return request.family == PointwiseFamily::kConvolution &&
         request.operation == "convolution_wgrad" &&
         (image.data_type == "float32" ||
          image.data_type == "float16" ||
          image.data_type == "bfloat16") &&
         loss.data_type == image.data_type &&
         output.data_type == image.data_type &&
         (stride2 || standard3x3 || standard1x1) &&
         is_row_major_contiguous(image) &&
         is_row_major_contiguous(loss) &&
         is_row_major_contiguous(output) &&
         request.convolution_spatial_rank == 2 &&
         request.convolution_groups == 1 &&
         request.convolution_mode == 0 &&
         request.convolution_dilation ==
             std::vector<std::int64_t>{1, 1};
}

bool uses_nd_packed_wgrad(const PointwiseRequest& request) {
  if (request.input_precision == 1) return false;
  const TensorSpec& image = request.convolution_image;
  const TensorSpec& loss = request.convolution_result;
  const TensorSpec& output = request.convolution_filter;
  const bool one_dimensional =
      request.convolution_spatial_rank == 1 &&
      image.dimensions == std::vector<std::int64_t>{16, 32, 256} &&
      loss.dimensions == std::vector<std::int64_t>{16, 64, 256} &&
      output.dimensions == std::vector<std::int64_t>{64, 32, 3} &&
      request.convolution_stride == std::vector<std::int64_t>{1} &&
      request.convolution_pre_padding == std::vector<std::int64_t>{1} &&
      request.convolution_post_padding == std::vector<std::int64_t>{1} &&
      request.convolution_dilation == std::vector<std::int64_t>{1};
  const bool symmetric_three_dimensional =
      request.convolution_spatial_rank == 3 &&
      image.dimensions ==
          std::vector<std::int64_t>{2, 8, 8, 16, 16} &&
      loss.dimensions ==
          std::vector<std::int64_t>{2, 16, 8, 16, 16} &&
      output.dimensions ==
          std::vector<std::int64_t>{16, 8, 3, 3, 3} &&
      request.convolution_stride ==
          std::vector<std::int64_t>{1, 1, 1} &&
      request.convolution_pre_padding ==
          std::vector<std::int64_t>{1, 1, 1} &&
      request.convolution_post_padding ==
          std::vector<std::int64_t>{1, 1, 1} &&
      request.convolution_dilation ==
          std::vector<std::int64_t>{1, 1, 1};
  const bool asymmetric_three_dimensional =
      request.convolution_spatial_rank == 3 &&
      image.dimensions ==
          std::vector<std::int64_t>{1, 8, 10, 12, 14} &&
      loss.dimensions ==
          std::vector<std::int64_t>{1, 12, 10, 11, 15} &&
      output.dimensions ==
          std::vector<std::int64_t>{12, 8, 2, 3, 3} &&
      request.convolution_stride ==
          std::vector<std::int64_t>{1, 1, 1} &&
      request.convolution_pre_padding ==
          std::vector<std::int64_t>{1, 0, 1} &&
      request.convolution_post_padding ==
          std::vector<std::int64_t>{0, 1, 2} &&
      request.convolution_dilation ==
          std::vector<std::int64_t>{1, 1, 1};
  return request.family == PointwiseFamily::kConvolution &&
         request.operation == "convolution_wgrad" &&
         (image.data_type == "float32" ||
          image.data_type == "float16" ||
          image.data_type == "bfloat16") &&
         loss.data_type == image.data_type &&
         output.data_type == image.data_type &&
         (one_dimensional || symmetric_three_dimensional ||
          asymmetric_three_dimensional) &&
         is_row_major_contiguous(image) &&
         is_row_major_contiguous(loss) &&
         is_row_major_contiguous(output) &&
         request.convolution_groups == 1 &&
         request.convolution_mode == 0;
}

std::size_t expected_standard_wgrad_workspace_size(
    const PointwiseRequest& request) {
  const std::uint64_t kh = static_cast<std::uint64_t>(
      request.convolution_filter.dimensions[2]);
  const std::uint64_t kw = static_cast<std::uint64_t>(
      request.convolution_filter.dimensions[3]);
  const bool one_by_one = kh == 1 && kw == 1;
  const std::uint64_t num_splits = one_by_one
      ? 8ULL
      : static_cast<std::uint64_t>(request.convolution_batch);
  const std::uint64_t partial_raw =
      num_splits *
      static_cast<std::uint64_t>(request.convolution_out_per_group) *
      static_cast<std::uint64_t>(request.convolution_in_per_group) * kh * kw *
      sizeof(float);
  const std::uint64_t partial_aligned =
      (partial_raw + kPhaseOneWorkspaceAlignment - 1) /
      kPhaseOneWorkspaceAlignment * kPhaseOneWorkspaceAlignment;
  const std::uint64_t column_size =
      static_cast<std::uint64_t>(request.convolution_batch) *
          static_cast<std::uint64_t>(
              request.convolution_result.dimensions[2]) *
          static_cast<std::uint64_t>(
              request.convolution_result.dimensions[3]) *
          static_cast<std::uint64_t>(request.convolution_in_per_group) * kh *
          kw * static_cast<std::uint64_t>(
                   element_size(request.convolution_image.data_type));
  const std::uint64_t raw =
      one_by_one ? partial_aligned : partial_aligned + column_size;
  if (raw > std::numeric_limits<std::size_t>::max() -
                (kPhaseOneWorkspaceAlignment - 1)) {
    artifact_error("mthreads standard Wgrad workspace size overflows size_t");
  }
  const std::size_t aligned =
      (static_cast<std::size_t>(raw) + kPhaseOneWorkspaceAlignment - 1) /
      kPhaseOneWorkspaceAlignment * kPhaseOneWorkspaceAlignment;
  return std::max(kPhaseOneWorkspaceSize, aligned);
}


std::size_t expected_nd_packed_wgrad_workspace_size(
    const PointwiseRequest& request) {
  std::uint64_t output_area = 1;
  for (std::size_t axis = 2;
       axis < request.convolution_result.dimensions.size(); ++axis) {
    output_area *= static_cast<std::uint64_t>(
        request.convolution_result.dimensions[axis]);
  }
  std::uint64_t kernel_volume = 1;
  for (std::size_t axis = 2;
       axis < request.convolution_filter.dimensions.size(); ++axis) {
    kernel_volume *= static_cast<std::uint64_t>(
        request.convolution_filter.dimensions[axis]);
  }
  const std::uint64_t total_rows =
      static_cast<std::uint64_t>(request.convolution_batch) * output_area;
  const std::uint64_t reduction_extent =
      static_cast<std::uint64_t>(request.convolution_in_per_group) *
      kernel_volume;
  const std::uint64_t total_weights =
      static_cast<std::uint64_t>(request.convolution_out_per_group) *
      reduction_extent;
  const std::uint64_t num_splits = total_rows >= 4096 ? 16ULL : 8ULL;
  const std::uint64_t partial_raw =
      num_splits * total_weights * sizeof(float);
  const std::uint64_t partial_aligned =
      (partial_raw + kPhaseOneWorkspaceAlignment - 1) /
      kPhaseOneWorkspaceAlignment * kPhaseOneWorkspaceAlignment;
  const std::uint64_t column_size =
      total_rows * reduction_extent * static_cast<std::uint64_t>(
          element_size(request.convolution_image.data_type));
  const std::uint64_t raw = partial_aligned + column_size;
  if (raw > std::numeric_limits<std::size_t>::max() -
                (kPhaseOneWorkspaceAlignment - 1)) {
    artifact_error(
        "mthreads ND packed Wgrad workspace size overflows size_t");
  }
  const std::size_t aligned =
      (static_cast<std::size_t>(raw) + kPhaseOneWorkspaceAlignment - 1) /
      kPhaseOneWorkspaceAlignment * kPhaseOneWorkspaceAlignment;
  return std::max(kPhaseOneWorkspaceSize, aligned);
}
bool uses_stem_wgrad(const PointwiseRequest& request) {
  if (request.input_precision == 1) return false;
  const TensorSpec& image = request.convolution_image;
  const TensorSpec& loss = request.convolution_result;
  const TensorSpec& output = request.convolution_filter;
  return request.family == PointwiseFamily::kConvolution &&
         request.operation == "convolution_wgrad" &&
         (image.data_type == "float32" ||
          image.data_type == "float16" ||
          image.data_type == "bfloat16") &&
         loss.data_type == image.data_type &&
         output.data_type == image.data_type &&
         image.dimensions ==
             std::vector<std::int64_t>{1, 3, 640, 640} &&
         loss.dimensions.size() == 4 && loss.dimensions[0] == 1 &&
         (loss.dimensions[1] == 16 || loss.dimensions[1] == 32 ||
          loss.dimensions[1] == 64 || loss.dimensions[1] == 96) &&
         loss.dimensions[2] == 320 && loss.dimensions[3] == 320 &&
         output.dimensions ==
             std::vector<std::int64_t>{loss.dimensions[1], 3, 3, 3} &&
         is_row_major_contiguous(image) &&
         is_row_major_contiguous(loss) &&
         is_row_major_contiguous(output) &&
         request.convolution_spatial_rank == 2 &&
         request.convolution_groups == 1 &&
         request.convolution_mode == 0 &&
         request.convolution_stride == std::vector<std::int64_t>{2, 2} &&
         request.convolution_pre_padding ==
             std::vector<std::int64_t>{1, 1} &&
         request.convolution_post_padding ==
             std::vector<std::int64_t>{1, 1} &&
         request.convolution_dilation ==
             std::vector<std::int64_t>{1, 1};
}

std::size_t expected_stem_wgrad_workspace_size(
    const PointwiseRequest& request) {
  const std::uint64_t raw =
      64ULL * static_cast<std::uint64_t>(request.convolution_out_per_group) *
      static_cast<std::uint64_t>(request.convolution_in_per_group) * 9ULL *
      sizeof(float);
  const std::size_t aligned = static_cast<std::size_t>(raw) /
      kPhaseOneWorkspaceAlignment * kPhaseOneWorkspaceAlignment;
  return std::max(kPhaseOneWorkspaceSize, aligned);
}

bool uses_p5_wgrad(const PointwiseRequest& request) {
  if (request.input_precision == 1) return false;
  const TensorSpec& image = request.convolution_image;
  const TensorSpec& loss = request.convolution_result;
  const TensorSpec& output = request.convolution_filter;
  return request.family == PointwiseFamily::kConvolution &&
         request.operation == "convolution_wgrad" &&
         (image.data_type == "float32" ||
          image.data_type == "float16" ||
          image.data_type == "bfloat16") &&
         loss.data_type == image.data_type &&
         output.data_type == image.data_type &&
         image.dimensions.size() == 4 &&
         loss.dimensions.size() == 4 &&
         output.dimensions.size() == 4 &&
         image.dimensions[0] == 1 &&
         image.dimensions[2] == 40 && image.dimensions[3] == 40 &&
         loss.dimensions[0] == 1 &&
         loss.dimensions[2] == 20 && loss.dimensions[3] == 20 &&
         output.dimensions[2] == 3 && output.dimensions[3] == 3 &&
         image.dimensions[1] == output.dimensions[1] &&
         loss.dimensions[1] == output.dimensions[0] &&
         is_row_major_contiguous(image) &&
         is_row_major_contiguous(loss) &&
         is_row_major_contiguous(output) &&
         request.convolution_spatial_rank == 2 &&
         request.convolution_groups == 1 &&
         request.convolution_mode == 0 &&
         request.convolution_stride == std::vector<std::int64_t>{2, 2} &&
         request.convolution_pre_padding ==
             std::vector<std::int64_t>{1, 1} &&
         request.convolution_post_padding ==
             std::vector<std::int64_t>{1, 1} &&
         request.convolution_dilation ==
             std::vector<std::int64_t>{1, 1};
}

std::size_t expected_p5_wgrad_workspace_size(
    const PointwiseRequest& request) {
  const std::uint64_t bytes_per_channel =
      400ULL * 9ULL * static_cast<std::uint64_t>(
                           element_size(request.convolution_image.data_type));
  const auto channels = static_cast<std::uint64_t>(
      request.convolution_in_per_group);
  if (channels > std::numeric_limits<std::size_t>::max() /
                         bytes_per_channel) {
    artifact_error("mthreads P5 Wgrad workspace size overflows size_t");
  }
  const std::size_t raw =
      static_cast<std::size_t>(channels * bytes_per_channel);
  const std::size_t aligned =
      (raw + kPhaseOneWorkspaceAlignment - 1) /
      kPhaseOneWorkspaceAlignment * kPhaseOneWorkspaceAlignment;
  return std::max(kPhaseOneWorkspaceSize, aligned);
}

unsigned int matmul_block_n(
    const PointwiseRequest& request, unsigned int block_m) {
  return request.left.data_type != "float32" && block_m <= 64U &&
                 request.matmul_n >= static_cast<std::int64_t>(block_m * 2U)
             ? block_m * 2U
             : block_m;
}

struct PaddedConvolutionTensor {
  std::array<std::int64_t, 5> dimensions = {1, 1, 1, 1, 1};
  std::array<std::int64_t, 5> strides = {0, 0, 0, 0, 0};
};

PaddedConvolutionTensor padded_convolution_tensor(
    const TensorSpec& tensor, std::size_t spatial_rank) {
  if (spatial_rank < 1 || spatial_rank > 3 ||
      tensor.dimensions.size() != spatial_rank + 2 ||
      tensor.strides.size() != tensor.dimensions.size()) {
    artifact_error("mthreads convolution padded tensor is invalid");
  }
  PaddedConvolutionTensor result;
  result.dimensions[0] = tensor.dimensions[0];
  result.dimensions[1] = tensor.dimensions[1];
  result.strides[0] = tensor.strides[0];
  result.strides[1] = tensor.strides[1];
  const std::size_t leading = 3 - spatial_rank;
  for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
    const std::size_t destination = 2 + leading + axis;
    result.dimensions[destination] = tensor.dimensions[axis + 2];
    result.strides[destination] = tensor.strides[axis + 2];
  }
  return result;
}

std::array<std::int64_t, 3> padded_convolution_spatial(
    const std::vector<std::int64_t>& values, std::int64_t fill) {
  if (values.empty() || values.size() > 3) {
    artifact_error("mthreads convolution spatial vector is invalid");
  }
  std::array<std::int64_t, 3> result = {fill, fill, fill};
  const std::size_t leading = 3 - values.size();
  std::copy(values.begin(), values.end(), result.begin() + leading);
  return result;
}

std::vector<std::int64_t> convolution_constants(
    const PointwiseRequest& request,
    std::string_view function,
    unsigned int block_size) {
  const std::size_t spatial_rank =
      static_cast<std::size_t>(request.convolution_spatial_rank);
  const auto image = padded_convolution_tensor(
      request.convolution_image, spatial_rank);
  const auto filter = padded_convolution_tensor(
      request.convolution_filter, spatial_rank);
  const auto loss = padded_convolution_tensor(
      request.convolution_result, spatial_rank);
  const auto stride =
      padded_convolution_spatial(request.convolution_stride, 1);
  const auto padding =
      padded_convolution_spatial(request.convolution_pre_padding, 0);
  const auto dilation =
      padded_convolution_spatial(request.convolution_dilation, 1);
  const std::int64_t xd = image.dimensions[2];
  const std::int64_t xh = image.dimensions[3];
  const std::int64_t xw = image.dimensions[4];
  const std::int64_t kd = filter.dimensions[2];
  const std::int64_t kh = filter.dimensions[3];
  const std::int64_t kw = filter.dimensions[4];
  const std::int64_t od = loss.dimensions[2];
  const std::int64_t oh = loss.dimensions[3];
  const std::int64_t ow = loss.dimensions[4];
  const std::int64_t block = static_cast<std::int64_t>(block_size);
  const std::int64_t dtype_id =
      request.convolution_image.data_type == "float32"
          ? 0
          : (request.convolution_image.data_type == "float16" ? 1 : 2);
  const auto capped_reduction_factor = [](std::int64_t value) {
    return std::min<std::uint64_t>(
        64U, static_cast<std::uint64_t>(value));
  };
  std::uint64_t fprop_reduction_extent =
      capped_reduction_factor(request.convolution_in_per_group);
  for (std::size_t index = 2;
       index < request.convolution_filter.dimensions.size(); ++index) {
    fprop_reduction_extent = std::min<std::uint64_t>(
        64U, fprop_reduction_extent * capped_reduction_factor(
            request.convolution_filter.dimensions[index]));
  }
  const bool fprop_operation =
      request.operation == "conv2d_fprop" ||
      request.operation == "convolution_fprop";
  const std::int64_t input_precision =
      request.convolution_image.data_type == "float32" &&
              (request.input_precision == 2 ||
               (request.input_precision == 0 &&
                (!fprop_operation || fprop_reduction_extent >= 64U)))
          ? 1
          : 0;

  std::vector<std::int64_t> result;
  if (function == "conv1d_gemm_kernel") {
    result = {
        request.convolution_batch * ow,
        xw,
        ow,
        dtype_id};
    result.insert(
        result.end(),
        request.convolution_image.strides.begin(),
        request.convolution_image.strides.end());
    result.insert(
        result.end(),
        request.convolution_filter.strides.begin(),
        request.convolution_filter.strides.end());
    result.push_back(1);
    result.insert(
        result.end(),
        request.convolution_result.strides.begin(),
        request.convolution_result.strides.end());
    result.insert(
        result.end(),
        {request.convolution_in_per_group,
         request.convolution_out_per_group,
         kw,
         stride[2],
         padding[2],
         dilation[2],
         0,
         block,
         block,
         block,
         8,
         input_precision});
    return result;
  }
  if (function == "conv2d_spatial_nchw_kernel") {
    return {
        xh,
        xw,
        oh,
        ow,
        request.convolution_in_channels,
        request.convolution_out_channels,
        request.convolution_in_per_group,
        request.convolution_out_per_group,
        request.convolution_groups,
        stride[1],
        stride[2],
        padding[1],
        padding[2],
        dilation[1],
        dilation[2],
        kh,
        kw,
        0,
        block,
        block,
        block,
        8,
        dtype_id,
        input_precision,
        image.strides[0],
        image.strides[1],
        image.strides[3],
        image.strides[4],
        filter.strides[0],
        filter.strides[1],
        filter.strides[3],
        filter.strides[4],
        loss.strides[0],
        loss.strides[1],
        loss.strides[3],
        loss.strides[4]};
  }
  if (function == "conv3d_spatial_ncdhw_m_kernel") {
    result = {
        request.convolution_batch * od * oh * ow,
        xd,
        xh,
        xw,
        od,
        oh,
        ow,
        request.convolution_in_channels,
        request.convolution_out_channels,
        request.convolution_in_per_group,
        request.convolution_out_per_group,
        stride[0],
        stride[1],
        stride[2],
        padding[0],
        padding[1],
        padding[2],
        dilation[0],
        dilation[1],
        dilation[2],
        kd,
        kh,
        kw,
        0,
        block,
        block,
        block,
        8};
    result.insert(
        result.end(), image.strides.begin(), image.strides.end());
    result.insert(
        result.end(), filter.strides.begin(), filter.strides.end());
    result.insert(result.end(), loss.strides.begin(), loss.strides.end());
    result.push_back(input_precision);
    return result;
  }

  result = {
      xd,
      xh,
      xw,
      od,
      oh,
      ow,
      kd,
      kh,
      kw,
      request.convolution_in_per_group,
      request.convolution_out_per_group,
      stride[0],
      stride[1],
      stride[2],
      padding[0],
      padding[1],
      padding[2],
      dilation[0],
      dilation[1],
      dilation[2],
      request.convolution_mode};
  result.insert(result.end(), loss.strides.begin(), loss.strides.end());
  result.insert(result.end(), image.strides.begin(), image.strides.end());
  result.insert(result.end(), filter.strides.begin(), filter.strides.end());
  result.push_back(
      request.input_precision
          ? static_cast<int>(request.input_precision == 2)
          : static_cast<int>(request.convolution_image.data_type ==
                                 "float32" &&
                             (request.operation == "convolution_wgrad" ||
                              uses_stride2_tile4_dgrad(request) ||
                              uses_stride2_packed2_1d_dgrad(request))));
  if (function == "conv_dgrad_nd_kernel") {
    result.insert(
        result.end(),
        {request.convolution_batch * xd * xh * xw,
         stride2_packed4_dgrad_block_m(request, block),
         block,
         block,
         8});
  } else if (function == "conv_wgrad_nd_kernel") {
    result.insert(
        result.end(),
        {request.convolution_batch * od * oh * ow,
         block,
         block,
         block});
  } else {
    artifact_error("mthreads convolution function is unsupported");
  }
  return result;
}

unsigned int reduction_block_n(const PointwiseRequest& request) {
  unsigned int result = 1;
  const auto extent = static_cast<unsigned int>(request.extent);
  while (result < extent) {
    result <<= 1U;
  }
  return result;
}

bool uses_batchnorm_inference_nchw(const PointwiseRequest& request) {
  return request.family == PointwiseFamily::kBatchnormInference &&
         request.dense && request.normalization_spatial > 1;
}

bool uses_packed_identity(const PointwiseRequest& request) {
  return request.family == PointwiseFamily::kUnary &&
         request.operation == "identity" && request.dense &&
         request.input.alignment >= 8 && request.output.alignment >= 8;
}

unsigned int identity_pack_size(const PointwiseRequest& request) {
  if (request.input.data_type == "float32" ||
      request.input.data_type == "int32") {
    return 2U;
  }
  if (request.input.data_type == "float16" ||
      request.input.data_type == "bfloat16") {
    return 4U;
  }
  if (element_size(request.input.data_type) == 1) return 8U;
  artifact_error("mthreads packed Identity data type is invalid");
}

unsigned int identity_tiles_per_program(const PointwiseRequest& request) {
  return uses_packed_identity(request) ? 4U : (request.dense ? 8U : 1U);
}

unsigned int unary_tiles_per_program(const PointwiseRequest& request) {
  static_cast<void>(request);
  return 1U;
}

bool uses_contiguous_reshape(const PointwiseRequest& request) {
  return request.family == PointwiseFamily::kLayout &&
         request.operation == "reshape" && request.input_base == 0 &&
         is_row_major_contiguous(request.input) &&
         is_row_major_contiguous(request.output);
}

bool uses_physical_transpose(const PointwiseRequest& request) {
  return request.family == PointwiseFamily::kLayout &&
         request.operation == "transpose" && request.input_base == 0 &&
         request.logical_input_dimensions == request.output.dimensions &&
         request.logical_input_strides == request.output.strides &&
         request.input.storage_size == request.output.storage_size;
}

bool uses_i32_slice(const PointwiseRequest& request) {
  if (request.family != PointwiseFamily::kLayout ||
      request.operation != "slice") {
    return false;
  }
  constexpr std::uint64_t limit =
      static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
  const auto last_offset = [&](const std::vector<std::int64_t>& dimensions,
                               const std::vector<std::int64_t>& strides,
                               std::uint64_t base) {
    if (base > limit || dimensions.size() != strides.size()) {
      return false;
    }
    for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
      const std::uint64_t extent =
          static_cast<std::uint64_t>(dimensions[axis] - 1);
      const std::uint64_t stride =
          static_cast<std::uint64_t>(strides[axis]);
      if (extent != 0 && stride > (limit - base) / extent) {
        return false;
      }
      base += extent * stride;
    }
    return true;
  };
  return last_offset(
             request.logical_input_dimensions,
             request.logical_input_strides,
             static_cast<std::uint64_t>(request.input_base)) &&
         last_offset(
             request.output.dimensions, request.output.strides, 0);
}

std::string expected_signature(
    const PointwiseRequest& request, unsigned int block_size) {
  std::vector<std::string> tokens;
  const auto append_normalization_metadata = [&]() {
    const std::size_t leading =
        kMaximumTensorRank - request.normalization_x.dimensions.size();
    for (std::size_t index = 0; index < leading; ++index) {
      tokens.emplace_back("1");
    }
    for (const std::int64_t value :
         request.normalization_x.dimensions) {
      tokens.push_back(std::to_string(value));
    }
    for (std::size_t index = 0; index < leading; ++index) {
      tokens.emplace_back("0");
    }
    for (const std::int64_t value : request.normalization_x.strides) {
      tokens.push_back(std::to_string(value));
    }
    for (std::size_t index = 0; index < leading; ++index) {
      tokens.emplace_back("0");
    }
    for (const std::int64_t value : request.normalization_y.strides) {
      tokens.push_back(std::to_string(value));
    }
  };
  if (request.family == PointwiseFamily::kNormalization) {
    if (request.operation == "layernorm") {
      tokens = {pointer_token(request.normalization_x),
                pointer_token(request.normalization_y),
                pointer_token(request.normalization_mean),
                pointer_token(request.normalization_inv_variance),
                pointer_token(request.normalization_scale),
                pointer_token(request.normalization_bias),
                "i32",
                python_float_token(request.normalization_epsilon),
                std::to_string(request.normalization_elements),
                std::to_string(block_size),
                "1",
                "1",
                "1",
                "1",
                "0",
                "0",
                "0"};
    } else {
      tokens = {pointer_token(request.normalization_x),
                pointer_token(request.normalization_y),
                pointer_token(request.normalization_scale),
                pointer_token(request.normalization_bias),
                pointer_token(request.normalization_inv_variance),
                "i32",
                std::to_string(request.normalization_elements),
                python_float_token(request.normalization_epsilon),
                std::to_string(block_size),
                "1",
                "1",
                "1",
                "1",
                "0"};
    }
  } else if (request.family == PointwiseFamily::kBatchnorm) {
    tokens = {pointer_token(request.normalization_x),
              pointer_token(request.normalization_y),
              pointer_token(request.normalization_previous_running_mean),
              pointer_token(request.normalization_previous_running_variance),
              pointer_token(request.normalization_scale),
              pointer_token(request.normalization_bias),
              pointer_token(request.normalization_mean),
              pointer_token(request.normalization_inv_variance),
              pointer_token(request.normalization_next_running_mean),
              pointer_token(request.normalization_next_running_variance)};
    if (request.dense) {
      for (const std::string& value : {
               std::to_string(request.normalization_batch),
               std::to_string(request.normalization_channels),
               std::to_string(request.normalization_spatial),
               python_float_token(request.normalization_epsilon),
               python_float_token(request.normalization_momentum),
               std::to_string(block_size),
               std::string("1"),
               std::string("1"),
               std::string("1"),
               std::string("1"),
               std::string("1")}) {
        tokens.push_back(value);
      }
    } else {
      tokens.insert(tokens.end(), {"i32", "i32", "i32"});
      tokens.push_back(
          python_float_token(request.normalization_epsilon));
      tokens.push_back(
          python_float_token(request.normalization_momentum));
      tokens.push_back(std::to_string(block_size));
      tokens.insert(tokens.end(), {"1", "1", "1", "1", "1", "1"});
      append_normalization_metadata();
    }
  } else if (request.family == PointwiseFamily::kBatchnormInference) {
    tokens = {
        pointer_token(request.normalization_x),
        pointer_token(request.normalization_mean),
        pointer_token(request.normalization_inv_variance),
        pointer_token(request.normalization_scale),
        pointer_token(request.normalization_bias),
        pointer_token(request.normalization_y)};
    if (uses_batchnorm_inference_nchw(request)) {
      tokens.push_back(std::to_string(request.normalization_channels));
      tokens.push_back(std::to_string(request.normalization_spatial));
      tokens.push_back(python_float_token(0.0F));
      tokens.push_back(std::to_string(block_size));
      tokens.insert(tokens.end(), {"1", "1", "1"});
    } else {
      tokens.insert(tokens.end(), {"i32", "i32", "i32"});
      tokens.push_back(python_float_token(0.0F));
      tokens.push_back(std::to_string(block_size));
      tokens.insert(tokens.end(), {"1", "1", "1", "1"});
      append_normalization_metadata();
    }
  } else if (request.family == PointwiseFamily::kAddSquare) {
    tokens = {
        pointer_token(request.left),
        pointer_token(request.right),
        pointer_token(request.output),
        "i32",
        "1",
        std::to_string(block_size),
        "1"};
  } else if (request.family == PointwiseFamily::kConvBiasRelu) {
    tokens = {
        pointer_token(request.convolution_image),
        pointer_token(request.convolution_filter),
        pointer_token(request.convolution_bias),
        pointer_token(request.convolution_result)};
    auto constants = convolution_constants(
        request, "conv2d_spatial_nchw_kernel", block_size);
    if (constants.size() != 36) {
      artifact_error("mthreads ConvBiasRelu signature constants are invalid");
    }
    constants[17] = 1;
    constants[23] = 0;
    for (const std::int64_t value : constants) {
      tokens.push_back(std::to_string(value));
    }
  } else if (request.family == PointwiseFamily::kBinary) {
    tokens = {
        pointer_token(request.left),
        pointer_token(request.right),
        pointer_token(request.output),
        "i32"};
    if (!request.dense) {
      for (const std::int64_t value :
           binary_pointwise_constants(request)) {
        tokens.push_back(std::to_string(value));
      }
    }
    tokens.push_back(std::to_string(request.pointwise_mode));
    tokens.push_back(python_float_token(request.alpha));
    tokens.push_back(std::to_string(block_size));
  } else if (request.family == PointwiseFamily::kUnary) {
    tokens = {
        pointer_token(request.input),
        pointer_token(request.output),
        "i32"};
    if (request.operation == "identity") {
      if (uses_packed_identity(request)) {
        tokens.push_back(std::to_string(identity_pack_size(request)));
        tokens.push_back(
            std::to_string(identity_tiles_per_program(request)));
      } else if (request.dense) {
        tokens.push_back(
            std::to_string(identity_tiles_per_program(request)));
      } else {
        for (const std::int64_t value :
             unary_pointwise_constants(request)) {
          tokens.push_back(std::to_string(value));
        }
      }
    } else {
      if (!request.dense) {
        for (const std::int64_t value :
             unary_pointwise_constants(request)) {
          tokens.push_back(std::to_string(value));
        }
        tokens.emplace_back("1");
      }
      tokens.push_back(std::to_string(request.pointwise_mode));
      tokens.push_back(python_float_token(request.negative_slope));
      tokens.push_back(python_float_token(request.lower_clip));
      tokens.push_back(python_float_token(request.upper_clip));
      tokens.push_back(std::to_string(request.has_upper_clip));
      tokens.push_back(python_float_token(request.swish_beta));
      tokens.push_back(python_float_token(request.elu_alpha));
      tokens.push_back(python_float_token(request.softplus_beta));
      tokens.push_back(std::to_string(unary_tiles_per_program(request)));
    }
    tokens.push_back(std::to_string(block_size));
  } else if (request.family == PointwiseFamily::kTernary) {
    tokens = {
        pointer_token(request.left),
        pointer_token(request.right),
        pointer_token(request.predicate),
        pointer_token(request.output),
        "i32"};
    if (!request.dense) {
      for (const std::int64_t value :
           ternary_pointwise_constants(request)) {
        tokens.push_back(std::to_string(value));
      }
    }
    tokens.push_back(std::to_string(block_size));
  } else if (request.family == PointwiseFamily::kLayout) {
    tokens = {
        pointer_token(request.input),
        pointer_token(request.output),
        "i32"};
    if (uses_i32_slice(request)) {
      tokens.push_back(std::to_string(request.input_base));
      tokens.push_back(
          std::to_string(request.logical_input_dimensions.size()));
      const std::size_t leading =
          kMaximumTensorRank - request.logical_input_dimensions.size();
      tokens.insert(tokens.end(), leading, "1");
      for (const std::int64_t value :
           request.logical_input_dimensions) {
        tokens.push_back(std::to_string(value));
      }
      tokens.insert(tokens.end(), leading, "0");
      for (const std::int64_t value : request.logical_input_strides) {
        tokens.push_back(std::to_string(value));
      }
      tokens.insert(tokens.end(), leading, "0");
      for (const std::int64_t value : request.output.strides) {
        tokens.push_back(std::to_string(value));
      }
    } else if (!uses_contiguous_reshape(request) &&
               !uses_physical_transpose(request)) {
      for (const std::int64_t value : layout_constants(request)) {
        tokens.push_back(std::to_string(value));
      }
    }
    tokens.push_back(std::to_string(block_size));
  } else if (request.family == PointwiseFamily::kMatmul) {
    if (const auto tle = matmul_tle_config(request); tle.has_value()) {
      const std::string dtype =
          request.left.data_type == "float16" ? "fp16" : "bf16";
      tokens = {
          "tensordesc<" + dtype + "[" +
              std::to_string(tle->block_m) + "," +
              std::to_string(tle->block_k) + "]>",
          "tensordesc<" + dtype + "[" +
              std::to_string(tle->block_k) + "," +
              std::to_string(tle->block_n) + "]>",
          pointer_token(request.output),
          std::to_string(request.matmul_m),
          std::to_string(request.matmul_n),
          std::to_string(request.matmul_k),
          std::to_string(request.matmul_batch),
          std::to_string(tle->block_m),
          std::to_string(tle->block_n),
          std::to_string(tle->block_k),
          std::to_string(tle->pipeline_stages),
          request.left.data_type == "bfloat16" ? "1" : "0",
          std::to_string(tle->panel_width)};
    } else if (uses_matmul_descriptor(request)) {
      const std::string dtype =
          request.left.data_type == "float16" ? "fp16" : "bf16";
      const unsigned int block_n = matmul_block_n(request, block_size);
      tokens = {
          "tensordesc<" + dtype + "[" +
              std::to_string(block_size) + "," +
              "64]>",
          "tensordesc<" + dtype + "[" +
              "64," +
              std::to_string(block_n) + "]>",
          "tensordesc<" + dtype + "[" +
              std::to_string(block_size) + "," +
              std::to_string(block_n) + "]>",
          std::to_string(request.matmul_m),
          std::to_string(request.matmul_n),
          std::to_string(request.matmul_k),
          std::to_string(request.matmul_batch),
          std::to_string(block_size),
          std::to_string(block_n),
          "64",
          "8"};
    } else {
      tokens = {
          pointer_token(request.left),
          pointer_token(request.right),
          pointer_token(request.output)};
      for (const std::int64_t value : matmul_constants(request)) {
        tokens.push_back(std::to_string(value));
      }
      tokens.push_back(std::to_string(block_size));
      tokens.push_back(
          std::to_string(matmul_block_n(request, block_size)));
      tokens.push_back(
          request.left.data_type == "float32" ? "32" : "64");
      tokens.emplace_back("8");
    }
  } else if (request.family == PointwiseFamily::kConvolution) {
    const bool fprop =
        request.operation == "conv2d_fprop" ||
        request.operation == "convolution_fprop";
    std::string function;
    if (fprop) {
      tokens = {
          pointer_token(request.convolution_image),
          pointer_token(request.convolution_filter),
          pointer_token(request.convolution_image),
          pointer_token(request.convolution_result)};
      function =
          request.convolution_spatial_rank == 1
              ? "conv1d_gemm_kernel"
              : (request.convolution_spatial_rank == 2
                     ? "conv2d_spatial_nchw_kernel"
                     : "conv3d_spatial_ncdhw_m_kernel");
    } else if (request.operation == "convolution_dgrad") {
      tokens = {
          pointer_token(request.convolution_result),
          pointer_token(request.convolution_filter),
          pointer_token(request.convolution_image)};
      function = "conv_dgrad_nd_kernel";
    } else {
      tokens = {
          pointer_token(request.convolution_result),
          pointer_token(request.convolution_image),
          pointer_token(request.convolution_filter)};
      function = "conv_wgrad_nd_kernel";
    }
    for (const std::int64_t value :
         convolution_constants(request, function, block_size)) {
      tokens.push_back(std::to_string(value));
    }
  } else {
    tokens = {
        pointer_token(request.input),
        pointer_token(request.output),
        "i32"};
    const unsigned int block_n = reduction_block_n(request);
    if (request.dense && request.inner == 1) {
      for (const std::int64_t value :
           {request.extent,
            request.extent,
            std::int64_t{1},
            request.reduction_mode + 1,
            static_cast<std::int64_t>(block_size),
            static_cast<std::int64_t>(block_n)}) {
        tokens.push_back(std::to_string(value));
      }
    } else if (request.dense) {
      for (const std::int64_t value :
           {request.extent,
            request.inner,
            request.extent * request.inner,
            request.inner,
            std::int64_t{1},
            request.reduction_mode + 1,
            static_cast<std::int64_t>(block_size),
            static_cast<std::int64_t>(block_n)}) {
        tokens.push_back(std::to_string(value));
      }
    } else {
      tokens.push_back(std::to_string(request.extent));
      for (const std::int64_t value :
           reduction_strided_constants(request)) {
        tokens.push_back(std::to_string(value));
      }
      tokens.push_back(std::to_string(request.reduction_mode + 1));
      tokens.push_back(std::to_string(block_size));
      tokens.push_back(std::to_string(block_n));
    }
  }

  std::string result;
  for (const std::string& token : tokens) {
    if (!result.empty()) {
      result += ',';
    }
    result += token;
  }
  return result;
}

std::string i32_bits(std::int32_t value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.reserve(8);
  for (unsigned int byte_index = 0; byte_index < 4; ++byte_index) {
    const unsigned int byte = (bits >> (byte_index * 8)) & 0xffU;
    result.push_back(digits[byte >> 4]);
    result.push_back(digits[byte & 0x0fU]);
  }
  return result;
}

std::string f32_bits(float value) {
  return i32_bits(std::bit_cast<std::int32_t>(value));
}

ArgumentSpec parse_expected_argument(
    const JsonValue& value,
    ArgumentKind expected_kind,
    std::string_view expected_name,
    std::int64_t expected_uid,
    std::string_view expected_bits) {
  require_exact_keys(
      value, {"kind", "semantic_name", "uid", "scalar_bits"},
      "kernel argument");
  ArgumentSpec result;
  result.semantic_name = value.at("semantic_name").as_string();
  if (result.semantic_name != expected_name) {
    artifact_error("mthreads kernel argument semantic name differs");
  }
  const std::string& kind = value.at("kind").as_string();
  if (expected_kind == ArgumentKind::kTensor) {
    if (kind != "tensor" || value.at("uid").is_null() ||
        value.at("uid").as_int() != expected_uid ||
        !value.at("scalar_bits").is_null()) {
      artifact_error("mthreads tensor argument differs from Graph IR");
    }
    result.kind = ArgumentKind::kTensor;
    result.uid = expected_uid;
    return result;
  }
  if (expected_kind == ArgumentKind::kWorkspace) {
    if (kind != "workspace" || !value.at("uid").is_null() ||
        !value.at("scalar_bits").is_null()) {
      artifact_error("mthreads workspace argument differs from Graph IR");
    }
    result.kind = ArgumentKind::kWorkspace;
    result.alignment = kPhaseOneWorkspaceAlignment;
    return result;
  }
  if (expected_kind == ArgumentKind::kScalarI32) {
    if (kind != "scalar_i32" || !value.at("uid").is_null() ||
        value.at("scalar_bits").is_null() ||
        value.at("scalar_bits").as_string() != expected_bits) {
      artifact_error("mthreads int32 scalar argument differs from Graph IR");
    }
    result.kind = ArgumentKind::kScalarI32;
    result.scalar_bits =
        std::bit_cast<std::uint32_t>(
            static_cast<std::int32_t>(expected_uid));
    return result;
  }
  if (expected_kind == ArgumentKind::kScalarF32) {
    if (kind != "scalar_f32" || !value.at("uid").is_null() ||
        value.at("scalar_bits").is_null() ||
        value.at("scalar_bits").as_string() != expected_bits ||
        expected_uid < 0 ||
        static_cast<std::uint64_t>(expected_uid) >
            std::numeric_limits<std::uint32_t>::max()) {
      artifact_error("mthreads float32 scalar argument differs from Graph IR");
    }
    result.kind = ArgumentKind::kScalarF32;
    result.scalar_bits = static_cast<std::uint32_t>(expected_uid);
    return result;
  }
  artifact_error("mthreads artifact argument kind is unsupported");
}

std::array<unsigned int, 3> parse_grid(const JsonValue& value) {
  const auto& values = value.as_array();
  if (values.size() != 3) {
    artifact_error("mthreads launch grid must contain three dimensions");
  }
  return {
      checked_positive_unsigned(values[0].as_int(), "grid.x"),
      checked_positive_unsigned(values[1].as_int(), "grid.y"),
      checked_positive_unsigned(values[2].as_int(), "grid.z")};
}

struct Candidate {
  unsigned int block_size;
  unsigned int num_warps;
  unsigned int num_stages;
};

std::array<unsigned int, 3> expected_convolution_grid(
    const PointwiseRequest& request, unsigned int block_size) {
  const std::size_t spatial_rank =
      static_cast<std::size_t>(request.convolution_spatial_rank);
  const auto image = padded_convolution_tensor(
      request.convolution_image, spatial_rank);
  const auto filter = padded_convolution_tensor(
      request.convolution_filter, spatial_rank);
  const auto loss = padded_convolution_tensor(
      request.convolution_result, spatial_rank);
  const auto ceil = [block_size](std::int64_t value) {
    return static_cast<std::uint64_t>(
        (value + block_size - 1) / block_size);
  };
  std::array<std::uint64_t, 3> grid{};
  if (request.operation == "convolution_dgrad") {
    std::int64_t rows = 0;
    if (uses_stride2_packed2_1d_dgrad(request)) {
      rows = request.convolution_batch *
          ((image.dimensions[4] + 1) / 2);
    } else if (uses_stride2_tile4_dgrad(request)) {
      rows = request.convolution_batch * loss.dimensions[2] *
          loss.dimensions[3] * loss.dimensions[4];
    } else {
      rows = request.convolution_batch * image.dimensions[2] *
          image.dimensions[3] * image.dimensions[4];
    }
    const unsigned int channel_block =
        uses_stride2_packed4_dgrad(request) ? block_size / 4 : block_size;
    const unsigned int block_m =
        stride2_packed4_dgrad_block_m(request, block_size);
    const auto ceil_channels = [channel_block](std::int64_t value) {
      return static_cast<std::uint64_t>(
          (value + channel_block - 1) / channel_block);
    };
    grid = {
        static_cast<std::uint64_t>((rows + block_m - 1) / block_m) *
            ceil_channels(request.convolution_in_per_group),
        static_cast<std::uint64_t>(request.convolution_groups),
        1};
  } else if (request.operation == "convolution_wgrad") {
    grid = {
        ceil(request.convolution_out_per_group) *
            ceil(request.convolution_in_per_group),
        static_cast<std::uint64_t>(
            filter.dimensions[2] * filter.dimensions[3] *
            filter.dimensions[4]),
        static_cast<std::uint64_t>(request.convolution_groups)};
  } else if (request.convolution_spatial_rank == 1) {
    grid = {
        ceil(request.convolution_batch * loss.dimensions[4]) *
            ceil(request.convolution_out_per_group),
        static_cast<std::uint64_t>(request.convolution_groups),
        1};
  } else if (request.convolution_spatial_rank == 2) {
    grid = {
        ceil(loss.dimensions[3] * loss.dimensions[4]) *
            ceil(request.convolution_out_per_group),
        static_cast<std::uint64_t>(
            request.convolution_batch * request.convolution_groups),
        1};
  } else {
    grid = {
        ceil(request.convolution_batch * loss.dimensions[2] *
             loss.dimensions[3] * loss.dimensions[4]) *
            ceil(request.convolution_out_per_group),
        static_cast<std::uint64_t>(request.convolution_groups),
        1};
  }
  if (std::any_of(
          grid.begin(), grid.end(), [](std::uint64_t value) {
            return value == 0 ||
                   value > std::numeric_limits<unsigned int>::max();
          })) {
    artifact_error("mthreads convolution grid exceeds uint32");
  }
  return {
      static_cast<unsigned int>(grid[0]),
      static_cast<unsigned int>(grid[1]),
      static_cast<unsigned int>(grid[2])};
}

std::array<unsigned int, 3> expected_normalization_grid(
    const PointwiseRequest& request, unsigned int block_size) {
  if (request.family == PointwiseFamily::kNormalization) {
    return {
        checked_positive_unsigned(
            request.normalization_rows, "normalization rows"),
        1,
        1};
  }
  if (request.family == PointwiseFamily::kBatchnorm) {
    return {
        checked_positive_unsigned(
            request.normalization_channels, "BatchNorm channels"),
        1,
        1};
  }
  if (request.family != PointwiseFamily::kBatchnormInference) {
    artifact_error("mthreads normalization grid family is invalid");
  }
  if (!uses_batchnorm_inference_nchw(request)) {
    return {
        static_cast<unsigned int>(
            (request.n_elements + block_size - 1) / block_size),
        1,
        1};
  }
  const std::uint64_t batch_channels =
      static_cast<std::uint64_t>(request.normalization_batch) *
      static_cast<std::uint64_t>(request.normalization_channels);
  const std::uint64_t spatial_blocks =
      (static_cast<std::uint64_t>(request.normalization_spatial) +
       block_size - 1) /
      block_size;
  if (batch_channels == 0 || spatial_blocks == 0 ||
      batch_channels > std::numeric_limits<unsigned int>::max() ||
      spatial_blocks > std::numeric_limits<unsigned int>::max()) {
    artifact_error("mthreads BatchNorm inference grid exceeds uint32");
  }
  return {static_cast<unsigned int>(batch_channels),
          static_cast<unsigned int>(spatial_blocks),
          1};
}

std::vector<Candidate> expected_candidates(const PointwiseRequest& request) {
  if (request.family == PointwiseFamily::kNormalization) {
    Candidate steady_state{256, 4, 1};
    if (request.normalization_elements > 513) {
      unsigned int block = 1;
      while (block < request.normalization_elements && block < 4096U) {
        block <<= 1U;
      }
      steady_state = {block, block <= 1024U ? 4U : 8U, 1};
    }
    if (!request.autotune) {
      return {steady_state};
    }
    std::vector<Candidate> result{steady_state};
    for (const unsigned int block : {256U, 512U}) {
      for (const unsigned int num_warps : {4U, 8U}) {
        if (block != steady_state.block_size ||
            num_warps != steady_state.num_warps) {
          result.push_back({block, num_warps, 1});
        }
      }
    }
    return result;
  }
  if (request.family == PointwiseFamily::kBatchnorm ||
      request.family == PointwiseFamily::kBatchnormInference) {
    if (!request.autotune) {
      return {{256, 4, 1}};
    }
    unsigned int batch_block = 1;
    while (batch_block <
           static_cast<unsigned int>(request.normalization_batch)) {
      batch_block <<= 1U;
    }
    std::vector<Candidate> result;
    if (request.family == PointwiseFamily::kBatchnorm && request.dense) {
      const std::uint64_t items_per_channel =
          static_cast<std::uint64_t>(request.normalization_batch) *
          static_cast<std::uint64_t>(request.normalization_spatial);
      if (items_per_channel > 512U) {
        unsigned int shape_block = 1U;
        while (shape_block < items_per_channel && shape_block < 16384U) {
          shape_block <<= 1U;
        }
        result.push_back(
            {shape_block, shape_block <= 1024U ? 4U : 8U, 1U});
      }
    }
    for (const unsigned int block : {128U, 256U, 512U}) {
      if (request.family == PointwiseFamily::kBatchnorm &&
          request.dense && block < batch_block) {
        continue;
      }
      for (const unsigned int num_warps : {4U, 8U}) {
        result.push_back({block, num_warps, 1});
      }
    }
    return result;
  }
  if (request.family == PointwiseFamily::kConvolution ||
      request.family == PointwiseFamily::kConvBiasRelu) {
    if (!request.autotune) {
      return {{32, 4, 2}};
    }
    std::vector<Candidate> result;
    for (const unsigned int block : {16U, 32U}) {
      for (const unsigned int num_warps : {4U, 8U}) {
        for (const unsigned int num_stages : {1U, 2U}) {
          result.push_back({block, num_warps, num_stages});
        }
      }
    }
    return result;
  }
  if (request.family == PointwiseFamily::kMatmul) {
    if (const auto tle = matmul_tle_config(request); tle.has_value()) {
      return {{tle->block_m, 16U, tle->pipeline_stages}};
    }
    if (uses_matmul_descriptor(request)) {
      // Pipeline stages greater than one currently fail in the MTGPU LLVM
      // greedy register allocator for descriptor SQMMA kernels.
      return {{128, 4, 1}};
    }
    if (!request.autotune) {
      return request.left.data_type != "float32" &&
                     request.matmul_m >= 64 &&
                     request.matmul_n >= 128 &&
                     request.matmul_k >= 64
                 ? std::vector<Candidate>{{64, 8, 2}}
                 : std::vector<Candidate>{{64, 4, 2}};
    }
    std::vector<Candidate> result;
    for (const unsigned int block : {32U, 64U}) {
      for (const unsigned int num_warps : {4U, 8U}) {
        for (const unsigned int num_stages : {1U, 2U}) {
          result.push_back({block, num_warps, num_stages});
        }
      }
    }
    return result;
  }
  if (request.family == PointwiseFamily::kReduction) {
    if (!request.autotune) {
      return reduction_block_n(request) * 16U <= 65536U
                 ? std::vector<Candidate>{{16, 2, 1}}
                 : std::vector<Candidate>{{1, 4, 1}};
    }
    const unsigned int block_n = reduction_block_n(request);
    std::vector<Candidate> result;
    for (const unsigned int block_m : {1U, 4U, 16U}) {
      for (const unsigned int num_warps : {2U, 4U}) {
        if (block_m * block_n <= 65536U) {
          result.push_back({block_m, num_warps, 1});
        }
      }
    }
    return result;
  }
  if (!request.autotune) {
    return {{256, 4, 1}};
  }
  const std::vector<unsigned int> blocks =
      request.dense ? std::vector<unsigned int>{128, 256, 512}
                    : std::vector<unsigned int>{128, 256};
  const std::vector<unsigned int> warps = {2, 4, 8};
  std::vector<Candidate> result;
  for (const unsigned int block : blocks) {
    for (const unsigned int warp : warps) {
      if (!request.dense && warp == 8) {
        continue;
      }
      result.push_back({block, warp, 1});
    }
  }
  return result;
}

PointwiseRequest parse_normalization_request(
    const JsonValue& request,
    std::string_view expected_target) {
  require_exact_keys(
      request,
      {"schema_version", "flagdnn_version", "backend", "target",
       "build_options", "graph", "compiler_identity"},
      "request");
  PointwiseRequest result;
  result.flagdnn_version = request.at("flagdnn_version").as_string();
  result.target = request.at("target").as_string();
  result.compiler_identity =
      request.at("compiler_identity").as_string();
  validate_target(result.target);
  if (request.at("schema_version").as_int() != 3 ||
      result.flagdnn_version != FLAGDNN_VERSION_STRING ||
      request.at("backend").as_string() != "mthreads" ||
      result.target != expected_target ||
      !is_lower_sha256(result.compiler_identity)) {
    artifact_error("mthreads normalization Graph IR identity is invalid");
  }

  const JsonValue& build_options = request.at("build_options");
  require_exact_keys(
      build_options, {"heuristic_modes", "autotune"}, "build_options");
  const auto& heuristic_modes =
      build_options.at("heuristic_modes").as_array();
  if (heuristic_modes.empty()) {
    artifact_error("mthreads heuristic modes are empty");
  }
  std::set<std::string> unique_modes;
  for (const JsonValue& mode_value : heuristic_modes) {
    const std::string& mode = mode_value.as_string();
    if ((mode != "A" && mode != "FALLBACK") ||
        !unique_modes.insert(mode).second) {
      artifact_error("mthreads heuristic modes are invalid");
    }
  }
  result.autotune = build_options.at("autotune").as_bool();

  const JsonValue& graph = request.at("graph");
  require_exact_keys(
      graph, {"name", "tensor_count", "tensors", "node_count", "nodes"},
      "graph");
  const std::string& graph_name = graph.at("name").as_string();
  if (graph_name.size() > (1U << 20) ||
      graph_name.find('\0') != std::string::npos) {
    artifact_error("mthreads normalization Graph name is invalid");
  }
  const auto& tensor_values = graph.at("tensors").as_array();
  if (tensor_values.empty() || tensor_values.size() > 10 ||
      graph.at("tensor_count").as_int() !=
          static_cast<std::int64_t>(tensor_values.size())) {
    artifact_error("mthreads normalization tensor count is invalid");
  }
  std::map<std::int64_t, TensorSpec> tensors;
  for (const JsonValue& tensor_value : tensor_values) {
    TensorSpec tensor = parse_tensor(tensor_value);
    if (tensor.is_virtual) {
      artifact_error(
          "mthreads normalization tensors must be externally bound");
    }
    result.external_binding_uids.push_back(tensor.uid);
    if (!tensors.emplace(tensor.uid, std::move(tensor)).second) {
      artifact_error("mthreads normalization tensor UID is duplicated");
    }
  }

  const auto& nodes = graph.at("nodes").as_array();
  if (graph.at("node_count").as_int() != 1 || nodes.size() != 1) {
    artifact_error("mthreads normalization requires exactly one node");
  }
  const JsonValue& node = nodes.front();
  require_exact_keys(
      node,
      {"id", "type", "name", "compute_data_type", "inputs", "outputs",
       "attributes"},
      "normalization node");
  result.node_id = node.at("id").as_int();
  result.operation = node.at("type").as_string();
  const std::string& node_name = node.at("name").as_string();
  if (result.node_id < 0 || node_name.size() > 4096 ||
      node_name.find('\0') != std::string::npos ||
      node.at("compute_data_type").as_string() != "float32") {
    artifact_error("mthreads normalization node identity/type is invalid");
  }
  const auto inputs =
      parse_ports(node.at("inputs"), tensors, "normalization input");
  const auto outputs =
      parse_ports(node.at("outputs"), tensors, "normalization output");

  const auto require_all_referenced =
      [&](const std::set<std::int64_t>& referenced,
          std::size_t expected_count) {
        if (tensor_values.size() != expected_count ||
            referenced.size() != expected_count ||
            referenced.size() != tensors.size()) {
          artifact_error(
              "mthreads normalization contains duplicate/unreferenced tensors");
        }
      };
  const auto parameter_valid =
      [](const TensorSpec& tensor,
         std::int64_t channels,
         std::string_view data_type) {
        return tensor.data_type == data_type &&
               element_count(tensor) == channels &&
               is_row_major_contiguous(tensor);
      };
  const auto parse_batch_attributes =
      [&](const JsonValue& attributes,
          bool training) {
        if (training) {
          require_exact_keys(
              attributes,
              {"batch", "channels", "dimensions", "epsilon", "momentum",
               "n_elements", "rank", "spatial", "x_strides", "y_strides"},
              "BatchNorm attributes");
        } else {
          require_exact_keys(
              attributes,
              {"channels", "dimensions", "n_elements", "rank", "spatial",
               "x_strides", "y_strides"},
              "BatchNorm inference attributes");
        }
        const std::size_t rank = result.normalization_x.dimensions.size();
        if (rank < 2 || rank > kMaximumTensorRank ||
            result.normalization_y.data_type !=
                result.normalization_x.data_type ||
            !is_floating_data_type(result.normalization_x.data_type) ||
            result.normalization_y.dimensions !=
                result.normalization_x.dimensions) {
          artifact_error("mthreads BatchNorm X/Y metadata is invalid");
        }
        result.normalization_rank = static_cast<std::int64_t>(rank);
        result.normalization_batch =
            result.normalization_x.dimensions[0];
        result.normalization_channels =
            result.normalization_x.dimensions[1];
        result.normalization_spatial = 1;
        for (std::size_t axis = 2; axis < rank; ++axis) {
          const std::int64_t dimension =
              result.normalization_x.dimensions[axis];
          if (result.normalization_spatial >
              std::numeric_limits<std::int32_t>::max() / dimension) {
            artifact_error("mthreads BatchNorm spatial extent exceeds int32");
          }
          result.normalization_spatial *= dimension;
        }
        result.n_elements = element_count(result.normalization_x);
        const auto dimensions = positive_i32_array(
            attributes.at("dimensions"), "BatchNorm dimensions");
        const auto x_strides = positive_i32_array(
            attributes.at("x_strides"), "BatchNorm X strides");
        const auto y_strides = positive_i32_array(
            attributes.at("y_strides"), "BatchNorm Y strides");
        if (dimensions != result.normalization_x.dimensions ||
            x_strides != result.normalization_x.strides ||
            y_strides != result.normalization_y.strides ||
            attributes.at("rank").as_int() !=
                result.normalization_rank ||
            attributes.at("channels").as_int() !=
                result.normalization_channels ||
            attributes.at("spatial").as_int() !=
                result.normalization_spatial ||
            attributes.at("n_elements").as_int() != result.n_elements ||
            (training &&
             attributes.at("batch").as_int() !=
                 result.normalization_batch)) {
          artifact_error(
              "mthreads BatchNorm attributes differ from tensor metadata");
        }
      };

  if (result.operation == "layernorm" ||
      result.operation == "rmsnorm") {
    const bool layernorm = result.operation == "layernorm";
    const std::set<std::string> expected_inputs = {"x", "scale", "bias"};
    const std::set<std::string> expected_outputs =
        layernorm
            ? std::set<std::string>{"y", "mean", "inv_variance"}
            : std::set<std::string>{"y", "inv_variance"};
    if (inputs.size() != expected_inputs.size() ||
        outputs.size() != expected_outputs.size() ||
        !std::all_of(
            expected_inputs.begin(), expected_inputs.end(),
            [&](const std::string& role) {
              return inputs.find(role) != inputs.end();
            }) ||
        !std::all_of(
            expected_outputs.begin(), expected_outputs.end(),
            [&](const std::string& role) {
              return outputs.find(role) != outputs.end();
            })) {
      artifact_error("mthreads normalization port roles are invalid");
    }
    result.family = PointwiseFamily::kNormalization;
    result.normalization_x = tensors.at(inputs.at("x"));
    result.normalization_scale = tensors.at(inputs.at("scale"));
    result.normalization_bias = tensors.at(inputs.at("bias"));
    result.normalization_y = tensors.at(outputs.at("y"));
    result.normalization_inv_variance =
        tensors.at(outputs.at("inv_variance"));
    std::set<std::int64_t> referenced = {
        inputs.at("x"), inputs.at("scale"), inputs.at("bias"),
        outputs.at("y"), outputs.at("inv_variance")};
    if (layernorm) {
      result.normalization_mean = tensors.at(outputs.at("mean"));
      referenced.insert(outputs.at("mean"));
    }
    require_all_referenced(referenced, layernorm ? 6U : 5U);
    const auto& x = result.normalization_x;
    const auto& scale = result.normalization_scale;
    const auto& bias = result.normalization_bias;
    const auto& y = result.normalization_y;
    const auto& inv_variance = result.normalization_inv_variance;
    if (!is_floating_data_type(x.data_type) ||
        y.data_type != x.data_type || scale.data_type != x.data_type ||
        bias.data_type != x.data_type || x.dimensions != y.dimensions ||
        x.strides != y.strides || x.dimensions.empty() ||
        !is_row_major_contiguous(x) || !is_row_major_contiguous(y) ||
        !is_row_major_contiguous(scale) ||
        !is_row_major_contiguous(bias) ||
        scale.dimensions != bias.dimensions || scale.dimensions.empty() ||
        scale.dimensions.size() > x.dimensions.size() ||
        inv_variance.data_type != "float32" ||
        !is_row_major_contiguous(inv_variance) ||
        (layernorm &&
         (result.normalization_mean.data_type != "float32" ||
          !is_row_major_contiguous(result.normalization_mean)))) {
      artifact_error("mthreads normalization tensor metadata is invalid");
    }
    const std::size_t leading =
        x.dimensions.size() - scale.dimensions.size();
    std::size_t normalized_start = x.dimensions.size();
    for (std::size_t axis = 0; axis < x.dimensions.size(); ++axis) {
      const std::int64_t scale_dimension =
          axis < leading ? 1 : scale.dimensions[axis - leading];
      if (scale_dimension != 1) {
        if (scale_dimension != x.dimensions[axis]) {
          artifact_error(
              "mthreads normalization scale does not match input suffix");
        }
        if (normalized_start == x.dimensions.size()) {
          normalized_start = axis;
        }
      } else if (normalized_start != x.dimensions.size() &&
                 x.dimensions[axis] != 1) {
        artifact_error(
            "mthreads normalization scale does not describe a suffix");
      }
    }
    if (normalized_start == x.dimensions.size()) {
      normalized_start = x.dimensions.size() - 1;
    }
    result.normalization_rows = 1;
    result.normalization_elements = 1;
    for (std::size_t axis = 0; axis < x.dimensions.size(); ++axis) {
      std::int64_t& product =
          axis < normalized_start ? result.normalization_rows
                                  : result.normalization_elements;
      if (product > std::numeric_limits<std::int32_t>::max() /
                        x.dimensions[axis]) {
        artifact_error("mthreads normalization extent exceeds int32");
      }
      product *= x.dimensions[axis];
    }
    std::vector<std::int64_t> expected_statistics = x.dimensions;
    std::fill(
        expected_statistics.begin() +
            static_cast<std::ptrdiff_t>(normalized_start),
        expected_statistics.end(), 1);
    if (element_count(scale) != result.normalization_elements ||
        inv_variance.dimensions != expected_statistics ||
        element_count(inv_variance) != result.normalization_rows ||
        (layernorm &&
         (result.normalization_mean.dimensions != expected_statistics ||
          element_count(result.normalization_mean) !=
              result.normalization_rows))) {
      artifact_error("mthreads normalization row geometry is invalid");
    }
    const JsonValue& attributes = node.at("attributes");
    require_exact_keys(
        attributes,
        {"epsilon", "forward_phase", "normalized_elements", "rows"},
        "normalization attributes");
    result.normalization_epsilon =
        checked_float32(attributes.at("epsilon"), "normalization epsilon");
    if (result.normalization_epsilon <= 0.0F ||
        attributes.at("forward_phase").as_int() != 2 ||
        attributes.at("normalized_elements").as_int() !=
            result.normalization_elements ||
        attributes.at("rows").as_int() != result.normalization_rows) {
      artifact_error("mthreads normalization attributes are invalid");
    }
    result.n_elements =
        static_cast<std::int32_t>(result.normalization_rows);
    result.dense = true;
    return result;
  }

  if (result.operation == "batchnorm") {
    if (inputs.size() != 5 || outputs.size() != 5 ||
        inputs.find("x") == inputs.end() ||
        inputs.find("scale") == inputs.end() ||
        inputs.find("bias") == inputs.end() ||
        inputs.find("previous_running_mean") == inputs.end() ||
        inputs.find("previous_running_variance") == inputs.end() ||
        outputs.find("y") == outputs.end() ||
        outputs.find("mean") == outputs.end() ||
        outputs.find("inv_variance") == outputs.end() ||
        outputs.find("next_running_mean") == outputs.end() ||
        outputs.find("next_running_variance") == outputs.end()) {
      artifact_error("mthreads BatchNorm port roles are invalid");
    }
    result.family = PointwiseFamily::kBatchnorm;
    result.normalization_x = tensors.at(inputs.at("x"));
    result.normalization_scale = tensors.at(inputs.at("scale"));
    result.normalization_bias = tensors.at(inputs.at("bias"));
    result.normalization_previous_running_mean =
        tensors.at(inputs.at("previous_running_mean"));
    result.normalization_previous_running_variance =
        tensors.at(inputs.at("previous_running_variance"));
    result.normalization_y = tensors.at(outputs.at("y"));
    result.normalization_mean = tensors.at(outputs.at("mean"));
    result.normalization_inv_variance =
        tensors.at(outputs.at("inv_variance"));
    result.normalization_next_running_mean =
        tensors.at(outputs.at("next_running_mean"));
    result.normalization_next_running_variance =
        tensors.at(outputs.at("next_running_variance"));
    require_all_referenced(
        {inputs.at("x"),
         inputs.at("scale"),
         inputs.at("bias"),
         inputs.at("previous_running_mean"),
         inputs.at("previous_running_variance"),
         outputs.at("y"),
         outputs.at("mean"),
         outputs.at("inv_variance"),
         outputs.at("next_running_mean"),
         outputs.at("next_running_variance")},
        10U);
    parse_batch_attributes(node.at("attributes"), true);
    const std::int64_t channels = result.normalization_channels;
    if (!parameter_valid(
            result.normalization_scale,
            channels,
            result.normalization_x.data_type) ||
        !parameter_valid(
            result.normalization_bias,
            channels,
            result.normalization_x.data_type)) {
      artifact_error("mthreads BatchNorm scale/bias metadata is invalid");
    }
    for (const TensorSpec* statistic : {
             &result.normalization_previous_running_mean,
             &result.normalization_previous_running_variance,
             &result.normalization_mean,
             &result.normalization_inv_variance,
             &result.normalization_next_running_mean,
             &result.normalization_next_running_variance}) {
      if (!parameter_valid(*statistic, channels, "float32")) {
        artifact_error("mthreads BatchNorm statistic metadata is invalid");
      }
    }
    result.normalization_epsilon = checked_float32(
        node.at("attributes").at("epsilon"), "BatchNorm epsilon");
    result.normalization_momentum = checked_float32(
        node.at("attributes").at("momentum"), "BatchNorm momentum");
    if (result.normalization_epsilon <= 0.0F ||
        result.normalization_momentum < 0.0F ||
        result.normalization_momentum > 1.0F) {
      artifact_error("mthreads BatchNorm scalar attributes are invalid");
    }
    unsigned int batch_block = 1;
    while (batch_block <
           static_cast<unsigned int>(result.normalization_batch)) {
      batch_block <<= 1U;
    }
    result.dense = is_row_major_contiguous(result.normalization_x) &&
                   is_row_major_contiguous(result.normalization_y) &&
                   batch_block <= 256U;
    return result;
  }

  if (result.operation == "batchnorm_inference") {
    if (inputs.size() != 5 || outputs.size() != 1 ||
        inputs.find("x") == inputs.end() ||
        inputs.find("mean") == inputs.end() ||
        inputs.find("inv_variance") == inputs.end() ||
        inputs.find("scale") == inputs.end() ||
        inputs.find("bias") == inputs.end() ||
        outputs.find("y") == outputs.end()) {
      artifact_error("mthreads BatchNorm inference port roles are invalid");
    }
    result.family = PointwiseFamily::kBatchnormInference;
    result.normalization_x = tensors.at(inputs.at("x"));
    result.normalization_mean = tensors.at(inputs.at("mean"));
    result.normalization_inv_variance =
        tensors.at(inputs.at("inv_variance"));
    result.normalization_scale = tensors.at(inputs.at("scale"));
    result.normalization_bias = tensors.at(inputs.at("bias"));
    result.normalization_y = tensors.at(outputs.at("y"));
    require_all_referenced(
        {inputs.at("x"),
         inputs.at("mean"),
         inputs.at("inv_variance"),
         inputs.at("scale"),
         inputs.at("bias"),
         outputs.at("y")},
        6U);
    parse_batch_attributes(node.at("attributes"), false);
    for (const TensorSpec* parameter : {
             &result.normalization_mean,
             &result.normalization_inv_variance,
             &result.normalization_scale,
             &result.normalization_bias}) {
      if (!parameter_valid(
              *parameter, result.normalization_channels, "float32")) {
        artifact_error(
            "mthreads BatchNorm inference parameter metadata is invalid");
      }
    }
    result.dense = is_row_major_contiguous(result.normalization_x) &&
                   is_row_major_contiguous(result.normalization_y);
    return result;
  }

  artifact_error("mthreads normalization operation is unsupported");
}

bool is_attention_operation(std::string_view operation) {
  return operation == "sdpa" || operation == "sdpa_backward" ||
         operation == "sdpa_fp8" ||
         operation == "sdpa_fp8_backward";
}

const TensorSpec& attention_tensor(
    const PointwiseRequest& request, std::string_view name) {
  const auto found = request.attention_tensors.find(std::string(name));
  if (found == request.attention_tensors.end()) {
    artifact_error(
        "mthreads Attention tensor role is missing: " +
        std::string(name));
  }
  return found->second;
}

PointwiseRequest parse_attention_request(
    const JsonValue& request,
    std::string_view expected_target) {
  require_exact_keys(
      request,
      {"schema_version", "flagdnn_version", "backend", "target",
       "build_options", "graph", "compiler_identity"},
      "request");
  PointwiseRequest result;
  result.family = PointwiseFamily::kAttention;
  result.flagdnn_version = request.at("flagdnn_version").as_string();
  result.target = request.at("target").as_string();
  result.compiler_identity =
      request.at("compiler_identity").as_string();
  validate_target(result.target);
  if (request.at("schema_version").as_int() != 3 ||
      result.flagdnn_version != FLAGDNN_VERSION_STRING ||
      request.at("backend").as_string() != "mthreads" ||
      result.target != expected_target ||
      !is_lower_sha256(result.compiler_identity)) {
    artifact_error("mthreads Attention Graph IR identity is invalid");
  }

  const JsonValue& build_options = request.at("build_options");
  require_exact_keys(
      build_options, {"heuristic_modes", "autotune"}, "build_options");
  const auto& heuristic_modes =
      build_options.at("heuristic_modes").as_array();
  if (heuristic_modes.empty()) {
    artifact_error("mthreads Attention heuristic modes are empty");
  }
  std::set<std::string> unique_modes;
  for (const JsonValue& mode_value : heuristic_modes) {
    const std::string& mode = mode_value.as_string();
    if ((mode != "A" && mode != "FALLBACK") ||
        !unique_modes.insert(mode).second) {
      artifact_error("mthreads Attention heuristic modes are invalid");
    }
  }
  result.autotune = build_options.at("autotune").as_bool();

  const JsonValue& graph = request.at("graph");
  require_exact_keys(
      graph, {"name", "tensor_count", "tensors", "node_count", "nodes"},
      "graph");
  const std::string& graph_name = graph.at("name").as_string();
  if (graph_name.size() > (1U << 20) ||
      graph_name.find('\0') != std::string::npos) {
    artifact_error("mthreads Attention Graph name is invalid");
  }
  const auto& tensor_values = graph.at("tensors").as_array();
  if (tensor_values.empty() || tensor_values.size() > 32 ||
      graph.at("tensor_count").as_int() !=
          static_cast<std::int64_t>(tensor_values.size())) {
    artifact_error("mthreads Attention tensor count is invalid");
  }
  std::map<std::int64_t, TensorSpec> tensors;
  for (const JsonValue& tensor_value : tensor_values) {
    TensorSpec tensor = parse_tensor(tensor_value);
    if (!tensor.is_virtual) {
      result.external_binding_uids.push_back(tensor.uid);
    }
    if (!tensors.emplace(tensor.uid, std::move(tensor)).second) {
      artifact_error("mthreads Attention tensor UID is duplicated");
    }
  }

  const auto& nodes = graph.at("nodes").as_array();
  if (graph.at("node_count").as_int() != 1 || nodes.size() != 1) {
    artifact_error("mthreads Attention requires exactly one node");
  }
  const JsonValue& node = nodes.front();
  require_exact_keys(
      node,
      {"id", "type", "name", "compute_data_type", "inputs", "outputs",
       "attributes"},
      "Attention node");
  result.node_id = node.at("id").as_int();
  result.operation = node.at("type").as_string();
  const std::string& node_name = node.at("name").as_string();
  if (result.node_id < 0 || node_name.size() > 4096 ||
      node_name.find('\0') != std::string::npos ||
      !is_attention_operation(result.operation) ||
      node.at("compute_data_type").as_string() != "float32") {
    artifact_error("mthreads Attention node identity/type is invalid");
  }

  const JsonValue& attributes = node.at("attributes");
  require_exact_keys(
      attributes,
      {"attn_scale", "attn_scale_set", "banded", "batch",
       "causal_top_left", "diagonal_alignment",
       "diagonal_band_left_bound", "diagonal_band_right_bound",
       "generate_stats", "has_bias", "has_dbias", "head_dimension",
       "heads", "key_heads", "left_bound_set", "max_diag", "min_diag",
       "q_per_k", "q_per_v", "reverse_causal", "right_bound_set",
       "sequence_kv", "sequence_q", "value_dimension", "value_heads"},
      "Attention attributes");
  const auto flag = [&](std::string_view name) {
    const std::int64_t value = attributes.at(name).as_int();
    if (value != 0 && value != 1) {
      artifact_error(
          "mthreads Attention flag is invalid: " + std::string(name));
    }
    return value == 1;
  };
  result.attention_has_bias = flag("has_bias");
  result.attention_has_dbias = flag("has_dbias");
  result.attention_generate_stats = flag("generate_stats");
  const bool fp8 = result.operation == "sdpa_fp8" ||
                   result.operation == "sdpa_fp8_backward";
  const bool backward = result.operation == "sdpa_backward" ||
                        result.operation == "sdpa_fp8_backward";
  if (fp8 &&
      (result.attention_has_bias || result.attention_has_dbias)) {
    artifact_error("mthreads FP8 Attention does not support bias");
  }

  std::set<std::string> expected_inputs;
  std::set<std::string> expected_outputs;
  if (result.operation == "sdpa") {
    expected_inputs = {"q", "k", "v"};
    if (result.attention_has_bias) {
      expected_inputs.insert("bias");
    }
    expected_outputs = {"o", "stats"};
  } else if (result.operation == "sdpa_backward") {
    expected_inputs = {"q", "k", "v", "o", "do", "stats"};
    if (result.attention_has_bias) {
      expected_inputs.insert("bias");
    }
    expected_outputs = {"dq", "dk", "dv"};
    if (result.attention_has_dbias) {
      expected_outputs.insert("dbias");
    }
  } else if (result.operation == "sdpa_fp8") {
    expected_inputs = {
        "q",         "k",         "v",        "descale_q",
        "descale_k", "descale_v", "descale_s", "scale_s",
        "scale_o"};
    expected_outputs = {"o", "stats", "amax_s", "amax_o"};
  } else {
    expected_inputs = {
        "q",          "k",          "v",          "o",
        "do",         "stats",      "descale_q",  "descale_k",
        "descale_v",  "descale_o",  "descale_do", "descale_s",
        "descale_dp", "scale_s",    "scale_dq",   "scale_dk",
        "scale_dv",   "scale_dp"};
    expected_outputs = {
        "dq", "dk", "dv", "amax_dq", "amax_dk", "amax_dv",
        "amax_dp"};
  }
  const auto inputs =
      parse_ports(node.at("inputs"), tensors, "Attention input");
  const auto outputs =
      parse_ports(node.at("outputs"), tensors, "Attention output");
  const auto exact_roles = [](
                               const std::map<std::string, std::int64_t>& ports,
                               const std::set<std::string>& expected) {
    if (ports.size() != expected.size()) {
      return false;
    }
    return std::all_of(
        expected.begin(), expected.end(), [&](const std::string& role) {
          return ports.find(role) != ports.end();
        });
  };
  if (!exact_roles(inputs, expected_inputs) ||
      !exact_roles(outputs, expected_outputs)) {
    artifact_error("mthreads Attention Graph port roles are invalid");
  }
  std::set<std::int64_t> referenced;
  for (const auto& [name, uid] : inputs) {
    result.attention_tensors.emplace(name, tensors.at(uid));
    referenced.insert(uid);
  }
  for (const auto& [name, uid] : outputs) {
    if (!result.attention_tensors.emplace(name, tensors.at(uid)).second) {
      artifact_error("mthreads Attention tensor role is duplicated");
    }
    referenced.insert(uid);
  }
  if (referenced.size() != tensors.size() ||
      referenced.size() != inputs.size() + outputs.size()) {
    artifact_error(
        "mthreads Attention contains duplicate/unreferenced tensors");
  }

  const TensorSpec& q = attention_tensor(result, "q");
  const TensorSpec& k = attention_tensor(result, "k");
  const TensorSpec& v = attention_tensor(result, "v");
  const bool valid_q_type =
      fp8 ? (q.data_type == "fp8_e4m3" || q.data_type == "fp8_e5m2")
          : is_floating_data_type(q.data_type);
  if (!valid_q_type || k.data_type != q.data_type ||
      v.data_type != q.data_type || q.is_virtual || k.is_virtual ||
      v.is_virtual || q.dimensions.size() != 4 ||
      k.dimensions.size() != 4 || v.dimensions.size() != 4) {
    artifact_error("mthreads Attention Q/K/V metadata is invalid");
  }
  if (q.dimensions[0] != k.dimensions[0] ||
      q.dimensions[0] != v.dimensions[0] ||
      q.dimensions[3] != k.dimensions[3] ||
      k.dimensions[2] != v.dimensions[2] ||
      q.dimensions[1] % k.dimensions[1] != 0 ||
      q.dimensions[1] % v.dimensions[1] != 0) {
    artifact_error("mthreads Attention Q/K/V shapes are inconsistent");
  }
  result.attention_batch = q.dimensions[0];
  result.attention_heads = q.dimensions[1];
  result.attention_sequence_q = q.dimensions[2];
  result.attention_head_dimension = q.dimensions[3];
  result.attention_key_heads = k.dimensions[1];
  result.attention_value_heads = v.dimensions[1];
  result.attention_sequence_kv = k.dimensions[2];
  result.attention_value_dimension = v.dimensions[3];
  result.attention_q_per_k =
      result.attention_heads / result.attention_key_heads;
  result.attention_q_per_v =
      result.attention_heads / result.attention_value_heads;
  if (result.attention_head_dimension > 256 ||
      result.attention_value_dimension > 256 ||
      (result.operation == "sdpa_backward" &&
       result.attention_key_heads != result.attention_value_heads) ||
      (result.operation == "sdpa_fp8_backward" &&
       (result.attention_key_heads != result.attention_value_heads ||
        result.attention_head_dimension != result.attention_value_dimension ||
        result.attention_head_dimension > 128))) {
    artifact_error("mthreads Attention head geometry is unsupported");
  }

  const std::vector<std::int64_t> output_dimensions = {
      result.attention_batch, result.attention_heads,
      result.attention_sequence_q, result.attention_value_dimension};
  const std::vector<std::int64_t> stats_dimensions = {
      result.attention_batch, result.attention_heads,
      result.attention_sequence_q, 1};
  const auto require_tensor = [&result](
                                  std::string_view name,
                                  const std::vector<std::int64_t>& dimensions,
                                  std::string_view data_type,
                                  bool allow_virtual = false) {
    const TensorSpec& tensor = attention_tensor(result, name);
    if (tensor.dimensions != dimensions || tensor.data_type != data_type ||
        (tensor.is_virtual && !allow_virtual)) {
      artifact_error(
          "mthreads Attention tensor metadata is invalid: " +
          std::string(name));
    }
  };
  require_tensor("o", output_dimensions, q.data_type);
  const TensorSpec& stats = attention_tensor(result, "stats");
  require_tensor(
      "stats", stats_dimensions, "float32",
      !result.attention_generate_stats);
  if (stats.is_virtual == result.attention_generate_stats ||
      (backward && !result.attention_generate_stats)) {
    artifact_error("mthreads Attention stats state is invalid");
  }
  if (backward) {
    require_tensor("do", output_dimensions, q.data_type);
    const std::array<
        std::pair<std::string_view, std::string_view>, 3>
        gradient_pairs = {{{"q", "dq"},
                           {"k", "dk"},
                           {"v", "dv"}}};
    for (const auto& [primal, gradient] : gradient_pairs) {
      require_tensor(
          gradient, attention_tensor(result, primal).dimensions, q.data_type);
    }
  }
  if (result.attention_has_bias) {
    const TensorSpec& bias = attention_tensor(result, "bias");
    if (bias.data_type != q.data_type || bias.is_virtual ||
        bias.dimensions.size() != 4 ||
        (bias.dimensions[0] != 1 &&
         bias.dimensions[0] != result.attention_batch) ||
        (bias.dimensions[1] != 1 &&
         bias.dimensions[1] != result.attention_heads) ||
        bias.dimensions[2] != result.attention_sequence_q ||
        bias.dimensions[3] != result.attention_sequence_kv) {
      artifact_error("mthreads Attention bias metadata is invalid");
    }
  }
  if (result.attention_has_dbias) {
    if (!result.attention_has_bias) {
      artifact_error("mthreads Attention dbias requires bias");
    }
    require_tensor(
        "dbias", attention_tensor(result, "bias").dimensions, q.data_type);
  }

  if (fp8) {
    const std::vector<std::string_view> scalar_names =
        result.operation == "sdpa_fp8"
            ? std::vector<std::string_view>{
                  "descale_q", "descale_k", "descale_v", "descale_s",
                  "scale_s", "scale_o", "amax_s", "amax_o"}
            : std::vector<std::string_view>{
                  "descale_q", "descale_k", "descale_v", "descale_o",
                  "descale_do", "descale_s", "descale_dp", "scale_s",
                  "scale_dq", "scale_dk", "scale_dv", "scale_dp",
                  "amax_dq", "amax_dk", "amax_dv", "amax_dp"};
    for (const std::string_view name : scalar_names) {
      const TensorSpec& scalar = attention_tensor(result, name);
      if (scalar.data_type != "float32" || scalar.is_virtual ||
          element_count(scalar) != 1) {
        artifact_error("mthreads FP8 Attention scalar metadata is invalid");
      }
    }
  }

  const std::array<std::pair<std::string_view, std::int64_t>, 10>
      shape_attributes = {{{"batch", result.attention_batch},
                           {"heads", result.attention_heads},
                           {"key_heads", result.attention_key_heads},
                           {"value_heads", result.attention_value_heads},
                           {"sequence_q", result.attention_sequence_q},
                           {"sequence_kv", result.attention_sequence_kv},
                           {"head_dimension",
                            result.attention_head_dimension},
                           {"value_dimension",
                            result.attention_value_dimension},
                           {"q_per_k", result.attention_q_per_k},
                           {"q_per_v", result.attention_q_per_v}}};
  for (const auto& [name, expected] : shape_attributes) {
    if (attributes.at(name).as_int() != expected) {
      artifact_error("mthreads Attention shape attributes are inconsistent");
    }
  }

  const bool scale_set = attributes.at("attn_scale_set").as_bool();
  result.attention_scale =
      checked_float32(attributes.at("attn_scale"), "Attention scale");
  if (result.attention_scale <= 0.0F ||
      (!scale_set &&
       result.attention_scale != static_cast<float>(
                                     1.0 / std::sqrt(static_cast<double>(
                                               result.attention_head_dimension))))) {
    artifact_error("mthreads Attention scale is invalid");
  }

  const bool left_set = attributes.at("left_bound_set").as_bool();
  const bool right_set = attributes.at("right_bound_set").as_bool();
  const std::int64_t left =
      attributes.at("diagonal_band_left_bound").as_int();
  const std::int64_t right =
      attributes.at("diagonal_band_right_bound").as_int();
  const std::int64_t alignment =
      attributes.at("diagonal_alignment").as_int();
  if (left < 0 || right < 0 || (left_set && left < 1) ||
      (alignment != 0 && alignment != 1)) {
    artifact_error("mthreads Attention diagonal bounds are invalid");
  }
  constexpr std::int64_t unbounded_diagonal = 1LL << 30;
  const std::int64_t shift =
      alignment == 1
          ? result.attention_sequence_kv - result.attention_sequence_q
          : 0;
  const std::int64_t expected_min =
      left_set ? 1 - left + shift : -unbounded_diagonal;
  const std::int64_t expected_max =
      right_set ? right + shift : unbounded_diagonal;
  result.attention_min_diag = attributes.at("min_diag").as_int();
  result.attention_max_diag = attributes.at("max_diag").as_int();
  result.attention_banded = flag("banded");
  result.attention_causal_top_left = flag("causal_top_left");
  result.attention_reverse_causal = flag("reverse_causal");
  const bool expected_causal =
      alignment == 0 && !left_set && right_set && right == 0 &&
      result.attention_sequence_q == result.attention_sequence_kv;
  const bool expected_reverse =
      alignment == 0 && !left_set && right_set && right == 0;
  if (result.attention_min_diag != expected_min ||
      result.attention_max_diag != expected_max ||
      result.attention_min_diag > result.attention_max_diag ||
      result.attention_banded != (left_set || right_set) ||
      result.attention_causal_top_left != expected_causal ||
      result.attention_reverse_causal != expected_reverse) {
    artifact_error("mthreads Attention diagonal metadata is inconsistent");
  }
  return result;
}

struct ExpectedAttentionArgument {
  ArgumentKind kind = ArgumentKind::kTensor;
  std::string semantic_name;
  const TensorSpec* tensor = nullptr;
  std::int64_t scalar_value = 0;
  std::string scalar_bits;
};

struct AttentionVariantExpectation {
  std::string signature;
  std::array<unsigned int, 3> grid = {1, 1, 1};
  std::vector<ExpectedAttentionArgument> arguments;
};

class AttentionSignatureBuilder final {
 public:
  void tensor(const TensorSpec& value, std::string_view semantic_name) {
    tokens_.push_back(pointer_token(value));
    arguments_.push_back(
        {ArgumentKind::kTensor, std::string(semantic_name), &value, 0, {}});
  }

  void workspace(
      std::string_view semantic_name,
      std::string_view data_type = "float32") {
    TensorSpec workspace;
    workspace.data_type = std::string(data_type);
    workspace.alignment = 16;
    tokens_.push_back(pointer_token(workspace));
    arguments_.push_back(
        {ArgumentKind::kWorkspace, std::string(semantic_name), nullptr, 0,
         {}});
  }

  void scalar_i32(std::string_view name, std::int64_t value) {
    if (value < std::numeric_limits<std::int32_t>::min() ||
        value > std::numeric_limits<std::int32_t>::max()) {
      artifact_error("mthreads Attention runtime int32 is out of range");
    }
    tokens_.push_back("i32");
    arguments_.push_back(
        {ArgumentKind::kScalarI32,
         std::string(name),
         nullptr,
         value,
         i32_bits(static_cast<std::int32_t>(value))});
  }

  void scalar_f32(std::string_view name, float value) {
    if (!std::isfinite(value)) {
      artifact_error("mthreads Attention runtime float32 is nonfinite");
    }
    tokens_.push_back("fp32");
    arguments_.push_back(
        {ArgumentKind::kScalarF32,
         std::string(name),
         nullptr,
         static_cast<std::int64_t>(std::bit_cast<std::uint32_t>(value)),
         f32_bits(value)});
  }

  void constant(std::int64_t value) {
    tokens_.push_back(std::to_string(value));
  }

  void constant(bool value) {
    tokens_.push_back(value ? "true" : "false");
  }

  [[nodiscard]] AttentionVariantExpectation finish(
      std::array<unsigned int, 3> grid) && {
    std::string signature;
    for (const std::string& token : tokens_) {
      if (!signature.empty()) {
        signature += ',';
      }
      signature += token;
    }
    return {std::move(signature), grid, std::move(arguments_)};
  }

 private:
  std::vector<std::string> tokens_;
  std::vector<ExpectedAttentionArgument> arguments_;
};

struct ExpectedAttentionStage {
  std::string function;
  std::vector<std::size_t> dependencies;
  bool fixed = false;
};

bool attention_dbias_reduces(const PointwiseRequest& request) {
  if (!request.attention_has_dbias) {
    return false;
  }
  const TensorSpec& dbias = attention_tensor(request, "dbias");
  return dbias.dimensions[0] != request.attention_batch ||
         dbias.dimensions[1] != request.attention_heads;
}

std::vector<ExpectedAttentionStage> expected_attention_stages(
    const PointwiseRequest& request) {
  if (request.operation == "sdpa") {
    return {{"_sdpa_fwd_kernel", {}, false}};
  }
  if (request.operation == "sdpa_fp8") {
    return {{"_zero_sdpa_fp8_fwd_amax_kernel", {}, true},
            {"_sdpa_fp8_fwd_kernel", {0}, false}};
  }
  if (request.operation == "sdpa_fp8_backward") {
    return {{"_zero_sdpa_fp8_bwd_amax_kernel", {}, true},
            {"_sdpa_fp8_bwd_dq_kernel", {0}, false},
            {"_sdpa_fp8_bwd_dkdv_kernel", {1}, false}};
  }

  std::vector<ExpectedAttentionStage> result;
  std::vector<std::size_t> dq_dependencies;
  if (attention_dbias_reduces(request)) {
    const TensorSpec& dbias = attention_tensor(request, "dbias");
    if (!is_row_major_contiguous(dbias)) {
      artifact_error(
          "mthreads broadcast Attention dbias must be contiguous");
    }
    result.push_back({"_zero_contiguous_kernel", {}, true});
    dq_dependencies.push_back(0);
  }
  const std::size_t dq_id = result.size();
  result.push_back(
      {"_sdpa_bwd_dq_dbias_kernel", dq_dependencies, false});
  const std::size_t gradient_id = result.size();
  if (request.attention_key_heads == request.attention_value_heads &&
      request.attention_head_dimension ==
          request.attention_value_dimension) {
    result.push_back(
        {"_sdpa_bwd_dkdv_kernel", {dq_id}, false});
  } else {
    result.push_back({"_sdpa_bwd_dk_kernel", {dq_id}, false});
    result.push_back(
        {"_sdpa_bwd_dv_kernel", {gradient_id}, false});
  }
  return result;
}

std::vector<Candidate> expected_attention_candidates(
    const PointwiseRequest& request,
    bool fixed) {
  if (fixed) {
    return {{1, 4, 1}};
  }
  if (!request.autotune) {
    return {{32, 4, 2}};
  }
  std::vector<Candidate> result;
  for (const unsigned int block : {16U, 32U}) {
    for (const unsigned int warps : {2U, 4U}) {
      for (const unsigned int stages : {1U, 2U}) {
        result.push_back({block, warps, stages});
      }
    }
  }
  return result;
}

std::size_t expected_attention_workspace_size(
    const PointwiseRequest& request) {
  if (request.operation != "sdpa_backward") {
    return kPhaseOneWorkspaceSize;
  }
  const std::uint64_t elements =
      static_cast<std::uint64_t>(request.attention_batch) *
      static_cast<std::uint64_t>(request.attention_heads) *
      static_cast<std::uint64_t>(request.attention_sequence_q);
  if (elements >
      std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    artifact_error("mthreads Attention workspace size overflows size_t");
  }
  const std::size_t raw =
      static_cast<std::size_t>(elements) * sizeof(float);
  if (raw > std::numeric_limits<std::size_t>::max() -
                (kPhaseOneWorkspaceAlignment - 1)) {
    artifact_error("mthreads Attention workspace alignment overflows");
  }
  const std::size_t aligned =
      (raw + kPhaseOneWorkspaceAlignment - 1) /
      kPhaseOneWorkspaceAlignment * kPhaseOneWorkspaceAlignment;
  return std::max(kPhaseOneWorkspaceSize, aligned);
}

std::array<std::int64_t, 4> attention_bias_strides(
    const PointwiseRequest& request) {
  if (!request.attention_has_bias) {
    return {0, 0, 0, 0};
  }
  const TensorSpec& bias = attention_tensor(request, "bias");
  return {bias.dimensions[0] == 1 ? 0 : bias.strides[0],
          bias.dimensions[1] == 1 ? 0 : bias.strides[1],
          bias.strides[2],
          bias.strides[3]};
}

unsigned int attention_full_dimension_block(std::int64_t value) {
  unsigned int result = 16;
  while (result < static_cast<std::uint64_t>(value)) {
    result <<= 1U;
  }
  return result;
}

unsigned int attention_output_dimension_block(
    std::int64_t value,
    unsigned int block_size) {
  return std::min(block_size, attention_full_dimension_block(value));
}

void append_attention_bias_pointer(
    AttentionSignatureBuilder& builder,
    const PointwiseRequest& request) {
  if (request.attention_has_bias) {
    builder.tensor(attention_tensor(request, "bias"), "bias");
  } else {
    builder.tensor(
        attention_tensor(request, "q"), "bias_placeholder_q");
  }
}

void append_attention_stats_pointer(
    AttentionSignatureBuilder& builder,
    const PointwiseRequest& request) {
  const TensorSpec& stats = attention_tensor(request, "stats");
  if (stats.is_virtual) {
    builder.workspace("stats_placeholder");
  } else {
    builder.tensor(stats, "stats");
  }
}

void append_attention_strides4(
    AttentionSignatureBuilder& builder,
    const TensorSpec& tensor) {
  for (const std::int64_t stride : tensor.strides) {
    builder.constant(stride);
  }
}

void append_attention_stats_strides(
    AttentionSignatureBuilder& builder,
    const TensorSpec& tensor) {
  builder.constant(tensor.strides[0]);
  builder.constant(tensor.strides[1]);
  builder.constant(tensor.strides[2]);
}

std::array<unsigned int, 3> attention_grid(
    std::uint64_t x,
    std::uint64_t y,
    std::uint64_t z) {
  if (x == 0 || y == 0 || z == 0 ||
      x > std::numeric_limits<unsigned int>::max() ||
      y > std::numeric_limits<unsigned int>::max() ||
      z > std::numeric_limits<unsigned int>::max()) {
    artifact_error("mthreads Attention grid exceeds uint32");
  }
  return {static_cast<unsigned int>(x),
          static_cast<unsigned int>(y),
          static_cast<unsigned int>(z)};
}

std::uint64_t attention_ceil_div(
    std::int64_t value,
    unsigned int divisor) {
  return (static_cast<std::uint64_t>(value) + divisor - 1) / divisor;
}

std::vector<Candidate> expected_im2col_fprop_candidates(
    std::size_t stage_index) {
  if (stage_index == 0) {
    return {{64U, 4U, 1U}};
  }
  if (stage_index == 1) {
    return {{64U, 8U, 1U}};
  }
  artifact_error("mthreads im2col Fprop stage index is invalid");
}

AttentionVariantExpectation expected_im2col_fprop_variant(
    const PointwiseRequest& request,
    std::string_view function,
    unsigned int block_size) {
  const Im2colFpropGeometry geometry =
      im2col_fprop_geometry(request);
  AttentionSignatureBuilder builder;
  const auto constant = [&builder](const auto value) {
    builder.constant(static_cast<std::int64_t>(value));
  };
  if (function == "_conv_fprop2d_im2col_kernel") {
    builder.tensor(request.convolution_image, "input");
    builder.workspace(
        "fprop_columns", request.convolution_image.data_type);
    constant(geometry.output_area);
    constant(request.convolution_image.dimensions[2]);
    constant(request.convolution_image.dimensions[3]);
    constant(request.convolution_result.dimensions[2]);
    constant(request.convolution_result.dimensions[3]);
    constant(request.convolution_in_per_group);
    constant(request.convolution_filter.dimensions[2]);
    constant(request.convolution_filter.dimensions[3]);
    for (const std::int64_t value : request.convolution_stride) {
      constant(value);
    }
    for (const std::int64_t value :
         request.convolution_pre_padding) {
      constant(value);
    }
    for (const std::int64_t value : request.convolution_dilation) {
      constant(value);
    }
    for (const std::int64_t value :
         request.convolution_image.strides) {
      constant(value);
    }
    for (const std::int64_t value : geometry.column_strides) {
      constant(value);
    }
    constant(block_size);
    constant(32);
    const std::uint64_t spatial_tiles =
        attention_ceil_div(geometry.output_area, block_size);
    const std::uint64_t reduction_tiles =
        attention_ceil_div(geometry.reduction_extent, 32);
    if (reduction_tiles != 0 &&
        spatial_tiles >
            std::numeric_limits<std::uint64_t>::max() /
                reduction_tiles) {
      artifact_error("mthreads im2col Fprop pack grid overflows uint64");
    }
    return std::move(builder).finish(attention_grid(
        spatial_tiles * reduction_tiles,
        request.convolution_batch,
        1));
  }
  if (function == "_conv_fprop2d_im2col_mm_kernel") {
    builder.tensor(request.convolution_filter, "filter");
    builder.workspace(
        "fprop_columns", request.convolution_image.data_type);
    builder.tensor(request.convolution_result, "output");
    constant(geometry.output_area);
    constant(request.convolution_out_per_group);
    constant(request.convolution_in_per_group);
    constant(request.convolution_filter.dimensions[2]);
    constant(request.convolution_filter.dimensions[3]);
    for (const std::int64_t value :
         request.convolution_filter.strides) {
      constant(value);
    }
    for (const std::int64_t value :
         request.convolution_result.strides) {
      constant(value);
    }
    constant(request.convolution_result.dimensions[3]);
    for (const std::int64_t value : geometry.column_strides) {
      constant(value);
    }
    constant(request.convolution_image.data_type == "float32" &&
                     request.input_precision != 1
                 ? 1
                 : 0);
    const Im2colFpropMmBlocks blocks =
        im2col_fprop_mm_blocks(request, geometry);
    constant(blocks.output_channels);
    constant(blocks.output_spatial);
    constant(blocks.reduction);
    constant(8);
    const std::uint64_t output_tiles = attention_ceil_div(
        request.convolution_out_per_group, blocks.output_channels);
    const std::uint64_t spatial_tiles =
        attention_ceil_div(geometry.output_area, blocks.output_spatial);
    if (spatial_tiles != 0 &&
        output_tiles >
            std::numeric_limits<std::uint64_t>::max() /
                spatial_tiles) {
      artifact_error("mthreads im2col Fprop MM grid overflows uint64");
    }
    return std::move(builder).finish(attention_grid(
        output_tiles * spatial_tiles,
        request.convolution_batch,
        1));
  }
  artifact_error("mthreads im2col Fprop kernel function is invalid");
}

std::vector<Candidate> expected_dense_dgrad_candidates(
    const PointwiseRequest& request, std::size_t stage_index) {
  if (stage_index > 2) {
    artifact_error("mthreads dense Dgrad stage index is invalid");
  }
  if (stage_index < 2) {
    return {{32U, 8U, 1U}};
  }
  if (!request.autotune) {
    return {{64U, 8U, 1U}};
  }
  std::vector<Candidate> result;
  for (const unsigned int num_warps : {4U, 8U}) {
    for (const unsigned int num_stages : {1U, 2U}) {
      result.push_back({64U, num_warps, num_stages});
    }
  }
  return result;
}

AttentionVariantExpectation expected_dense_dgrad_variant(
    const PointwiseRequest& request,
    std::string_view function,
    unsigned int block_size) {
  const DenseDgradGeometry geometry = dense_dgrad_geometry(request);
  AttentionSignatureBuilder builder;
  const auto constant = [&builder](const auto value) {
    builder.constant(static_cast<std::int64_t>(value));
  };
  if (function == "_conv_dgrad2d_dense_pack_filter_kernel") {
    builder.tensor(request.convolution_filter, "w");
    builder.workspace(
        "dgrad_dense_filter", request.convolution_image.data_type);
    constant(request.convolution_in_per_group);
    constant(request.convolution_out_per_group);
    for (const std::int64_t stride :
         request.convolution_filter.strides) {
      constant(stride);
    }
    constant(block_size);
    constant(block_size);
    return std::move(builder).finish(attention_grid(
        attention_ceil_div(geometry.matrix_rows, block_size),
        attention_ceil_div(geometry.reduction_extent, block_size),
        1));
  }
  if (function == "_conv_dgrad2d_dense_pack_loss_kernel") {
    builder.tensor(request.convolution_result, "dy");
    builder.workspace(
        "dgrad_dense_loss", request.convolution_image.data_type);
    constant(geometry.loss_offset);
    constant(geometry.loss_rows);
    constant(request.convolution_result.dimensions[2]);
    constant(request.convolution_result.dimensions[3]);
    constant(request.convolution_out_per_group);
    for (const std::int64_t stride :
         request.convolution_result.strides) {
      constant(stride);
    }
    constant(block_size);
    constant(block_size);
    return std::move(builder).finish(attention_grid(
        attention_ceil_div(geometry.reduction_extent, block_size),
        attention_ceil_div(geometry.loss_rows, block_size),
        1));
  }
  if (function == "_conv_dgrad2d_dense_mm_kernel") {
    builder.workspace(
        "dgrad_dense_filter", request.convolution_image.data_type);
    builder.workspace(
        "dgrad_dense_loss", request.convolution_image.data_type);
    builder.tensor(request.convolution_image, "dx");
    constant(geometry.loss_offset);
    constant(geometry.loss_rows);
    constant(request.convolution_result.dimensions[2]);
    constant(request.convolution_result.dimensions[3]);
    constant(request.convolution_image.dimensions[2]);
    constant(request.convolution_image.dimensions[3]);
    constant(request.convolution_in_per_group);
    constant(request.convolution_out_per_group);
    for (const std::int64_t stride :
         request.convolution_image.strides) {
      constant(stride);
    }
    constant(request.convolution_image.data_type == "float32" &&
                     request.input_precision != 1
                 ? 1
                 : 0);
    constant(block_size);
    constant(block_size);
    constant(block_size);
    constant(8);
    const std::uint64_t row_tiles =
        attention_ceil_div(geometry.matrix_rows, block_size);
    const std::uint64_t loss_tiles =
        attention_ceil_div(geometry.loss_rows, block_size);
    if (loss_tiles != 0 &&
        row_tiles >
            std::numeric_limits<std::uint64_t>::max() / loss_tiles) {
      artifact_error("mthreads dense Dgrad grid overflows uint64");
    }
    return std::move(builder).finish(
        attention_grid(row_tiles * loss_tiles, 1, 1));
  }
  artifact_error("mthreads dense Dgrad kernel function is invalid");
}

AttentionVariantExpectation expected_p5_wgrad_variant(
    const PointwiseRequest& request,
    std::string_view function,
    unsigned int block_size) {
  if (request.convolution_in_per_group >
      std::numeric_limits<std::int64_t>::max() / 9) {
    artifact_error("mthreads P5 Wgrad packed width overflows int64");
  }
  const std::int64_t dtype_id =
      request.convolution_image.data_type == "float32"
          ? 0
          : (request.convolution_image.data_type == "float16" ? 1 : 2);
  const std::int64_t packed_columns =
      request.convolution_in_per_group * 9;
  AttentionSignatureBuilder builder;
  const auto constant = [&builder](const auto value) {
    builder.constant(static_cast<std::int64_t>(value));
  };
  if (function == "_conv_wgrad2d_p5_pack_image_kernel") {
    builder.tensor(request.convolution_image, "x");
    builder.workspace(
        "packed_image", request.convolution_image.data_type);
    constant(request.convolution_in_per_group);
    constant(request.convolution_image.strides[1]);
    constant(request.convolution_image.strides[2]);
    constant(request.convolution_image.strides[3]);
    constant(400);
    constant(packed_columns);
    constant(block_size);
    constant(block_size);
    constant(block_size);
    constant(8);
    return std::move(builder).finish(attention_grid(
        attention_ceil_div(400, block_size),
        attention_ceil_div(packed_columns, block_size),
        1));
  }
  if (function == "_conv_wgrad2d_p5_mm_kernel") {
    builder.tensor(request.convolution_result, "dy");
    builder.workspace(
        "packed_image", request.convolution_image.data_type);
    builder.tensor(request.convolution_filter, "dw");
    constant(request.convolution_out_per_group);
    constant(packed_columns);
    constant(400);
    constant(dtype_id);
    constant(block_size);
    constant(block_size);
    constant(block_size);
    constant(8);
    const std::uint64_t output_tiles = attention_ceil_div(
        request.convolution_out_per_group, block_size);
    const std::uint64_t column_tiles =
        attention_ceil_div(packed_columns, block_size);
    if (output_tiles >
        std::numeric_limits<std::uint64_t>::max() / column_tiles) {
      artifact_error("mthreads P5 Wgrad grid overflows uint64");
    }
    return std::move(builder).finish(
        attention_grid(output_tiles * column_tiles, 1, 1));
  }
  artifact_error("mthreads P5 Wgrad kernel function is invalid");
}

std::vector<Candidate> expected_standard_wgrad_candidates(
    const PointwiseRequest& request, std::size_t stage_index) {
  const bool one_by_one =
      request.convolution_filter.dimensions.size() == 4 &&
      request.convolution_filter.dimensions[2] == 1 &&
      request.convolution_filter.dimensions[3] == 1;
  const std::size_t maximum_stage = one_by_one ? 1 : 2;
  if (stage_index > maximum_stage) {
    artifact_error("mthreads standard Wgrad stage index is invalid");
  }
  if (!request.autotune) {
    if (one_by_one) {
      return stage_index == 0 ? std::vector<Candidate>{{16, 4, 2}}
                              : std::vector<Candidate>{{256, 4, 1}};
    }
    if (stage_index == 0) {
      return {{64, 4, 1}};
    }
    return stage_index == 1 ? std::vector<Candidate>{{64, 8, 1}}
                            : std::vector<Candidate>{{256, 4, 1}};
  }
  std::vector<Candidate> result;
  if (one_by_one) {
    if (stage_index == 0) {
      for (const unsigned int block : {16U, 32U}) {
        for (const unsigned int warps : {4U, 8U}) {
          for (const unsigned int stages : {1U, 2U}) {
            result.push_back({block, warps, stages});
          }
        }
      }
    } else {
      for (const unsigned int block : {128U, 256U}) {
        for (const unsigned int warps : {4U, 8U}) {
          result.push_back({block, warps, 1});
        }
      }
    }
  } else {
    if (stage_index == 2) {
      for (const unsigned int block : {128U, 256U}) {
        for (const unsigned int warps : {4U, 8U}) {
          result.push_back({block, warps, 1});
        }
      }
    } else {
      for (const unsigned int block : {32U, 64U}) {
        for (const unsigned int warps : {4U, 8U}) {
          if (stage_index == 0) {
            result.push_back({block, warps, 1});
          } else {
          for (const unsigned int stages : {1U, 2U}) {
            result.push_back({block, warps, stages});
          }
          }
        }
      }
    }
  }
  return result;
}

AttentionVariantExpectation expected_standard_wgrad_variant(
    const PointwiseRequest& request,
    std::string_view function,
    unsigned int block_size) {
  const TensorSpec& image = request.convolution_image;
  const TensorSpec& loss = request.convolution_result;
  const TensorSpec& weight = request.convolution_filter;
  const std::int64_t xh = image.dimensions[2];
  const std::int64_t xw = image.dimensions[3];
  const std::int64_t oh = loss.dimensions[2];
  const std::int64_t ow = loss.dimensions[3];
  const std::int64_t kh = weight.dimensions[2];
  const std::int64_t kw = weight.dimensions[3];
  const std::int64_t cik = request.convolution_in_per_group * kh * kw;
  const std::int64_t total = request.convolution_out_per_group * cik;
  const std::int64_t num_splits = kh == 1 && kw == 1
      ? 8
      : request.convolution_batch;
  const std::int64_t partial_raw = num_splits * total * sizeof(float);
  const std::int64_t partial_aligned =
      (partial_raw + kPhaseOneWorkspaceAlignment - 1) /
      kPhaseOneWorkspaceAlignment * kPhaseOneWorkspaceAlignment;
  const std::int64_t column_offset = partial_aligned /
      static_cast<std::int64_t>(element_size(image.data_type));
  const std::int64_t total_rows = request.convolution_batch * oh * ow;
  const std::int64_t rows_per_split =
      (total_rows + num_splits - 1) / num_splits;
  const std::int64_t dtype_id = image.data_type == "float32"
      ? 0
      : (image.data_type == "float16" ? 1 : 2);
  AttentionSignatureBuilder builder;
  const auto constant = [&builder](const auto value) {
    builder.constant(static_cast<std::int64_t>(value));
  };
  if (function == "_conv_wgrad2d_im2row_kernel") {
    builder.tensor(image, "x");
    builder.workspace("wgrad_columns", image.data_type);
    constant(oh * ow);
    constant(xh);
    constant(xw);
    constant(ow);
    constant(request.convolution_in_per_group);
    constant(kh);
    constant(kw);
    for (const std::int64_t value : request.convolution_stride) {
      constant(value);
    }
    for (const std::int64_t value : request.convolution_pre_padding) {
      constant(value);
    }
    for (const std::int64_t value : request.convolution_dilation) {
      constant(value);
    }
    for (const std::int64_t stride : image.strides) {
      constant(stride);
    }
    constant(cik);
    constant(column_offset);
    constant(1);
    constant(block_size);
    constant(block_size);
    const std::uint64_t row_tiles =
        attention_ceil_div(oh * ow, block_size);
    const std::uint64_t column_tiles =
        attention_ceil_div(cik, block_size);
    return std::move(builder).finish(attention_grid(
        row_tiles * column_tiles,
        static_cast<std::uint64_t>(request.convolution_batch),
        1));
  }
  if (function == "_conv_wgrad2d_rowmajor_kernel") {
    builder.tensor(loss, "dy");
    builder.workspace("wgrad_columns", image.data_type);
    builder.workspace("wgrad_partial");
    constant(oh * ow);
    constant(request.convolution_out_per_group);
    constant(request.convolution_in_per_group);
    constant(kh);
    constant(kw);
    for (const std::int64_t stride : loss.strides) {
      constant(stride);
    }
    constant(column_offset);
    constant(cik);
    constant(1);
    constant(total);
    constant(cik);
    constant(1);
    constant(dtype_id);
    constant(block_size);
    constant(block_size);
    constant(block_size);
    const std::uint64_t output_tiles = attention_ceil_div(
        request.convolution_out_per_group, block_size);
    const std::uint64_t column_tiles =
        attention_ceil_div(cik, block_size);
    return std::move(builder).finish(attention_grid(
        output_tiles * column_tiles,
        static_cast<std::uint64_t>(request.convolution_batch),
        1));
  }
  if (function == "_conv_wgrad2d_direct_split_kernel") {
    builder.tensor(loss, "dy");
    builder.tensor(image, "x");
    builder.workspace("wgrad_partial");
    constant(total_rows);
    constant(rows_per_split);
    constant(oh * ow);
    constant(xh);
    constant(xw);
    constant(ow);
    constant(request.convolution_out_per_group);
    constant(request.convolution_in_per_group);
    constant(kh);
    constant(kw);
    for (const std::int64_t value : request.convolution_stride) {
      constant(value);
    }
    for (const std::int64_t value : request.convolution_pre_padding) {
      constant(value);
    }
    for (const std::int64_t value : request.convolution_dilation) {
      constant(value);
    }
    for (const std::int64_t stride : loss.strides) {
      constant(stride);
    }
    for (const std::int64_t stride : image.strides) {
      constant(stride);
    }
    constant(total);
    constant(cik);
    constant(1);
    constant(dtype_id);
    constant(block_size);
    constant(block_size);
    constant(64);
    const std::uint64_t output_tiles = attention_ceil_div(
        request.convolution_out_per_group, block_size);
    const std::uint64_t column_tiles = attention_ceil_div(cik, block_size);
    return std::move(builder).finish(attention_grid(
        output_tiles * column_tiles,
        static_cast<std::uint64_t>(num_splits),
        1));
  }
  if (function == "_conv_wgrad2d_1x1_split_kernel") {
    builder.tensor(loss, "dy");
    builder.tensor(image, "x");
    builder.workspace("wgrad_partial");
    constant(total_rows);
    constant(rows_per_split);
    constant(oh * ow);
    constant(image.dimensions[1]);
    constant(loss.dimensions[1]);
    constant(request.convolution_in_per_group);
    constant(request.convolution_out_per_group);
    constant(request.convolution_groups);
    constant(total);
    constant(cik);
    constant(1);
    constant(dtype_id);
    constant(block_size);
    constant(block_size);
    constant(64);
    const std::uint64_t output_tiles = attention_ceil_div(
        request.convolution_out_per_group, block_size);
    const std::uint64_t input_tiles = attention_ceil_div(
        request.convolution_in_per_group, block_size);
    return std::move(builder).finish(attention_grid(
        output_tiles * input_tiles,
        static_cast<std::uint64_t>(
            num_splits * request.convolution_groups),
        1));
  }
  if (function == "_conv_wgrad2d_stem_reduce_kernel") {
    builder.workspace("wgrad_partial");
    builder.tensor(weight, "dw");
    constant(total);
    constant(cik);
    constant(request.convolution_in_per_group);
    constant(kh);
    constant(kw);
    constant(num_splits);
    constant(total);
    constant(cik);
    constant(1);
    for (const std::int64_t stride : weight.strides) {
      constant(stride);
    }
    constant(block_size);
    return std::move(builder).finish(attention_grid(
        attention_ceil_div(total, block_size), 1, 1));
  }
  artifact_error("mthreads standard Wgrad kernel function is invalid");
}

AttentionVariantExpectation expected_nd_packed_wgrad_variant(
    const PointwiseRequest& request,
    std::string_view function,
    unsigned int block_size) {
  const TensorSpec& image = request.convolution_image;
  const TensorSpec& loss = request.convolution_result;
  const TensorSpec& weight = request.convolution_filter;
  const std::size_t spatial_rank =
      static_cast<std::size_t>(request.convolution_spatial_rank);
  std::array<std::int64_t, 3> input_spatial = {1, 1, 1};
  std::array<std::int64_t, 3> output_spatial = {1, 1, 1};
  std::array<std::int64_t, 3> kernel_spatial = {1, 1, 1};
  std::array<std::int64_t, 3> stride = {1, 1, 1};
  std::array<std::int64_t, 3> padding = {0, 0, 0};
  std::array<std::int64_t, 3> dilation = {1, 1, 1};
  std::array<std::int64_t, 5> input_strides = {
      image.strides[0], image.strides[1], 0, 0, 0};
  const std::size_t spatial_offset = 3 - spatial_rank;
  for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
    const std::size_t normalized_axis = spatial_offset + axis;
    input_spatial[normalized_axis] = image.dimensions[2 + axis];
    output_spatial[normalized_axis] = loss.dimensions[2 + axis];
    kernel_spatial[normalized_axis] = weight.dimensions[2 + axis];
    stride[normalized_axis] = request.convolution_stride[axis];
    padding[normalized_axis] = request.convolution_pre_padding[axis];
    dilation[normalized_axis] = request.convolution_dilation[axis];
    input_strides[2 + normalized_axis] = image.strides[2 + axis];
  }

  std::int64_t output_area = 1;
  std::int64_t kernel_volume = 1;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    output_area *= output_spatial[axis];
    kernel_volume *= kernel_spatial[axis];
  }
  const std::int64_t reduction_extent =
      request.convolution_in_per_group * kernel_volume;
  const std::int64_t total_rows = request.convolution_batch * output_area;
  const std::int64_t total_weights =
      request.convolution_out_per_group * reduction_extent;
  const std::int64_t num_splits = total_rows >= 4096 ? 16 : 8;
  const std::int64_t rows_per_split =
      (total_rows + num_splits - 1) / num_splits;
  const std::int64_t partial_raw =
      num_splits * total_weights * sizeof(float);
  const std::int64_t partial_aligned =
      (partial_raw + kPhaseOneWorkspaceAlignment - 1) /
      kPhaseOneWorkspaceAlignment * kPhaseOneWorkspaceAlignment;
  const std::int64_t column_offset = partial_aligned /
      static_cast<std::int64_t>(element_size(image.data_type));
  const std::int64_t dtype_id = image.data_type == "float32"
      ? 0
      : (image.data_type == "float16" ? 1 : 2);

  AttentionSignatureBuilder builder;
  const auto constant = [&builder](const auto value) {
    builder.constant(static_cast<std::int64_t>(value));
  };
  if (function == "_conv_wgrad_nd_im2row_kernel") {
    builder.tensor(image, "x");
    builder.workspace("wgrad_nd_columns", image.data_type);
    constant(output_area);
    for (const std::int64_t value : input_spatial) {
      constant(value);
    }
    constant(output_spatial[1]);
    constant(output_spatial[2]);
    constant(request.convolution_in_per_group);
    for (const std::int64_t value : kernel_spatial) {
      constant(value);
    }
    for (const std::int64_t value : stride) {
      constant(value);
    }
    for (const std::int64_t value : padding) {
      constant(value);
    }
    for (const std::int64_t value : dilation) {
      constant(value);
    }
    for (const std::int64_t value : input_strides) {
      constant(value);
    }
    constant(reduction_extent);
    constant(column_offset);
    constant(1);
    constant(block_size);
    constant(block_size);
    return std::move(builder).finish(attention_grid(
        attention_ceil_div(output_area, block_size) *
            attention_ceil_div(reduction_extent, block_size),
        static_cast<std::uint64_t>(request.convolution_batch),
        1));
  }
  if (function == "_conv_wgrad_nd_rowmajor_kernel") {
    builder.tensor(loss, "dy");
    builder.workspace("wgrad_nd_columns", image.data_type);
    builder.workspace("wgrad_nd_partial");
    constant(total_rows);
    constant(rows_per_split);
    constant(output_area);
    constant(request.convolution_out_per_group);
    constant(reduction_extent);
    constant(loss.strides[0]);
    constant(loss.strides[1]);
    constant(loss.strides.back());
    constant(reduction_extent);
    constant(column_offset);
    constant(1);
    constant(total_weights);
    constant(reduction_extent);
    constant(1);
    constant(dtype_id);
    constant(block_size);
    constant(block_size);
    constant(block_size);
    return std::move(builder).finish(attention_grid(
        attention_ceil_div(request.convolution_out_per_group, block_size) *
            attention_ceil_div(reduction_extent, block_size),
        static_cast<std::uint64_t>(num_splits),
        1));
  }
  if (function == "_conv_wgrad_nd_reduce_kernel") {
    builder.workspace("wgrad_nd_partial");
    builder.tensor(weight, "dw");
    constant(total_weights);
    constant(num_splits);
    constant(total_weights);
    constant(block_size);
    return std::move(builder).finish(attention_grid(
        attention_ceil_div(total_weights, block_size), 1, 1));
  }
  artifact_error("mthreads ND packed Wgrad kernel function is invalid");
}

std::vector<Candidate> expected_stem_wgrad_candidates(
    const PointwiseRequest& request, std::size_t stage_index) {
  if (stage_index > 1) {
    artifact_error("mthreads stem Wgrad stage index is invalid");
  }
  if (!request.autotune) {
    return stage_index == 0 ? std::vector<Candidate>{{64, 4, 2}}
                            : std::vector<Candidate>{{256, 4, 1}};
  }
  std::vector<Candidate> result;
  if (stage_index == 0) {
    for (const unsigned int block : {32U, 64U}) {
      for (const unsigned int warps : {4U, 8U}) {
        for (const unsigned int stages : {1U, 2U}) {
          result.push_back({block, warps, stages});
        }
      }
    }
  } else {
    for (const unsigned int block : {128U, 256U}) {
      for (const unsigned int warps : {4U, 8U}) {
        result.push_back({block, warps, 1});
      }
    }
  }
  return result;
}

AttentionVariantExpectation expected_stem_wgrad_variant(
    const PointwiseRequest& request,
    std::string_view function,
    unsigned int block_size) {
  const std::int64_t dtype_id =
      request.convolution_image.data_type == "float32"
          ? 0
          : (request.convolution_image.data_type == "float16" ? 1 : 2);
  const std::int64_t partial_stride_oc =
      request.convolution_in_per_group * 9;
  const std::int64_t partial_stride_split =
      request.convolution_out_per_group * partial_stride_oc;
  AttentionSignatureBuilder builder;
  const auto constant = [&builder](const auto value) {
    builder.constant(static_cast<std::int64_t>(value));
  };
  if (function == "_conv_wgrad2d_stem_split_kernel") {
    builder.tensor(request.convolution_result, "dy");
    builder.tensor(request.convolution_image, "x");
    builder.workspace("wgrad_partial");
    constant(5);
    constant(320);
    constant(320);
    constant(640);
    constant(640);
    constant(request.convolution_out_per_group);
    constant(request.convolution_in_per_group);
    constant(3);
    constant(3);
    constant(2);
    constant(2);
    constant(1);
    constant(1);
    constant(1);
    constant(1);
    constant(request.convolution_result.strides[1]);
    constant(request.convolution_result.strides[2]);
    constant(request.convolution_result.strides[3]);
    constant(request.convolution_image.strides[1]);
    constant(request.convolution_image.strides[2]);
    constant(request.convolution_image.strides[3]);
    constant(partial_stride_split);
    constant(partial_stride_oc);
    constant(1);
    constant(dtype_id);
    constant(block_size);
    constant(32);
    constant(64);
    return std::move(builder).finish(attention_grid(
        attention_ceil_div(request.convolution_out_per_group, block_size),
        64,
        1));
  }
  if (function == "_conv_wgrad2d_stem_reduce_kernel") {
    builder.workspace("wgrad_partial");
    builder.tensor(request.convolution_filter, "dw");
    constant(partial_stride_split);
    constant(partial_stride_oc);
    constant(request.convolution_in_per_group);
    constant(3);
    constant(3);
    constant(64);
    constant(partial_stride_split);
    constant(partial_stride_oc);
    constant(1);
    for (const std::int64_t stride : request.convolution_filter.strides) {
      constant(stride);
    }
    constant(block_size);
    return std::move(builder).finish(attention_grid(
        attention_ceil_div(partial_stride_split, block_size), 1, 1));
  }
  artifact_error("mthreads stem Wgrad kernel function is invalid");
}

void append_attention_forward_scalars(
    AttentionSignatureBuilder& builder,
    const PointwiseRequest& request,
    bool fp8) {
  constexpr double kLog2E = 1.4426950408889634;
  builder.scalar_f32(
      fp8 ? "attn_scale" : "qk_scale",
      fp8 ? request.attention_scale
          : static_cast<float>(
                static_cast<double>(request.attention_scale) * kLog2E));
  const TensorSpec& q = attention_tensor(request, "q");
  const TensorSpec& k = attention_tensor(request, "k");
  const TensorSpec& v = attention_tensor(request, "v");
  const TensorSpec& o = attention_tensor(request, "o");
  const TensorSpec& stats = attention_tensor(request, "stats");
  const auto bias_strides = attention_bias_strides(request);
  const std::array<std::pair<std::string_view, std::int64_t>, 30> values = {{
      {"HQ", request.attention_heads},
      {"SQ", request.attention_sequence_q},
      {"SKV", request.attention_sequence_kv},
      {"q_per_k", request.attention_q_per_k},
      {"q_per_v", request.attention_q_per_v},
      {"min_diag", request.attention_min_diag},
      {"max_diag", request.attention_max_diag},
      {"stride_qb", q.strides[0]},
      {"stride_qh", q.strides[1]},
      {"stride_qm", q.strides[2]},
      {"stride_qd", q.strides[3]},
      {"stride_kb", k.strides[0]},
      {"stride_kh", k.strides[1]},
      {"stride_kn", k.strides[2]},
      {"stride_kd", k.strides[3]},
      {"stride_vb", v.strides[0]},
      {"stride_vh", v.strides[1]},
      {"stride_vn", v.strides[2]},
      {"stride_vd", v.strides[3]},
      {"stride_bias_b", bias_strides[0]},
      {"stride_bias_h", bias_strides[1]},
      {"stride_bias_m", bias_strides[2]},
      {"stride_bias_n", bias_strides[3]},
      {"stride_ob", o.strides[0]},
      {"stride_oh", o.strides[1]},
      {"stride_om", o.strides[2]},
      {"stride_od", o.strides[3]},
      {"stride_sb", stats.strides[0]},
      {"stride_sh", stats.strides[1]},
      {"stride_sm", stats.strides[2]},
  }};
  for (const auto& [name, value] : values) {
    builder.scalar_i32(name, value);
  }
}

void append_attention_forward_constants(
    AttentionSignatureBuilder& builder,
    const PointwiseRequest& request,
    unsigned int block_size,
    bool fp8) {
  builder.constant(request.attention_head_dimension);
  builder.constant(request.attention_value_dimension);
  if (!fp8) {
    builder.constant(static_cast<std::int64_t>(
        element_size(attention_tensor(request, "q").data_type)));
  }
  builder.constant(static_cast<std::int64_t>(block_size));
  builder.constant(static_cast<std::int64_t>(block_size));
  builder.constant(static_cast<std::int64_t>(
      attention_full_dimension_block(request.attention_head_dimension)));
  builder.constant(static_cast<std::int64_t>(
      attention_full_dimension_block(request.attention_value_dimension)));
  builder.constant(request.attention_has_bias);
  builder.constant(request.attention_banded);
  builder.constant(request.attention_generate_stats);
  builder.constant(request.attention_reverse_causal);
}

AttentionVariantExpectation expected_attention_forward_variant(
    const PointwiseRequest& request,
    unsigned int block_size,
    bool fp8) {
  AttentionSignatureBuilder builder;
  builder.tensor(attention_tensor(request, "q"), "q");
  builder.tensor(attention_tensor(request, "k"), "k");
  builder.tensor(attention_tensor(request, "v"), "v");
  append_attention_bias_pointer(builder, request);
  builder.tensor(attention_tensor(request, "o"), "o");
  append_attention_stats_pointer(builder, request);
  if (fp8) {
    for (const std::string_view port :
         {"amax_s", "amax_o", "descale_q", "descale_k", "descale_v",
          "descale_s", "scale_s", "scale_o"}) {
      builder.tensor(attention_tensor(request, port), port);
    }
  }
  append_attention_forward_scalars(builder, request, fp8);
  append_attention_forward_constants(builder, request, block_size, fp8);
  return std::move(builder).finish(attention_grid(
      attention_ceil_div(request.attention_sequence_q, block_size),
      static_cast<std::uint64_t>(request.attention_batch) *
          static_cast<std::uint64_t>(request.attention_heads),
      1));
}

void append_attention_qkv_bias_strides(
    AttentionSignatureBuilder& builder,
    const PointwiseRequest& request,
    bool include_v) {
  append_attention_strides4(builder, attention_tensor(request, "q"));
  append_attention_strides4(builder, attention_tensor(request, "k"));
  if (include_v) {
    append_attention_strides4(builder, attention_tensor(request, "v"));
  }
  for (const std::int64_t stride : attention_bias_strides(request)) {
    builder.constant(stride);
  }
}

AttentionVariantExpectation expected_attention_dq_variant(
    const PointwiseRequest& request,
    unsigned int block_size) {
  AttentionSignatureBuilder builder;
  builder.tensor(attention_tensor(request, "q"), "q");
  builder.tensor(attention_tensor(request, "k"), "k");
  builder.tensor(attention_tensor(request, "v"), "v");
  append_attention_bias_pointer(builder, request);
  builder.tensor(attention_tensor(request, "o"), "o");
  builder.tensor(attention_tensor(request, "do"), "do");
  builder.tensor(attention_tensor(request, "stats"), "stats");
  builder.workspace("delta");
  builder.tensor(attention_tensor(request, "dq"), "dq");
  if (request.attention_has_dbias) {
    builder.tensor(attention_tensor(request, "dbias"), "dbias");
  } else {
    builder.tensor(
        attention_tensor(request, "dq"), "dbias_placeholder_dq");
  }
  builder.scalar_f32("attn_scale", request.attention_scale);
  builder.constant(request.attention_heads);
  builder.scalar_i32("SQ", request.attention_sequence_q);
  builder.scalar_i32("SKV", request.attention_sequence_kv);
  builder.constant(request.attention_q_per_k);
  builder.constant(request.attention_q_per_v);
  builder.scalar_i32("min_diag", request.attention_min_diag);
  builder.scalar_i32("max_diag", request.attention_max_diag);
  append_attention_qkv_bias_strides(builder, request, true);
  append_attention_strides4(builder, attention_tensor(request, "o"));
  append_attention_strides4(builder, attention_tensor(request, "do"));
  append_attention_stats_strides(
      builder, attention_tensor(request, "stats"));
  builder.constant(request.attention_heads * request.attention_sequence_q);
  builder.constant(request.attention_sequence_q);
  builder.constant(std::int64_t{1});
  append_attention_strides4(builder, attention_tensor(request, "dq"));
  if (request.attention_has_dbias) {
    append_attention_strides4(
        builder, attention_tensor(request, "dbias"));
  } else {
    for (int index = 0; index < 4; ++index) {
      builder.constant(std::int64_t{0});
    }
  }
  builder.constant(request.attention_head_dimension);
  builder.constant(request.attention_value_dimension);
  builder.constant(
      request.attention_has_dbias
          ? attention_tensor(request, "dbias").dimensions[0]
          : 1);
  builder.constant(
      request.attention_has_dbias
          ? attention_tensor(request, "dbias").dimensions[1]
          : 1);
  builder.constant(static_cast<std::int64_t>(block_size));
  builder.constant(static_cast<std::int64_t>(block_size));
  builder.constant(static_cast<std::int64_t>(
      attention_full_dimension_block(request.attention_head_dimension)));
  const unsigned int block_d_out = attention_output_dimension_block(
      request.attention_head_dimension, block_size);
  builder.constant(static_cast<std::int64_t>(block_d_out));
  builder.constant(static_cast<std::int64_t>(
      attention_full_dimension_block(request.attention_value_dimension)));
  builder.constant(false);
  builder.constant(request.attention_has_bias);
  builder.constant(request.attention_has_dbias);
  builder.constant(attention_dbias_reduces(request));
  builder.constant(request.attention_banded);
  builder.constant(request.attention_causal_top_left);
  return std::move(builder).finish(attention_grid(
      attention_ceil_div(request.attention_sequence_q, block_size),
      attention_ceil_div(
          request.attention_head_dimension, block_d_out),
      static_cast<std::uint64_t>(request.attention_batch) *
          static_cast<std::uint64_t>(request.attention_heads)));
}

AttentionVariantExpectation expected_attention_dkdv_variant(
    const PointwiseRequest& request,
    unsigned int block_size) {
  AttentionSignatureBuilder builder;
  builder.tensor(attention_tensor(request, "q"), "q");
  builder.tensor(attention_tensor(request, "k"), "k");
  builder.tensor(attention_tensor(request, "v"), "v");
  append_attention_bias_pointer(builder, request);
  builder.tensor(attention_tensor(request, "do"), "do");
  builder.tensor(attention_tensor(request, "stats"), "stats");
  builder.workspace("delta");
  builder.tensor(attention_tensor(request, "dk"), "dk");
  builder.tensor(attention_tensor(request, "dv"), "dv");
  builder.scalar_f32("attn_scale", request.attention_scale);
  builder.constant(request.attention_key_heads);
  builder.scalar_i32("SQ", request.attention_sequence_q);
  builder.scalar_i32("SKV", request.attention_sequence_kv);
  builder.scalar_i32("min_diag", request.attention_min_diag);
  builder.scalar_i32("max_diag", request.attention_max_diag);
  append_attention_qkv_bias_strides(builder, request, true);
  append_attention_strides4(builder, attention_tensor(request, "do"));
  append_attention_stats_strides(
      builder, attention_tensor(request, "stats"));
  builder.constant(request.attention_heads * request.attention_sequence_q);
  builder.constant(request.attention_sequence_q);
  builder.constant(std::int64_t{1});
  append_attention_strides4(builder, attention_tensor(request, "dk"));
  append_attention_strides4(builder, attention_tensor(request, "dv"));
  builder.constant(request.attention_head_dimension);
  builder.constant(request.attention_q_per_k);
  builder.constant(static_cast<std::int64_t>(block_size));
  builder.constant(static_cast<std::int64_t>(block_size));
  builder.constant(static_cast<std::int64_t>(
      attention_full_dimension_block(request.attention_head_dimension)));
  const unsigned int block_d_out = attention_output_dimension_block(
      request.attention_head_dimension, block_size);
  builder.constant(static_cast<std::int64_t>(block_d_out));
  builder.constant(false);
  builder.constant(request.attention_has_bias);
  builder.constant(request.attention_banded);
  builder.constant(request.attention_causal_top_left);
  return std::move(builder).finish(attention_grid(
      attention_ceil_div(request.attention_sequence_kv, block_size),
      attention_ceil_div(
          request.attention_head_dimension, block_d_out),
      static_cast<std::uint64_t>(request.attention_batch) *
          static_cast<std::uint64_t>(request.attention_key_heads)));
}

AttentionVariantExpectation expected_attention_dk_variant(
    const PointwiseRequest& request,
    unsigned int block_size) {
  AttentionSignatureBuilder builder;
  builder.tensor(attention_tensor(request, "q"), "q");
  builder.tensor(attention_tensor(request, "k"), "k");
  builder.tensor(attention_tensor(request, "v"), "v");
  append_attention_bias_pointer(builder, request);
  builder.tensor(attention_tensor(request, "do"), "do");
  builder.tensor(attention_tensor(request, "stats"), "stats");
  builder.workspace("delta");
  builder.tensor(attention_tensor(request, "dk"), "dk");
  builder.scalar_f32("attn_scale", request.attention_scale);
  builder.constant(request.attention_key_heads);
  builder.scalar_i32("SQ", request.attention_sequence_q);
  builder.scalar_i32("SKV", request.attention_sequence_kv);
  builder.scalar_i32("min_diag", request.attention_min_diag);
  builder.scalar_i32("max_diag", request.attention_max_diag);
  append_attention_qkv_bias_strides(builder, request, true);
  append_attention_strides4(builder, attention_tensor(request, "do"));
  append_attention_stats_strides(
      builder, attention_tensor(request, "stats"));
  builder.constant(request.attention_heads * request.attention_sequence_q);
  builder.constant(request.attention_sequence_q);
  builder.constant(std::int64_t{1});
  append_attention_strides4(builder, attention_tensor(request, "dk"));
  builder.constant(request.attention_head_dimension);
  builder.constant(request.attention_value_dimension);
  builder.constant(request.attention_q_per_k);
  builder.constant(static_cast<std::int64_t>(block_size));
  builder.constant(static_cast<std::int64_t>(block_size));
  builder.constant(static_cast<std::int64_t>(
      attention_full_dimension_block(request.attention_head_dimension)));
  const unsigned int block_d_out = attention_output_dimension_block(
      request.attention_head_dimension, block_size);
  builder.constant(static_cast<std::int64_t>(block_d_out));
  builder.constant(static_cast<std::int64_t>(
      attention_full_dimension_block(request.attention_value_dimension)));
  builder.constant(false);
  builder.constant(request.attention_has_bias);
  builder.constant(request.attention_banded);
  builder.constant(request.attention_causal_top_left);
  return std::move(builder).finish(attention_grid(
      attention_ceil_div(request.attention_sequence_kv, block_size),
      attention_ceil_div(
          request.attention_head_dimension, block_d_out),
      static_cast<std::uint64_t>(request.attention_batch) *
          static_cast<std::uint64_t>(request.attention_key_heads)));
}

AttentionVariantExpectation expected_attention_dv_variant(
    const PointwiseRequest& request,
    unsigned int block_size) {
  AttentionSignatureBuilder builder;
  builder.tensor(attention_tensor(request, "q"), "q");
  builder.tensor(attention_tensor(request, "k"), "k");
  append_attention_bias_pointer(builder, request);
  builder.tensor(attention_tensor(request, "do"), "do");
  builder.tensor(attention_tensor(request, "stats"), "stats");
  builder.tensor(attention_tensor(request, "dv"), "dv");
  builder.scalar_f32("attn_scale", request.attention_scale);
  builder.constant(request.attention_value_heads);
  builder.scalar_i32("SQ", request.attention_sequence_q);
  builder.scalar_i32("SKV", request.attention_sequence_kv);
  builder.scalar_i32("min_diag", request.attention_min_diag);
  builder.scalar_i32("max_diag", request.attention_max_diag);
  append_attention_qkv_bias_strides(builder, request, false);
  append_attention_strides4(builder, attention_tensor(request, "do"));
  append_attention_stats_strides(
      builder, attention_tensor(request, "stats"));
  append_attention_strides4(builder, attention_tensor(request, "dv"));
  builder.constant(request.attention_head_dimension);
  builder.constant(request.attention_value_dimension);
  builder.constant(request.attention_q_per_v);
  builder.constant(static_cast<std::int64_t>(block_size));
  builder.constant(static_cast<std::int64_t>(block_size));
  builder.constant(static_cast<std::int64_t>(
      attention_full_dimension_block(request.attention_head_dimension)));
  const unsigned int block_dv_out = attention_output_dimension_block(
      request.attention_value_dimension, block_size);
  builder.constant(static_cast<std::int64_t>(block_dv_out));
  builder.constant(false);
  builder.constant(request.attention_has_bias);
  builder.constant(request.attention_banded);
  builder.constant(request.attention_causal_top_left);
  return std::move(builder).finish(attention_grid(
      attention_ceil_div(request.attention_sequence_kv, block_size),
      attention_ceil_div(
          request.attention_value_dimension, block_dv_out),
      static_cast<std::uint64_t>(request.attention_batch) *
      static_cast<std::uint64_t>(request.attention_value_heads)));
}

AttentionVariantExpectation expected_attention_fixed_variant(
    const PointwiseRequest& request,
    std::string_view function) {
  AttentionSignatureBuilder builder;
  if (function == "_zero_contiguous_kernel") {
    const TensorSpec& dbias = attention_tensor(request, "dbias");
    const std::int64_t elements = element_count(dbias);
    builder.tensor(dbias, "dbias");
    builder.scalar_i32("n_elements", elements);
    builder.constant(std::int64_t{256});
    return std::move(builder).finish(
        attention_grid(attention_ceil_div(elements, 256), 1, 1));
  }
  if (function == "_zero_sdpa_fp8_fwd_amax_kernel") {
    builder.tensor(attention_tensor(request, "amax_s"), "amax_s");
    builder.tensor(attention_tensor(request, "amax_o"), "amax_o");
    return std::move(builder).finish({1, 1, 1});
  }
  if (function == "_zero_sdpa_fp8_bwd_amax_kernel") {
    for (const std::string_view port :
         {"amax_dq", "amax_dk", "amax_dv", "amax_dp"}) {
      builder.tensor(attention_tensor(request, port), port);
    }
    return std::move(builder).finish({1, 1, 1});
  }
  artifact_error("mthreads Attention fixed-stage function is invalid");
}

AttentionVariantExpectation expected_attention_fp8_dq_variant(
    const PointwiseRequest& request,
    unsigned int block_size) {
  AttentionSignatureBuilder builder;
  for (const std::string_view port :
       {"q",          "k",          "v",         "o",
        "do",         "stats",      "dq",        "amax_dq",
        "descale_q",  "descale_k",  "descale_v", "descale_o",
        "descale_do", "descale_dp", "scale_dq",  "scale_dp"}) {
    builder.tensor(attention_tensor(request, port), port);
  }
  builder.scalar_f32("attn_scale", request.attention_scale);
  builder.scalar_i32("HQ", request.attention_heads);
  builder.scalar_i32("SQ", request.attention_sequence_q);
  builder.scalar_i32("SKV", request.attention_sequence_kv);
  builder.constant(request.attention_q_per_k);
  builder.constant(request.attention_q_per_v);
  builder.scalar_i32("min_diag", request.attention_min_diag);
  builder.scalar_i32("max_diag", request.attention_max_diag);
  append_attention_strides4(builder, attention_tensor(request, "q"));
  append_attention_strides4(builder, attention_tensor(request, "k"));
  append_attention_strides4(builder, attention_tensor(request, "v"));
  append_attention_strides4(builder, attention_tensor(request, "o"));
  append_attention_strides4(builder, attention_tensor(request, "do"));
  append_attention_stats_strides(
      builder, attention_tensor(request, "stats"));
  append_attention_strides4(builder, attention_tensor(request, "dq"));
  builder.constant(request.attention_head_dimension);
  builder.constant(static_cast<std::int64_t>(block_size));
  builder.constant(static_cast<std::int64_t>(block_size));
  builder.constant(static_cast<std::int64_t>(
      attention_full_dimension_block(request.attention_head_dimension)));
  builder.constant(request.attention_banded);
  builder.constant(false);
  builder.constant(request.attention_causal_top_left);
  return std::move(builder).finish(attention_grid(
      attention_ceil_div(request.attention_sequence_q, block_size),
      static_cast<std::uint64_t>(request.attention_batch) *
          static_cast<std::uint64_t>(request.attention_heads),
      1));
}

AttentionVariantExpectation expected_attention_fp8_dkdv_variant(
    const PointwiseRequest& request,
    unsigned int block_size) {
  AttentionSignatureBuilder builder;
  for (const std::string_view port :
       {"q",          "k",          "v",          "o",
        "do",         "stats",      "dk",         "dv",
        "amax_dk",    "amax_dv",    "amax_dp",    "descale_q",
        "descale_k",  "descale_v",  "descale_o",  "descale_do",
        "descale_s",  "descale_dp", "scale_s",    "scale_dk",
        "scale_dv",   "scale_dp"}) {
    builder.tensor(attention_tensor(request, port), port);
  }
  builder.scalar_f32("attn_scale", request.attention_scale);
  builder.constant(request.attention_key_heads);
  builder.scalar_i32("SQ", request.attention_sequence_q);
  builder.scalar_i32("SKV", request.attention_sequence_kv);
  builder.scalar_i32("min_diag", request.attention_min_diag);
  builder.scalar_i32("max_diag", request.attention_max_diag);
  append_attention_strides4(builder, attention_tensor(request, "q"));
  append_attention_strides4(builder, attention_tensor(request, "k"));
  append_attention_strides4(builder, attention_tensor(request, "v"));
  append_attention_strides4(builder, attention_tensor(request, "o"));
  append_attention_strides4(builder, attention_tensor(request, "do"));
  append_attention_stats_strides(
      builder, attention_tensor(request, "stats"));
  append_attention_strides4(builder, attention_tensor(request, "dk"));
  append_attention_strides4(builder, attention_tensor(request, "dv"));
  builder.constant(request.attention_head_dimension);
  builder.constant(request.attention_q_per_k);
  builder.constant(static_cast<std::int64_t>(block_size));
  builder.constant(static_cast<std::int64_t>(block_size));
  builder.constant(static_cast<std::int64_t>(
      attention_full_dimension_block(request.attention_head_dimension)));
  builder.constant(request.attention_banded);
  builder.constant(false);
  builder.constant(request.attention_causal_top_left);
  return std::move(builder).finish(attention_grid(
      attention_ceil_div(request.attention_sequence_kv, block_size),
      static_cast<std::uint64_t>(request.attention_batch) *
          static_cast<std::uint64_t>(request.attention_key_heads),
      1));
}

AttentionVariantExpectation expected_attention_variant(
    const PointwiseRequest& request,
    std::string_view function,
    unsigned int block_size,
    bool fixed) {
  if (fixed) {
    return expected_attention_fixed_variant(request, function);
  }
  if (function == "_sdpa_fwd_kernel") {
    return expected_attention_forward_variant(
        request, block_size, false);
  }
  if (function == "_sdpa_fp8_fwd_kernel") {
    return expected_attention_forward_variant(request, block_size, true);
  }
  if (function == "_sdpa_bwd_dq_dbias_kernel") {
    return expected_attention_dq_variant(request, block_size);
  }
  if (function == "_sdpa_bwd_dkdv_kernel") {
    return expected_attention_dkdv_variant(request, block_size);
  }
  if (function == "_sdpa_bwd_dk_kernel") {
    return expected_attention_dk_variant(request, block_size);
  }
  if (function == "_sdpa_bwd_dv_kernel") {
    return expected_attention_dv_variant(request, block_size);
  }
  if (function == "_sdpa_fp8_bwd_dq_kernel") {
    return expected_attention_fp8_dq_variant(request, block_size);
  }
  if (function == "_sdpa_fp8_bwd_dkdv_kernel") {
    return expected_attention_fp8_dkdv_variant(request, block_size);
  }
  artifact_error("mthreads Attention kernel function is invalid");
}

void parse_attention_program(
    const PointwiseRequest& request,
    const JsonValue& program,
    const std::filesystem::path& source,
    std::string_view source_hash,
    std::string_view expected_source_path,
    const std::filesystem::path& canonical_root,
    MthreadsArtifact& result) {
  require_exact_keys(
      program, {"schema_version", "stage_count", "stages"}, "program");
  const auto expected_stages = expected_attention_stages(request);
  const auto& stage_values = program.at("stages").as_array();
  const std::size_t stage_count =
      checked_size(program.at("stage_count").as_int(), "stage_count");
  if (program.at("schema_version").as_int() !=
          kMthreadsExecutionProgramVersion ||
      stage_count == 0 || stage_count > kMaximumStages ||
      stage_count != expected_stages.size() ||
      stage_values.size() != stage_count) {
    artifact_error("mthreads Attention stage count is invalid");
  }

  result.stages.reserve(stage_count);
  for (std::size_t stage_index = 0; stage_index < stage_count;
       ++stage_index) {
    const JsonValue& stage_value = stage_values[stage_index];
    const ExpectedAttentionStage& expected_stage =
        expected_stages[stage_index];
    require_exact_keys(
        stage_value,
        {"id", "node_id", "operation", "dependencies", "source",
         "function", "variants", "autotune"},
        "Attention stage");
    StageArtifact stage;
    stage.id = checked_size(stage_value.at("id").as_int(), "stage.id");
    stage.node_id = stage_value.at("node_id").as_int();
    stage.function_name = stage_value.at("function").as_string();
    if (stage.id != stage_index || stage.node_id != request.node_id ||
        stage_value.at("operation").as_string() != request.operation ||
        stage_value.at("source").as_string() != expected_source_path ||
        stage.function_name != expected_stage.function ||
        !printable_ascii_identifier(stage.function_name, 256)) {
      artifact_error("mthreads Attention stage identity is invalid");
    }
    stage.source = source;

    const auto& dependencies =
        stage_value.at("dependencies").as_array();
    if (dependencies.size() != expected_stage.dependencies.size()) {
      artifact_error("mthreads Attention stage dependencies differ");
    }
    for (std::size_t index = 0; index < dependencies.size(); ++index) {
      const std::size_t dependency = checked_size(
          dependencies[index].as_int(), "Attention dependency");
      if (dependency != expected_stage.dependencies[index] ||
          dependency >= stage_index) {
        artifact_error("mthreads Attention stage dependency is invalid");
      }
      stage.dependencies.push_back(dependency);
    }

    const JsonValue& autotune = stage_value.at("autotune");
    require_exact_keys(
        autotune,
        {"enabled", "warmup", "repetitions", "selection_cache"},
        "Attention autotune");
    stage.autotune = autotune.at("enabled").as_bool();
    stage.warmup = checked_positive_unsigned(
        autotune.at("warmup").as_int(), "Attention warmup");
    stage.repetitions = checked_positive_unsigned(
        autotune.at("repetitions").as_int(), "Attention repetitions");
    const std::string expected_cache =
        "tuning/stage-" + std::to_string(stage_index) + ".json";
    const bool expected_autotune =
        !expected_stage.fixed && request.autotune;
    if (stage.autotune != expected_autotune || stage.warmup != 3 ||
        stage.repetitions != 10 ||
        autotune.at("selection_cache").as_string() != expected_cache) {
      artifact_error("mthreads Attention autotune metadata differs");
    }
    stage.selection_cache = canonical_root / expected_cache;

    const auto candidates = expected_attention_candidates(
        request, expected_stage.fixed);
    const auto& variants = stage_value.at("variants").as_array();
    if (variants.empty() || variants.size() > kMaximumVariants ||
        variants.size() != candidates.size()) {
      artifact_error("mthreads Attention variant count is invalid");
    }
    std::set<std::string> variant_ids;
    stage.variants.reserve(variants.size());
    for (std::size_t variant_index = 0;
         variant_index < variants.size(); ++variant_index) {
      const JsonValue& variant_value = variants[variant_index];
      const Candidate candidate = candidates[variant_index];
      require_exact_keys(
          variant_value,
          {"variant_id", "source", "source_sha256", "function",
           "full_signature", "grid", "num_warps", "num_stages",
           "arguments"},
          "Attention variant");
      KernelVariantArtifact variant;
      variant.variant_id = variant_value.at("variant_id").as_string();
      variant.function_name = variant_value.at("function").as_string();
      variant.full_signature =
          variant_value.at("full_signature").as_string();
      variant.grid = parse_grid(variant_value.at("grid"));
      variant.num_warps = checked_positive_unsigned(
          variant_value.at("num_warps").as_int(), "Attention num_warps");
      variant.num_stages = checked_positive_unsigned(
          variant_value.at("num_stages").as_int(), "Attention num_stages");
      const std::string expected_variant_id =
          expected_stage.fixed
              ? "fixed-warps-4-stages-1"
              : "block-" + std::to_string(candidate.block_size) +
                    "-warps-" + std::to_string(candidate.num_warps) +
                    "-stages-" + std::to_string(candidate.num_stages);
      const AttentionVariantExpectation expectation =
          expected_attention_variant(
              request,
              expected_stage.function,
              candidate.block_size,
              expected_stage.fixed);
      if (variant.variant_id != expected_variant_id ||
          !variant_ids.insert(variant.variant_id).second ||
          variant_value.at("source").as_string() != expected_source_path ||
          variant_value.at("source_sha256").as_string() != source_hash ||
          variant.function_name != expected_stage.function ||
          variant.full_signature != expectation.signature ||
          variant.grid != expectation.grid ||
          variant.num_warps != candidate.num_warps ||
          variant.num_stages != candidate.num_stages) {
        artifact_error(
            "mthreads Attention variant launch/signature differs");
      }
      variant.source = source;

      const auto& arguments =
          variant_value.at("arguments").as_array();
      if (arguments.size() > kMaximumArguments ||
          arguments.size() != expectation.arguments.size()) {
        artifact_error("mthreads Attention argument count differs");
      }
      for (std::size_t argument_index = 0;
           argument_index < arguments.size(); ++argument_index) {
        const ExpectedAttentionArgument& expected_argument =
            expectation.arguments[argument_index];
        const std::int64_t expected_value =
            expected_argument.kind == ArgumentKind::kTensor
                ? expected_argument.tensor->uid
                : expected_argument.scalar_value;
        variant.arguments.push_back(parse_expected_argument(
            arguments[argument_index],
            expected_argument.kind,
            expected_argument.semantic_name,
            expected_value,
            expected_argument.scalar_bits));
        ArgumentSpec& parsed = variant.arguments.back();
        if (expected_argument.kind == ArgumentKind::kTensor) {
          parsed.storage_size = expected_argument.tensor->storage_size;
          parsed.alignment = expected_argument.tensor->alignment;
          parsed.data_type = expected_argument.tensor->data_type;
        } else if (expected_argument.kind == ArgumentKind::kWorkspace) {
          parsed.storage_size = result.workspace_size;
          parsed.alignment = result.workspace_alignment;
        }
      }
      stage.variants.push_back(std::move(variant));
    }
    result.stages.push_back(std::move(stage));
  }
}

void parse_im2col_fprop_program(
    const PointwiseRequest& request,
    const JsonValue& program,
    const std::filesystem::path& source,
    std::string_view source_hash,
    std::string_view expected_source_path,
    const std::filesystem::path& canonical_root,
    MthreadsArtifact& result) {
  require_exact_keys(
      program, {"schema_version", "stage_count", "stages"}, "program");
  const auto& stage_values = program.at("stages").as_array();
  const std::size_t stage_count =
      checked_size(program.at("stage_count").as_int(), "stage_count");
  if (program.at("schema_version").as_int() !=
          kMthreadsExecutionProgramVersion ||
      stage_count != 2 || stage_values.size() != stage_count) {
    artifact_error("mthreads im2col Fprop stage count is invalid");
  }

  constexpr std::array<std::string_view, 2> kFunctions = {
      "_conv_fprop2d_im2col_kernel",
      "_conv_fprop2d_im2col_mm_kernel"};
  result.stages.reserve(stage_count);
  for (std::size_t stage_index = 0; stage_index < stage_count;
       ++stage_index) {
    const JsonValue& stage_value = stage_values[stage_index];
    const std::string_view expected_function = kFunctions[stage_index];
    require_exact_keys(
        stage_value,
        {"id", "node_id", "operation", "dependencies", "source",
         "function", "variants", "autotune"},
        "im2col Fprop stage");
    StageArtifact stage;
    stage.id = checked_size(stage_value.at("id").as_int(), "stage.id");
    stage.node_id = stage_value.at("node_id").as_int();
    stage.function_name = stage_value.at("function").as_string();
    if (stage.id != stage_index || stage.node_id != request.node_id ||
        stage_value.at("operation").as_string() != request.operation ||
        stage_value.at("source").as_string() != expected_source_path ||
        stage.function_name != expected_function ||
        !printable_ascii_identifier(stage.function_name, 256)) {
      artifact_error("mthreads im2col Fprop stage identity is invalid");
    }
    stage.source = source;

    const auto& dependencies =
        stage_value.at("dependencies").as_array();
    const std::size_t expected_dependency_count =
        stage_index == 0 ? 0 : 1;
    if (dependencies.size() != expected_dependency_count) {
      artifact_error("mthreads im2col Fprop dependencies differ");
    }
    if (stage_index == 1) {
      const std::size_t dependency = checked_size(
          dependencies.front().as_int(), "im2col Fprop dependency");
      if (dependency != 0) {
        artifact_error("mthreads im2col Fprop dependency is invalid");
      }
      stage.dependencies.push_back(dependency);
    }

    const JsonValue& autotune = stage_value.at("autotune");
    require_exact_keys(
        autotune,
        {"enabled", "warmup", "repetitions", "selection_cache"},
        "im2col Fprop autotune");
    stage.autotune = autotune.at("enabled").as_bool();
    stage.warmup = checked_positive_unsigned(
        autotune.at("warmup").as_int(), "im2col Fprop warmup");
    stage.repetitions = checked_positive_unsigned(
        autotune.at("repetitions").as_int(),
        "im2col Fprop repetitions");
    const std::string expected_cache =
        "tuning/stage-" + std::to_string(stage_index) + ".json";
    if (stage.autotune || stage.warmup != 3 ||
        stage.repetitions != 10 ||
        autotune.at("selection_cache").as_string() != expected_cache) {
      artifact_error("mthreads im2col Fprop autotune metadata differs");
    }
    stage.selection_cache = canonical_root / expected_cache;

    const auto candidates =
        expected_im2col_fprop_candidates(stage_index);
    const auto& variants = stage_value.at("variants").as_array();
    if (variants.size() != 1 || variants.size() != candidates.size()) {
      artifact_error("mthreads im2col Fprop variant count is invalid");
    }
    const JsonValue& variant_value = variants.front();
    const Candidate candidate = candidates.front();
    require_exact_keys(
        variant_value,
        {"variant_id", "source", "source_sha256", "function",
         "full_signature", "grid", "num_warps", "num_stages",
         "arguments"},
        "im2col Fprop variant");
    KernelVariantArtifact variant;
    variant.variant_id = variant_value.at("variant_id").as_string();
    variant.function_name = variant_value.at("function").as_string();
    variant.full_signature =
        variant_value.at("full_signature").as_string();
    variant.grid = parse_grid(variant_value.at("grid"));
    variant.num_warps = checked_positive_unsigned(
        variant_value.at("num_warps").as_int(),
        "im2col Fprop num_warps");
    variant.num_stages = checked_positive_unsigned(
        variant_value.at("num_stages").as_int(),
        "im2col Fprop num_stages");
    const std::string expected_variant_id =
        "block-" + std::to_string(candidate.block_size) +
        "-warps-" + std::to_string(candidate.num_warps) +
        "-stages-" + std::to_string(candidate.num_stages);
    const AttentionVariantExpectation expectation =
        expected_im2col_fprop_variant(
            request, expected_function, candidate.block_size);
    if (variant.variant_id != expected_variant_id ||
        variant_value.at("source").as_string() != expected_source_path ||
        variant_value.at("source_sha256").as_string() != source_hash ||
        variant.function_name != expected_function ||
        variant.full_signature != expectation.signature ||
        variant.grid != expectation.grid ||
        variant.num_warps != candidate.num_warps ||
        variant.num_stages != candidate.num_stages) {
      artifact_error(
          "mthreads im2col Fprop variant launch/signature differs");
    }
    variant.source = source;

    const auto& arguments =
        variant_value.at("arguments").as_array();
    if (arguments.size() > kMaximumArguments ||
        arguments.size() != expectation.arguments.size()) {
      artifact_error("mthreads im2col Fprop argument count differs");
    }
    for (std::size_t argument_index = 0;
         argument_index < arguments.size(); ++argument_index) {
      const ExpectedAttentionArgument& expected_argument =
          expectation.arguments[argument_index];
      const std::int64_t expected_value =
          expected_argument.kind == ArgumentKind::kTensor
              ? expected_argument.tensor->uid
              : expected_argument.scalar_value;
      variant.arguments.push_back(parse_expected_argument(
          arguments[argument_index],
          expected_argument.kind,
          expected_argument.semantic_name,
          expected_value,
          expected_argument.scalar_bits));
      ArgumentSpec& parsed = variant.arguments.back();
      if (expected_argument.kind == ArgumentKind::kTensor) {
        parsed.storage_size = expected_argument.tensor->storage_size;
        parsed.alignment = expected_argument.tensor->alignment;
        parsed.data_type = expected_argument.tensor->data_type;
      } else if (expected_argument.kind == ArgumentKind::kWorkspace) {
        parsed.storage_size = result.workspace_size;
        parsed.alignment = result.workspace_alignment;
      }
    }
    stage.variants.push_back(std::move(variant));
    result.stages.push_back(std::move(stage));
  }
}

void parse_dense_dgrad_program(
    const PointwiseRequest& request,
    const JsonValue& program,
    const std::filesystem::path& source,
    std::string_view source_hash,
    std::string_view expected_source_path,
    const std::filesystem::path& canonical_root,
    MthreadsArtifact& result) {
  require_exact_keys(
      program, {"schema_version", "stage_count", "stages"}, "program");
  const auto& stage_values = program.at("stages").as_array();
  const std::size_t stage_count =
      checked_size(program.at("stage_count").as_int(), "stage_count");
  if (program.at("schema_version").as_int() !=
          kMthreadsExecutionProgramVersion ||
      stage_count != 3 || stage_values.size() != stage_count) {
    artifact_error("mthreads dense Dgrad stage count is invalid");
  }

  constexpr std::array<std::string_view, 3> kFunctions = {
      "_conv_dgrad2d_dense_pack_filter_kernel",
      "_conv_dgrad2d_dense_pack_loss_kernel",
      "_conv_dgrad2d_dense_mm_kernel"};
  result.stages.reserve(stage_count);
  for (std::size_t stage_index = 0; stage_index < stage_count;
       ++stage_index) {
    const JsonValue& stage_value = stage_values[stage_index];
    const std::string_view expected_function = kFunctions[stage_index];
    require_exact_keys(
        stage_value,
        {"id", "node_id", "operation", "dependencies", "source",
         "function", "variants", "autotune"},
        "dense Dgrad stage");
    StageArtifact stage;
    stage.id = checked_size(stage_value.at("id").as_int(), "stage.id");
    stage.node_id = stage_value.at("node_id").as_int();
    stage.function_name = stage_value.at("function").as_string();
    if (stage.id != stage_index || stage.node_id != request.node_id ||
        stage_value.at("operation").as_string() != request.operation ||
        stage_value.at("source").as_string() != expected_source_path ||
        stage.function_name != expected_function ||
        !printable_ascii_identifier(stage.function_name, 256)) {
      artifact_error("mthreads dense Dgrad stage identity is invalid");
    }
    stage.source = source;

    const auto& dependencies =
        stage_value.at("dependencies").as_array();
    const std::size_t expected_dependency_count =
        stage_index == 2 ? 2 : 0;
    if (dependencies.size() != expected_dependency_count) {
      artifact_error("mthreads dense Dgrad stage dependencies differ");
    }
    if (stage_index == 2) {
      for (std::size_t dependency_index = 0;
           dependency_index < dependencies.size(); ++dependency_index) {
        const std::size_t dependency = checked_size(
            dependencies[dependency_index].as_int(),
            "dense Dgrad dependency");
        if (dependency != dependency_index || dependency >= stage_index) {
          artifact_error(
              "mthreads dense Dgrad stage dependency is invalid");
        }
        stage.dependencies.push_back(dependency);
      }
    }

    const JsonValue& autotune = stage_value.at("autotune");
    require_exact_keys(
        autotune,
        {"enabled", "warmup", "repetitions", "selection_cache"},
        "dense Dgrad autotune");
    stage.autotune = autotune.at("enabled").as_bool();
    stage.warmup = checked_positive_unsigned(
        autotune.at("warmup").as_int(), "dense Dgrad warmup");
    stage.repetitions = checked_positive_unsigned(
        autotune.at("repetitions").as_int(),
        "dense Dgrad repetitions");
    const std::string expected_cache =
        "tuning/stage-" + std::to_string(stage_index) + ".json";
    const bool expected_autotune =
        stage_index == 2 && request.autotune;
    if (stage.autotune != expected_autotune || stage.warmup != 3 ||
        stage.repetitions != 10 ||
        autotune.at("selection_cache").as_string() != expected_cache) {
      artifact_error("mthreads dense Dgrad autotune metadata differs");
    }
    stage.selection_cache = canonical_root / expected_cache;

    const auto candidates =
        expected_dense_dgrad_candidates(request, stage_index);
    const auto& variants = stage_value.at("variants").as_array();
    if (variants.empty() || variants.size() > kMaximumVariants ||
        variants.size() != candidates.size()) {
      artifact_error("mthreads dense Dgrad variant count is invalid");
    }
    std::set<std::string> variant_ids;
    stage.variants.reserve(variants.size());
    for (std::size_t variant_index = 0;
         variant_index < variants.size(); ++variant_index) {
      const JsonValue& variant_value = variants[variant_index];
      const Candidate candidate = candidates[variant_index];
      require_exact_keys(
          variant_value,
          {"variant_id", "source", "source_sha256", "function",
           "full_signature", "grid", "num_warps", "num_stages",
           "arguments"},
          "dense Dgrad variant");
      KernelVariantArtifact variant;
      variant.variant_id = variant_value.at("variant_id").as_string();
      variant.function_name = variant_value.at("function").as_string();
      variant.full_signature =
          variant_value.at("full_signature").as_string();
      variant.grid = parse_grid(variant_value.at("grid"));
      variant.num_warps = checked_positive_unsigned(
          variant_value.at("num_warps").as_int(),
          "dense Dgrad num_warps");
      variant.num_stages = checked_positive_unsigned(
          variant_value.at("num_stages").as_int(),
          "dense Dgrad num_stages");
      const std::string expected_variant_id =
          "block-" + std::to_string(candidate.block_size) +
          "-warps-" + std::to_string(candidate.num_warps) +
          "-stages-" + std::to_string(candidate.num_stages);
      const AttentionVariantExpectation expectation =
          expected_dense_dgrad_variant(
              request, expected_function, candidate.block_size);
      if (variant.variant_id != expected_variant_id ||
          !variant_ids.insert(variant.variant_id).second ||
          variant_value.at("source").as_string() != expected_source_path ||
          variant_value.at("source_sha256").as_string() != source_hash ||
          variant.function_name != expected_function ||
          variant.full_signature != expectation.signature ||
          variant.grid != expectation.grid ||
          variant.num_warps != candidate.num_warps ||
          variant.num_stages != candidate.num_stages) {
        artifact_error(
            "mthreads dense Dgrad variant launch/signature differs");
      }
      variant.source = source;

      const auto& arguments =
          variant_value.at("arguments").as_array();
      if (arguments.size() > kMaximumArguments ||
          arguments.size() != expectation.arguments.size()) {
        artifact_error("mthreads dense Dgrad argument count differs");
      }
      for (std::size_t argument_index = 0;
           argument_index < arguments.size(); ++argument_index) {
        const ExpectedAttentionArgument& expected_argument =
            expectation.arguments[argument_index];
        const std::int64_t expected_value =
            expected_argument.kind == ArgumentKind::kTensor
                ? expected_argument.tensor->uid
                : expected_argument.scalar_value;
        variant.arguments.push_back(parse_expected_argument(
            arguments[argument_index],
            expected_argument.kind,
            expected_argument.semantic_name,
            expected_value,
            expected_argument.scalar_bits));
        ArgumentSpec& parsed = variant.arguments.back();
        if (expected_argument.kind == ArgumentKind::kTensor) {
          parsed.storage_size = expected_argument.tensor->storage_size;
          parsed.alignment = expected_argument.tensor->alignment;
          parsed.data_type = expected_argument.tensor->data_type;
        } else if (expected_argument.kind == ArgumentKind::kWorkspace) {
          parsed.storage_size = result.workspace_size;
          parsed.alignment = result.workspace_alignment;
        }
      }
      stage.variants.push_back(std::move(variant));
    }
    result.stages.push_back(std::move(stage));
  }
}

void parse_p5_wgrad_program(
    const PointwiseRequest& request,
    const JsonValue& program,
    const std::filesystem::path& source,
    std::string_view source_hash,
    std::string_view expected_source_path,
    const std::filesystem::path& canonical_root,
    MthreadsArtifact& result) {
  require_exact_keys(
      program, {"schema_version", "stage_count", "stages"}, "program");
  const auto& stage_values = program.at("stages").as_array();
  const std::size_t stage_count =
      checked_size(program.at("stage_count").as_int(), "stage_count");
  if (program.at("schema_version").as_int() !=
          kMthreadsExecutionProgramVersion ||
      stage_count != 2 || stage_values.size() != stage_count) {
    artifact_error("mthreads P5 Wgrad stage count is invalid");
  }

  constexpr std::array<std::string_view, 2> kFunctions = {
      "_conv_wgrad2d_p5_pack_image_kernel",
      "_conv_wgrad2d_p5_mm_kernel"};
  result.stages.reserve(stage_count);
  for (std::size_t stage_index = 0; stage_index < stage_count;
       ++stage_index) {
    const JsonValue& stage_value = stage_values[stage_index];
    const std::string_view expected_function = kFunctions[stage_index];
    require_exact_keys(
        stage_value,
        {"id", "node_id", "operation", "dependencies", "source",
         "function", "variants", "autotune"},
        "P5 Wgrad stage");
    StageArtifact stage;
    stage.id = checked_size(stage_value.at("id").as_int(), "stage.id");
    stage.node_id = stage_value.at("node_id").as_int();
    stage.function_name = stage_value.at("function").as_string();
    if (stage.id != stage_index || stage.node_id != request.node_id ||
        stage_value.at("operation").as_string() != request.operation ||
        stage_value.at("source").as_string() != expected_source_path ||
        stage.function_name != expected_function ||
        !printable_ascii_identifier(stage.function_name, 256)) {
      artifact_error("mthreads P5 Wgrad stage identity is invalid");
    }
    stage.source = source;

    const auto& dependencies =
        stage_value.at("dependencies").as_array();
    const std::size_t expected_dependency_count = stage_index == 0 ? 0 : 1;
    if (dependencies.size() != expected_dependency_count) {
      artifact_error("mthreads P5 Wgrad stage dependencies differ");
    }
    if (stage_index == 1) {
      const std::size_t dependency = checked_size(
          dependencies.front().as_int(), "P5 Wgrad dependency");
      if (dependency != 0) {
        artifact_error("mthreads P5 Wgrad stage dependency is invalid");
      }
      stage.dependencies.push_back(dependency);
    }

    const JsonValue& autotune = stage_value.at("autotune");
    require_exact_keys(
        autotune,
        {"enabled", "warmup", "repetitions", "selection_cache"},
        "P5 Wgrad autotune");
    stage.autotune = autotune.at("enabled").as_bool();
    stage.warmup = checked_positive_unsigned(
        autotune.at("warmup").as_int(), "P5 Wgrad warmup");
    stage.repetitions = checked_positive_unsigned(
        autotune.at("repetitions").as_int(), "P5 Wgrad repetitions");
    const std::string expected_cache =
        "tuning/stage-" + std::to_string(stage_index) + ".json";
    if (stage.autotune != request.autotune || stage.warmup != 3 ||
        stage.repetitions != 10 ||
        autotune.at("selection_cache").as_string() != expected_cache) {
      artifact_error("mthreads P5 Wgrad autotune metadata differs");
    }
    stage.selection_cache = canonical_root / expected_cache;

    auto candidates = expected_candidates(request);
    if (stage_index == 1 && request.autotune &&
        request.convolution_image.data_type != "float32") {
      for (const unsigned int num_warps : {4U, 8U}) {
        for (const unsigned int num_stages : {1U, 2U}) {
          candidates.push_back({64U, num_warps, num_stages});
        }
      }
    }
    const auto& variants = stage_value.at("variants").as_array();
    if (variants.empty() || variants.size() > kMaximumVariants ||
        variants.size() != candidates.size()) {
      artifact_error("mthreads P5 Wgrad variant count is invalid");
    }
    std::set<std::string> variant_ids;
    stage.variants.reserve(variants.size());
    for (std::size_t variant_index = 0;
         variant_index < variants.size(); ++variant_index) {
      const JsonValue& variant_value = variants[variant_index];
      const Candidate candidate = candidates[variant_index];
      require_exact_keys(
          variant_value,
          {"variant_id", "source", "source_sha256", "function",
           "full_signature", "grid", "num_warps", "num_stages",
           "arguments"},
          "P5 Wgrad variant");
      KernelVariantArtifact variant;
      variant.variant_id = variant_value.at("variant_id").as_string();
      variant.function_name = variant_value.at("function").as_string();
      variant.full_signature =
          variant_value.at("full_signature").as_string();
      variant.grid = parse_grid(variant_value.at("grid"));
      variant.num_warps = checked_positive_unsigned(
          variant_value.at("num_warps").as_int(),
          "P5 Wgrad num_warps");
      variant.num_stages = checked_positive_unsigned(
          variant_value.at("num_stages").as_int(),
          "P5 Wgrad num_stages");
      const std::string expected_variant_id =
          "block-" + std::to_string(candidate.block_size) +
          "-warps-" + std::to_string(candidate.num_warps) +
          "-stages-" + std::to_string(candidate.num_stages);
      const AttentionVariantExpectation expectation =
          expected_p5_wgrad_variant(
              request, expected_function, candidate.block_size);
      if (variant.variant_id != expected_variant_id ||
          !variant_ids.insert(variant.variant_id).second ||
          variant_value.at("source").as_string() != expected_source_path ||
          variant_value.at("source_sha256").as_string() != source_hash ||
          variant.function_name != expected_function ||
          variant.full_signature != expectation.signature ||
          variant.grid != expectation.grid ||
          variant.num_warps != candidate.num_warps ||
          variant.num_stages != candidate.num_stages) {
        artifact_error("mthreads P5 Wgrad variant launch/signature differs");
      }
      variant.source = source;

      const auto& arguments =
          variant_value.at("arguments").as_array();
      if (arguments.size() > kMaximumArguments ||
          arguments.size() != expectation.arguments.size()) {
        artifact_error("mthreads P5 Wgrad argument count differs");
      }
      for (std::size_t argument_index = 0;
           argument_index < arguments.size(); ++argument_index) {
        const ExpectedAttentionArgument& expected_argument =
            expectation.arguments[argument_index];
        const std::int64_t expected_value =
            expected_argument.kind == ArgumentKind::kTensor
                ? expected_argument.tensor->uid
                : expected_argument.scalar_value;
        variant.arguments.push_back(parse_expected_argument(
            arguments[argument_index],
            expected_argument.kind,
            expected_argument.semantic_name,
            expected_value,
            expected_argument.scalar_bits));
        ArgumentSpec& parsed = variant.arguments.back();
        if (expected_argument.kind == ArgumentKind::kTensor) {
          parsed.storage_size = expected_argument.tensor->storage_size;
          parsed.alignment = expected_argument.tensor->alignment;
          parsed.data_type = expected_argument.tensor->data_type;
        } else if (expected_argument.kind == ArgumentKind::kWorkspace) {
          parsed.storage_size = result.workspace_size;
          parsed.alignment = result.workspace_alignment;
        }
      }
      stage.variants.push_back(std::move(variant));
    }
    result.stages.push_back(std::move(stage));
  }
}

void parse_standard_wgrad_program(
    const PointwiseRequest& request,
    const JsonValue& program,
    const std::filesystem::path& source,
    std::string_view source_hash,
    std::string_view expected_source_path,
    const std::filesystem::path& canonical_root,
    MthreadsArtifact& result) {
  require_exact_keys(
      program, {"schema_version", "stage_count", "stages"}, "program");
  const auto& stage_values = program.at("stages").as_array();
  const std::size_t stage_count =
      checked_size(program.at("stage_count").as_int(), "stage_count");
  const bool one_by_one =
      request.convolution_filter.dimensions.size() == 4 &&
      request.convolution_filter.dimensions[2] == 1 &&
      request.convolution_filter.dimensions[3] == 1;
  const bool nd_packed = uses_nd_packed_wgrad(request);
  const std::size_t expected_stage_count = one_by_one ? 2 : 3;
  if (program.at("schema_version").as_int() !=
          kMthreadsExecutionProgramVersion ||
      stage_count != expected_stage_count ||
      stage_values.size() != stage_count) {
    artifact_error("mthreads standard Wgrad stage count is invalid");
  }

  std::vector<std::string_view> functions;
  if (one_by_one) {
    functions = {
        "_conv_wgrad2d_1x1_split_kernel",
        "_conv_wgrad2d_stem_reduce_kernel"};
  } else if (nd_packed) {
    functions = {
        "_conv_wgrad_nd_im2row_kernel",
        "_conv_wgrad_nd_rowmajor_kernel",
        "_conv_wgrad_nd_reduce_kernel"};
  } else {
    functions = {
        "_conv_wgrad2d_im2row_kernel",
        "_conv_wgrad2d_rowmajor_kernel",
        "_conv_wgrad2d_stem_reduce_kernel"};
  }
  result.stages.reserve(stage_count);
  for (std::size_t stage_index = 0; stage_index < stage_count;
       ++stage_index) {
    const JsonValue& stage_value = stage_values[stage_index];
    const std::string_view expected_function = functions[stage_index];
    require_exact_keys(
        stage_value,
        {"id", "node_id", "operation", "dependencies", "source",
         "function", "variants", "autotune"},
        "standard Wgrad stage");
    StageArtifact stage;
    stage.id = checked_size(stage_value.at("id").as_int(), "stage.id");
    stage.node_id = stage_value.at("node_id").as_int();
    stage.function_name = stage_value.at("function").as_string();
    if (stage.id != stage_index || stage.node_id != request.node_id ||
        stage_value.at("operation").as_string() != request.operation ||
        stage_value.at("source").as_string() != expected_source_path ||
        stage.function_name != expected_function ||
        !printable_ascii_identifier(stage.function_name, 256)) {
      artifact_error("mthreads standard Wgrad stage identity is invalid");
    }
    stage.source = source;

    const auto& dependencies =
        stage_value.at("dependencies").as_array();
    const std::size_t expected_dependency_count = stage_index == 0 ? 0 : 1;
    if (dependencies.size() != expected_dependency_count) {
      artifact_error("mthreads standard Wgrad stage dependencies differ");
    }
    if (stage_index > 0) {
      const std::size_t dependency = checked_size(
          dependencies.front().as_int(), "standard Wgrad dependency");
      if (dependency != stage_index - 1) {
        artifact_error(
            "mthreads standard Wgrad stage dependency is invalid");
      }
      stage.dependencies.push_back(dependency);
    }

    const JsonValue& autotune = stage_value.at("autotune");
    require_exact_keys(
        autotune,
        {"enabled", "warmup", "repetitions", "selection_cache"},
        "standard Wgrad autotune");
    stage.autotune = autotune.at("enabled").as_bool();
    stage.warmup = checked_positive_unsigned(
        autotune.at("warmup").as_int(), "standard Wgrad warmup");
    stage.repetitions = checked_positive_unsigned(
        autotune.at("repetitions").as_int(),
        "standard Wgrad repetitions");
    const std::string expected_cache =
        "tuning/stage-" + std::to_string(stage_index) + ".json";
    if (stage.autotune != request.autotune || stage.warmup != 3 ||
        stage.repetitions != 10 ||
        autotune.at("selection_cache").as_string() != expected_cache) {
      artifact_error("mthreads standard Wgrad autotune metadata differs");
    }
    stage.selection_cache = canonical_root / expected_cache;

    const auto candidates =
        expected_standard_wgrad_candidates(request, stage_index);
    const auto& variants = stage_value.at("variants").as_array();
    if (variants.empty() || variants.size() > kMaximumVariants ||
        variants.size() != candidates.size()) {
      artifact_error("mthreads standard Wgrad variant count is invalid");
    }
    std::set<std::string> variant_ids;
    stage.variants.reserve(variants.size());
    for (std::size_t variant_index = 0;
         variant_index < variants.size(); ++variant_index) {
      const JsonValue& variant_value = variants[variant_index];
      const Candidate candidate = candidates[variant_index];
      require_exact_keys(
          variant_value,
          {"variant_id", "source", "source_sha256", "function",
           "full_signature", "grid", "num_warps", "num_stages",
           "arguments"},
          "standard Wgrad variant");
      KernelVariantArtifact variant;
      variant.variant_id = variant_value.at("variant_id").as_string();
      variant.function_name = variant_value.at("function").as_string();
      variant.full_signature =
          variant_value.at("full_signature").as_string();
      variant.grid = parse_grid(variant_value.at("grid"));
      variant.num_warps = checked_positive_unsigned(
          variant_value.at("num_warps").as_int(),
          "standard Wgrad num_warps");
      variant.num_stages = checked_positive_unsigned(
          variant_value.at("num_stages").as_int(),
          "standard Wgrad num_stages");
      const std::string expected_variant_id =
          "block-" + std::to_string(candidate.block_size) +
          "-warps-" + std::to_string(candidate.num_warps) +
          "-stages-" + std::to_string(candidate.num_stages);
      const AttentionVariantExpectation expectation = nd_packed
          ? expected_nd_packed_wgrad_variant(
                request, expected_function, candidate.block_size)
          : expected_standard_wgrad_variant(
                request, expected_function, candidate.block_size);
      if (variant.variant_id != expected_variant_id ||
          !variant_ids.insert(variant.variant_id).second ||
          variant_value.at("source").as_string() != expected_source_path ||
          variant_value.at("source_sha256").as_string() != source_hash ||
          variant.function_name != expected_function ||
          variant.full_signature != expectation.signature ||
          variant.grid != expectation.grid ||
          variant.num_warps != candidate.num_warps ||
          variant.num_stages != candidate.num_stages) {
        artifact_error(
            "mthreads standard Wgrad variant launch/signature differs");
      }
      variant.source = source;

      const auto& arguments =
          variant_value.at("arguments").as_array();
      if (arguments.size() > kMaximumArguments ||
          arguments.size() != expectation.arguments.size()) {
        artifact_error("mthreads standard Wgrad argument count differs");
      }
      for (std::size_t argument_index = 0;
           argument_index < arguments.size(); ++argument_index) {
        const ExpectedAttentionArgument& expected_argument =
            expectation.arguments[argument_index];
        const std::int64_t expected_value =
            expected_argument.kind == ArgumentKind::kTensor
                ? expected_argument.tensor->uid
                : expected_argument.scalar_value;
        variant.arguments.push_back(parse_expected_argument(
            arguments[argument_index],
            expected_argument.kind,
            expected_argument.semantic_name,
            expected_value,
            expected_argument.scalar_bits));
        ArgumentSpec& parsed = variant.arguments.back();
        if (expected_argument.kind == ArgumentKind::kTensor) {
          parsed.storage_size = expected_argument.tensor->storage_size;
          parsed.alignment = expected_argument.tensor->alignment;
          parsed.data_type = expected_argument.tensor->data_type;
        } else if (expected_argument.kind == ArgumentKind::kWorkspace) {
          parsed.storage_size = result.workspace_size;
          parsed.alignment = result.workspace_alignment;
        }
      }
      stage.variants.push_back(std::move(variant));
    }
    result.stages.push_back(std::move(stage));
  }
}

void parse_stem_wgrad_program(
    const PointwiseRequest& request,
    const JsonValue& program,
    const std::filesystem::path& source,
    std::string_view source_hash,
    std::string_view expected_source_path,
    const std::filesystem::path& canonical_root,
    MthreadsArtifact& result) {
  require_exact_keys(
      program, {"schema_version", "stage_count", "stages"}, "program");
  const auto& stage_values = program.at("stages").as_array();
  const std::size_t stage_count =
      checked_size(program.at("stage_count").as_int(), "stage_count");
  if (program.at("schema_version").as_int() !=
          kMthreadsExecutionProgramVersion ||
      stage_count != 2 || stage_values.size() != stage_count) {
    artifact_error("mthreads stem Wgrad stage count is invalid");
  }

  constexpr std::array<std::string_view, 2> kFunctions = {
      "_conv_wgrad2d_stem_split_kernel",
      "_conv_wgrad2d_stem_reduce_kernel"};
  result.stages.reserve(stage_count);
  for (std::size_t stage_index = 0; stage_index < stage_count;
       ++stage_index) {
    const JsonValue& stage_value = stage_values[stage_index];
    const std::string_view expected_function = kFunctions[stage_index];
    require_exact_keys(
        stage_value,
        {"id", "node_id", "operation", "dependencies", "source",
         "function", "variants", "autotune"},
        "stem Wgrad stage");
    StageArtifact stage;
    stage.id = checked_size(stage_value.at("id").as_int(), "stage.id");
    stage.node_id = stage_value.at("node_id").as_int();
    stage.function_name = stage_value.at("function").as_string();
    if (stage.id != stage_index || stage.node_id != request.node_id ||
        stage_value.at("operation").as_string() != request.operation ||
        stage_value.at("source").as_string() != expected_source_path ||
        stage.function_name != expected_function ||
        !printable_ascii_identifier(stage.function_name, 256)) {
      artifact_error("mthreads stem Wgrad stage identity is invalid");
    }
    stage.source = source;

    const auto& dependencies =
        stage_value.at("dependencies").as_array();
    const std::size_t expected_dependency_count = stage_index == 0 ? 0 : 1;
    if (dependencies.size() != expected_dependency_count) {
      artifact_error("mthreads stem Wgrad stage dependencies differ");
    }
    if (stage_index == 1) {
      const std::size_t dependency = checked_size(
          dependencies.front().as_int(), "stem Wgrad dependency");
      if (dependency != 0) {
        artifact_error("mthreads stem Wgrad stage dependency is invalid");
      }
      stage.dependencies.push_back(dependency);
    }

    const JsonValue& autotune = stage_value.at("autotune");
    require_exact_keys(
        autotune,
        {"enabled", "warmup", "repetitions", "selection_cache"},
        "stem Wgrad autotune");
    stage.autotune = autotune.at("enabled").as_bool();
    stage.warmup = checked_positive_unsigned(
        autotune.at("warmup").as_int(), "stem Wgrad warmup");
    stage.repetitions = checked_positive_unsigned(
        autotune.at("repetitions").as_int(), "stem Wgrad repetitions");
    const std::string expected_cache =
        "tuning/stage-" + std::to_string(stage_index) + ".json";
    if (stage.autotune != request.autotune || stage.warmup != 3 ||
        stage.repetitions != 10 ||
        autotune.at("selection_cache").as_string() != expected_cache) {
      artifact_error("mthreads stem Wgrad autotune metadata differs");
    }
    stage.selection_cache = canonical_root / expected_cache;

    const auto candidates =
        expected_stem_wgrad_candidates(request, stage_index);
    const auto& variants = stage_value.at("variants").as_array();
    if (variants.empty() || variants.size() > kMaximumVariants ||
        variants.size() != candidates.size()) {
      artifact_error("mthreads stem Wgrad variant count is invalid");
    }
    std::set<std::string> variant_ids;
    stage.variants.reserve(variants.size());
    for (std::size_t variant_index = 0;
         variant_index < variants.size(); ++variant_index) {
      const JsonValue& variant_value = variants[variant_index];
      const Candidate candidate = candidates[variant_index];
      require_exact_keys(
          variant_value,
          {"variant_id", "source", "source_sha256", "function",
           "full_signature", "grid", "num_warps", "num_stages",
           "arguments"},
          "stem Wgrad variant");
      KernelVariantArtifact variant;
      variant.variant_id = variant_value.at("variant_id").as_string();
      variant.function_name = variant_value.at("function").as_string();
      variant.full_signature =
          variant_value.at("full_signature").as_string();
      variant.grid = parse_grid(variant_value.at("grid"));
      variant.num_warps = checked_positive_unsigned(
          variant_value.at("num_warps").as_int(),
          "stem Wgrad num_warps");
      variant.num_stages = checked_positive_unsigned(
          variant_value.at("num_stages").as_int(),
          "stem Wgrad num_stages");
      const std::string expected_variant_id =
          "block-" + std::to_string(candidate.block_size) +
          "-warps-" + std::to_string(candidate.num_warps) +
          "-stages-" + std::to_string(candidate.num_stages);
      const AttentionVariantExpectation expectation =
          expected_stem_wgrad_variant(
              request, expected_function, candidate.block_size);
      if (variant.variant_id != expected_variant_id ||
          !variant_ids.insert(variant.variant_id).second ||
          variant_value.at("source").as_string() != expected_source_path ||
          variant_value.at("source_sha256").as_string() != source_hash ||
          variant.function_name != expected_function ||
          variant.full_signature != expectation.signature ||
          variant.grid != expectation.grid ||
          variant.num_warps != candidate.num_warps ||
          variant.num_stages != candidate.num_stages) {
        artifact_error(
            "mthreads stem Wgrad variant launch/signature differs");
      }
      variant.source = source;

      const auto& arguments =
          variant_value.at("arguments").as_array();
      if (arguments.size() > kMaximumArguments ||
          arguments.size() != expectation.arguments.size()) {
        artifact_error("mthreads stem Wgrad argument count differs");
      }
      for (std::size_t argument_index = 0;
           argument_index < arguments.size(); ++argument_index) {
        const ExpectedAttentionArgument& expected_argument =
            expectation.arguments[argument_index];
        const std::int64_t expected_value =
            expected_argument.kind == ArgumentKind::kTensor
                ? expected_argument.tensor->uid
                : expected_argument.scalar_value;
        variant.arguments.push_back(parse_expected_argument(
            arguments[argument_index],
            expected_argument.kind,
            expected_argument.semantic_name,
            expected_value,
            expected_argument.scalar_bits));
        ArgumentSpec& parsed = variant.arguments.back();
        if (expected_argument.kind == ArgumentKind::kTensor) {
          parsed.storage_size = expected_argument.tensor->storage_size;
          parsed.alignment = expected_argument.tensor->alignment;
          parsed.data_type = expected_argument.tensor->data_type;
        } else if (expected_argument.kind == ArgumentKind::kWorkspace) {
          parsed.storage_size = result.workspace_size;
          parsed.alignment = result.workspace_alignment;
        }
      }
      stage.variants.push_back(std::move(variant));
    }
    result.stages.push_back(std::move(stage));
  }
}

PointwiseRequest parse_pointwise_request(
    const JsonValue& request,
    std::string_view expected_target) {
  const auto& nodes =
      request.at("graph").at("nodes").as_array();
  if (nodes.size() == 3) {
    return parse_conv_bias_relu_request(request, expected_target);
  }
  if (nodes.size() == 2) {
    return parse_add_square_request(request, expected_target);
  }
  if (nodes.size() != 1) {
    artifact_error(
        "mthreads compiler requires exactly one node");
  }
  const std::string& operation = nodes.front().at("type").as_string();
  if (operation == "reshape" || operation == "transpose" ||
      operation == "slice") {
    return parse_layout_request(request, expected_target);
  }
  if (operation == "reduction_sum" || operation == "reduction_avg" ||
      operation == "reduction_mul") {
    return parse_reduction_request(request, expected_target);
  }
  if (operation == "matmul") {
    return parse_matmul_request(request, expected_target);
  }
  if (operation == "conv2d_fprop" ||
      operation == "convolution_fprop" ||
      operation == "convolution_dgrad" ||
      operation == "convolution_wgrad") {
    return parse_convolution_request(request, expected_target);
  }
  if (operation == "layernorm" || operation == "rmsnorm" ||
      operation == "batchnorm" ||
      operation == "batchnorm_inference") {
    return parse_normalization_request(request, expected_target);
  }
  if (is_attention_operation(operation)) {
    return parse_attention_request(request, expected_target);
  }
  const std::int64_t mode =
      nodes.front().at("attributes").at("mode").as_int();
  if (is_unary_mode(mode)) {
    return parse_unary_request(request, expected_target);
  }
  if (mode == 41) {
    return parse_ternary_request(request, expected_target);
  }
  return parse_binary_request(request, expected_target);
}

std::string_view source_path(const PointwiseRequest& request) {
  if (request.family == PointwiseFamily::kAttention) {
    return kAttentionSourcePath;
  }
  if (request.family == PointwiseFamily::kNormalization ||
      request.family == PointwiseFamily::kBatchnorm ||
      request.family == PointwiseFamily::kBatchnormInference) {
    return kNormalizationSourcePath;
  }
  if (request.family == PointwiseFamily::kAddSquare) {
    return kCompositeSourcePath;
  }
  if (request.family == PointwiseFamily::kConvBiasRelu) {
    return kConvBiasReluSourcePath;
  }
  if (request.family == PointwiseFamily::kBinary) {
    return kBinarySourcePath;
  }
  if (request.family == PointwiseFamily::kUnary) {
    return request.operation == "identity" ? kIdentitySourcePath
                                             : kUnarySourcePath;
  }
  if (request.family == PointwiseFamily::kTernary) {
    return kTernarySourcePath;
  }
  if (request.family == PointwiseFamily::kReduction) {
    return kReductionSourcePath;
  }
  if (request.family == PointwiseFamily::kMatmul) {
    return kMatmulSourcePath;
  }
  if (request.family == PointwiseFamily::kConvolution) {
    return kConvolutionSourcePath;
  }
  return kLayoutSourcePath;
}

std::string expected_function(const PointwiseRequest& request) {
  if (request.family == PointwiseFamily::kNormalization) {
    return request.operation == "layernorm" ? "layer_norm_kernel"
                                             : "rms_norm_kernel";
  }
  if (request.family == PointwiseFamily::kBatchnorm) {
    return request.dense ? "batch_norm_nchw_kernel"
                         : "batch_norm_kernel";
  }
  if (request.family == PointwiseFamily::kBatchnormInference) {
    return uses_batchnorm_inference_nchw(request)
               ? "batch_norm_inference_nchw_kernel"
               : "batch_norm_inference_kernel";
  }
  if (request.family == PointwiseFamily::kAddSquare) {
    return "add_square_tensor_kernel";
  }
  if (request.family == PointwiseFamily::kConvBiasRelu) {
    return "conv_bias_relu_2d_kernel";
  }
  if (request.family == PointwiseFamily::kBinary) {
    return request.dense ? "binary_contiguous_kernel"
                         : "binary_strided_kernel";
  }
  if (request.family == PointwiseFamily::kUnary) {
    if (request.operation == "identity") {
      if (uses_packed_identity(request)) {
        return "identity_contiguous_packed_kernel";
      }
      return request.dense ? "identity_contiguous_kernel"
                           : "identity_strided_kernel";
    }
    return request.dense ? "unary_pointwise_contiguous_kernel"
                         : "unary_pointwise_strided_kernel";
  }
  if (request.family == PointwiseFamily::kTernary) {
    return request.dense ? "binary_select_tensor_kernel"
                         : "binary_select_strided_kernel";
  }
  if (request.family == PointwiseFamily::kReduction) {
    if (request.dense && request.inner == 1) {
      return "reduction_2d_kernel";
    }
    return request.dense ? "reduction_3d_kernel"
                         : "reduction_strided_kernel";
  }
  if (request.family == PointwiseFamily::kMatmul) {
    return uses_matmul_tle(request)
               ? "matmul_tle_kernel"
               : uses_matmul_descriptor(request)
               ? "matmul_descriptor_kernel"
               : "matmul_strided_kernel";
  }
  if (request.family == PointwiseFamily::kConvolution) {
    if (request.operation == "convolution_dgrad") {
      return "conv_dgrad_nd_kernel";
    }
    if (request.operation == "convolution_wgrad") {
      return "conv_wgrad_nd_kernel";
    }
    if (request.convolution_spatial_rank == 1) {
      return "conv1d_gemm_kernel";
    }
    return request.convolution_spatial_rank == 2
               ? "conv2d_spatial_nchw_kernel"
               : "conv3d_spatial_ncdhw_m_kernel";
  }
  if (request.family == PointwiseFamily::kLayout &&
      uses_contiguous_reshape(request)) {
    return "reshape_contiguous_kernel";
  }
  if (request.family == PointwiseFamily::kLayout &&
      uses_physical_transpose(request)) {
    return "transpose_physical_copy_kernel";
  }
  if (request.family == PointwiseFamily::kLayout &&
      uses_i32_slice(request)) {
    return "slice_copy_kernel";
  }
  return "layout_copy_kernel";
}

std::filesystem::path validate_source(
    const std::filesystem::path& canonical_root,
    const JsonValue& descriptor,
    std::string_view expected_path,
    std::string& source_hash) {
  require_exact_keys(descriptor, {"path", "size", "sha256"}, "file");
  const std::string& path_text = descriptor.at("path").as_string();
  if (path_text != expected_path) {
    artifact_error("mthreads artifact source path is not the registered path");
  }
  const std::filesystem::path relative(path_text);
  const std::filesystem::path parent = canonical_root / relative.parent_path();
  std::error_code error;
  const auto parent_status = std::filesystem::symlink_status(parent, error);
  if (error || std::filesystem::is_symlink(parent_status) ||
      !std::filesystem::is_directory(parent_status)) {
    artifact_error("mthreads artifact source parent is unsafe");
  }
  const std::filesystem::path source_path = canonical_root / relative;
  const auto source_status =
      std::filesystem::symlink_status(source_path, error);
  if (error || std::filesystem::is_symlink(source_status) ||
      !std::filesystem::is_regular_file(source_status)) {
    artifact_error("mthreads artifact source is missing or unsafe");
  }
  const std::filesystem::path canonical_source =
      std::filesystem::canonical(source_path, error);
  if (error || !path_is_below(canonical_root, canonical_source)) {
    artifact_error("mthreads artifact source escapes its directory");
  }
  const std::size_t expected_size =
      checked_size(descriptor.at("size").as_int(), "source.size");
  const std::uintmax_t actual_size =
      std::filesystem::file_size(source_path, error);
  if (error || expected_size == 0 || expected_size > kMaximumSourceSize ||
      actual_size != expected_size) {
    artifact_error("mthreads artifact source size differs");
  }
  source_hash = descriptor.at("sha256").as_string();
  if (!is_lower_sha256(source_hash) ||
      flagdnn::native::sha256_file(source_path) != source_hash) {
    artifact_error("mthreads artifact source SHA-256 differs");
  }
  return canonical_source;
}

void validate_artifact_tree(
    const std::filesystem::path& canonical_root,
    std::string_view graph_ir,
    std::string_view expected_source_path,
    std::size_t stage_count) {
  const std::set<std::string> always_allowed_files = {
      "manifest.json", std::string(expected_source_path)};
  std::set<std::string> potentially_allowed_files = {"request.json"};
  for (std::size_t stage = 0; stage < stage_count; ++stage) {
    potentially_allowed_files.insert(
        "tuning/stage-" + std::to_string(stage) + ".json");
  }
  const std::set<std::string> potentially_allowed_directories = {
      "kernels", "tuning"};
  bool saw_request = false;
  bool saw_cache = false;
  bool saw_tuning_directory = false;
  std::uintmax_t total_size = 0;
  std::error_code error;
  std::filesystem::recursive_directory_iterator iterator(
      canonical_root,
      std::filesystem::directory_options::none,
      error);
  if (error) {
    artifact_error("cannot enumerate mthreads artifact directory");
  }
  const std::filesystem::recursive_directory_iterator end;
  for (; iterator != end; iterator.increment(error)) {
    if (error) {
      artifact_error("cannot enumerate mthreads artifact directory");
    }
    const std::filesystem::directory_entry& entry = *iterator;
    const auto status = entry.symlink_status(error);
    if (error || std::filesystem::is_symlink(status)) {
      artifact_error("mthreads artifact contains a symbolic link");
    }
    const std::string relative =
        entry.path().lexically_relative(canonical_root).generic_string();
    if (std::filesystem::is_directory(status)) {
      if (potentially_allowed_directories.find(relative) ==
          potentially_allowed_directories.end()) {
        artifact_error("mthreads artifact contains an extra directory");
      }
      saw_tuning_directory = saw_tuning_directory || relative == "tuning";
      continue;
    }
    if (!std::filesystem::is_regular_file(status)) {
      artifact_error("mthreads artifact contains a non-regular entry");
    }
    if (always_allowed_files.find(relative) ==
            always_allowed_files.end() &&
        potentially_allowed_files.find(relative) ==
            potentially_allowed_files.end()) {
      artifact_error("mthreads artifact contains an extra file");
    }
    const std::uintmax_t size = entry.file_size(error);
    if (error || size > kMaximumArtifactSize - total_size) {
      artifact_error("mthreads artifact exceeds its total size limit");
    }
    total_size += size;
    if (relative == "request.json") {
      saw_request = true;
      if (read_file(
              entry.path(), kMaximumMetadataSize, "artifact request") !=
          graph_ir) {
        artifact_error(
            "mthreads artifact request.json differs from Graph IR");
      }
    } else if (relative.starts_with("tuning/stage-") &&
               relative.ends_with(".json")) {
      saw_cache = true;
      static_cast<void>(
          read_file(entry.path(), kMaximumMetadataSize, "autotune cache"));
    }
  }
  static_cast<void>(saw_request);
  if (saw_tuning_directory != saw_cache) {
    artifact_error(
        "mthreads artifact tuning directory/cache layout is incomplete");
  }
}

#include "backends/mthreads/codegen/extended_artifact.inc"

}  // namespace

MthreadsArtifact parse_mthreads_artifact(
    const EngineBuildContext& context,
    const flagdnnBackendBuildInputV2& input) {
  try {
    if (input.struct_size < sizeof(flagdnnBackendBuildInputV2) ||
        input.graph_ir == nullptr || input.graph_ir_size == 0 ||
        input.graph_ir_size > kMaximumMetadataSize ||
        input.artifact_directory == nullptr ||
        *input.artifact_directory == '\0' ||
        input.request_sha256 == nullptr) {
      artifact_error("mthreads build input is incomplete");
    }
    const std::string_view graph_ir(
        static_cast<const char*>(input.graph_ir), input.graph_ir_size);
    const std::string request_hash(input.request_sha256);
    if (!is_lower_sha256(request_hash) ||
        flagdnn::native::sha256(graph_ir) != request_hash) {
      artifact_error(
          "mthreads Graph IR request SHA-256 does not match build input");
    }
    const JsonValue raw_request = flagdnn::native::json::parse(graph_ir);
    if (is_extended_request(raw_request))
      return parse_extended_artifact(context, input, raw_request, graph_ir,
                                     request_hash);
    const JsonValue request = normalize_pointwise_attributes(raw_request);
    const PointwiseRequest pointwise_request =
        parse_pointwise_request(request, context.target_fingerprint);
    const std::string_view expected_source_path =
        source_path(pointwise_request);

    const std::filesystem::path artifact_directory(
        input.artifact_directory);
    std::error_code error;
    const auto root_status =
        std::filesystem::symlink_status(artifact_directory, error);
    if (error || std::filesystem::is_symlink(root_status) ||
        !std::filesystem::is_directory(root_status)) {
      artifact_error(
          "mthreads artifact root is missing, non-directory, or a symlink");
    }
    const std::filesystem::path canonical_root =
        std::filesystem::canonical(artifact_directory, error);
    if (error || !std::filesystem::is_directory(canonical_root)) {
      artifact_error("mthreads artifact root cannot be canonicalized");
    }
    const std::string manifest_bytes = read_file(
        canonical_root / "manifest.json",
        kMaximumMetadataSize,
        "artifact manifest");
    const JsonValue manifest =
        flagdnn::native::json::parse(manifest_bytes);
    require_exact_keys(
        manifest,
        {"schema_version", "artifact_kind", "flagdnn_version", "backend",
         "target", "engine", "request_sha256", "compiler_identity",
         "source_sha256", "workspace_size", "workspace_alignment",
         "external_binding_uids", "program", "files"},
        "manifest");
    if (manifest.at("schema_version").as_int() !=
            kMthreadsArtifactSchemaVersion ||
        manifest.at("artifact_kind").as_string() !=
            "flagdnn_execution_program" ||
        manifest.at("flagdnn_version").as_string() !=
            pointwise_request.flagdnn_version ||
        manifest.at("backend").as_string() != "mthreads" ||
        manifest.at("target").as_string() != pointwise_request.target ||
        manifest.at("target").as_string() != context.target_fingerprint ||
        manifest.at("engine").as_string() != "libtriton_jit" ||
        manifest.at("request_sha256").as_string() != request_hash ||
        manifest.at("compiler_identity").as_string() !=
            pointwise_request.compiler_identity) {
      artifact_error(
          "mthreads artifact identity does not match Graph IR/context");
    }

    MthreadsArtifact result;
    result.workspace_size = checked_size(
        manifest.at("workspace_size").as_int(), "workspace_size");
    result.workspace_alignment = checked_size(
        manifest.at("workspace_alignment").as_int(),
        "workspace_alignment");
    std::size_t expected_workspace_size = kPhaseOneWorkspaceSize;
    if (pointwise_request.family == PointwiseFamily::kAttention) {
      expected_workspace_size =
          expected_attention_workspace_size(pointwise_request);
    } else if (uses_im2col_fprop(pointwise_request)) {
      expected_workspace_size =
          expected_im2col_fprop_workspace_size(pointwise_request);
    } else if (uses_dense_stride2_dgrad(pointwise_request)) {
      expected_workspace_size =
          expected_dense_dgrad_workspace_size(pointwise_request);
    } else if (uses_standard_wgrad(pointwise_request)) {
      expected_workspace_size =
          expected_standard_wgrad_workspace_size(pointwise_request);
    } else if (uses_stem_wgrad(pointwise_request)) {
      expected_workspace_size =
          expected_stem_wgrad_workspace_size(pointwise_request);
    } else if (uses_nd_packed_wgrad(pointwise_request)) {
      expected_workspace_size =
          expected_nd_packed_wgrad_workspace_size(pointwise_request);
    } else if (uses_p5_wgrad(pointwise_request)) {
      expected_workspace_size =
          expected_p5_wgrad_workspace_size(pointwise_request);
    }
    if (!is_power_of_two(result.workspace_alignment) ||
        result.workspace_alignment < kMinimumWorkspaceAlignment ||
        result.workspace_alignment > kMaximumWorkspaceAlignment ||
        result.workspace_size % result.workspace_alignment != 0 ||
        result.workspace_size != expected_workspace_size ||
        result.workspace_alignment != kPhaseOneWorkspaceAlignment) {
      artifact_error("mthreads artifact workspace contract is invalid");
    }

    const auto& binding_values =
        manifest.at("external_binding_uids").as_array();
    if (binding_values.size() !=
        pointwise_request.external_binding_uids.size()) {
      artifact_error("mthreads artifact binding count differs from Graph IR");
    }
    for (std::size_t index = 0; index < binding_values.size(); ++index) {
      const std::int64_t uid = binding_values[index].as_int();
      if (uid != pointwise_request.external_binding_uids[index]) {
        artifact_error(
            "mthreads artifact binding order differs from Graph IR");
      }
      result.binding_uids.push_back(uid);
    }

    const auto& file_values = manifest.at("files").as_array();
    if (file_values.size() != 1) {
      artifact_error("mthreads artifact must contain one source descriptor");
    }
    std::string source_hash;
    const std::filesystem::path source = validate_source(
        canonical_root,
        file_values.front(),
        expected_source_path,
        source_hash);
    if (manifest.at("source_sha256").as_string() != source_hash) {
      artifact_error("mthreads manifest source hashes disagree");
    }

    const JsonValue& program = manifest.at("program");
    if (uses_im2col_fprop(pointwise_request)) {
      parse_im2col_fprop_program(
          pointwise_request,
          program,
          source,
          source_hash,
          expected_source_path,
          canonical_root,
          result);
      validate_artifact_tree(
          canonical_root,
          graph_ir,
          expected_source_path,
          result.stages.size());
      return result;
    }
    if (pointwise_request.family == PointwiseFamily::kAttention) {
      parse_attention_program(
          pointwise_request,
          program,
          source,
          source_hash,
          expected_source_path,
          canonical_root,
          result);
      validate_artifact_tree(
          canonical_root,
          graph_ir,
          expected_source_path,
          result.stages.size());
      return result;
    }
    if (uses_dense_stride2_dgrad(pointwise_request)) {
      parse_dense_dgrad_program(
          pointwise_request,
          program,
          source,
          source_hash,
          expected_source_path,
          canonical_root,
          result);
      validate_artifact_tree(
          canonical_root,
          graph_ir,
          expected_source_path,
          result.stages.size());
      return result;
    }
    if (uses_standard_wgrad(pointwise_request) ||
        uses_nd_packed_wgrad(pointwise_request)) {
      parse_standard_wgrad_program(
          pointwise_request,
          program,
          source,
          source_hash,
          expected_source_path,
          canonical_root,
          result);
      validate_artifact_tree(
          canonical_root,
          graph_ir,
          expected_source_path,
          result.stages.size());
      return result;
    }
    if (uses_stem_wgrad(pointwise_request)) {
      parse_stem_wgrad_program(
          pointwise_request,
          program,
          source,
          source_hash,
          expected_source_path,
          canonical_root,
          result);
      validate_artifact_tree(
          canonical_root,
          graph_ir,
          expected_source_path,
          result.stages.size());
      return result;
    }
    if (uses_p5_wgrad(pointwise_request)) {
      parse_p5_wgrad_program(
          pointwise_request,
          program,
          source,
          source_hash,
          expected_source_path,
          canonical_root,
          result);
      validate_artifact_tree(
          canonical_root,
          graph_ir,
          expected_source_path,
          result.stages.size());
      return result;
    }
    require_exact_keys(
        program, {"schema_version", "stage_count", "stages"}, "program");
    const auto& stage_values = program.at("stages").as_array();
    const std::size_t stage_count =
        checked_size(program.at("stage_count").as_int(), "stage_count");
    if (program.at("schema_version").as_int() !=
            kMthreadsExecutionProgramVersion ||
        stage_count == 0 || stage_count > kMaximumStages ||
        stage_values.size() != stage_count || stage_count != 1) {
      artifact_error("mthreads execution program stage count is invalid");
    }

    const std::string function = expected_function(pointwise_request);
    const auto candidates = expected_candidates(pointwise_request);
    result.stages.reserve(stage_count);
    for (std::size_t stage_index = 0;
         stage_index < stage_values.size(); ++stage_index) {
      const JsonValue& stage_value = stage_values[stage_index];
      require_exact_keys(
          stage_value,
          {"id", "node_id", "operation", "dependencies", "source",
           "function", "variants", "autotune"},
          "stage");
      StageArtifact stage;
      stage.id = checked_size(stage_value.at("id").as_int(), "stage.id");
      stage.node_id = stage_value.at("node_id").as_int();
      if (stage.id != stage_index || stage.id != 0 ||
          stage.node_id != pointwise_request.node_id ||
          stage_value.at("operation").as_string() !=
              pointwise_request.operation ||
          stage_value.at("source").as_string() != expected_source_path ||
          stage_value.at("function").as_string() != function ||
          !printable_ascii_identifier(
              stage_value.at("function").as_string(), 256)) {
        artifact_error(
            "mthreads execution stage differs from the pointwise Graph");
      }
      stage.source = source;
      stage.function_name = function;
      const auto& dependencies =
          stage_value.at("dependencies").as_array();
      if (!dependencies.empty()) {
        artifact_error(
            "mthreads single-stage pointwise must not have dependencies");
      }

      const JsonValue& autotune = stage_value.at("autotune");
      require_exact_keys(
          autotune,
          {"enabled", "warmup", "repetitions", "selection_cache"},
          "autotune");
      stage.autotune = autotune.at("enabled").as_bool();
      stage.warmup =
          checked_positive_unsigned(autotune.at("warmup").as_int(), "warmup");
      stage.repetitions = checked_positive_unsigned(
          autotune.at("repetitions").as_int(), "repetitions");
      const bool expected_autotune =
          pointwise_request.autotune &&
          !(pointwise_request.family == PointwiseFamily::kMatmul &&
            uses_matmul_descriptor(pointwise_request));
      if (stage.autotune != expected_autotune ||
          stage.warmup != 3 || stage.repetitions != 10 ||
          autotune.at("selection_cache").as_string() !=
              kSelectionCachePath) {
        artifact_error(
            "mthreads autotune metadata differs from Graph/options");
      }
      stage.selection_cache =
          canonical_root / std::filesystem::path(kSelectionCachePath);

      const auto& variants = stage_value.at("variants").as_array();
      if (variants.empty() || variants.size() > kMaximumVariants ||
          variants.size() != candidates.size()) {
        artifact_error("mthreads pointwise variant count is invalid");
      }
      std::set<std::string> variant_ids;
      stage.variants.reserve(variants.size());
      for (std::size_t variant_index = 0;
           variant_index < variants.size(); ++variant_index) {
        const JsonValue& variant_value = variants[variant_index];
        require_exact_keys(
            variant_value,
            {"variant_id", "source", "source_sha256", "function",
             "full_signature", "grid", "num_warps", "num_stages",
             "arguments"},
            "variant");
        const Candidate candidate = candidates[variant_index];
        KernelVariantArtifact variant;
        variant.variant_id =
            variant_value.at("variant_id").as_string();
        const std::string expected_variant_id =
            "block-" + std::to_string(candidate.block_size) +
            "-warps-" + std::to_string(candidate.num_warps) +
            "-stages-" + std::to_string(candidate.num_stages);
        variant.function_name =
            variant_value.at("function").as_string();
        variant.full_signature =
            variant_value.at("full_signature").as_string();
        variant.grid = parse_grid(variant_value.at("grid"));
        variant.num_warps = checked_positive_unsigned(
            variant_value.at("num_warps").as_int(), "num_warps");
        variant.num_stages = checked_positive_unsigned(
            variant_value.at("num_stages").as_int(), "num_stages");
        std::array<unsigned int, 3> expected_grid{};
        if (pointwise_request.family == PointwiseFamily::kMatmul) {
          if (const auto tle = matmul_tle_config(pointwise_request);
              tle.has_value()) {
            const auto tiles_m = static_cast<unsigned int>(
                pointwise_request.matmul_m / tle->block_m);
            const auto tiles_n = static_cast<unsigned int>(
                pointwise_request.matmul_n / tle->block_n);
            expected_grid = {
                static_cast<unsigned int>(
                    pointwise_request.matmul_batch) *
                    tiles_m * tiles_n,
                1,
                1};
          } else {
            const auto tiles_m = static_cast<unsigned int>(
                (pointwise_request.matmul_m + candidate.block_size - 1) /
                candidate.block_size);
            const auto tiles_n = static_cast<unsigned int>(
                (pointwise_request.matmul_n +
                 matmul_block_n(
                     pointwise_request, candidate.block_size) -
                 1) /
                matmul_block_n(
                    pointwise_request, candidate.block_size));
            expected_grid = {
                tiles_m * tiles_n,
                static_cast<unsigned int>(
                    pointwise_request.matmul_batch),
                1};
          }
        } else if (
            pointwise_request.family == PointwiseFamily::kConvolution ||
            pointwise_request.family == PointwiseFamily::kConvBiasRelu) {
          expected_grid = expected_convolution_grid(
              pointwise_request, candidate.block_size);
        } else if (
            pointwise_request.family == PointwiseFamily::kNormalization ||
            pointwise_request.family == PointwiseFamily::kBatchnorm ||
            pointwise_request.family ==
                PointwiseFamily::kBatchnormInference) {
          expected_grid = expected_normalization_grid(
              pointwise_request, candidate.block_size);
        } else if (
            pointwise_request.family == PointwiseFamily::kUnary) {
          unsigned int work_items =
              static_cast<unsigned int>(pointwise_request.n_elements);
          unsigned int tiles = 1;
          if (pointwise_request.operation == "identity") {
            tiles = identity_tiles_per_program(pointwise_request);
            if (uses_packed_identity(pointwise_request)) {
              work_items /= identity_pack_size(pointwise_request);
            }
          } else {
            tiles = unary_tiles_per_program(pointwise_request);
          }
          const unsigned int work_per_program =
              candidate.block_size * tiles;
          expected_grid = {
              std::max(
                  1U,
                  (work_items + work_per_program - 1) / work_per_program),
              1,
              1};
        } else {
          expected_grid = {
              (static_cast<unsigned int>(pointwise_request.n_elements) +
               candidate.block_size - 1) /
                  candidate.block_size,
              1,
              1};
        }
        if (variant.variant_id != expected_variant_id ||
            !variant_ids.insert(variant.variant_id).second ||
            variant_value.at("source").as_string() != expected_source_path ||
            variant_value.at("source_sha256").as_string() != source_hash ||
            variant.function_name != function ||
            !printable_ascii_identifier(variant.function_name, 256) ||
            variant.full_signature !=
                expected_signature(pointwise_request, candidate.block_size) ||
            variant.grid != expected_grid ||
            variant.num_warps != candidate.num_warps ||
            variant.num_stages != candidate.num_stages) {
          artifact_error(
              "mthreads pointwise variant launch/signature differs");
        }
        variant.source = source;

        const auto& arguments =
            variant_value.at("arguments").as_array();
        const std::size_t expected_argument_count = [&]() {
          if (pointwise_request.family ==
              PointwiseFamily::kNormalization) {
            return pointwise_request.operation == "layernorm" ? 7U : 6U;
          }
          if (pointwise_request.family == PointwiseFamily::kBatchnorm) {
            return pointwise_request.dense ? 10U : 13U;
          }
          if (pointwise_request.family ==
              PointwiseFamily::kBatchnormInference) {
            return uses_batchnorm_inference_nchw(pointwise_request) ? 6U
                                                                    : 9U;
          }
          if (pointwise_request.family == PointwiseFamily::kBinary ||
              pointwise_request.family == PointwiseFamily::kAddSquare ||
              pointwise_request.family ==
                  PointwiseFamily::kConvBiasRelu) {
            return 4U;
          }
          if (pointwise_request.family == PointwiseFamily::kUnary ||
              pointwise_request.family == PointwiseFamily::kLayout ||
              pointwise_request.family == PointwiseFamily::kReduction) {
            return 3U;
          }
          if (pointwise_request.family == PointwiseFamily::kMatmul) {
            return 3U;
          }
          if (pointwise_request.family == PointwiseFamily::kConvolution) {
            return pointwise_request.operation == "conv2d_fprop" ||
                           pointwise_request.operation ==
                               "convolution_fprop"
                       ? 4U
                       : 3U;
          }
          return 5U;
        }();
        if (arguments.size() > kMaximumArguments ||
            arguments.size() != expected_argument_count) {
          artifact_error(
              "mthreads pointwise runtime argument count is invalid");
        }
        const auto append_tensor_argument =
            [&](std::size_t index,
                std::string_view name,
                const TensorSpec& tensor) {
              variant.arguments.push_back(parse_expected_argument(
                  arguments[index],
                  ArgumentKind::kTensor,
                  name,
                  tensor.uid,
                  ""));
              variant.arguments.back().storage_size = tensor.storage_size;
              variant.arguments.back().alignment = tensor.alignment;
              variant.arguments.back().data_type = tensor.data_type;
            };
        const auto append_i32_argument =
            [&](std::size_t index,
                std::string_view name,
                std::int64_t value) {
              variant.arguments.push_back(parse_expected_argument(
                  arguments[index],
                  ArgumentKind::kScalarI32,
                  name,
                  value,
                  i32_bits(static_cast<std::int32_t>(value))));
            };
        if (pointwise_request.family ==
            PointwiseFamily::kNormalization) {
          append_tensor_argument(0, "x", pointwise_request.normalization_x);
          append_tensor_argument(1, "y", pointwise_request.normalization_y);
          if (pointwise_request.operation == "layernorm") {
            append_tensor_argument(
                2, "mean", pointwise_request.normalization_mean);
            append_tensor_argument(
                3,
                "inv_variance",
                pointwise_request.normalization_inv_variance);
            append_tensor_argument(
                4, "scale", pointwise_request.normalization_scale);
            append_tensor_argument(
                5, "bias", pointwise_request.normalization_bias);
            append_i32_argument(
                6, "rows", pointwise_request.normalization_rows);
          } else {
            append_tensor_argument(
                2, "scale", pointwise_request.normalization_scale);
            append_tensor_argument(
                3, "bias", pointwise_request.normalization_bias);
            append_tensor_argument(
                4,
                "inv_variance",
                pointwise_request.normalization_inv_variance);
            append_i32_argument(
                5, "rows", pointwise_request.normalization_rows);
          }
        } else if (
            pointwise_request.family == PointwiseFamily::kBatchnorm) {
          append_tensor_argument(0, "x", pointwise_request.normalization_x);
          append_tensor_argument(1, "y", pointwise_request.normalization_y);
          append_tensor_argument(
              2,
              "previous_running_mean",
              pointwise_request.normalization_previous_running_mean);
          append_tensor_argument(
              3,
              "previous_running_variance",
              pointwise_request.normalization_previous_running_variance);
          append_tensor_argument(
              4, "scale", pointwise_request.normalization_scale);
          append_tensor_argument(
              5, "bias", pointwise_request.normalization_bias);
          append_tensor_argument(
              6, "mean", pointwise_request.normalization_mean);
          append_tensor_argument(
              7,
              "inv_variance",
              pointwise_request.normalization_inv_variance);
          append_tensor_argument(
              8,
              "next_running_mean",
              pointwise_request.normalization_next_running_mean);
          append_tensor_argument(
              9,
              "next_running_variance",
              pointwise_request.normalization_next_running_variance);
          if (!pointwise_request.dense) {
            append_i32_argument(
                10, "batch", pointwise_request.normalization_batch);
            append_i32_argument(
                11, "channels", pointwise_request.normalization_channels);
            append_i32_argument(
                12, "spatial", pointwise_request.normalization_spatial);
          }
        } else if (
            pointwise_request.family ==
            PointwiseFamily::kBatchnormInference) {
          append_tensor_argument(0, "x", pointwise_request.normalization_x);
          append_tensor_argument(
              1, "mean", pointwise_request.normalization_mean);
          append_tensor_argument(
              2,
              "inv_variance",
              pointwise_request.normalization_inv_variance);
          append_tensor_argument(
              3, "scale", pointwise_request.normalization_scale);
          append_tensor_argument(
              4, "bias", pointwise_request.normalization_bias);
          append_tensor_argument(5, "y", pointwise_request.normalization_y);
          if (!uses_batchnorm_inference_nchw(pointwise_request)) {
            append_i32_argument(
                6, "n_elements", pointwise_request.n_elements);
            append_i32_argument(
                7, "channels", pointwise_request.normalization_channels);
            append_i32_argument(
                8, "spatial", pointwise_request.normalization_spatial);
          }
        } else if (
            pointwise_request.family == PointwiseFamily::kConvBiasRelu) {
          append_tensor_argument(
              0, "input", pointwise_request.convolution_image);
          append_tensor_argument(
              1, "filter", pointwise_request.convolution_filter);
          append_tensor_argument(
              2, "bias", pointwise_request.convolution_bias);
          append_tensor_argument(
              3, "output", pointwise_request.convolution_result);
        } else if (pointwise_request.family == PointwiseFamily::kBinary ||
            pointwise_request.family == PointwiseFamily::kAddSquare) {
          append_tensor_argument(0, "left", pointwise_request.left);
          append_tensor_argument(1, "right", pointwise_request.right);
          append_tensor_argument(2, "output", pointwise_request.output);
        } else if (
            pointwise_request.family == PointwiseFamily::kUnary) {
          append_tensor_argument(0, "input", pointwise_request.input);
          append_tensor_argument(1, "output", pointwise_request.output);
        } else if (
            pointwise_request.family == PointwiseFamily::kTernary) {
          append_tensor_argument(0, "a", pointwise_request.left);
          append_tensor_argument(1, "b", pointwise_request.right);
          append_tensor_argument(2, "t", pointwise_request.predicate);
          append_tensor_argument(3, "output", pointwise_request.output);
        } else if (
            pointwise_request.family == PointwiseFamily::kMatmul) {
          append_tensor_argument(0, "a", pointwise_request.left);
          append_tensor_argument(1, "b", pointwise_request.right);
          append_tensor_argument(2, "output", pointwise_request.output);
          const auto tle = matmul_tle_config(pointwise_request);
          if (uses_matmul_descriptor(pointwise_request)) {
            const unsigned int block_m =
                tle.has_value() ? tle->block_m : candidate.block_size;
            const unsigned int block_n =
                tle.has_value()
                    ? tle->block_n
                    : matmul_block_n(pointwise_request, block_m);
            const unsigned int block_k =
                tle.has_value() ? tle->block_k : 64U;
            const auto configure_descriptor =
                [](ArgumentSpec& argument,
                   std::int64_t rows,
                   std::int64_t columns,
                   std::uint32_t block_rows,
                   std::uint32_t block_columns) {
                  argument.tensor_descriptor = true;
                  argument.descriptor_shape = {
                      static_cast<std::int32_t>(rows),
                      static_cast<std::int32_t>(columns)};
                  argument.descriptor_strides = {columns, 1};
                  argument.descriptor_block_shape = {
                      block_rows, block_columns};
                };
            configure_descriptor(
                variant.arguments[0],
                pointwise_request.matmul_batch *
                    pointwise_request.matmul_m,
                pointwise_request.matmul_k,
                block_m,
                block_k);
            configure_descriptor(
                variant.arguments[1],
                pointwise_request.matmul_batch *
                    pointwise_request.matmul_k,
                pointwise_request.matmul_n,
                block_k,
                block_n);
            if (!tle.has_value()) {
              configure_descriptor(
                  variant.arguments[2],
                  pointwise_request.matmul_batch *
                      pointwise_request.matmul_m,
                  pointwise_request.matmul_n,
                  block_m,
                  block_n);
            }
          }
        } else if (
            pointwise_request.family == PointwiseFamily::kConvolution) {
          if (pointwise_request.operation == "conv2d_fprop" ||
              pointwise_request.operation == "convolution_fprop") {
            append_tensor_argument(
                0, "input", pointwise_request.convolution_image);
            append_tensor_argument(
                1, "filter", pointwise_request.convolution_filter);
            append_tensor_argument(
                2,
                "bias_placeholder",
                pointwise_request.convolution_image);
            append_tensor_argument(
                3, "output", pointwise_request.convolution_result);
          } else if (
              pointwise_request.operation == "convolution_dgrad") {
            append_tensor_argument(
                0, "dy", pointwise_request.convolution_result);
            append_tensor_argument(
                1, "w", pointwise_request.convolution_filter);
            append_tensor_argument(
                2, "dx", pointwise_request.convolution_image);
          } else {
            append_tensor_argument(
                0, "dy", pointwise_request.convolution_result);
            append_tensor_argument(
                1, "x", pointwise_request.convolution_image);
            append_tensor_argument(
                2, "dw", pointwise_request.convolution_filter);
          }
        } else {
          append_tensor_argument(0, "input", pointwise_request.input);
          append_tensor_argument(1, "output", pointwise_request.output);
        }
        if (pointwise_request.family != PointwiseFamily::kMatmul &&
            pointwise_request.family != PointwiseFamily::kConvolution &&
            pointwise_request.family != PointwiseFamily::kConvBiasRelu &&
            pointwise_request.family != PointwiseFamily::kNormalization &&
            pointwise_request.family != PointwiseFamily::kBatchnorm &&
            pointwise_request.family !=
                PointwiseFamily::kBatchnormInference) {
          const std::size_t scalar_index = expected_argument_count - 1;
          const bool reduction_2d =
              pointwise_request.family == PointwiseFamily::kReduction &&
              function == "reduction_2d_kernel";
          const std::string_view scalar_name =
              pointwise_request.family == PointwiseFamily::kReduction
                  ? (reduction_2d ? "outer" : "output_elements")
                  : "n_elements";
          const std::int64_t scalar_value =
              reduction_2d ? pointwise_request.outer
                           : pointwise_request.n_elements;
          variant.arguments.push_back(parse_expected_argument(
              arguments[scalar_index],
              ArgumentKind::kScalarI32,
              scalar_name,
              scalar_value,
              i32_bits(static_cast<std::int32_t>(scalar_value))));
        }
        stage.variants.push_back(std::move(variant));
      }
      result.stages.push_back(std::move(stage));
    }

    validate_artifact_tree(
        canonical_root,
        graph_ir,
        expected_source_path,
        result.stages.size());
    return result;
  } catch (const MthreadsError&) {
    throw;
  } catch (const std::exception& error) {
    artifact_error(
        "invalid mthreads artifact: " + std::string(error.what()));
  }
}

}  // namespace flagdnn::mthreads
