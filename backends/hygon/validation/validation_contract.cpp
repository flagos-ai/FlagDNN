/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "activation_layout.hpp"
#include "benchmark/benchmark_phase.hpp"
#include "benchmark/ops.hpp"
#include "benchmark/pointwise_matcher.hpp"
#include "convolution_reference.hpp"
#include "functional/accuracy.hpp"
#include "functional/binding_address.hpp"
#include "hipdnn_reference.hpp"
#include "pointwise_reference.hpp"
#include "tensor_io.hpp"
#include "tensor_reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace hb = flagdnn::benchmarking;
namespace hd = flagdnn::benchmarking::hipdnn_detail;
namespace hf = flagdnn::testing::hygon_functional;
namespace hv = flagdnn::validation::hygon;

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

template <typename Function>
void require_runtime_error(Function &&function, std::string_view message) {
  try {
    function();
  } catch (const std::runtime_error &) {
    return;
  }
  throw std::runtime_error(std::string(message));
}

template <typename Function>
void require_invalid_argument(Function &&function, std::string_view message) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error(std::string(message));
}

hb::BenchmarkCase add_square_case() {
  hb::BenchmarkCase result;
  result.operation = hb::Operation::kGraph;
  result.output_count = 1;
  result.tensors = {
      hb::tensor(11, {2, 3}),
      hb::tensor(12, {2, 3}),
      hb::tensor(14, {2, 3}),
  };
  result.graph.intermediates = {hb::tensor(13, {2, 3})};

  hb::GraphNodeSpec square;
  square.operation = hb::Operation::kPointwise;
  square.input_uids = {12, 12};
  square.output_uid = 13;
  square.pointwise_mode = FLAGDNN_POINTWISE_MUL;

  hb::GraphNodeSpec add;
  add.operation = hb::Operation::kPointwise;
  add.input_uids = {11, 13};
  add.output_uid = 14;
  add.pointwise_mode = FLAGDNN_POINTWISE_ADD;
  result.graph.nodes = {square, add};
  return result;
}

hb::BenchmarkCase convolution_case(hb::Operation operation,
                                   flagdnnDataType_t data_type) {
  hb::BenchmarkCase result;
  result.operation = operation;
  result.output_count = 1;
  result.tensors = {
      hb::tensor(101, {1, 1, 3, 3}, data_type),
      hb::tensor(102, {1, 1, 1, 1}, data_type),
      hb::tensor(103, {1, 1, 3, 3}, data_type),
  };
  result.convolution.spatial_rank = 2;
  result.convolution.pre_padding = {0, 0};
  result.convolution.post_padding = {0, 0};
  result.convolution.stride = {1, 1};
  result.convolution.dilation = {1, 1};
  return result;
}

bool same_tensor_spec(const hb::TensorSpec &left, const hb::TensorSpec &right) {
  return left.uid == right.uid && left.data_type == right.data_type &&
         left.dimensions == right.dimensions && left.strides == right.strides &&
         left.binding_byte_offset == right.binding_byte_offset;
}

void test_status_classification() {
  require(hv::hipdnn_status_is_capability(HIPDNN_STATUS_NOT_SUPPORTED),
          "NOT_SUPPORTED must be a capability status");
  require(!hv::hipdnn_status_is_capability(HIPDNN_STATUS_ARCH_MISMATCH),
          "ARCH_MISMATCH must be a hard failure");
  require(!hv::hipdnn_status_is_capability(
              HIPDNN_STATUS_RUNTIME_PREREQUISITE_MISSING),
          "RUNTIME_PREREQUISITE_MISSING must be a hard failure");

  for (const hipdnnStatus_t status :
       {HIPDNN_STATUS_ALLOC_FAILED, HIPDNN_STATUS_BAD_PARAM,
        HIPDNN_STATUS_INTERNAL_ERROR, HIPDNN_STATUS_INVALID_VALUE,
        HIPDNN_STATUS_EXECUTION_FAILED, HIPDNN_STATUS_VERSION_MISMATCH}) {
    require(!hv::hipdnn_status_is_capability(status),
            "hard hipDNN error was incorrectly classified as capability");
  }

  try {
    hv::check_hipdnn_status(HIPDNN_STATUS_BAD_PARAM, "contract probe");
  } catch (const hv::HipdnnStatusError &error) {
    require(error.status() == HIPDNN_STATUS_BAD_PARAM,
            "typed hipDNN error lost its native status");
    require(std::string_view(error.what()).find("HIPDNN_STATUS_BAD_PARAM") !=
                std::string_view::npos,
            "typed hipDNN error message lost its status name");
    return;
  }
  throw std::runtime_error("check_hipdnn_status did not throw typed error");
}

void test_convolution_heuristic_fallback_classification() {
  for (const hipdnnStatus_t status :
       {HIPDNN_STATUS_NOT_SUPPORTED, HIPDNN_STATUS_EXECUTION_FAILED}) {
    require(hv::hipdnn_convolution_heuristic_allows_enum_fallback(status),
            "known heuristic metadata-query status did not allow public-enum "
            "fallback");
  }

  for (const hipdnnStatus_t status :
       {HIPDNN_STATUS_SUCCESS, HIPDNN_STATUS_NOT_INITIALIZED,
        HIPDNN_STATUS_ALLOC_FAILED, HIPDNN_STATUS_BAD_PARAM,
        HIPDNN_STATUS_INTERNAL_ERROR, HIPDNN_STATUS_INVALID_VALUE,
        HIPDNN_STATUS_ARCH_MISMATCH, HIPDNN_STATUS_MAPPING_ERROR,
        HIPDNN_STATUS_LICENSE_ERROR, HIPDNN_STATUS_RUNTIME_PREREQUISITE_MISSING,
        HIPDNN_STATUS_RUNTIME_IN_PROGRESS, HIPDNN_STATUS_RUNTIME_FP_OVERFLOW,
        HIPDNN_STATUS_VERSION_MISMATCH}) {
    require(!hv::hipdnn_convolution_heuristic_allows_enum_fallback(status),
            "hard hipDNN convolution heuristic status incorrectly allowed "
            "public-enum fallback");
  }
}

void test_convolution_algorithm_policy() {
  using Policy = hv::HipdnnConvolutionAlgorithmPolicy;
  const std::array<int, 0> no_heuristic{};

  const auto require_oracle = [&](hv::HipdnnConvolutionKind kind, int expected,
                                  std::string_view expected_name) {
    const std::vector<int> order = hv::hipdnn_convolution_algorithm_order(
        kind, Policy::kCorrectnessOracle, no_heuristic);
    require(order == std::vector<int>{expected},
            "correctness policy did not select exactly one symbolic oracle");
    require(hv::hipdnn_convolution_algorithm_name(kind, expected) ==
                expected_name,
            "correctness oracle lost its symbolic algorithm name");
  };

  require_oracle(hv::HipdnnConvolutionKind::kFprop,
                 static_cast<int>(HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM),
                 "HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM");
  require_oracle(hv::HipdnnConvolutionKind::kConvBiasRelu,
                 static_cast<int>(HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM),
                 "HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM");
  require_oracle(hv::HipdnnConvolutionKind::kDgrad,
                 static_cast<int>(HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1),
                 "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1");
  require_oracle(hv::HipdnnConvolutionKind::kWgrad,
                 static_cast<int>(HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1),
                 "HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1");

  const std::array<int, 3> heuristic = {
      static_cast<int>(HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD),
      static_cast<int>(HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM),
      static_cast<int>(HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD),
  };
  const std::vector<int> performance = hv::hipdnn_convolution_algorithm_order(
      hv::HipdnnConvolutionKind::kFprop, Policy::kPerformance, heuristic);
  require(!performance.empty() &&
              performance.front() ==
                  static_cast<int>(HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD),
          "performance policy did not preserve the v7 heuristic order");
  require(performance.size() ==
              static_cast<std::size_t>(HIPDNN_CONVOLUTION_FWD_ALGO_COUNT),
          "performance policy did not append every public enum candidate");
  require(performance.back() ==
              static_cast<int>(HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM),
          "performance policy did not retain the oracle as its final fallback");
  require(std::count(
              performance.begin(), performance.end(),
              static_cast<int>(HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)) == 1,
          "performance policy lost or duplicated its correctness oracle");

  const std::array<int, 1> invalid = {
      static_cast<int>(HIPDNN_CONVOLUTION_FWD_ALGO_COUNT)};
  require_invalid_argument(
      [&] {
        (void)hv::hipdnn_convolution_algorithm_order(
            hv::HipdnnConvolutionKind::kFprop, Policy::kPerformance, invalid);
      },
      "performance policy accepted a non-algorithm enum sentinel");
  require_invalid_argument(
      [&] {
        (void)hv::hipdnn_convolution_algorithm_order(
            hv::HipdnnConvolutionKind::kUnavailable, Policy::kCorrectnessOracle,
            no_heuristic);
      },
      "unavailable convolution kind acquired an oracle algorithm");

  require(hv::hipdnn_convolution_algorithm_policy_name(
              Policy::kCorrectnessOracle) == "correctness_oracle" &&
              hv::hipdnn_convolution_algorithm_policy_name(
                  Policy::kPerformance) == "performance",
          "convolution algorithm policies lost their stable output names");
}

void test_benchmark_fp16_convolution_oracle_policy() {
  using Policy = hb::HipdnnReferencePolicy;
  hb::BenchmarkCase fp16 =
      convolution_case(hb::Operation::kConvolutionFprop, FLAGDNN_DATA_FLOAT16);
  fp16.tensors[0].binding_byte_offset = 0;
  fp16.tensors[1].binding_byte_offset = 2;
  fp16.tensors[2].binding_byte_offset = 6;

  require(hd::convolution_uses_fp32_correctness_oracle(fp16),
          "benchmark FP16 convolution did not select its FP32 oracle policy");
  const std::vector<hb::TensorSpec> correctness =
      hd::convolution_reference_specs(fp16, Policy::kCorrectnessOracle);
  const std::vector<hb::TensorSpec> performance =
      hd::convolution_reference_specs(fp16, Policy::kPerformance);
  require(correctness.size() == fp16.tensors.size() &&
              performance.size() == fp16.tensors.size(),
          "convolution reference policy changed tensor arity");
  const std::array<std::size_t, 3> expected_offsets = {0, 4, 12};
  for (std::size_t index = 0; index < fp16.tensors.size(); ++index) {
    require(
        correctness[index].data_type == FLAGDNN_DATA_FLOAT32 &&
            correctness[index].uid == fp16.tensors[index].uid &&
            correctness[index].dimensions == fp16.tensors[index].dimensions &&
            correctness[index].strides == fp16.tensors[index].strides &&
            correctness[index].binding_byte_offset == expected_offsets[index],
        "FP32 oracle policy did not preserve element-addressed metadata");
    require(same_tensor_spec(performance[index], fp16.tensors[index]),
            "performance policy did not retain the original FP16 tensor");
  }

  const std::array<std::pair<hb::Operation, std::array<std::int64_t, 3>>, 3>
      semantic_orders = {{
          {hb::Operation::kConvolutionFprop, {101, 102, 103}},
          {hb::Operation::kConvolutionDgrad, {103, 102, 101}},
          {hb::Operation::kConvolutionWgrad, {102, 103, 101}},
      }};
  for (const auto &[operation, expected_uids] : semantic_orders) {
    hb::BenchmarkCase test_case =
        convolution_case(operation, FLAGDNN_DATA_FLOAT16);
    const std::vector<hv::ReferenceTensor> semantic =
        hd::convolution_semantic_reference_tensors(test_case,
                                                   Policy::kCorrectnessOracle);
    require(semantic.size() == expected_uids.size(),
            "convolution semantic policy changed tensor arity");
    for (std::size_t index = 0; index < semantic.size(); ++index) {
      require(semantic[index].uid == expected_uids[index] &&
                  semantic[index].data_type == FLAGDNN_DATA_FLOAT32,
              "FP32 oracle policy changed a convolution semantic UID");
    }
  }

  for (flagdnnDataType_t data_type :
       {FLAGDNN_DATA_FLOAT32, FLAGDNN_DATA_BFLOAT16}) {
    const hb::BenchmarkCase unchanged =
        convolution_case(hb::Operation::kConvolutionFprop, data_type);
    require(!hd::convolution_uses_fp32_correctness_oracle(unchanged),
            "non-FP16 convolution entered the FP32-upcast policy");
    const std::vector<hb::TensorSpec> specs =
        hd::convolution_reference_specs(unchanged, Policy::kCorrectnessOracle);
    require(specs.size() == unchanged.tensors.size(),
            "non-FP16 convolution reference changed tensor arity");
    for (std::size_t index = 0; index < specs.size(); ++index) {
      require(same_tensor_spec(specs[index], unchanged.tensors[index]),
              "non-FP16 convolution was changed by the FP16 oracle policy");
    }
  }

  hb::BenchmarkCase misaligned = fp16;
  misaligned.tensors[1].binding_byte_offset = 1;
  require_invalid_argument(
      [&] {
        (void)hd::convolution_reference_specs(misaligned,
                                              Policy::kCorrectnessOracle);
      },
      "FP16 oracle policy accepted a sub-element binding offset");

  hb::BenchmarkCase overflowing = fp16;
  overflowing.tensors[1].binding_byte_offset =
      std::numeric_limits<std::size_t>::max() - 1;
  require_runtime_error(
      [&] {
        (void)hd::convolution_reference_specs(overflowing,
                                              Policy::kCorrectnessOracle);
      },
      "FP16 oracle policy accepted an overflowing FP32 binding offset");

  require_invalid_argument(
      [&] {
        (void)hd::convolution_reference_specs(
            fp16, static_cast<Policy>(std::numeric_limits<int>::max()));
      },
      "convolution reference accepted an invalid policy");

  const std::array<float, 1> source = {1.0F / 3.0F};
  const std::vector<std::uint8_t> fp16_bytes =
      hv::tensor_io::encode(source, FLAGDNN_DATA_FLOAT16);
  const std::vector<float> quantized =
      hv::tensor_io::decode(fp16_bytes, FLAGDNN_DATA_FLOAT16, source.size());
  const std::vector<std::uint8_t> fp32_bytes =
      hv::tensor_io::encode(quantized, FLAGDNN_DATA_FLOAT32);
  const std::vector<float> upcast =
      hv::tensor_io::decode(fp32_bytes, FLAGDNN_DATA_FLOAT32, source.size());
  require(quantized[0] != source[0] && upcast[0] == quantized[0],
          "FP32 oracle inputs were not quantized through source FP16 first");
}

void test_extended_storage_codecs() {
  namespace io = hv::tensor_io;
  for (const auto type :
       {FLAGDNN_DATA_FP8_E4M3, FLAGDNN_DATA_FP8_E5M2, FLAGDNN_DATA_FP8_E8M0}) {
    for (unsigned value = 0; value < 256; ++value) {
      const std::array<std::uint8_t, 1> bytes{static_cast<std::uint8_t>(value)};
      const auto decoded = io::decode(bytes, type, 1);
      if (std::isfinite(decoded[0]))
        require(io::encode(decoded, type)[0] == bytes[0],
                "finite FP8 byte did not round-trip exactly");
    }
  }
  const std::array<float, 6> values{1.0625F, 1.1875F, 1.0e6F,
                                    -1.0e6F, 0.0F,    -0.0F};
  const auto bytes = io::encode(values, FLAGDNN_DATA_FP8_E4M3);
  require(bytes == std::vector<std::uint8_t>{0x38, 0x3a, 0x7e, 0xfe, 0, 0x80},
          "E4M3 rounding, saturation or signed zero changed");
  const auto e5 = io::decode(
      io::encode(std::span(values).subspan(2, 2), FLAGDNN_DATA_FP8_E5M2),
      FLAGDNN_DATA_FP8_E5M2, 2);
  require(e5 == std::vector<float>{57344, -57344}, "E5M2 saturation changed");
  const std::array<float, 9> scales{1.0F,
                                    1.125F,
                                    2.25F,
                                    0.5625F,
                                    std::nextafter(1.0F, 2.0F),
                                    std::nextafter(1.0F, 0.0F),
                                    std::numeric_limits<float>::denorm_min(),
                                    std::ldexp(1.0F, 127),
                                    std::numeric_limits<float>::max()};
  require(
      io::encode(scales, FLAGDNN_DATA_FP8_E8M0) ==
          std::vector<std::uint8_t>{127, 128, 129, 127, 128, 127, 0, 254, 255},
      "E8M0 scale rounding must match NVIDIA's upward conversion");
  const std::array<float, 7> invalid_scales{
      0.0F,
      -0.0F,
      -1.0F,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
      io::kPaddingSentinel};
  require(io::encode(invalid_scales, FLAGDNN_DATA_FP8_E8M0) ==
              std::vector<std::uint8_t>{255, 255, 255, 255, 255, 255, 0},
          "E8M0 invalid values and padding must match NVIDIA's encoding");
  const std::array<float, 5> integers{-2147483648.0F, -16777216.0F, 0,
                                      16777216.0F, 2147483520.0F};
  const auto integer_result =
      io::decode(io::encode(integers, FLAGDNN_DATA_INT32), FLAGDNN_DATA_INT32,
                 integers.size());
  require(std::equal(integers.begin(), integers.end(), integer_result.begin()),
          "representable INT32 values changed");
  for (const float value :
       {2147483648.0F, std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity()}) {
    require_invalid_argument(
        [&] { (void)io::encode(std::span(&value, 1), FLAGDNN_DATA_INT32); },
        "invalid INT32 float conversion accepted");
  }
}

void test_accuracy_contract() {
  const auto scalar = hb::tensor(1, {});
  require(hv::tensor_io::storage_element_count(scalar) == 1 &&
              hv::tensor_io::scatter(std::array{3.0F}, scalar) ==
                  std::vector<float>{3.0F},
          "rank-zero tensors must have one storage element");
  const std::array<float, 2> exact = {1.0F, -2.0F};
  (void)hf::compare_outputs(exact, exact, 0.0, 0.0, "finite_exact");

  const std::array<float, 1> nan = {std::numeric_limits<float>::quiet_NaN()};
  require_runtime_error(
      [&] { (void)hf::compare_outputs(nan, nan, 0.0, 0.0, "nan"); },
      "matching NaNs must fail like the NVIDIA functional comparator");

  const std::array<float, 1> infinity = {
      std::numeric_limits<float>::infinity()};
  require_runtime_error(
      [&] {
        (void)hf::compare_outputs(infinity, infinity, 0.0, 0.0, "infinity");
      },
      "matching infinities must fail like the NVIDIA functional comparator");
  const std::array<float, 2> pool_boundaries = {
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity()};
  (void)hf::compare_outputs(pool_boundaries, pool_boundaries, 0.0, 0.0,
                            "pool_boundaries", "CPU",
                            hf::ComparisonRule::kCombinedTolerance);
  require_runtime_error(
      [&] {
        (void)hf::compare_outputs(nan, nan, 0.0, 0.0, "nan", "CPU",
                                  hf::ComparisonRule::kCombinedTolerance);
      },
      "allowing matching infinities must still reject NaNs");
  const std::array<float, 1> half_actual{-3.697265625F};
  const std::array<float, 1> half_expected{-3.6953125F};
  (void)hf::compare_outputs(half_actual, half_expected, 5e-4, 5e-4,
                            "NV_paired_tolerance", "CPU",
                            hf::ComparisonRule::kCombinedTolerance);
  require_runtime_error(
      [&] {
        (void)hf::compare_outputs(half_actual, half_expected, 5e-4, 5e-4,
                                  "either_tolerance", "CPU");
      },
      "paired comparison must not change existing comparison semantics");
  const std::array<float, 1> negative_infinity = {-infinity[0]};
  require_runtime_error(
      [&] {
        (void)hf::compare_outputs(infinity, negative_infinity, 0.0, 0.0,
                                  "opposite_infinities", "CPU",
                                  hf::ComparisonRule::kCombinedTolerance);
      },
      "opposite infinities must fail");
}

void test_binding_contract() {
  std::array<std::uint8_t, 64> storage{};
  require(hf::binding_pointer(storage.data(), 16,
                              hf::BindingAddress::kStorageBase) ==
              storage.data(),
          "hipDNN binding must retain the storage base");
  require(hf::binding_pointer(storage.data(), 16,
                              hf::BindingAddress::kTensorEntrance) ==
              storage.data() + 16,
          "FlagDNN binding must point at the tensor entrance");
}

void test_reduction_dtype_capability() {
  const hv::HipdnnTensorOperation operation =
      hv::make_hipdnn_reduction_operation(FLAGDNN_REDUCTION_ADD, 1, false);
  std::array<hv::ReferenceTensor, 2> tensors = {
      hv::ReferenceTensor{1, FLAGDNN_DATA_BFLOAT16, {2, 4}, {4, 1}, 0},
      hv::ReferenceTensor{2, FLAGDNN_DATA_BFLOAT16, {2}, {1}, 0},
  };
  const hv::HipdnnCapability bfloat16 =
      hv::hipdnn_tensor_capability(operation, tensors);
  require(!bfloat16.supported &&
              bfloat16.classification ==
                  hv::HipdnnCapabilityClass::kVendorUnsupported &&
              bfloat16.reason.find("no validated executable primitive") !=
                  std::string::npos,
          "BF16 reduction must remain outside the validated hipDNN allowlist");

  tensors[0].data_type = FLAGDNN_DATA_FLOAT32;
  tensors[1].data_type = FLAGDNN_DATA_FLOAT32;
  require(hv::hipdnn_tensor_capability(operation, tensors).supported,
          "FP32 reduction unexpectedly left the validated hipDNN allowlist");
}

void test_slice_stride_capability() {
  const auto check = [](std::vector<std::int64_t> dimensions,
                        std::vector<std::pair<std::int64_t, std::int64_t>>
                            slices,
                        std::vector<std::int64_t> steps, bool supported) {
    hv::ReferenceTensor input = hv::dense_reference_tensor(
        {1, FLAGDNN_DATA_FLOAT32, std::move(dimensions), {}, 0});
    std::vector<std::int64_t> output_dimensions;
    for (std::size_t axis = 0; axis < slices.size(); ++axis) {
      output_dimensions.push_back(
          (slices[axis].second - slices[axis].first + steps[axis] - 1) /
          steps[axis]);
    }
    const std::array<hv::ReferenceTensor, 2> tensors = {
        input, hv::dense_reference_tensor(
                   {2, FLAGDNN_DATA_FLOAT32, output_dimensions, {}, 0})};
    const auto operation =
        hv::make_hipdnn_slice_operation(std::move(slices), std::move(steps));
    const auto capability = hv::hipdnn_tensor_capability(operation, tensors);
    require(capability.supported == supported,
            "Slice stride capability differs from the validated DTK behavior");
    if (!supported) {
      require(capability.classification ==
                      hv::HipdnnCapabilityClass::kVendorUnsupported &&
                  capability.reason.find("non-integral packed stride ratios") !=
                      std::string::npos,
              "Slice reference limitation must have a specific vendor reason");
    }
  };

  // These descriptors return incorrect row/batch offsets with hipDNN runtime
  // 8910.
  check({3, 7}, {{0, 3}, {1, 7}}, {1, 2}, false);
  check({2, 3, 9}, {{0, 2}, {0, 3}, {1, 9}}, {1, 1, 2}, false);
  check({2, 16, 31}, {{0, 2}, {1, 16}, {0, 31}}, {1, 3, 2}, false);
  check({4, 8, 65}, {{1, 4}, {0, 8}, {1, 65}}, {2, 1, 4}, false);
  // Preserve adjacent aligned/padded rows and the existing 3D Slice cases.
  check({3, 6}, {{0, 3}, {0, 6}}, {1, 2}, true);
  check({3, 8}, {{0, 3}, {1, 7}}, {1, 2}, true);
  check({2, 4, 5}, {{0, 2}, {1, 4}, {0, 5}}, {1, 2, 1}, true);
  check({4, 6, 8}, {{1, 4}, {0, 6}, {2, 8}}, {1, 2, 3}, true);
  check({3, 5, 7}, {{1, 3}, {0, 5}, {1, 7}}, {1, 2, 2}, true);
}

void test_capability_classification() {
  hv::HipdnnPointwiseOperation operation;
  operation.kind = hv::HipdnnPointwiseKind::kAdd;
  operation.unavailable_reason.clear();
  const std::array<hv::ReferenceTensor, 2> invalid_arity = {
      hv::ReferenceTensor{1, FLAGDNN_DATA_FLOAT32, {1, 1, 4}, {4, 4, 1}, 0},
      hv::ReferenceTensor{2, FLAGDNN_DATA_FLOAT32, {1, 1, 4}, {4, 4, 1}, 0},
  };
  const hv::HipdnnCapability invalid =
      hv::hipdnn_pointwise_capability(operation, invalid_arity, true);
  require(!invalid.supported &&
              invalid.classification ==
                  hv::HipdnnCapabilityClass::kInvalidAdapterContract,
          "pointwise arity error was not classified as adapter-invalid");
  require_invalid_argument(
      [&] { hv::require_valid_hipdnn_adapter_contract(invalid, "contract"); },
      "adapter-invalid capability did not hard fail");

  const hv::HipdnnCapability vendor =
      hv::HipdnnCapability::vendor_unsupported("no exact primitive");
  hv::require_valid_hipdnn_adapter_contract(vendor, "contract");

  const hv::HipdnnCapability malformed_supported = {
      false, hv::HipdnnCapabilityClass::kSupported, "contradictory"};
  require_invalid_argument(
      [&] {
        hv::require_valid_hipdnn_adapter_contract(malformed_supported,
                                                  "contract");
      },
      "contradictory supported/classification capability was accepted");
  const hv::HipdnnCapability malformed_reason = {
      false, hv::HipdnnCapabilityClass::kVendorUnsupported, {}};
  require_invalid_argument(
      [&] {
        hv::require_valid_hipdnn_adapter_contract(malformed_reason, "contract");
      },
      "unsupported capability without a reason was accepted");
}

void test_pointwise_broadcast_stride_capability() {
  hv::HipdnnPointwiseOperation operation;
  operation.kind = hv::HipdnnPointwiseKind::kAdd;
  operation.unavailable_reason.clear();
  std::array<hv::ReferenceTensor, 3> tensors = {
      hv::ReferenceTensor{1, FLAGDNN_DATA_FLOAT32, {2, 3, 4}, {31, 9, 2}, 0},
      hv::ReferenceTensor{2, FLAGDNN_DATA_FLOAT32, {1, 4}, {13, 3}, 0},
      hv::ReferenceTensor{3, FLAGDNN_DATA_FLOAT32, {2, 3, 4}, {37, 11, 2}, 0},
  };
  const hv::HipdnnCapability strided =
      hv::hipdnn_pointwise_capability(operation, tensors, true);
  require(!strided.supported &&
              strided.classification ==
                  hv::HipdnnCapabilityClass::kVendorUnsupported &&
              strided.reason.find("outside the validated capability") !=
                  std::string::npos,
          "hipDNN broadcast B must remain outside the allowlist");
  tensors[1].strides = {4, 1};
  require(!hv::hipdnn_pointwise_capability(operation, tensors, true).supported,
          "packed hipDNN broadcast B was incorrectly admitted");
  tensors[1].dimensions = {2, 3, 4};
  tensors[1].strides = tensors[2].strides;
  require(!hv::hipdnn_pointwise_capability(operation, tensors, true).supported,
          "different hipDNN A/C strides were incorrectly admitted");
  tensors[0].strides = tensors[2].strides;
  require(hv::hipdnn_pointwise_capability(operation, tensors, true).supported,
          "same-shape same-stride hipDNN tensors left the allowlist");
}

void test_pointwise_capture_capability() {
  hv::HipdnnPointwiseOperation operation;
  operation.kind = hv::HipdnnPointwiseKind::kIdentity;
  operation.alpha = 1.0;
  operation.unavailable_reason.clear();
  require(hv::hipdnn_pointwise_capture_capability(operation).supported,
          "unit-alpha hipDNN identity left the capture allowlist");

  operation.alpha = -1.0;
  const std::array<hv::ReferenceTensor, 2> tensors = {
      hv::ReferenceTensor{1, FLAGDNN_DATA_FLOAT32, {1, 1, 4}, {4, 4, 1}, 0},
      hv::ReferenceTensor{2, FLAGDNN_DATA_FLOAT32, {1, 1, 4}, {4, 4, 1}, 0},
  };
  require(hv::hipdnn_pointwise_capability(operation, tensors, true).supported,
          "hipDNN neg direct primitive left the functional allowlist");
  const hv::HipdnnCapability neg =
      hv::hipdnn_pointwise_capture_capability(operation);
  require(
      !neg.supported &&
          neg.classification == hv::HipdnnCapabilityClass::kVendorUnsupported &&
          neg.reason.find("phase=capture-replay") != std::string::npos &&
          neg.reason.find("HIPDNN_CAPTURE_REPLAY_UNSAFE") != std::string::npos,
      "hipDNN neg capture-replay hazard left the allowlist");

  operation.kind = hv::HipdnnPointwiseKind::kAdd;
  require(hv::hipdnn_pointwise_capture_capability(operation).supported,
          "non-NEG alpha was incorrectly classified by the capture gate");
}

void test_benchmark_phase_names() {
  using Phase = hb::BenchmarkPhase;
  using Provider = hb::BenchmarkProviderKind;
  const std::array<std::pair<Phase, std::string_view>, 5> phases = {{
      {Phase::kProbe, "probe"},
      {Phase::kWarmup, "warmup"},
      {Phase::kCaptureBuild, "capture-build"},
      {Phase::kCaptureReplay, "capture-replay"},
      {Phase::kTiming, "timing"},
  }};

  require(hb::benchmark_provider_name(Provider::kFlagdnn) == "flagdnn" &&
              hb::benchmark_provider_name(Provider::kHipdnn) == "hipdnn",
          "benchmark provider names are not stable");
  for (const auto &[phase, expected] : phases) {
    require(hb::benchmark_phase_name(phase) == expected,
            "benchmark phase name is not stable");
    require(hb::benchmark_phase_context(Provider::kFlagdnn, phase) ==
                "provider=flagdnn phase=" + std::string(expected),
            "FlagDNN benchmark phase context is incomplete");
    require(hb::benchmark_phase_context(Provider::kHipdnn, phase) ==
                "provider=hipdnn phase=" + std::string(expected),
            "hipDNN benchmark phase context is incomplete");
  }

  require_invalid_argument(
      [] {
        (void)hb::benchmark_phase_name(
            static_cast<Phase>(std::numeric_limits<int>::max()));
      },
      "unknown benchmark phase acquired a diagnostic name");
  require_invalid_argument(
      [] {
        (void)hb::benchmark_provider_name(
            static_cast<Provider>(std::numeric_limits<int>::max()));
      },
      "unknown benchmark provider acquired a diagnostic name");
}

void test_activation_descriptor_canonicalization() {
  const auto require_canonical = [](const auto &tensors, std::int64_t batches,
                                    std::int64_t channels,
                                    std::string_view message) {
    const std::vector<std::int64_t> expected_dimensions = {batches, channels, 1,
                                                           1};
    const std::vector<std::int64_t> expected_strides = {channels, 1, 1, 1};
    require(!tensors.empty(), message);
    for (const hv::ReferenceTensor &tensor : tensors) {
      require(tensor.dimensions == expected_dimensions &&
                  tensor.strides == expected_strides,
              message);
    }
  };

  hv::HipdnnPointwiseOperation abs;
  abs.kind = hv::HipdnnPointwiseKind::kAbs;
  abs.unavailable_reason.clear();
  const std::array<hv::ReferenceTensor, 2> compact_thousand = {
      hv::ReferenceTensor{
          11, FLAGDNN_DATA_FLOAT32, {1, 1, 1000}, {1000, 1000, 1}, 0},
      hv::ReferenceTensor{
          12, FLAGDNN_DATA_FLOAT32, {1, 1, 1000}, {1000, 1000, 1}, 0},
  };
  const std::vector<hv::ReferenceTensor> thousand =
      hv::hipdnn_pointwise_descriptor_tensors(abs, compact_thousand);
  require_canonical(thousand, 1, 1000,
                    "compact 1x1x1000 activation was not canonicalized");
  require(thousand[0].uid == compact_thousand[0].uid &&
              thousand[1].uid == compact_thousand[1].uid &&
              thousand[0].data_type == compact_thousand[0].data_type &&
              thousand[0].binding_byte_offset == 0,
          "activation canonicalization changed non-layout metadata");

  auto channel_selection = compact_thousand;
  for (hv::ReferenceTensor &tensor : channel_selection) {
    tensor.dimensions = {2, 5, 1};
    tensor.strides = {5, 1, 1};
  }
  require_canonical(
      hv::hipdnn_pointwise_descriptor_tensors(abs, channel_selection), 2, 5,
      "activation channel was not the rightmost non-unit dimension");

  auto compact_permutation = compact_thousand;
  for (hv::ReferenceTensor &tensor : compact_permutation) {
    tensor.dimensions = {2, 3, 4};
    tensor.strides = {1, 2, 6};
  }
  require_canonical(
      hv::hipdnn_activation_descriptor_tensors(compact_permutation), 6, 4,
      "dense compact permutation was not canonicalized");

  auto unit_axis_strides = compact_thousand;
  unit_axis_strides[0].strides = {4000, 2000, 1};
  unit_axis_strides[1].strides = {9000, 3000, 1};
  require_canonical(
      hv::hipdnn_pointwise_descriptor_tensors(abs, unit_axis_strides), 1, 1000,
      "unit-axis strides incorrectly changed physical mapping eligibility");

  auto all_unit = compact_thousand;
  for (hv::ReferenceTensor &tensor : all_unit) {
    tensor.dimensions = {1, 1, 1};
  }
  all_unit[0].strides = {19, 7, 1};
  all_unit[1].strides = {23, 11, 1};
  require_canonical(hv::hipdnn_pointwise_descriptor_tensors(abs, all_unit), 1,
                    1, "all-unit activation did not use the scalar 4D view");

  hv::HipdnnPointwiseOperation sigmoid_backward;
  sigmoid_backward.kind = hv::HipdnnPointwiseKind::kSigmoidBackward;
  sigmoid_backward.unavailable_reason.clear();
  const std::array<hv::ReferenceTensor, 3> backward_tensors = {
      compact_thousand[0],
      compact_thousand[1],
      hv::ReferenceTensor{
          13, FLAGDNN_DATA_FLOAT32, {1, 1, 1000}, {1000, 1000, 1}, 0},
  };
  require_canonical(
      hv::hipdnn_pointwise_descriptor_tensors(sigmoid_backward,
                                              backward_tensors),
      1, 1000, "sigmoid backward activation sequence was not canonicalized");
}

void test_activation_descriptor_rejections() {
  const auto require_unchanged = [](const auto &actual, const auto &expected,
                                    std::string_view message) {
    require(actual.size() == expected.size(), message);
    for (std::size_t index = 0; index < actual.size(); ++index) {
      require(actual[index].uid == expected[index].uid &&
                  actual[index].data_type == expected[index].data_type &&
                  actual[index].dimensions == expected[index].dimensions &&
                  actual[index].strides == expected[index].strides &&
                  actual[index].binding_byte_offset ==
                      expected[index].binding_byte_offset,
              message);
    }
  };

  const std::array<hv::ReferenceTensor, 2> compact = {
      hv::ReferenceTensor{21, FLAGDNN_DATA_FLOAT32, {2, 3, 4}, {12, 4, 1}, 0},
      hv::ReferenceTensor{22, FLAGDNN_DATA_FLOAT32, {2, 3, 4}, {12, 4, 1}, 0},
  };

  auto offset = compact;
  offset[1].binding_byte_offset = sizeof(float);
  require_unchanged(hv::hipdnn_activation_descriptor_tensors(offset), offset,
                    "activation tensor with an offset was canonicalized");

  auto padded = compact;
  for (hv::ReferenceTensor &tensor : padded) {
    tensor.strides = {16, 4, 1};
  }
  require_unchanged(hv::hipdnn_activation_descriptor_tensors(padded), padded,
                    "padded activation tensor was canonicalized");

  auto aliased = compact;
  for (hv::ReferenceTensor &tensor : aliased) {
    tensor.dimensions = {2, 2, 2};
    tensor.strides = {1, 1, 5};
  }
  require_unchanged(
      hv::hipdnn_activation_descriptor_tensors(aliased), aliased,
      "alias-and-hole layout with a compact-sized span was canonicalized");

  auto different_mapping = compact;
  different_mapping[1].strides = {1, 2, 6};
  require_unchanged(hv::hipdnn_activation_descriptor_tensors(different_mapping),
                    different_mapping,
                    "activation input/output with different physical mappings "
                    "was canonicalized");

  const std::int64_t maximum_int = std::numeric_limits<int>::max();
  auto oversized_batch = compact;
  for (hv::ReferenceTensor &tensor : oversized_batch) {
    tensor.dimensions = {maximum_int, 2, 2};
    tensor.strides = {4, 2, 1};
  }
  require_unchanged(
      hv::hipdnn_activation_descriptor_tensors(oversized_batch),
      oversized_batch,
      "activation canonical batch exceeding hipDNN int was accepted");

  auto oversized_metadata = compact;
  for (hv::ReferenceTensor &tensor : oversized_metadata) {
    tensor.dimensions = {1, 1, 1000};
    tensor.strides = {maximum_int + 1, 1000, 1};
  }
  require_unchanged(
      hv::hipdnn_activation_descriptor_tensors(oversized_metadata),
      oversized_metadata,
      "activation metadata exceeding hipDNN int was canonicalized");

  hv::HipdnnPointwiseOperation binary;
  binary.kind = hv::HipdnnPointwiseKind::kAdd;
  binary.unavailable_reason.clear();
  const std::array<hv::ReferenceTensor, 3> binary_tensors = {
      compact[0],
      compact[1],
      hv::ReferenceTensor{23, FLAGDNN_DATA_FLOAT32, {2, 3, 4}, {12, 4, 1}, 0},
  };
  require_unchanged(
      hv::hipdnn_pointwise_descriptor_tensors(binary, binary_tensors),
      binary_tensors,
      "binary pointwise descriptors were treated as an activation");
}

void test_add_square_matcher() {
  const hb::BenchmarkCase valid = add_square_case();
  require(hd::matches_add_square_graph(valid),
          "canonical add-square graph was rejected");

  hb::BenchmarkCase mutated = valid;
  mutated.graph.nodes[0].input_uids[0] = mutated.tensors[0].uid;
  require(!hd::matches_add_square_graph(mutated),
          "matcher accepted a graph that does not square right");

  mutated = valid;
  mutated.graph.nodes[1].input_uids[1] = mutated.tensors[1].uid;
  require(!hd::matches_add_square_graph(mutated),
          "matcher accepted an invalid add-square UID topology");

  mutated = valid;
  mutated.graph.nodes[0].alpha = 0.5;
  require(!hd::matches_add_square_graph(mutated),
          "matcher ignored square alpha");

  mutated = valid;
  mutated.graph.nodes[1].alpha = 2.0;
  require(!hd::matches_add_square_graph(mutated), "matcher ignored add alpha");

  mutated = valid;
  mutated.graph.nodes[0].pointwise_attributes.flags =
      FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP;
  require(!hd::matches_add_square_graph(mutated),
          "matcher ignored active pointwise attributes");

  mutated = valid;
  mutated.graph.intermediates[0].dimensions = {3, 2};
  mutated.graph.intermediates[0].strides = {2, 1};
  require(!hd::matches_add_square_graph(mutated),
          "matcher ignored the intermediate tensor layout");

  mutated = valid;
  mutated.graph.intermediates[0].uid = mutated.tensors[1].uid;
  mutated.graph.nodes[0].output_uid = mutated.tensors[1].uid;
  mutated.graph.nodes[1].input_uids[1] = mutated.tensors[1].uid;
  require(!hd::matches_add_square_graph(mutated),
          "matcher accepted a non-unique intermediate UID");
}

} // namespace

int main() {
  test_status_classification();
  test_convolution_heuristic_fallback_classification();
  test_convolution_algorithm_policy();
  test_benchmark_fp16_convolution_oracle_policy();
  test_extended_storage_codecs();
  test_accuracy_contract();
  test_binding_contract();
  test_reduction_dtype_capability();
  test_slice_stride_capability();
  test_capability_classification();
  test_pointwise_broadcast_stride_capability();
  test_pointwise_capture_capability();
  test_benchmark_phase_names();
  test_activation_descriptor_canonicalization();
  test_activation_descriptor_rejections();
  test_add_square_matcher();
  return 0;
}
