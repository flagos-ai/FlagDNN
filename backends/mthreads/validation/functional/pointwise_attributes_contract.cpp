/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/mthreads/validation/functional/pointwise_attributes.hpp"
#include "reference/cpu/pointwise.hpp"
#include "reference/cpu/benchmark.hpp"
#include "benchmark/common/case.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cpu = flagdnn::reference::cpu;
namespace mv = flagdnn::validation::mthreads;

namespace {

struct Attribute {
  std::uint64_t flag;
  double flagdnnPointwiseAttributes_t::*field;
};

constexpr std::array<Attribute, 6> kAttributes = {{
    {FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP,
     &flagdnnPointwiseAttributes_t::relu_lower_clip},
    {FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP,
     &flagdnnPointwiseAttributes_t::relu_upper_clip},
    {FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE,
     &flagdnnPointwiseAttributes_t::relu_lower_clip_slope},
    {FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA,
     &flagdnnPointwiseAttributes_t::swish_beta},
    {FLAGDNN_POINTWISE_ATTRIBUTE_ELU_ALPHA,
     &flagdnnPointwiseAttributes_t::elu_alpha},
    {FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA,
     &flagdnnPointwiseAttributes_t::softplus_beta},
}};

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void verify_attribute_presence() {
  flagdnnPointwiseAttributes_t attributes =
      FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  for (const auto& attribute : kAttributes) {
    attributes.*attribute.field = 0.1;
  }
  for (const auto& selected : kAttributes) {
    attributes.flags = selected.flag;
    const auto normalized =
        mv::normalize_frontend_pointwise_attributes(attributes);
    require(normalized.struct_size == attributes.struct_size &&
                normalized.version == attributes.version &&
                normalized.flags == attributes.flags,
            "attribute normalization changed ABI or presence bits");
    for (const auto& attribute : kAttributes) {
      // The exact float encoding of 0.1, widened to the public double field.
      const double expected =
          attribute.flag == selected.flag ? 0x1.99999ap-4 : 0.1;
      require(normalized.*attribute.field == expected,
              "enabled or absent attribute precision differs");
      require(attributes.*attribute.field == 0.1,
              "attribute normalization mutated its input");
    }
  }
  attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_FLAGS_ALL;
  const auto normalized =
      mv::normalize_frontend_pointwise_attributes(attributes);
  for (const auto& attribute : kAttributes) {
    require(normalized.*attribute.field == 0x1.99999ap-4,
            "normalizing all enabled attributes missed a field");
  }

  // Absent fields retain their bits and remain absent, including NaN/Inf.
  constexpr std::array<double, 6> unused = {
      0.1, std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN(), -0.0, 1.0e300};
  attributes.flags = 0;
  for (std::size_t i = 0; i < kAttributes.size(); ++i) {
    attributes.*kAttributes[i].field = unused[i];
  }
  const auto absent =
      mv::normalize_frontend_pointwise_attributes(attributes);
  require(absent.flags == 0, "normalization enabled an absent attribute");
  for (const auto& attribute : kAttributes) {
    require(std::bit_cast<std::uint64_t>(absent.*attribute.field) ==
                std::bit_cast<std::uint64_t>(attributes.*attribute.field),
            "normalization altered an absent attribute");
  }
  const std::vector<float> dy{2};
  const std::vector<float> x{0};
  require(cpu::evaluate_activation_backward(
              FLAGDNN_POINTWISE_ELU_BWD, dy, x, absent) ==
              std::vector<float>{2},
          "absent attributes changed the CPU default");
}

void verify_relu_boundaries() {
  const auto evaluate = [](const flagdnnPointwiseAttributes_t& attributes,
                            const std::vector<float>& x) {
    const std::vector<float> dy(x.size(), 1.0F);
    const auto result = cpu::evaluate_activation_backward(
        FLAGDNN_POINTWISE_RELU_BWD, dy, x, attributes);
    flagdnn::benchmarking::BenchmarkCase specification;
    specification.operation = flagdnn::benchmarking::Operation::kPointwise;
    specification.pointwise_mode = FLAGDNN_POINTWISE_RELU_BWD;
    specification.pointwise_attributes = attributes;
    for (std::int64_t uid = 1; uid <= 3; ++uid) {
      specification.tensors.push_back(
          {uid, FLAGDNN_DATA_FLOAT32,
           {static_cast<std::int64_t>(x.size())}, {1}});
    }
    const auto benchmark = cpu::evaluate_benchmark(
        specification, {dy, x},
        [](std::span<const float> values, flagdnnDataType_t dtype) {
          require(dtype == FLAGDNN_DATA_FLOAT32, "unexpected benchmark dtype");
          return std::vector<float>(values.begin(), values.end());
        });
    require(benchmark == std::vector<std::vector<float>>{result},
            "functional and benchmark CPU boundary semantics differ");
    return result;
  };
  const auto neighbours = [](float value) {
    return std::vector<float>{
        std::nextafter(value, -std::numeric_limits<float>::infinity()),
        value,
        std::nextafter(value, std::numeric_limits<float>::infinity())};
  };
  flagdnnPointwiseAttributes_t attributes =
      FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP;
  attributes.relu_lower_clip = 0.1;
  const auto lower = neighbours(0.1F);
  require(evaluate(attributes, lower) == std::vector<float>({0, 1, 1}),
          "generic CPU oracle must retain double attribute semantics");
  require(evaluate(mv::normalize_frontend_pointwise_attributes(attributes),
                   lower) == std::vector<float>({0, 0, 1}),
          "frontend lower clip boundary or neighbours differ");

  attributes = FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP;
  attributes.relu_upper_clip = 0.7;
  const auto upper = neighbours(0.7F);
  require(evaluate(attributes, upper) == std::vector<float>({1, 1, 0}),
          "generic CPU oracle must retain double upper clip semantics");
  require(evaluate(mv::normalize_frontend_pointwise_attributes(attributes),
                   upper) == std::vector<float>({1, 0, 0}),
          "frontend upper clip boundary or neighbours differ");

  attributes.flags |= FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP |
                      FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE;
  attributes.relu_lower_clip = 0.1;
  attributes.relu_lower_clip_slope = 0.1;
  const auto normalized =
      mv::normalize_frontend_pointwise_attributes(attributes);
  require(evaluate(normalized, lower) ==
              std::vector<float>({0.1F, 0.1F, 1.0F}),
          "clipped leaky ReLU lower boundary differs");
  require(evaluate(normalized, upper) == std::vector<float>({1, 0, 0}),
          "clipped leaky ReLU upper boundary differs");
}

}  // namespace

int main() {
  try {
    verify_attribute_presence();
    verify_relu_boundaries();
    std::cout << "PASS MThreads frontend pointwise attribute contracts\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
