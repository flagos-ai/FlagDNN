// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef FLAGDNN_THEAD_UNSUPPORTED_CASES_HPP_
#define FLAGDNN_THEAD_UNSUPPORTED_CASES_HPP_
#include <stdexcept>

#include "common/causal_convolution.hpp"
#include "common/fp8_matmul.hpp"
#include "common/index.hpp"
#include "common/moe_matmul.hpp"
#include "common/normalization_extended.hpp"
#include "common/position_embedding.hpp"
#include "common/random.hpp"
namespace flagdnn::validation::thead {
struct UnsupportedCase {
  std::string name;
  flagdnn::testing::TestTensor tensor;
};
inline std::vector<UnsupportedCase> unsupported_cases(
    std::string_view operation) {
  using namespace flagdnn::testing;
  const auto convert = [](const auto& cases) {
    std::vector<UnsupportedCase> result;
    for (const auto& t : cases) {
      if constexpr (requires { t.output; })
        result.push_back({t.name, t.output});
      else
        result.push_back({t.name, t.outputs.front()});
    }
    return result;
  };
  if (operation == "moe_grouped_matmul" ||
      operation == "moe_grouped_matmul_bwd")
    return convert(make_moe_matmul_cases(operation.ends_with("_bwd")));
  if (operation == "matmul_fp8") return convert(make_fp8_matmul_cases());
  if (operation == "causal_conv1d")
    return convert(make_causal_convolution_cases());
  if (operation == "rng") return convert(make_rng_cases());
  if (operation == "rope" || operation == "rope_backward")
    return convert(make_rope_cases(operation.ends_with("_backward")));
  if (operation == "concatenate" || operation == "gen_index")
    return convert(make_index_cases(operation));
  if (operation == "instancenorm" || operation == "adalayernorm" ||
      operation == "instancenorm_backward" ||
      operation == "adalayernorm_backward" ||
      operation == "layernorm_backward" || operation == "rmsnorm_backward")
    return convert(make_extended_normalization_cases(std::string(operation)));
  throw std::invalid_argument("unknown acDNN capability gate operator");
}
}  // namespace flagdnn::validation::thead
#endif
