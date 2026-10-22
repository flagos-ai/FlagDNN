/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "common/attention.hpp"
#include "host_runner.hpp"
#include "reference/cpu/attention_fp8.hpp"
namespace flagdnn::testing::hygon_functional::host {
inline float attention_quantize(float value, flagdnnDataType_t type) {
  const std::array<float, 1> input{value};
  return io::decode(io::encode(input, type), type, 1)[0];
}
inline Values fp8_forward_reference(const SdpaFp8TestCase& c,
                                    const Values& values) {
  return reference::cpu::fp8_forward_reference(c, values, attention_quantize);
}
inline Values fp8_backward_reference(const SdpaFp8BackwardTestCase& c,
                                     const Values& values) {
  return reference::cpu::fp8_backward_reference(c, values, attention_quantize);
}
}  // namespace flagdnn::testing::hygon_functional::host
