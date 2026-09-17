// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef FLAGDNN_ILUVATAR_VALIDATION_CAPABILITY_SKIPS_HPP_
#define FLAGDNN_ILUVATAR_VALIDATION_CAPABILITY_SKIPS_HPP_

#include "common/fp8_matmul.hpp"
#include <cudnn.h>
#include <iostream>
#include <stdexcept>

namespace flagdnn::iluvatar::validation {
// NVIDIA registers this subset under the ordinary matmul API. Keep exactly
// the same shared cases visible; CoreX 7.6.5 has no FP8 DNN tensor ABI.
inline void emit_plain_fp8_matmul_skips() {
  if (CUDNN_VERSION != 7605 || cudnnGetVersion() != 7605)
    throw std::runtime_error(
        "CoreX FP8 DNN capability requires requalification");
  for (const auto &test_case : flagdnn::testing::make_fp8_matmul_cases()) {
    if (test_case.plain_matmul) {
      std::cout << "[SKIP][iluvatar] op=matmul case=" << test_case.name
                << " reason=DNN_FP8_UNAVAILABLE\n";
    }
  }
}
} // namespace flagdnn::iluvatar::validation
#endif
