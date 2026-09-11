// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef FLAGDNN_BACKENDS_THEAD_VALIDATION_ACDNN_FP8_CODEC_HPP_
#define FLAGDNN_BACKENDS_THEAD_VALIDATION_ACDNN_FP8_CODEC_HPP_

#include "common/common.hpp"
#include <memory>

namespace flagdnn::validation::thead {
// Finite E4M3/E5M2 values only. Encoding uses round-to-nearest, ties-to-even
// and saturation. Every conversion/arithmetic step executes through acDNN.
[[nodiscard]] std::unique_ptr<flagdnn::testing::TestExecutable>
make_acdnn_fp8_codec(const flagdnn::testing::TestTensor &input,
                    const flagdnn::testing::TestTensor &output);
}
#endif
