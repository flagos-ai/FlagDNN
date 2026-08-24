// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "common/attention.hpp"

#include <stdexcept>

namespace flagdnn::testing {
namespace {

[[noreturn]] void unsupported(const char *operation) {
  throw std::logic_error(std::string(operation) +
                         " has no exported CoreX cuDNN reference primitive");
}

} // namespace

std::unique_ptr<AttentionExecutable>
build_sdpa_reference(const SdpaTestCase &) {
  unsupported("SDPA");
}

std::unique_ptr<AttentionExecutable>
build_sdpa_backward_reference(const SdpaBackwardTestCase &) {
  unsupported("SDPA backward");
}

std::unique_ptr<AttentionExecutable>
build_sdpa_fp8_reference(const SdpaFp8TestCase &) {
  unsupported("FP8 SDPA");
}

std::unique_ptr<AttentionExecutable>
build_sdpa_fp8_backward_reference(const SdpaFp8BackwardTestCase &) {
  unsupported("FP8 SDPA backward");
}

} // namespace flagdnn::testing
