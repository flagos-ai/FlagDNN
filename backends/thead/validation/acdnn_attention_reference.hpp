// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef FLAGDNN_BACKENDS_THEAD_VALIDATION_ACDNN_ATTENTION_REFERENCE_HPP_
#define FLAGDNN_BACKENDS_THEAD_VALIDATION_ACDNN_ATTENTION_REFERENCE_HPP_
#include "capability.hpp"
#include "common/attention.hpp"
namespace flagdnn::validation::thead {
[[nodiscard]] std::vector<std::string> acdnn_attention_plan(bool backward);
[[nodiscard]] std::vector<std::string> acdnn_fp8_attention_plan(bool backward);
[[nodiscard]] std::unique_ptr<flagdnn::testing::AttentionExecutable>
make_acdnn_attention_reference(const flagdnn::testing::SdpaTestCase &test_case,
                               const CapabilityRecord &capability);
[[nodiscard]] std::unique_ptr<flagdnn::testing::AttentionExecutable>
make_acdnn_attention_reference(const flagdnn::testing::SdpaBackwardTestCase &test_case,
                               const CapabilityRecord &capability);
[[nodiscard]] std::unique_ptr<flagdnn::testing::AttentionExecutable>
make_acdnn_attention_reference(const flagdnn::testing::SdpaFp8TestCase &test_case,
                               const CapabilityRecord &capability);
[[nodiscard]] std::unique_ptr<flagdnn::testing::AttentionExecutable>
make_acdnn_attention_reference(const flagdnn::testing::SdpaFp8BackwardTestCase &test_case,
                               const CapabilityRecord &capability);
}  // namespace flagdnn::validation::thead
#endif
