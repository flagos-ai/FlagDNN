// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_THEAD_VALIDATION_NUMERIC_TYPES_HPP_
#define FLAGDNN_BACKENDS_THEAD_VALIDATION_NUMERIC_TYPES_HPP_

#include <flagdnn/flagdnn.h>

#include <cstddef>
#include <span>
#include <vector>

namespace flagdnn::validation::thead {

[[nodiscard]] std::size_t element_size(flagdnnDataType_t data_type);

[[nodiscard]] std::vector<std::byte>
encode_floating(flagdnnDataType_t data_type, std::span<const float> values);

[[nodiscard]] std::vector<float>
decode_floating(flagdnnDataType_t data_type,
                std::span<const std::byte> bytes);

}  // namespace flagdnn::validation::thead

#endif  // FLAGDNN_BACKENDS_THEAD_VALIDATION_NUMERIC_TYPES_HPP_
