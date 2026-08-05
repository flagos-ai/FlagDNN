/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_RUNTIME_SHA256_HPP_
#define FLAGDNN_RUNTIME_SHA256_HPP_

#include <filesystem>
#include <string>
#include <string_view>

namespace flagdnn::native {

std::string sha256(std::string_view input);
std::string sha256_file(const std::filesystem::path& path);

}  // namespace flagdnn::native

#endif  // FLAGDNN_RUNTIME_SHA256_HPP_
