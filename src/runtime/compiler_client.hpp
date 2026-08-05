/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_RUNTIME_COMPILER_CLIENT_HPP_
#define FLAGDNN_RUNTIME_COMPILER_CLIENT_HPP_

#include <filesystem>
#include <string>

namespace flagdnn::native {

class RuntimeContext;

std::string query_compiler_identity(
    RuntimeContext& context,
    const std::filesystem::path& graph_cache_directory);

void compile_external_artifact(
    const RuntimeContext& context,
    const std::filesystem::path& request,
    const std::filesystem::path& output_directory);

}  // namespace flagdnn::native

#endif  // FLAGDNN_RUNTIME_COMPILER_CLIENT_HPP_
