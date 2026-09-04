/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_ENGINES_RUNTIME_LAYOUT_HPP_
#define FLAGDNN_BACKENDS_ASCEND_ENGINES_RUNTIME_LAYOUT_HPP_

#include <filesystem>
#include <optional>
#include <string_view>

namespace flagdnn::ascend::detail {

/* Resolve a regular file or directory in the plugin-relative private runtime
 * tree. An absent tree returns nullopt for build-tree fallback. Once the tree
 * exists, missing, malformed, symlinked or escaping entries fail closed. */
[[nodiscard]] std::optional<std::filesystem::path> private_runtime_path(
    std::string_view relative);

}  // namespace flagdnn::ascend::detail

#endif  // FLAGDNN_BACKENDS_ASCEND_ENGINES_RUNTIME_LAYOUT_HPP_
