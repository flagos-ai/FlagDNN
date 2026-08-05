/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "runtime/artifact.hpp"

#include "error.hpp"

#include <filesystem>
#include <system_error>

namespace flagdnn::native {

void validate_artifact_directory(
    const std::filesystem::path& artifact_directory) {
  const std::filesystem::path manifest =
      artifact_directory / "manifest.json";
  std::error_code error;
  if (!std::filesystem::is_regular_file(manifest, error) || error) {
    throw ApiError(
        FLAGDNN_STATUS_COMPILATION_FAILED,
        "external compiler did not produce a regular manifest.json");
  }
}

}  // namespace flagdnn::native
