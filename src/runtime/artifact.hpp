/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_RUNTIME_ARTIFACT_HPP_
#define FLAGDNN_RUNTIME_ARTIFACT_HPP_

#include <filesystem>
#include <string>
#include <string_view>

namespace flagdnn::native {
class RuntimeContext;

struct ArtifactPackage {
  std::filesystem::path directory;
  std::string request_sha256;
  std::string compiler_identity_sha256;
  std::string build_request;
  bool cache_hit = false;
};

void validate_artifact_directory(
    const std::filesystem::path& artifact_directory);

ArtifactPackage prepare_artifact_package(RuntimeContext& context,
                                         std::string_view graph_ir);

void invalidate_cached_artifact(const ArtifactPackage& artifact);

}  // namespace flagdnn::native

#endif  /* FLAGDNN_RUNTIME_ARTIFACT_HPP_ */
