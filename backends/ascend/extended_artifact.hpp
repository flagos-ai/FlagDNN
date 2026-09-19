/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "backends/ascend/artifact.hpp"
#include "runtime/json.hpp"
namespace flagdnn::ascend {
AscendArtifact
parse_extended_artifact(const flagdnn::native::json::Value &request,
                        const flagdnn::native::json::Value &manifest,
                        const std::filesystem::path &directory);
}
