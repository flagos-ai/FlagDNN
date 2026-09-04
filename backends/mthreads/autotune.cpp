/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/mthreads/autotune.hpp"

#include "backends/autotune_policy.hpp"
#include "backends/mthreads/error.hpp"
#include "runtime/sha256.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>

namespace flagdnn::mthreads {
namespace {

bool logging_enabled() noexcept {
  const char* value = std::getenv("FLAGDNN_PRINT_AUTOTUNING");
  return value != nullptr && value[0] != '\0' &&
         std::string_view(value) != "0";
}

std::string candidate_identity(const StageArtifact& stage) {
  std::string material = "flagdnn-mthreads-autotune-candidates-v1";
  const auto append = [&](std::string_view value) {
    material.push_back('\0');
    material.append(value);
  };
  append(std::to_string(stage.id));
  append(stage.function_name);
  for (const KernelVariantArtifact& variant : stage.variants) {
    append(variant.variant_id);
    append(variant.full_signature);
    append(std::to_string(variant.grid[0]));
    append(std::to_string(variant.grid[1]));
    append(std::to_string(variant.grid[2]));
    append(std::to_string(variant.num_warps));
    append(std::to_string(variant.num_stages));
  }
  return flagdnn::native::sha256(material);
}

void prepare_cache_directory(const std::filesystem::path& cache,
                             std::size_t stage_id) {
  const std::filesystem::path parent = cache.parent_path();
  const std::filesystem::path root = parent.parent_path();
  const std::string expected_filename =
      "stage-" + std::to_string(stage_id) + ".json";
  require(
      cache.filename() == expected_filename &&
          parent.filename() == "tuning" &&
          std::filesystem::is_directory(root),
      "mthreads autotune cache path is invalid",
      FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);

  std::error_code error;
  const auto status = std::filesystem::symlink_status(parent, error);
  if (status.type() == std::filesystem::file_type::not_found ||
      error == std::errc::no_such_file_or_directory) {
    error.clear();
    if (!std::filesystem::create_directory(parent, error) || error) {
      throw MthreadsError(
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
          "cannot create mthreads autotune cache directory");
    }
  } else if (
      error || std::filesystem::is_symlink(status) ||
      !std::filesystem::is_directory(status)) {
    throw MthreadsError(
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
        "mthreads autotune cache parent is unsafe");
  }
  const std::filesystem::path canonical_parent =
      std::filesystem::canonical(parent, error);
  const std::filesystem::path canonical_root =
      std::filesystem::canonical(root, error);
  if (error || canonical_parent.parent_path() != canonical_root) {
    throw MthreadsError(
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
        "mthreads autotune cache escapes its artifact directory");
  }
}

}  // namespace

std::size_t select_mthreads_autotune_candidate(
    const EngineBuildContext& context,
    const StageArtifact& stage,
    std::string measurement_identity,
    const AutotuneCallbacks& callbacks) {
  require(
      stage.autotune && stage.variants.size() >= 2 &&
          callbacks.prepare && callbacks.warmup && callbacks.measure,
      "invalid mthreads autotune stage/callbacks",
      FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
  prepare_cache_directory(stage.selection_cache, stage.id);

  backend::autotune::SelectionRequest request;
  request.candidate_identity = candidate_identity(stage);
  request.device_identity = context.device_identity;
  request.measurement_identity = std::move(measurement_identity);
  request.cache_path = stage.selection_cache;
  request.warmup_milliseconds = stage.warmup;
  request.benchmark_milliseconds = stage.repetitions;
  request.candidate_ids.reserve(stage.variants.size());
  for (const KernelVariantArtifact& variant : stage.variants) {
    request.candidate_ids.push_back(variant.variant_id);
  }

  if (const auto cached =
          backend::autotune::find_cached_candidate(request)) {
    try {
      callbacks.prepare(*cached);
      if (logging_enabled()) {
        std::cerr << "[FlagDNN mthreads autotune] cache hit "
                  << request.candidate_identity.substr(0, 12) << " -> "
                  << request.candidate_ids[*cached] << '\n';
      }
      return *cached;
    } catch (const std::exception& error) {
      backend::autotune::discard_cached_candidate(request);
      if (logging_enabled()) {
        std::cerr << "[FlagDNN mthreads autotune] discarded cached "
                  << request.candidate_ids[*cached] << ": "
                  << error.what() << '\n';
      }
    }
  }

  for (std::size_t index = 0; index < stage.variants.size(); ++index) {
    callbacks.prepare(index);
  }
  const backend::autotune::SelectionResult result =
      backend::autotune::select_best_candidate(
          request, callbacks.warmup, callbacks.measure);
  require(
      result.candidate_index < stage.variants.size(),
      "mthreads autotune selected an invalid candidate",
      FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);
  if (logging_enabled()) {
    for (std::size_t index = 0;
         index < result.median_milliseconds.size(); ++index) {
      std::cerr << "[FlagDNN mthreads autotune] "
                << request.candidate_ids[index] << " median_ms="
                << result.median_milliseconds[index] << '\n';
    }
    std::cerr << "[FlagDNN mthreads autotune] selected "
              << request.candidate_ids[result.candidate_index] << '\n';
  }
  return result.candidate_index;
}

}  // namespace flagdnn::mthreads
