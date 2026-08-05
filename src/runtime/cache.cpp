/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "runtime/artifact.hpp"

#include "error.hpp"
#include "runtime/compiler_client.hpp"
#include "runtime/context.hpp"
#include "runtime/sha256.hpp"

#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace flagdnn::native {
namespace {

std::atomic<std::uint64_t> cache_temporary_counter{0};

void quarantine_artifact_directory(
    const std::filesystem::path& artifact_directory) {
  std::error_code exists_error;
  if (!std::filesystem::exists(artifact_directory, exists_error)) {
    if (exists_error) {
      throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                     "cannot inspect cached artifact: " +
                         exists_error.message());
    }
    return;
  }

  const std::uint64_t serial = cache_temporary_counter.fetch_add(1);
  const std::filesystem::path quarantine =
      artifact_directory.parent_path() /
      (artifact_directory.filename().string() + ".invalid." +
       std::to_string(getpid()) + "." + std::to_string(serial));
  std::error_code rename_error;
  std::filesystem::rename(artifact_directory, quarantine, rename_error);
  if (rename_error) {
    if (rename_error == std::errc::no_such_file_or_directory) {
      return;
    }
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   "cannot quarantine cached artifact: " +
                       rename_error.message());
  }
  std::error_code ignored;
  std::filesystem::remove_all(quarantine, ignored);
}

bool is_sha256(std::string_view value) {
  if (value.size() != 64) {
    return false;
  }
  for (const char character : value) {
    if (!((character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f'))) {
      return false;
    }
  }
  return true;
}

void write_file(const std::filesystem::path& path,
                std::string_view contents,
                const char* description) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   std::string("cannot create ") + description + ": " +
                       path.string());
  }
  output.write(contents.data(),
               static_cast<std::streamsize>(contents.size()));
  output.close();
  if (!output) {
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   std::string("cannot write ") + description + ": " +
                       path.string());
  }
}

class TemporaryPath {
 public:
  TemporaryPath(std::filesystem::path path, bool directory)
      : path_(std::move(path)), directory_(directory) {}

  ~TemporaryPath() {
    if (!released_) {
      std::error_code ignored;
      if (directory_) {
        std::filesystem::remove_all(path_, ignored);
      } else {
        std::filesystem::remove(path_, ignored);
      }
    }
  }

  void release() noexcept { released_ = true; }

 private:
  std::filesystem::path path_;
  bool directory_ = false;
  bool released_ = false;
};

std::string read_cached_identity(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {};
  }
  std::string value((std::istreambuf_iterator<char>(input)),
                    std::istreambuf_iterator<char>());
  if (input.bad() || value.size() > 128) {
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   "compiler identity index is invalid");
  }
  while (!value.empty() &&
         (value.back() == '\n' || value.back() == '\r')) {
    value.pop_back();
  }
  return is_sha256(value) ? value : std::string{};
}

std::string make_build_request(std::string_view graph_ir,
                               std::string_view compiler_identity) {
  if (graph_ir.size() < 2 || graph_ir.back() != '}' ||
      !is_sha256(compiler_identity)) {
    throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                   "cannot construct versioned compiler request");
  }
  std::string result(graph_ir.substr(0, graph_ir.size() - 1));
  result += ",\"compiler_identity\":\"";
  result += compiler_identity;
  result += "\"}";
  return result;
}

void publish_active_identity(const std::filesystem::path& path,
                             std::string_view identity) {
  const std::uint64_t serial = cache_temporary_counter.fetch_add(1);
  const std::filesystem::path temporary =
      path.parent_path() /
      (".active_identity.tmp." + std::to_string(getpid()) + "." +
       std::to_string(serial));
  TemporaryPath cleanup(temporary, false);
  write_file(temporary, std::string(identity) + "\n", "identity index");

  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  if (error) {
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   "cannot publish compiler identity index: " +
                       error.message());
  }
  cleanup.release();
}

}  // namespace

ArtifactPackage prepare_artifact_package(RuntimeContext& context,
                                         std::string_view graph_ir) {
  if (context.cache_directory().empty()) {
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   "compiler cache directory is not configured");
  }
  if (graph_ir.empty()) {
    throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                   "cannot compile an empty graph IR");
  }

  const std::string graph_hash = sha256(graph_ir);
  const std::filesystem::path graph_cache_directory =
      context.cache_directory() / context.backend_name() /
      context.target_fingerprint() / graph_hash;
  std::error_code error;
  std::filesystem::create_directories(graph_cache_directory, error);
  if (error) {
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   "cannot create artifact cache directory: " +
                       error.message());
  }

  const std::filesystem::path active_identity_path =
      graph_cache_directory / "active_identity";
  std::string compiler_identity;
  bool compiler_available = true;
  try {
    compiler_identity =
        query_compiler_identity(context, graph_cache_directory);
  } catch (const ApiError& identity_error) {
    compiler_available = false;
    compiler_identity = read_cached_identity(active_identity_path);
    if (compiler_identity.empty()) {
      throw ApiError(
          FLAGDNN_STATUS_COMPILATION_FAILED,
          std::string(identity_error.what()) +
              "; no identity-indexed cached artifact is available");
    }
  }

  const std::string build_request =
      make_build_request(graph_ir, compiler_identity);
  const std::string request_hash = sha256(build_request);
  const std::filesystem::path artifact_directory =
      graph_cache_directory / compiler_identity;
  const std::filesystem::path manifest = artifact_directory / "manifest.json";

  error.clear();
  if (std::filesystem::is_regular_file(manifest, error) && !error) {
    if (compiler_available) {
      publish_active_identity(active_identity_path, compiler_identity);
    }
    return {artifact_directory,
            request_hash,
            compiler_identity,
            build_request,
            true};
  }
  if (!compiler_available) {
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   "cached artifact selected by active compiler identity is "
                   "missing");
  }

  // An interrupted or older compiler may leave a directory without a usable
  // manifest.  Move it out of the identity slot before publishing a rebuild.
  quarantine_artifact_directory(artifact_directory);

  const std::uint64_t serial = cache_temporary_counter.fetch_add(1);
  const std::filesystem::path temporary =
      graph_cache_directory /
      (compiler_identity + ".tmp." + std::to_string(getpid()) + "." +
       std::to_string(serial));
  std::filesystem::create_directory(temporary, error);
  if (error) {
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   "cannot create temporary artifact directory: " +
                       error.message());
  }

  TemporaryPath cleanup(temporary, true);
  const std::filesystem::path request_path = temporary / "request.json";
  write_file(request_path, build_request, "compiler request");
  compile_external_artifact(context, request_path, temporary);
  validate_artifact_directory(temporary);

  error.clear();
  std::filesystem::rename(temporary, artifact_directory, error);
  bool cache_hit = false;
  if (!error) {
    cleanup.release();
  } else {
    std::error_code manifest_error;
    if (!std::filesystem::is_regular_file(manifest, manifest_error) ||
        manifest_error) {
      throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                     "cannot publish compiled artifact: " + error.message());
    }
    cache_hit = true;
  }

  publish_active_identity(active_identity_path, compiler_identity);
  return {artifact_directory,
          request_hash,
          compiler_identity,
          build_request,
          cache_hit};
}

void invalidate_cached_artifact(const ArtifactPackage& artifact) {
  if (!artifact.cache_hit) {
    throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                   "cannot invalidate a newly compiled artifact");
  }
  quarantine_artifact_directory(artifact.directory);
}

}  // namespace flagdnn::native
