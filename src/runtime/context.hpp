/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_RUNTIME_CONTEXT_HPP_
#define FLAGDNN_RUNTIME_CONTEXT_HPP_

#include <flagdnn/flagdnn.h>

#include "backend_loader.hpp"
#include "error.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::native {

struct ArtifactPackage;

struct CompilerDependencyContent {
  std::string fingerprint;
  std::string sha256;
};

using CompilerDependencyContents =
    std::map<std::filesystem::path, CompilerDependencyContent>;

// This exception is deliberately narrower than a generic compiler failure.
// Only a positive ENOENT result from posix_spawnp is classified as the
// compiler being offline, which lets the cache layer distinguish that case
// without inspecting an error message.
class CompilerExecutableUnavailable final : public ApiError {
public:
  explicit CompilerExecutableUnavailable(std::string message)
      : ApiError(FLAGDNN_STATUS_COMPILATION_FAILED, std::move(message)) {}
};

// Backends that must temporarily configure a process-global dependency take
// the exclusive side only while publishing/restoring that state. Runtime
// environment reads and compiler spawning take the shared side. The compiler
// child is independent after posix_spawn returns, so long compilations remain
// concurrent.
std::shared_mutex &process_environment_mutex();

class RuntimeContext {
public:
  RuntimeContext(flagdnnBackend_t backend, std::int32_t device_ordinal);
  RuntimeContext(std::string backend_name, std::int32_t device_ordinal);
  ~RuntimeContext();

  RuntimeContext(const RuntimeContext &) = delete;
  RuntimeContext &operator=(const RuntimeContext &) = delete;

  void set_compiler(std::string executable, std::string compiler,
                    std::string cache_directory);

  [[nodiscard]] std::int32_t device_ordinal() const noexcept {
    return device_ordinal_;
  }
  [[nodiscard]] const std::string &backend_name() const noexcept {
    return backend_name_;
  }
  [[nodiscard]] const std::string &execution_engine() const noexcept {
    return execution_engine_;
  }
  [[nodiscard]] const std::string &target_fingerprint() const noexcept {
    return target_fingerprint_;
  }

  [[nodiscard]] std::unique_ptr<BackendExecutable>
  create_executable(const ArtifactPackage &artifact) const;

  [[nodiscard]] const std::string &compiler_executable() const noexcept {
    return compiler_executable_;
  }
  [[nodiscard]] const std::string &compiler() const noexcept {
    return compiler_;
  }
  [[nodiscard]] const std::filesystem::path &cache_directory() const noexcept {
    return cache_directory_;
  }

private:
  friend std::string
  query_compiler_identity(RuntimeContext &context,
                          const std::filesystem::path &graph_cache_directory,
                          bool force_refresh);
  friend ArtifactPackage prepare_artifact_package(RuntimeContext &context,
                                                  std::string_view graph_ir);

  void initialize(std::string backend_name);

  std::int32_t device_ordinal_ = 0;
  std::string backend_name_;
  std::string target_fingerprint_;
  std::string execution_engine_;
  std::shared_ptr<BackendLibrary> backend_library_;
  std::shared_ptr<BackendContext> backend_context_;
  std::string compiler_executable_;
  std::string compiler_;
  std::filesystem::path cache_directory_;
  // Serializes a complete compiler transaction (configuration snapshot,
  // identity query, cache lookup/build and publication) against
  // set_compiler(). Holding only the identity-query portion would allow a
  // concurrent reconfiguration to mix a new compiler with an old cache path.
  std::mutex compiler_configuration_mutex_;
  std::string compiler_identity_snapshot_;
  std::vector<std::filesystem::path> compiler_identity_dependencies_;
  std::string compiler_identity_dependencies_snapshot_;
  std::string compiler_identity_;
  // Native-computed hashes for only the last accepted dependency set. Full
  // fingerprints include inode, ctime, and symlink state; entries are never
  // shared between handles or persisted to disk.
  CompilerDependencyContents compiler_dependency_contents_;
};

class Executable {
public:
  Executable(std::unique_ptr<BackendExecutable> executable,
             std::vector<std::int64_t> binding_uids,
             std::size_t operation_count);
  ~Executable();

  Executable(const Executable &) = delete;
  Executable &operator=(const Executable &) = delete;

  [[nodiscard]] std::size_t operation_count() const noexcept {
    return operation_count_;
  }
  [[nodiscard]] std::size_t workspace_size() const noexcept {
    return executable_->workspace_size();
  }

  void execute(const flagdnnBinding_t bindings[], std::size_t binding_count,
               void *workspace, std::size_t workspace_size,
               flagdnnStream_t stream) const;

private:
  std::unique_ptr<BackendExecutable> executable_;
  std::vector<std::int64_t> binding_uids_;
  std::size_t operation_count_ = 0;
};

} // namespace flagdnn::native

#endif // FLAGDNN_RUNTIME_CONTEXT_HPP_
