/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_RUNTIME_CONTEXT_HPP_
#define FLAGDNN_RUNTIME_CONTEXT_HPP_

#include <flagdnn/flagdnn.h>

#include "backend_loader.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace flagdnn::native {

struct ArtifactPackage;

class RuntimeContext {
 public:
  RuntimeContext(flagdnnBackend_t backend, std::int32_t device_ordinal);
  RuntimeContext(std::string backend_name, std::int32_t device_ordinal);
  ~RuntimeContext();

  RuntimeContext(const RuntimeContext&) = delete;
  RuntimeContext& operator=(const RuntimeContext&) = delete;

  void set_compiler(std::string executable,
                    std::string compiler,
                    std::string cache_directory);

  [[nodiscard]] std::int32_t device_ordinal() const noexcept {
    return device_ordinal_;
  }
  [[nodiscard]] const std::string& backend_name() const noexcept {
    return backend_name_;
  }
  [[nodiscard]] const std::string& execution_engine() const noexcept {
    return execution_engine_;
  }
  [[nodiscard]] const std::string& target_fingerprint() const noexcept {
    return target_fingerprint_;
  }

  [[nodiscard]] std::unique_ptr<BackendExecutable> create_executable(
      const ArtifactPackage& artifact) const;

  [[nodiscard]] const std::string& compiler_executable() const noexcept {
    return compiler_executable_;
  }
  [[nodiscard]] const std::string& compiler() const noexcept {
    return compiler_;
  }
  [[nodiscard]] const std::filesystem::path& cache_directory() const noexcept {
    return cache_directory_;
  }
 private:
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
};

class Executable {
 public:
  Executable(std::unique_ptr<BackendExecutable> executable,
             std::vector<std::int64_t> binding_uids,
             std::size_t operation_count);
  ~Executable();

  Executable(const Executable&) = delete;
  Executable& operator=(const Executable&) = delete;

  [[nodiscard]] std::size_t operation_count() const noexcept {
    return operation_count_;
  }
  [[nodiscard]] std::size_t workspace_size() const noexcept {
    return executable_->workspace_size();
  }

  void execute(const flagdnnBinding_t bindings[],
               std::size_t binding_count,
               void* workspace,
               std::size_t workspace_size,
               flagdnnStream_t stream) const;

 private:
  std::unique_ptr<BackendExecutable> executable_;
  std::vector<std::int64_t> binding_uids_;
  std::size_t operation_count_ = 0;
};

}  // namespace flagdnn::native

#endif  // FLAGDNN_RUNTIME_CONTEXT_HPP_
