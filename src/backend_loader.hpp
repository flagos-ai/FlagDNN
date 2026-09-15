/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKEND_LOADER_HPP_
#define FLAGDNN_BACKEND_LOADER_HPP_

#include <flagdnn/flagdnn.h>

#include "backends/backend_api.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace flagdnn::native {

class BackendExecutable;

struct BackendApiDispatch {
  flagdnnBackendBuildApiV1 build_environment{};
  const char* (*get_last_error)(void) = nullptr;
  flagdnnBackendResult_t (*create_context)(std::int32_t, void**) = nullptr;
  void (*destroy_context)(void*) = nullptr;
  flagdnnBackendResult_t (*get_target_fingerprint)(
      void*, char*, std::size_t, std::size_t*) = nullptr;
  flagdnnBackendResult_t (*create_executable)(
      void*,
      const flagdnnBackendBuildInputV2*,
      void**,
      std::size_t*) = nullptr;
  void (*destroy_executable)(void*) = nullptr;
  flagdnnBackendResult_t (*execute)(
      void*,
      void*,
      const flagdnnBackendBindingV2[],
      std::size_t,
      void*,
      std::size_t) = nullptr;
};

class BackendLibrary {
 public:
  static std::shared_ptr<BackendLibrary> load(std::string backend_name);

  ~BackendLibrary();

  BackendLibrary(const BackendLibrary&) = delete;
  BackendLibrary& operator=(const BackendLibrary&) = delete;

  [[nodiscard]] const std::string& name() const noexcept { return name_; }
  [[nodiscard]] const BackendApiDispatch& api() const noexcept {
    return api_;
  }
  void prepare_build_environment(
      void* context, const char* execution_engine,
      const flagdnnBackendBuildInputV2& input);

 private:
  BackendLibrary(void* dynamic_library,
                 BackendApiDispatch api,
                 std::string name);

  void* dynamic_library_ = nullptr;
  BackendApiDispatch api_{};
  std::string name_;
};

class BackendContext
    : public std::enable_shared_from_this<BackendContext> {
 public:
  BackendContext(std::shared_ptr<BackendLibrary> library,
                 std::int32_t device_ordinal);
  ~BackendContext();

  BackendContext(const BackendContext&) = delete;
  BackendContext& operator=(const BackendContext&) = delete;

  [[nodiscard]] const std::string& target_fingerprint() const noexcept {
    return target_fingerprint_;
  }

  [[nodiscard]] std::unique_ptr<BackendExecutable> create_executable(
      std::string_view graph_ir,
      const std::filesystem::path& artifact_directory,
      std::string_view request_sha256,
      std::string_view execution_engine) const;

 private:
  std::shared_ptr<BackendLibrary> library_;
  void* context_ = nullptr;
  std::string target_fingerprint_;
};

class BackendExecutable {
 public:
  BackendExecutable(std::shared_ptr<BackendLibrary> library,
                    std::shared_ptr<const BackendContext> context_keepalive,
                    void* executable,
                    std::size_t workspace_size);
  ~BackendExecutable();

  BackendExecutable(const BackendExecutable&) = delete;
  BackendExecutable& operator=(const BackendExecutable&) = delete;

  [[nodiscard]] std::size_t workspace_size() const noexcept {
    return workspace_size_;
  }

  void execute(flagdnnStream_t stream,
               const flagdnnBinding_t bindings[],
               std::size_t binding_count,
               void* workspace,
               std::size_t workspace_size) const;

 private:
  std::shared_ptr<BackendLibrary> library_;
  std::shared_ptr<const BackendContext> context_keepalive_;
  void* executable_ = nullptr;
  std::size_t workspace_size_ = 0;
};

}  // namespace flagdnn::native

#endif  /* FLAGDNN_BACKEND_LOADER_HPP_ */
