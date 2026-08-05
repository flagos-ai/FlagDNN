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

class BackendLibrary {
 public:
  static std::shared_ptr<BackendLibrary> load(std::string backend_name);

  ~BackendLibrary();

  BackendLibrary(const BackendLibrary&) = delete;
  BackendLibrary& operator=(const BackendLibrary&) = delete;

  [[nodiscard]] const std::string& name() const noexcept { return name_; }
  [[nodiscard]] const flagdnnBackendApiV2& api() const noexcept {
    return *api_;
  }

 private:
  BackendLibrary(void* dynamic_library,
                 const flagdnnBackendApiV2* api,
                 std::string name);

  void* dynamic_library_ = nullptr;
  const flagdnnBackendApiV2* api_ = nullptr;
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
      std::string_view request_sha256) const;

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
