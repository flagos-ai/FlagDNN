/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "runtime/context.hpp"

#include "error.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace flagdnn::native {

Executable::Executable(std::unique_ptr<BackendExecutable> executable,
                       std::vector<std::int64_t> binding_uids,
                       std::size_t operation_count)
    : executable_(std::move(executable)),
      binding_uids_(std::move(binding_uids)),
      operation_count_(operation_count) {
  if (executable_ == nullptr || binding_uids_.empty() ||
      operation_count_ == 0) {
    throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                   "invalid internal executable state");
  }
}

Executable::~Executable() = default;

void Executable::execute(const flagdnnBinding_t bindings[],
                         std::size_t binding_count,
                         void* workspace,
                         std::size_t workspace_size,
                         flagdnnStream_t stream) const {
  if (binding_count != binding_uids_.size() || bindings == nullptr) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "binding count does not match executable");
  }

  for (std::size_t left = 0; left < binding_count; ++left) {
    if (bindings[left].uid <= 0 || bindings[left].device_pointer == nullptr) {
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "binding UID and device pointer must be valid");
    }
  }

  for (const std::int64_t expected_uid : binding_uids_) {
    bool found = false;
    for (std::size_t supplied = 0; supplied < binding_count; ++supplied) {
      if (bindings[supplied].uid == expected_uid) {
        found = true;
        break;
      }
    }
    if (!found) {
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "a required tensor UID is missing or duplicated in bindings");
    }
  }

  executable_->execute(
      stream, bindings, binding_count, workspace, workspace_size);
}

}  // namespace flagdnn::native
