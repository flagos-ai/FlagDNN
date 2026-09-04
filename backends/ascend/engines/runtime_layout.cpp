/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/engines/runtime_layout.hpp"

#include <dlfcn.h>

#include <stdexcept>

#ifndef FLAGDNN_ASCEND_PRIVATE_RUNTIME_RELATIVE
#define FLAGDNN_ASCEND_PRIVATE_RUNTIME_RELATIVE ""
#endif

namespace flagdnn::ascend::detail {
namespace {

const int kRuntimeLayoutAnchor = 0;

[[nodiscard]] bool path_is_within(const std::filesystem::path& child,
                                  const std::filesystem::path& parent) {
  auto child_iterator = child.begin();
  for (auto parent_iterator = parent.begin();
       parent_iterator != parent.end();
       ++parent_iterator, ++child_iterator) {
    if (child_iterator == child.end() ||
        *child_iterator != *parent_iterator) {
      return false;
    }
  }
  return true;
}

}  // namespace

std::optional<std::filesystem::path> private_runtime_path(
    std::string_view relative) {
  if (FLAGDNN_ASCEND_PRIVATE_RUNTIME_RELATIVE[0] == '\0' ||
      relative.empty()) {
    return std::nullopt;
  }
  const std::filesystem::path relative_path(relative);
  if (relative_path.is_absolute() || relative_path.has_root_path()) {
    throw std::runtime_error("Ascend private runtime path must be relative");
  }

  Dl_info information{};
  if (::dladdr(&kRuntimeLayoutAnchor, &information) == 0 ||
      information.dli_fname == nullptr) {
    throw std::runtime_error("cannot locate the loaded Ascend plugin");
  }
  std::error_code error;
  const std::filesystem::path plugin =
      std::filesystem::canonical(information.dli_fname, error);
  if (error) {
    throw std::runtime_error("cannot canonicalize the loaded Ascend plugin");
  }
  const std::filesystem::path lexical_root =
      (plugin.parent_path() / FLAGDNN_ASCEND_PRIVATE_RUNTIME_RELATIVE)
          .lexically_normal();
  const bool root_exists = std::filesystem::exists(lexical_root, error);
  if (error) {
    throw std::runtime_error("cannot inspect the Ascend private runtime root");
  }
  if (!root_exists) {
    return std::nullopt;
  }
  const std::filesystem::file_status root_status =
      std::filesystem::symlink_status(lexical_root, error);
  if (error || std::filesystem::is_symlink(root_status) ||
      !std::filesystem::is_directory(root_status)) {
    throw std::runtime_error(
        "Ascend private runtime root is not a canonical directory");
  }
  const std::filesystem::path root =
      std::filesystem::canonical(lexical_root, error);
  if (error || root != lexical_root ||
      !path_is_within(root, plugin.parent_path())) {
    throw std::runtime_error(
        "Ascend private runtime root escapes the plugin directory");
  }
  const std::filesystem::path candidate =
      (root / relative_path).lexically_normal();
  if (!path_is_within(candidate, root)) {
    throw std::runtime_error("Ascend private runtime path escapes its root");
  }
  const std::filesystem::file_status status =
      std::filesystem::symlink_status(candidate, error);
  if (error || !std::filesystem::exists(status)) {
    throw std::runtime_error("Ascend private runtime entry is missing");
  }
  if (std::filesystem::is_symlink(status) ||
      (!std::filesystem::is_regular_file(status) &&
       !std::filesystem::is_directory(status))) {
    throw std::runtime_error(
        "Ascend private runtime entry has an invalid file type");
  }
  const std::filesystem::path canonical =
      std::filesystem::canonical(candidate, error);
  if (error || !path_is_within(canonical, root)) {
    throw std::runtime_error("Ascend private runtime entry is not canonical");
  }
  return canonical;
}

}  // namespace flagdnn::ascend::detail
