/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backend_loader.hpp"

#include "error.hpp"

#include <dlfcn.h>

#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::native {
namespace {

const int loader_anchor = 0;

struct DynamicLibraryGuard {
  ~DynamicLibraryGuard() {
    if (value != nullptr) {
      (void)dlclose(value);
    }
  }

  void* release() noexcept {
    void* result = value;
    value = nullptr;
    return result;
  }

  void* value = nullptr;
};

bool safe_backend_name(std::string_view value) {
  if (value.empty() || value.size() > 63 || value.front() < 'a' ||
      value.front() > 'z') {
    return false;
  }
  for (const char character : value) {
    if ((character < 'a' || character > 'z') &&
        (character < '0' || character > '9') && character != '_') {
      return false;
    }
  }
  return true;
}

bool safe_target_fingerprint(std::string_view value) {
  if (value.empty() ||
      value.size() >= FLAGDNN_BACKEND_MAX_TARGET_FINGERPRINT) {
    return false;
  }
  for (const unsigned char character : value) {
    if (std::isalnum(character) == 0 && character != '_' &&
        character != '-' && character != '.') {
      return false;
    }
  }
  return true;
}

std::vector<std::filesystem::path> backend_candidates(
    std::string_view library_name) {
  std::vector<std::filesystem::path> result;
  const char* configured_path = std::getenv("FLAGDNN_BACKEND_PATH");
  if (configured_path != nullptr) {
    std::string_view remaining(configured_path);
    while (!remaining.empty()) {
      const std::size_t separator = remaining.find(':');
      const std::string_view directory = remaining.substr(0, separator);
      if (!directory.empty()) {
        result.emplace_back(std::filesystem::path(directory) / library_name);
      }
      if (separator == std::string_view::npos) {
        break;
      }
      remaining.remove_prefix(separator + 1);
    }
  }

  Dl_info information{};
  if (dladdr(&loader_anchor, &information) != 0 &&
      information.dli_fname != nullptr) {
    result.emplace_back(
        std::filesystem::path(information.dli_fname).parent_path() /
        library_name);
  }
  result.emplace_back(library_name);
  return result;
}

std::string dynamic_loader_error() {
  const char* message = dlerror();
  return message == nullptr ? "unknown dynamic loader error"
                            : std::string(message);
}

flagdnnStatus_t public_status(flagdnnBackendResult_t result) {
  switch (result) {
    case FLAGDNN_BACKEND_RESULT_SUCCESS:
      return FLAGDNN_STATUS_SUCCESS;
    case FLAGDNN_BACKEND_RESULT_INVALID_VALUE:
      return FLAGDNN_STATUS_INVALID_VALUE;
    case FLAGDNN_BACKEND_RESULT_ALLOC_FAILED:
      return FLAGDNN_STATUS_ALLOC_FAILED;
    case FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED:
      return FLAGDNN_STATUS_NOT_SUPPORTED;
    case FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR:
      return FLAGDNN_STATUS_BACKEND_ERROR;
    case FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED:
      return FLAGDNN_STATUS_COMPILATION_FAILED;
    case FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR:
      return FLAGDNN_STATUS_INTERNAL_ERROR;
  }
  return FLAGDNN_STATUS_INTERNAL_ERROR;
}

void check_backend(const BackendLibrary& library,
                   flagdnnBackendResult_t result,
                   const char* operation) {
  if (result == FLAGDNN_BACKEND_RESULT_SUCCESS) {
    return;
  }
  std::ostringstream message;
  message << library.name() << " backend " << operation << " failed";
  const char* detail = library.api().get_last_error();
  if (detail != nullptr && detail[0] != '\0') {
    message << ": " << detail;
  }
  throw ApiError(public_status(result), message.str());
}

void validate_backend_api(const flagdnnBackendApiV2* api,
                          std::string_view expected_name) {
  if (api == nullptr || api->struct_size < sizeof(flagdnnBackendApiV2) ||
      api->abi_version != FLAGDNN_BACKEND_ABI_VERSION ||
      api->backend_name == nullptr ||
      std::string_view(api->backend_name) != expected_name ||
      api->get_last_error == nullptr || api->create_context == nullptr ||
      api->destroy_context == nullptr ||
      api->get_target_fingerprint == nullptr ||
      api->create_executable == nullptr ||
      api->destroy_executable == nullptr || api->execute == nullptr) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "backend plugin has an incompatible ABI");
  }
}

}  // namespace

std::shared_ptr<BackendLibrary> BackendLibrary::load(
    std::string backend_name) {
  if (!safe_backend_name(backend_name)) {
    throw ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "backend name must match [a-z][a-z0-9_]{0,62}");
  }

  const std::string library_name =
      "libflagdnn_backend_" + backend_name + ".so." +
      std::to_string(FLAGDNN_BACKEND_ABI_VERSION);
  std::string failures;
  DynamicLibraryGuard library;
  for (const std::filesystem::path& candidate :
       backend_candidates(library_name)) {
    dlerror();
    library.value = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (library.value != nullptr) {
      break;
    }
    if (!failures.empty()) {
      failures += "; ";
    }
    failures += candidate.string() + ": " + dynamic_loader_error();
  }
  if (library.value == nullptr) {
    throw ApiError(
        FLAGDNN_STATUS_NOT_SUPPORTED,
        "cannot load " + backend_name + " backend plugin; searched "
        "FLAGDNN_BACKEND_PATH, the libflagdnn directory, and the dynamic "
        "loader path (" + failures + ")");
  }

  dlerror();
  void* symbol = dlsym(library.value, FLAGDNN_BACKEND_GET_API_SYMBOL);
  const char* symbol_error = dlerror();
  if (symbol_error != nullptr || symbol == nullptr) {
    throw ApiError(
        FLAGDNN_STATUS_NOT_SUPPORTED,
        backend_name + " backend plugin does not export " +
            std::string(FLAGDNN_BACKEND_GET_API_SYMBOL) + ": " +
            (symbol_error == nullptr ? "symbol is null"
                                     : std::string(symbol_error)));
  }
  const auto get_api =
      reinterpret_cast<flagdnnBackendGetApiV2Function>(symbol);
  const flagdnnBackendApiV2* api = get_api();
  validate_backend_api(api, backend_name);
  return std::shared_ptr<BackendLibrary>(new BackendLibrary(
      library.release(), api, std::move(backend_name)));
}

BackendLibrary::BackendLibrary(void* dynamic_library,
                               const flagdnnBackendApiV2* api,
                               std::string name)
    : dynamic_library_(dynamic_library),
      api_(api),
      name_(std::move(name)) {}

BackendLibrary::~BackendLibrary() {
  if (dynamic_library_ != nullptr) {
    (void)dlclose(dynamic_library_);
  }
}

BackendContext::BackendContext(std::shared_ptr<BackendLibrary> library,
                               std::int32_t device_ordinal)
    : library_(std::move(library)) {
  check_backend(*library_,
                library_->api().create_context(device_ordinal, &context_),
                "create_context");
  if (context_ == nullptr) {
    throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                   "backend returned a null context");
  }

  try {
    std::array<char, FLAGDNN_BACKEND_MAX_TARGET_FINGERPRINT> buffer{};
    std::size_t required_size = 0;
    check_backend(*library_,
                  library_->api().get_target_fingerprint(context_,
                                                         buffer.data(),
                                                         buffer.size(),
                                                         &required_size),
                  "get_target_fingerprint");
    if (required_size < 2 || required_size > buffer.size() ||
        buffer[required_size - 1] != '\0') {
      throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                     "backend returned an invalid target fingerprint size");
    }
    target_fingerprint_.assign(buffer.data(), required_size - 1);
    if (!safe_target_fingerprint(target_fingerprint_)) {
      throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                     "backend returned an unsafe target fingerprint");
    }
  } catch (...) {
    library_->api().destroy_context(context_);
    context_ = nullptr;
    throw;
  }
}

BackendContext::~BackendContext() {
  if (context_ != nullptr) {
    library_->api().destroy_context(context_);
  }
}

std::unique_ptr<BackendExecutable> BackendContext::create_executable(
    std::string_view graph_ir,
    const std::filesystem::path& artifact_directory,
    std::string_view request_sha256) const {
  const std::string directory = artifact_directory.string();
  const std::string hash(request_sha256);
  flagdnnBackendBuildInputV2 input{};
  input.struct_size = sizeof(input);
  input.graph_ir = graph_ir.data();
  input.graph_ir_size = graph_ir.size();
  input.artifact_directory = directory.c_str();
  input.request_sha256 = hash.c_str();

  void* executable = nullptr;
  std::size_t workspace_size = 0;
  check_backend(*library_,
                library_->api().create_executable(
                    context_, &input, &executable, &workspace_size),
                "create_executable");
  if (executable == nullptr) {
    throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                   "backend returned a null executable");
  }
  try {
    return std::make_unique<BackendExecutable>(
        library_, shared_from_this(), executable, workspace_size);
  } catch (...) {
    library_->api().destroy_executable(executable);
    throw;
  }
}

BackendExecutable::BackendExecutable(
    std::shared_ptr<BackendLibrary> library,
    std::shared_ptr<const BackendContext> context_keepalive,
    void* executable,
    std::size_t workspace_size)
    : library_(std::move(library)),
      context_keepalive_(std::move(context_keepalive)),
      executable_(executable),
      workspace_size_(workspace_size) {}

BackendExecutable::~BackendExecutable() {
  if (executable_ != nullptr) {
    library_->api().destroy_executable(executable_);
  }
}

void BackendExecutable::execute(flagdnnStream_t stream,
                                const flagdnnBinding_t bindings[],
                                std::size_t binding_count,
                                void* workspace,
                                std::size_t workspace_size) const {
  if (binding_count != 0 && bindings == nullptr) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE, "bindings are null");
  }
  if (workspace_size < workspace_size_ ||
      (workspace_size_ != 0 && workspace == nullptr)) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "workspace is smaller than executable requirement");
  }

  static_assert(sizeof(flagdnnBinding_t) ==
                sizeof(flagdnnBackendBindingV2));
  static_assert(alignof(flagdnnBinding_t) ==
                alignof(flagdnnBackendBindingV2));
  static_assert(offsetof(flagdnnBinding_t, uid) ==
                offsetof(flagdnnBackendBindingV2, uid));
  static_assert(offsetof(flagdnnBinding_t, device_pointer) ==
                offsetof(flagdnnBackendBindingV2, device_pointer));
  const auto* packed =
      reinterpret_cast<const flagdnnBackendBindingV2*>(bindings);
  check_backend(*library_,
                library_->api().execute(executable_,
                                        stream,
                                        packed,
                                        binding_count,
                                        workspace,
                                        workspace_size),
                "execute");
}

}  // namespace flagdnn::native
