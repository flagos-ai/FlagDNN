/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "runtime/context.hpp"

#include "error.hpp"
#include "runtime/artifact.hpp"

#include <dlfcn.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <utility>

#ifndef FLAGDNN_DEFAULT_COMPILER_EXECUTABLE
#define FLAGDNN_DEFAULT_COMPILER_EXECUTABLE ""
#endif

#ifndef FLAGDNN_DEFAULT_EXECUTION_ENGINE
#define FLAGDNN_DEFAULT_EXECUTION_ENGINE "external_artifact"
#endif

#ifndef FLAGDNN_DEFAULT_BACKEND
#define FLAGDNN_DEFAULT_BACKEND ""
#endif

#ifndef FLAGDNN_DEFAULT_CODEGEN_COMPILER
#define FLAGDNN_DEFAULT_CODEGEN_COMPILER ""
#endif

#ifndef FLAGDNN_INSTALLED_RESOURCE_RELATIVE
#define FLAGDNN_INSTALLED_RESOURCE_RELATIVE ""
#endif

namespace flagdnn::native {
namespace {

const int runtime_context_anchor = 0;

std::string configured_value(const char* environment_name,
                             const char* compiled_default) {
  const char* environment_value = std::getenv(environment_name);
  if (environment_value != nullptr && environment_value[0] != '\0') {
    return environment_value;
  }
  return compiled_default == nullptr ? std::string{} : compiled_default;
}

struct CompilerDefaults {
  std::string executable;
  std::string entry;
};

CompilerDefaults default_compiler_config() {
  if (FLAGDNN_INSTALLED_RESOURCE_RELATIVE[0] != '\0') {
    Dl_info information{};
    if (dladdr(&runtime_context_anchor, &information) != 0 &&
        information.dli_fname != nullptr) {
      const std::filesystem::path candidate =
          (std::filesystem::path(information.dli_fname).parent_path() /
           FLAGDNN_INSTALLED_RESOURCE_RELATIVE /
           "compiler/flagdnn_codegen/main.py")
              .lexically_normal();
      std::error_code error;
      if (std::filesystem::is_regular_file(candidate, error) && !error) {
        return {"python3", candidate.string()};
      }
    }
  }
  return {FLAGDNN_DEFAULT_COMPILER_EXECUTABLE,
          FLAGDNN_DEFAULT_CODEGEN_COMPILER};
}

std::string selected_backend_name(flagdnnBackend_t backend) {
  switch (backend) {
    case FLAGDNN_BACKEND_AUTO: {
      const char* configured = std::getenv("FLAGDNN_BACKEND");
      const std::string selected =
          configured == nullptr || configured[0] == '\0'
              ? FLAGDNN_DEFAULT_BACKEND
              : configured;
      if (selected.empty()) {
        throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                       "no default backend is configured; set "
                       "FLAGDNN_BACKEND or create the handle by name");
      }
      return selected;
    }
    case FLAGDNN_BACKEND_NVIDIA:
      return "nvidia";
  }
  throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                 "requested backend enum is not supported");
}

}  // namespace

RuntimeContext::RuntimeContext(flagdnnBackend_t backend,
                               std::int32_t device_ordinal)
    : RuntimeContext(selected_backend_name(backend), device_ordinal) {}

RuntimeContext::RuntimeContext(std::string backend_name,
                               std::int32_t device_ordinal)
    : device_ordinal_(device_ordinal) {
  initialize(std::move(backend_name));
}

void RuntimeContext::initialize(std::string backend_name) {
  if (device_ordinal_ < 0) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "device ordinal must be nonnegative");
  }
  backend_library_ = BackendLibrary::load(std::move(backend_name));
  backend_name_ = backend_library_->name();
  backend_context_ =
      std::make_shared<BackendContext>(backend_library_, device_ordinal_);
  target_fingerprint_ = backend_context_->target_fingerprint();

  execution_engine_ = configured_value("FLAGDNN_EXECUTION_ENGINE",
                                       FLAGDNN_DEFAULT_EXECUTION_ENGINE);
  if (execution_engine_ != "external_artifact" &&
      execution_engine_ != "libtriton_jit") {
    throw ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "FLAGDNN_EXECUTION_ENGINE must be external_artifact or libtriton_jit");
  }

  const CompilerDefaults compiler_defaults = default_compiler_config();
  compiler_executable_ = configured_value(
      "FLAGDNN_COMPILER_EXECUTABLE", compiler_defaults.executable.c_str());
  compiler_ = configured_value("FLAGDNN_CODEGEN_COMPILER",
                               compiler_defaults.entry.c_str());
  std::string cache = configured_value("FLAGDNN_CACHE_DIRECTORY", "");
  if (cache.empty()) {
    cache_directory_ = std::filesystem::temp_directory_path() /
                       ("flagdnn-cache-" + std::to_string(getuid()));
  } else {
    cache_directory_ = std::move(cache);
  }
}

RuntimeContext::~RuntimeContext() = default;

void RuntimeContext::set_compiler(std::string executable,
                                  std::string compiler,
                                  std::string cache_directory) {
  if (executable.empty() || compiler.empty() || cache_directory.empty()) {
    throw ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "compiler executable, compiler path, and cache directory must be "
        "nonempty");
  }
  compiler_executable_ = std::move(executable);
  compiler_ = std::move(compiler);
  cache_directory_ = std::move(cache_directory);
}

std::unique_ptr<BackendExecutable> RuntimeContext::create_executable(
    const ArtifactPackage& artifact) const {
  return backend_context_->create_executable(artifact.build_request,
                                             artifact.directory,
                                             artifact.request_sha256);
}

}  // namespace flagdnn::native
