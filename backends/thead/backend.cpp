// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "backends/backend_api.h"

#include "backends/thead/artifact.hpp"
#include "backends/thead/context.hpp"
#include "backends/thead/engines/engine.hpp"
#include "backends/thead/error.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>

#if defined(__GNUC__) || defined(__clang__)
#define FLAGDNN_BACKEND_EXPORT __attribute__((visibility("default")))
#else
#define FLAGDNN_BACKEND_EXPORT
#endif

namespace flagdnn::thead {
namespace {

const char* get_last_error() noexcept {
  return last_error();
}

flagdnnBackendResult_t create_context(std::int32_t device_ordinal,
                                      void** context) noexcept {
  return plugin_call([&] {
    require(context != nullptr, "context output pointer is null");
    *context = nullptr;
    std::unique_ptr<TheadContext> result =
        std::make_unique<TheadContext>(device_ordinal);
    *context = result.release();
  });
}

void destroy_context(void* context) noexcept {
  delete static_cast<TheadContext*>(context);
}

flagdnnBackendResult_t get_target_fingerprint(
    void* context,
    char* buffer,
    std::size_t buffer_size,
    std::size_t* required_size) noexcept {
  return plugin_call([&] {
    require(context != nullptr, "context is null");
    require(required_size != nullptr, "required size output is null");
    const std::string& target =
        static_cast<TheadContext*>(context)->target_fingerprint();
    *required_size = target.size() + 1;
    require(buffer != nullptr && buffer_size >= *required_size,
            "target fingerprint buffer is too small");
    std::memcpy(buffer, target.c_str(), *required_size);
  });
}

flagdnnBackendResult_t create_executable(
    void* context,
    const flagdnnBackendBuildInputV2* input,
    void** executable,
    std::size_t* workspace_size) noexcept {
  return plugin_call([&] {
    require(context != nullptr, "context is null");
    require(input != nullptr, "build input is null");
    require(input->struct_size >= sizeof(flagdnnBackendBuildInputV2),
            "build input structure is too small");
    require(executable != nullptr, "executable output pointer is null");
    require(workspace_size != nullptr,
            "workspace size output pointer is null");
    *executable = nullptr;
    *workspace_size = 0;
    const EngineBuildContext build_context =
        static_cast<TheadContext*>(context)->engine_build_context();
    ExecutionProgramArtifact artifact =
        load_and_validate_artifact(build_context, *input);
    std::unique_ptr<ExecutionEngine> result =
        create_execution_engine(build_context, std::move(artifact));
    *workspace_size = result->workspace_size();
    *executable = result.release();
  });
}

void destroy_executable(void* executable) noexcept {
  delete static_cast<ExecutionEngine*>(executable);
}

flagdnnBackendResult_t execute(
    void* executable,
    void* native_stream,
    const flagdnnBackendBindingV2 bindings[],
    std::size_t binding_count,
    void* workspace,
    std::size_t workspace_size) noexcept {
  return plugin_call([&] {
    require(executable != nullptr, "executable is null");
    require(native_stream != nullptr, "native PPU stream is null");
    static_cast<ExecutionEngine*>(executable)->execute(
        reinterpret_cast<CUstream>(native_stream),
        bindings,
        binding_count,
        workspace,
        workspace_size);
  });
}

const flagdnnBackendApiV2 api = {
    sizeof(flagdnnBackendApiV2),
    FLAGDNN_BACKEND_ABI_VERSION,
    "thead",
    &get_last_error,
    &create_context,
    &destroy_context,
    &get_target_fingerprint,
    &create_executable,
    &destroy_executable,
    &execute};

}  // namespace
}  // namespace flagdnn::thead

extern "C" FLAGDNN_BACKEND_EXPORT const flagdnnBackendApiV2*
flagdnnBackendGetApiV2(void) {
  return &flagdnn::thead::api;
}
