/*
 * Copyright (c) 2025-2026 BAAI. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * FlagDNN private, versioned, backend-neutral plugin ABI. This header is
 * shared by the runtime loader and backend plugins, but is not installed as
 * public SDK surface.
 */

#ifndef FLAGDNN_BACKEND_API_H_
#define FLAGDNN_BACKEND_API_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FLAGDNN_BACKEND_ABI_VERSION 2U
#define FLAGDNN_BACKEND_GET_API_SYMBOL "flagdnnBackendGetApiV2"
#define FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS 4096U
#define FLAGDNN_BACKEND_MAX_EXECUTION_STAGES 65536U
#define FLAGDNN_BACKEND_MAX_TARGET_FINGERPRINT 128U

typedef enum flagdnnBackendResult {
  FLAGDNN_BACKEND_RESULT_SUCCESS = 0,
  FLAGDNN_BACKEND_RESULT_INVALID_VALUE = 1,
  FLAGDNN_BACKEND_RESULT_ALLOC_FAILED = 2,
  FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED = 3,
  FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR = 4,
  FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED = 5,
  FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR = 6
} flagdnnBackendResult_t;

/*
 * Core passes a versioned build request containing backend-neutral Graph IR,
 * compiler identity, and the directory emitted by the selected compiler
 * provider. The plugin exclusively owns interpretation of the Execution
 * Program, binary, entry point, launch and scratch metadata.
 */
typedef struct flagdnnBackendBuildInputV2 {
  uint32_t struct_size;
  const void* graph_ir;
  size_t graph_ir_size;
  const char* artifact_directory;
  const char* request_sha256;
} flagdnnBackendBuildInputV2;

typedef struct flagdnnBackendBindingV2 {
  int64_t uid;
  void* device_pointer;
} flagdnnBackendBindingV2;

typedef struct flagdnnBackendApiV2 {
  uint32_t struct_size;
  uint32_t abi_version;
  const char* backend_name;

  /* Thread-local diagnostic, valid until the next plugin call in this thread. */
  const char* (*get_last_error)(void);

  flagdnnBackendResult_t (*create_context)(int32_t device_ordinal,
                                           void** context);
  void (*destroy_context)(void* context);

  /*
   * Writes a NUL-terminated, file-system-safe opaque target identifier.
   * required_size includes the NUL terminator.
   */
  flagdnnBackendResult_t (*get_target_fingerprint)(
      void* context,
      char* buffer,
      size_t buffer_size,
      size_t* required_size);

  /*
   * Core keeps the creating context alive until destroy_executable returns.
   * The executable may therefore safely retain backend-context resources.
   * This is the build-time boundary: a backend may load modules and benchmark
   * artifact variants here, but temporary streams/allocations must be released
   * before returning.
   */
  flagdnnBackendResult_t (*create_executable)(
      void* context,
      const flagdnnBackendBuildInputV2* input,
      void** executable,
      size_t* workspace_size);
  void (*destroy_executable)(void* executable);

  /* Must not compile, allocate, or synchronize. */
  flagdnnBackendResult_t (*execute)(
      void* executable,
      void* native_stream,
      const flagdnnBackendBindingV2 bindings[],
      size_t binding_count,
      void* workspace,
      size_t workspace_size);
} flagdnnBackendApiV2;

typedef const flagdnnBackendApiV2* (*flagdnnBackendGetApiV2Function)(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* FLAGDNN_BACKEND_API_H_ */
