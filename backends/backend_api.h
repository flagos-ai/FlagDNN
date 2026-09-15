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

#define FLAGDNN_BACKEND_ABI_VERSION_V2 2U
#define FLAGDNN_BACKEND_ABI_VERSION_V3 3U
#define FLAGDNN_BACKEND_GET_API_V2_SYMBOL "flagdnnBackendGetApiV2"
#define FLAGDNN_BACKEND_GET_API_V3_SYMBOL "flagdnnBackendGetApiV3"

/* Existing backends keep the original v2 source contract by default. */
#define FLAGDNN_BACKEND_ABI_VERSION FLAGDNN_BACKEND_ABI_VERSION_V2
#define FLAGDNN_BACKEND_GET_API_SYMBOL FLAGDNN_BACKEND_GET_API_V2_SYMBOL
#define FLAGDNN_BACKEND_EXECUTION_CONTRACT_VERSION 2U
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

typedef struct flagdnnBackendApiV3 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t execution_contract_version;
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

  /*
   * Must not compile, autotune, or synchronize. Execution contract revision 2
   * permits bounded, reclaimable host/device temporaries but no persistent or
   * execution-count-dependent resource growth. Bindings, native_stream, and
   * Graph workspace remain caller-owned until queued work completes. A
   * nonzero workspace requirement uses a non-null, 256-byte-aligned base;
   * when the requirement is zero, workspace and its alignment are ignored.
   */
  flagdnnBackendResult_t (*execute)(
      void* executable,
      void* native_stream,
      const flagdnnBackendBindingV2 bindings[],
      size_t binding_count,
      void* workspace,
      size_t workspace_size);
} flagdnnBackendApiV3;

typedef const flagdnnBackendApiV3* (*flagdnnBackendGetApiV3Function)(void);

/*
 * Optional build-environment extension; independent of executable ABI v2/v3.
 * Without it, core retains the legacy exclusive environment lock for builds.
 * With it, prepare_environment runs under that lock, then create_executable
 * runs under a shared lock and MUST NOT mutate the process environment.
 * is_environment_prepared is thread-safe, non-mutating and cannot throw.
 * Readiness is monotonic for the selected execution engine until unload.
 * The actual build input/context allow preparation of a real requested kernel;
 * plugins must not require a synthetic bootstrap kernel or a different device.
 */
#define FLAGDNN_BACKEND_GET_BUILD_API_V1_SYMBOL "flagdnnBackendGetBuildApiV1"
typedef struct flagdnnBackendBuildApiV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  int (*is_environment_prepared)(const char* execution_engine);
  flagdnnBackendResult_t (*prepare_environment)(
      void* context, const char* execution_engine,
      const flagdnnBackendBuildInputV2* input);
} flagdnnBackendBuildApiV1;
typedef const flagdnnBackendBuildApiV1*
    (*flagdnnBackendGetBuildApiV1Function)(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* FLAGDNN_BACKEND_API_H_ */
