/*
 * Copyright (c) 2025-2026 BAAI. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FLAGDNN_FLAGDNN_H_
#define FLAGDNN_FLAGDNN_H_

#include <stddef.h>
#include <stdint.h>

#include <flagdnn/version.h>

#if defined(_WIN32)
#if defined(FLAGDNN_BUILD_SHARED)
#define FLAGDNN_API __declspec(dllexport)
#elif defined(FLAGDNN_USE_SHARED)
#define FLAGDNN_API __declspec(dllimport)
#else
#define FLAGDNN_API
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define FLAGDNN_API __attribute__((visibility("default")))
#else
#define FLAGDNN_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum flagdnnStatus {
  FLAGDNN_STATUS_SUCCESS = 0,
  FLAGDNN_STATUS_INVALID_VALUE = 1,
  FLAGDNN_STATUS_NOT_INITIALIZED = 2,
  FLAGDNN_STATUS_ALLOC_FAILED = 3,
  FLAGDNN_STATUS_NOT_SUPPORTED = 4,
  FLAGDNN_STATUS_COMPILATION_FAILED = 5,
  FLAGDNN_STATUS_BACKEND_ERROR = 6,
  FLAGDNN_STATUS_INTERNAL_ERROR = 7
} flagdnnStatus_t;

typedef enum flagdnnBackend {
  FLAGDNN_BACKEND_AUTO = 0,
  FLAGDNN_BACKEND_NVIDIA = 1,
  /* Compatibility alias: NVIDIA currently uses the CUDA runtime. */
  FLAGDNN_BACKEND_CUDA = FLAGDNN_BACKEND_NVIDIA
} flagdnnBackend_t;

typedef enum flagdnnDataType {
  FLAGDNN_DATA_FLOAT32 = 0,
  FLAGDNN_DATA_FLOAT16 = 1,
  FLAGDNN_DATA_BFLOAT16 = 2,
  /* One byte per logical value in caller-owned storage. */
  FLAGDNN_DATA_BOOLEAN = 3,
  /* NVIDIA-compatible finite E4M3 and IEEE-like E5M2 FP8 encodings. */
  FLAGDNN_DATA_FP8_E4M3 = 4,
  FLAGDNN_DATA_FP8_E5M2 = 5
} flagdnnDataType_t;

typedef enum flagdnnOperation {
  FLAGDNN_OPERATION_RELU = 0,
  FLAGDNN_OPERATION_ADD = 1,
  FLAGDNN_OPERATION_REDUCTION = 2,
  /* Source-compatible alias retained for the original SUM-only API. */
  FLAGDNN_OPERATION_REDUCTION_SUM = FLAGDNN_OPERATION_REDUCTION,
  FLAGDNN_OPERATION_CONVOLUTION_FPROP = 3,
  /* Source-compatible alias retained for the original 2D-only API. */
  FLAGDNN_OPERATION_CONV2D_FPROP = FLAGDNN_OPERATION_CONVOLUTION_FPROP,
  /*
   * Generic pointwise descriptor used by the frontend-style Graph API.
   * The dedicated ReLU/Add operation values above remain supported for
   * source and binary compatibility.
   */
  FLAGDNN_OPERATION_POINTWISE = 4,
  /* Descriptor created by flagdnnCreateOperationDescriptorByName. */
  FLAGDNN_OPERATION_CUSTOM = 5,
  FLAGDNN_OPERATION_MATMUL = 6,
  FLAGDNN_OPERATION_SDPA = 7,
  FLAGDNN_OPERATION_SDPA_BACKWARD = 8,
  FLAGDNN_OPERATION_SDPA_FP8 = 9,
  FLAGDNN_OPERATION_SDPA_FP8_BACKWARD = 10
} flagdnnOperation_t;

typedef enum flagdnnDiagonalAlignment {
  FLAGDNN_DIAGONAL_ALIGNMENT_TOP_LEFT = 0,
  FLAGDNN_DIAGONAL_ALIGNMENT_BOTTOM_RIGHT = 1
} flagdnnDiagonalAlignment_t;

#define FLAGDNN_SDPA_ATTRIBUTES_VERSION 1U
#define FLAGDNN_SDPA_ATTRIBUTE_ATTN_SCALE (UINT64_C(1) << 0)
#define FLAGDNN_SDPA_ATTRIBUTE_LEFT_BOUND (UINT64_C(1) << 1)
#define FLAGDNN_SDPA_ATTRIBUTE_RIGHT_BOUND (UINT64_C(1) << 2)
#define FLAGDNN_SDPA_ATTRIBUTE_FLAGS_ALL                                  \
  (FLAGDNN_SDPA_ATTRIBUTE_ATTN_SCALE | FLAGDNN_SDPA_ATTRIBUTE_LEFT_BOUND | \
   FLAGDNN_SDPA_ATTRIBUTE_RIGHT_BOUND)

/*
 * Versioned, backend-neutral attributes shared by SDPA forward and backward.
 * An unset attention scale is inferred as 1/sqrt(head_dim). Diagonal bounds
 * follow the cuDNN Frontend convention and are relative to the selected
 * diagonal alignment. Forward consumes generate_stats; backward ignores it.
 */
typedef struct flagdnnSdpaAttributes {
  size_t struct_size;
  uint32_t version;
  uint64_t flags;
  double attn_scale;
  int64_t diagonal_band_left_bound;
  int64_t diagonal_band_right_bound;
  flagdnnDiagonalAlignment_t diagonal_alignment;
  int32_t generate_stats;
} flagdnnSdpaAttributes_t;

#define FLAGDNN_SDPA_ATTRIBUTES_INITIALIZER                            \
  {                                                                    \
    sizeof(flagdnnSdpaAttributes_t), FLAGDNN_SDPA_ATTRIBUTES_VERSION,  \
        0U, 0.0, 0, 0, FLAGDNN_DIAGONAL_ALIGNMENT_TOP_LEFT, 0         \
  }

/*
 * Backend-neutral pointwise modes. Names intentionally follow the cuDNN
 * Frontend vocabulary, while numeric values are owned by FlagDNN and must
 * never be passed directly to a platform library.
 */
typedef enum flagdnnPointwiseMode {
  FLAGDNN_POINTWISE_NOT_SET = 0,
  FLAGDNN_POINTWISE_ADD = 1,
  FLAGDNN_POINTWISE_RELU_FWD = 2,
  FLAGDNN_POINTWISE_SQRT = 3,
  FLAGDNN_POINTWISE_ERF = 4,
  FLAGDNN_POINTWISE_IDENTITY = 5,
  FLAGDNN_POINTWISE_EXP = 6,
  FLAGDNN_POINTWISE_LOG = 7,
  FLAGDNN_POINTWISE_NEG = 8,
  FLAGDNN_POINTWISE_ABS = 9,
  FLAGDNN_POINTWISE_CEIL = 10,
  FLAGDNN_POINTWISE_COS = 11,
  FLAGDNN_POINTWISE_FLOOR = 12,
  FLAGDNN_POINTWISE_RSQRT = 13,
  FLAGDNN_POINTWISE_SIN = 14,
  FLAGDNN_POINTWISE_TAN = 15,
  FLAGDNN_POINTWISE_RECIPROCAL = 16,
  FLAGDNN_POINTWISE_SUB = 17,
  FLAGDNN_POINTWISE_MUL = 18,
  FLAGDNN_POINTWISE_DIV = 19,
  FLAGDNN_POINTWISE_MIN = 20,
  FLAGDNN_POINTWISE_MAX = 21,
  FLAGDNN_POINTWISE_MOD = 22,
  FLAGDNN_POINTWISE_POW = 23,
  FLAGDNN_POINTWISE_LOGICAL_NOT = 24,
  FLAGDNN_POINTWISE_CMP_EQ = 25,
  FLAGDNN_POINTWISE_CMP_NEQ = 26,
  FLAGDNN_POINTWISE_CMP_GT = 27,
  FLAGDNN_POINTWISE_CMP_GE = 28,
  FLAGDNN_POINTWISE_CMP_LT = 29,
  FLAGDNN_POINTWISE_CMP_LE = 30,
  FLAGDNN_POINTWISE_LOGICAL_AND = 31,
  FLAGDNN_POINTWISE_LOGICAL_OR = 32,
  FLAGDNN_POINTWISE_SIGMOID_FWD = 33,
  FLAGDNN_POINTWISE_TANH_FWD = 34,
  FLAGDNN_POINTWISE_ELU_FWD = 35,
  FLAGDNN_POINTWISE_GELU_FWD = 36,
  FLAGDNN_POINTWISE_SOFTPLUS_FWD = 37,
  FLAGDNN_POINTWISE_SWISH_FWD = 38,
  FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD = 39,
  FLAGDNN_POINTWISE_SIGMOID_BWD = 40,
  FLAGDNN_POINTWISE_BINARY_SELECT = 41
} flagdnnPointwiseMode_t;

#define FLAGDNN_POINTWISE_ATTRIBUTES_VERSION 1U

/*
 * Presence bits keep zero-valued attributes distinguishable from defaults
 * and leave room for future pointwise modes without changing this ABI.
 */
#define FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP (UINT64_C(1) << 0)
#define FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP (UINT64_C(1) << 1)
#define FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE \
  (UINT64_C(1) << 2)
#define FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA (UINT64_C(1) << 3)
#define FLAGDNN_POINTWISE_ATTRIBUTE_ELU_ALPHA (UINT64_C(1) << 4)
#define FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA (UINT64_C(1) << 5)
#define FLAGDNN_POINTWISE_ATTRIBUTE_FLAGS_ALL                         \
  (FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP |                      \
   FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP |                      \
   FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE |                \
   FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA |                           \
   FLAGDNN_POINTWISE_ATTRIBUTE_ELU_ALPHA |                            \
   FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA)

/*
 * Versioned, backend-neutral unary pointwise attributes. A field is applied
 * only when its corresponding flag is set. Defaults match cuDNN Frontend:
 * ReLU lower clip/slope are zero, SWISH/ELU/SOFTPLUS parameters are one,
 * and ReLU has no upper clip.
 */
typedef struct flagdnnPointwiseAttributes {
  size_t struct_size;
  uint32_t version;
  uint64_t flags;
  double relu_lower_clip;
  double relu_upper_clip;
  double relu_lower_clip_slope;
  double swish_beta;
  double elu_alpha;
  double softplus_beta;
} flagdnnPointwiseAttributes_t;

#define FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER                         \
  {                                                                      \
    sizeof(flagdnnPointwiseAttributes_t),                                \
        FLAGDNN_POINTWISE_ATTRIBUTES_VERSION, 0U, 0.0, 0.0, 0.0, 1.0,   \
        1.0, 1.0                                                         \
  }

typedef enum flagdnnReductionMode {
  FLAGDNN_REDUCTION_ADD = 0,
  FLAGDNN_REDUCTION_SUM = FLAGDNN_REDUCTION_ADD,
  FLAGDNN_REDUCTION_AVG = 1,
  FLAGDNN_REDUCTION_MUL = 2
} flagdnnReductionMode_t;

typedef struct flagdnnContext* flagdnnHandle_t;
typedef struct flagdnnTensorDescriptor* flagdnnTensorDescriptor_t;
typedef struct flagdnnOperationDescriptor* flagdnnOperationDescriptor_t;
typedef struct flagdnnGraph* flagdnnGraph_t;
typedef struct flagdnnExecutable* flagdnnExecutable_t;

/*
 * A stream is an opaque native backend stream. For CUDA this is a
 * cudaStream_t/CUstream value. A null value denotes that backend's default
 * stream.
 */
typedef void* flagdnnStream_t;

typedef struct flagdnnBinding {
  int64_t uid;
  void* device_pointer;
} flagdnnBinding_t;

#define FLAGDNN_BUILD_OPTIONS_VERSION 1U

/*
 * Backend-neutral execution-plan candidate sources. A zero flags value keeps
 * source compatibility and selects FLAGDNN_BUILD_OPTION_HEURISTIC_MODE_A.
 * Providers may reject a requested mode explicitly, but must not silently
 * reinterpret these bits as vendor-specific enum values.
 */
#define FLAGDNN_BUILD_OPTION_HEURISTIC_MODE_A (UINT64_C(1) << 0)
#define FLAGDNN_BUILD_OPTION_HEURISTIC_MODE_FALLBACK (UINT64_C(1) << 1)
#define FLAGDNN_BUILD_OPTION_AUTOTUNE (UINT64_C(1) << 2)
#define FLAGDNN_BUILD_OPTION_FLAGS_ALL                                      \
  (FLAGDNN_BUILD_OPTION_HEURISTIC_MODE_A |                                  \
   FLAGDNN_BUILD_OPTION_HEURISTIC_MODE_FALLBACK |                           \
   FLAGDNN_BUILD_OPTION_AUTOTUNE)

typedef struct flagdnnBuildOptions {
  size_t struct_size;
  uint32_t version;
  uint64_t flags;
} flagdnnBuildOptions_t;

#define FLAGDNN_BUILD_OPTIONS_INITIALIZER                            \
  {                                                                  \
    sizeof(flagdnnBuildOptions_t), FLAGDNN_BUILD_OPTIONS_VERSION, 0U \
  }

FLAGDNN_API size_t flagdnnGetVersion(void);
FLAGDNN_API const char* flagdnnGetVersionString(void);
FLAGDNN_API const char* flagdnnGetErrorString(flagdnnStatus_t status);

/*
 * Returns the current thread's diagnostic for the most recent failed API
 * call. The returned pointer remains valid until the next FlagDNN API call
 * on the same thread.
 */
FLAGDNN_API const char* flagdnnGetLastErrorString(void);

FLAGDNN_API flagdnnStatus_t flagdnnCreate(flagdnnHandle_t* handle);
FLAGDNN_API flagdnnStatus_t flagdnnCreateWithBackend(
    flagdnnBackend_t backend,
    int32_t device_ordinal,
    flagdnnHandle_t* handle);
/* Creates a handle by an extensible platform name such as "nvidia". */
FLAGDNN_API flagdnnStatus_t flagdnnCreateWithBackendName(
    const char* backend_name,
    int32_t device_ordinal,
    flagdnnHandle_t* handle);
FLAGDNN_API flagdnnStatus_t flagdnnDestroy(flagdnnHandle_t handle);

/* Returned strings remain valid for the lifetime of handle. */
FLAGDNN_API flagdnnStatus_t flagdnnGetBackendName(
    flagdnnHandle_t handle,
    const char** backend_name);
FLAGDNN_API flagdnnStatus_t flagdnnGetTargetFingerprint(
    flagdnnHandle_t handle,
    const char** target_fingerprint);


/*
 * Configures the external compiler used on an executable-cache miss. Strings
 * are copied by FlagDNN. The compiler executable may be Python today, but the
 * public contract is backend-neutral and never embeds it in the caller.
 */
FLAGDNN_API flagdnnStatus_t flagdnnSetCompilerConfig(
    flagdnnHandle_t handle,
    const char* compiler_executable,
    const char* compiler_path,
    const char* cache_directory);

FLAGDNN_API flagdnnStatus_t flagdnnCreateTensorDescriptor(
    flagdnnTensorDescriptor_t* descriptor);
FLAGDNN_API flagdnnStatus_t flagdnnDestroyTensorDescriptor(
    flagdnnTensorDescriptor_t descriptor);
/* Rank zero describes a scalar; dimensions and strides may then be null. */
FLAGDNN_API flagdnnStatus_t flagdnnSetTensorNdDescriptor(
    flagdnnTensorDescriptor_t descriptor,
    int64_t uid,
    flagdnnDataType_t data_type,
    int32_t rank,
    const int64_t dimensions[],
    const int64_t strides[]);
FLAGDNN_API flagdnnStatus_t flagdnnGetTensorNdDescriptor(
    flagdnnTensorDescriptor_t descriptor,
    int32_t requested_rank,
    int64_t* uid,
    flagdnnDataType_t* data_type,
    int32_t* actual_rank,
    int64_t dimensions[],
    int64_t strides[]);
FLAGDNN_API flagdnnStatus_t flagdnnGetTensorSizeInBytes(
    flagdnnTensorDescriptor_t descriptor,
    size_t* size_in_bytes);
/*
 * Marks storage as executable-owned graph workspace. Virtual tensors may be
 * shared between operations by UID and must not appear in execute bindings.
 * The default after flagdnnSetTensorNdDescriptor is non-virtual.
 */
FLAGDNN_API flagdnnStatus_t flagdnnSetTensorDescriptorVirtual(
    flagdnnTensorDescriptor_t descriptor,
    int32_t is_virtual);
FLAGDNN_API flagdnnStatus_t flagdnnGetTensorDescriptorVirtual(
    flagdnnTensorDescriptor_t descriptor,
    int32_t* is_virtual);
/* Matches cuDNN Frontend's 16-byte default tensor alignment contract. */
FLAGDNN_API flagdnnStatus_t flagdnnSetTensorDescriptorAlignment(
    flagdnnTensorDescriptor_t descriptor,
    int64_t alignment);
FLAGDNN_API flagdnnStatus_t flagdnnGetTensorDescriptorAlignment(
    flagdnnTensorDescriptor_t descriptor,
    int64_t* alignment);

FLAGDNN_API flagdnnStatus_t flagdnnCreateOperationDescriptor(
    flagdnnOperation_t operation,
    flagdnnOperationDescriptor_t* descriptor);
/*
 * Extensible descriptor path. operation_kind and all port/attribute names use
 * lower_snake_case and are copied. Finalize before adding the operation to a
 * graph. Existing typed setters remain the compatibility convenience layer.
 */
FLAGDNN_API flagdnnStatus_t flagdnnCreateOperationDescriptorByName(
    const char* operation_kind,
    flagdnnOperationDescriptor_t* descriptor);
FLAGDNN_API flagdnnStatus_t flagdnnSetOperationDescriptorInput(
    flagdnnOperationDescriptor_t descriptor,
    const char* port_name,
    flagdnnTensorDescriptor_t tensor,
    int32_t is_optional);
FLAGDNN_API flagdnnStatus_t flagdnnSetOperationDescriptorOutput(
    flagdnnOperationDescriptor_t descriptor,
    const char* port_name,
    flagdnnTensorDescriptor_t tensor,
    int32_t is_optional);
FLAGDNN_API flagdnnStatus_t flagdnnSetOperationDescriptorAttributeInt64(
    flagdnnOperationDescriptor_t descriptor,
    const char* attribute_name,
    int64_t value);
FLAGDNN_API flagdnnStatus_t flagdnnSetOperationDescriptorAttributeDouble(
    flagdnnOperationDescriptor_t descriptor,
    const char* attribute_name,
    double value);
FLAGDNN_API flagdnnStatus_t flagdnnSetOperationDescriptorAttributeBoolean(
    flagdnnOperationDescriptor_t descriptor,
    const char* attribute_name,
    int32_t value);
FLAGDNN_API flagdnnStatus_t flagdnnSetOperationDescriptorAttributeString(
    flagdnnOperationDescriptor_t descriptor,
    const char* attribute_name,
    const char* value);
FLAGDNN_API flagdnnStatus_t flagdnnSetOperationDescriptorAttributeInt64Array(
    flagdnnOperationDescriptor_t descriptor,
    const char* attribute_name,
    const int64_t values[],
    size_t value_count);
FLAGDNN_API flagdnnStatus_t flagdnnFinalizeOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor);
FLAGDNN_API flagdnnStatus_t flagdnnDestroyOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor);
FLAGDNN_API flagdnnStatus_t flagdnnGetOperationDescriptorType(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnOperation_t* operation);
/* Optional backend-neutral metadata copied into Graph IR. */
FLAGDNN_API flagdnnStatus_t flagdnnSetOperationDescriptorName(
    flagdnnOperationDescriptor_t descriptor,
    const char* name);
FLAGDNN_API flagdnnStatus_t flagdnnSetOperationDescriptorComputeDataType(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnDataType_t data_type);

/* Operation setters only describe computation and never access a device. */
FLAGDNN_API flagdnnStatus_t flagdnnSetPointwiseUnaryOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t input,
    flagdnnPointwiseMode_t mode,
    flagdnnTensorDescriptor_t output);
/*
 * Attribute-aware unary setter. A null attributes pointer selects defaults
 * and is equivalent to flagdnnSetPointwiseUnaryOperationDescriptor.
 */
FLAGDNN_API flagdnnStatus_t
flagdnnSetPointwiseUnaryOperationDescriptorWithAttributes(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t input,
    flagdnnPointwiseMode_t mode,
    flagdnnTensorDescriptor_t output,
    const flagdnnPointwiseAttributes_t* attributes);
FLAGDNN_API flagdnnStatus_t flagdnnSetPointwiseBinaryOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t left,
    flagdnnTensorDescriptor_t right,
    flagdnnPointwiseMode_t mode,
    flagdnnTensorDescriptor_t output);
/*
 * FlagDNN extension for modes with a scaled right operand. ADD computes
 * left + alpha * right and SUB computes left - alpha * right. Other binary
 * modes require alpha == 1.
 */
FLAGDNN_API flagdnnStatus_t
flagdnnSetPointwiseBinaryOperationDescriptorWithAlpha(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t left,
    flagdnnTensorDescriptor_t right,
    flagdnnPointwiseMode_t mode,
    flagdnnTensorDescriptor_t output,
    double alpha);
/*
 * Describes cuDNN-compatible ternary pointwise operations. BINARY_SELECT
 * computes output = T ? A : B, where T is a BOOLEAN predicate tensor.
 */
FLAGDNN_API flagdnnStatus_t flagdnnSetPointwiseTernaryOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t a,
    flagdnnTensorDescriptor_t b,
    flagdnnTensorDescriptor_t t,
    flagdnnPointwiseMode_t mode,
    flagdnnTensorDescriptor_t output);
FLAGDNN_API flagdnnStatus_t flagdnnSetReluOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t input,
    flagdnnTensorDescriptor_t output);
FLAGDNN_API flagdnnStatus_t flagdnnSetAddOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t left,
    flagdnnTensorDescriptor_t right,
    flagdnnTensorDescriptor_t output);
/*
 * Describes output = left + alpha * right. The legacy setter above is
 * equivalent to alpha == 1.0.
 */
FLAGDNN_API flagdnnStatus_t flagdnnSetAddOperationDescriptorWithAlpha(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t left,
    flagdnnTensorDescriptor_t right,
    flagdnnTensorDescriptor_t output,
    double alpha);
FLAGDNN_API flagdnnStatus_t flagdnnSetMatmulOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t a,
    flagdnnTensorDescriptor_t b,
    flagdnnTensorDescriptor_t output);
/*
 * Describes cuDNN-Frontend-style scaled dot-product attention over dense BHSD
 * tensors. bias may be null. stats is always described so one compiled ABI is
 * used for inference and training; it is written only when generate_stats is
 * nonzero and may be virtual.
 */
FLAGDNN_API flagdnnStatus_t flagdnnSetSdpaOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t q,
    flagdnnTensorDescriptor_t k,
    flagdnnTensorDescriptor_t v,
    flagdnnTensorDescriptor_t bias,
    flagdnnTensorDescriptor_t output,
    flagdnnTensorDescriptor_t stats,
    const flagdnnSdpaAttributes_t* attributes);
/* dBias may be null; when present it is an additional caller-visible output. */
FLAGDNN_API flagdnnStatus_t flagdnnSetSdpaBackwardOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t q,
    flagdnnTensorDescriptor_t k,
    flagdnnTensorDescriptor_t v,
    flagdnnTensorDescriptor_t output,
    flagdnnTensorDescriptor_t doutput,
    flagdnnTensorDescriptor_t stats,
    flagdnnTensorDescriptor_t bias,
    flagdnnTensorDescriptor_t dq,
    flagdnnTensorDescriptor_t dk,
    flagdnnTensorDescriptor_t dv,
    flagdnnTensorDescriptor_t dbias,
    const flagdnnSdpaAttributes_t* attributes);
/*
 * Current-scaling FP8 SDPA. Scaling operands are caller-bound float32 scalar
 * tensors, matching the cuDNN Frontend Graph API. bias may be null. stats is
 * always described and is only written when generate_stats is enabled.
 */
FLAGDNN_API flagdnnStatus_t flagdnnSetSdpaFp8OperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t q,
    flagdnnTensorDescriptor_t k,
    flagdnnTensorDescriptor_t v,
    flagdnnTensorDescriptor_t descale_q,
    flagdnnTensorDescriptor_t descale_k,
    flagdnnTensorDescriptor_t descale_v,
    flagdnnTensorDescriptor_t descale_s,
    flagdnnTensorDescriptor_t scale_s,
    flagdnnTensorDescriptor_t scale_o,
    flagdnnTensorDescriptor_t bias,
    flagdnnTensorDescriptor_t output,
    flagdnnTensorDescriptor_t stats,
    flagdnnTensorDescriptor_t amax_s,
    flagdnnTensorDescriptor_t amax_o,
    const flagdnnSdpaAttributes_t* attributes);
FLAGDNN_API flagdnnStatus_t flagdnnSetSdpaFp8BackwardOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t q,
    flagdnnTensorDescriptor_t k,
    flagdnnTensorDescriptor_t v,
    flagdnnTensorDescriptor_t output,
    flagdnnTensorDescriptor_t doutput,
    flagdnnTensorDescriptor_t stats,
    flagdnnTensorDescriptor_t descale_q,
    flagdnnTensorDescriptor_t descale_k,
    flagdnnTensorDescriptor_t descale_v,
    flagdnnTensorDescriptor_t descale_o,
    flagdnnTensorDescriptor_t descale_doutput,
    flagdnnTensorDescriptor_t descale_s,
    flagdnnTensorDescriptor_t descale_dp,
    flagdnnTensorDescriptor_t scale_s,
    flagdnnTensorDescriptor_t scale_dq,
    flagdnnTensorDescriptor_t scale_dk,
    flagdnnTensorDescriptor_t scale_dv,
    flagdnnTensorDescriptor_t scale_dp,
    flagdnnTensorDescriptor_t dq,
    flagdnnTensorDescriptor_t dk,
    flagdnnTensorDescriptor_t dv,
    flagdnnTensorDescriptor_t amax_dq,
    flagdnnTensorDescriptor_t amax_dk,
    flagdnnTensorDescriptor_t amax_dv,
    flagdnnTensorDescriptor_t amax_dp,
    const flagdnnSdpaAttributes_t* attributes);
FLAGDNN_API flagdnnStatus_t flagdnnSetReductionOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t input,
    flagdnnReductionMode_t mode,
    int32_t axis,
    int32_t keep_dimensions,
    flagdnnTensorDescriptor_t output);
/*
 * Compatibility entry point. Equivalent to
 * flagdnnSetReductionOperationDescriptor(..., FLAGDNN_REDUCTION_ADD, ...).
 */
FLAGDNN_API flagdnnStatus_t flagdnnSetReductionSumOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t input,
    int32_t axis,
    int32_t keep_dimensions,
    flagdnnTensorDescriptor_t output);
/*
 * Describes an N-D forward cross-correlation over logical NCX/KCX tensors.
 * The initial native implementation accepts spatial_rank 1, 2, or 3. Tensor
 * layout is expressed only by element strides; it is never encoded as a
 * backend-specific layout enum. All spatial arrays contain spatial_rank
 * elements and are copied by FlagDNN.
 */
FLAGDNN_API flagdnnStatus_t
flagdnnSetConvolutionFpropOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t input,
    flagdnnTensorDescriptor_t filter,
    int32_t spatial_rank,
    const int64_t pre_padding[],
    const int64_t post_padding[],
    const int64_t stride[],
    const int64_t dilation[],
    int64_t groups,
    flagdnnTensorDescriptor_t output);
/* Compatibility entry point for symmetric rank-2 padding. */
FLAGDNN_API flagdnnStatus_t flagdnnSetConv2dFpropOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t input,
    flagdnnTensorDescriptor_t filter,
    const int64_t padding[2],
    const int64_t stride[2],
    const int64_t dilation[2],
    int64_t groups,
    flagdnnTensorDescriptor_t output);
/*
 * Extended Conv2D FProp entry point for asymmetric spatial padding.
 * The original entry point is equivalent to passing the same `padding`
 * array as both `pre_padding` and `post_padding`.
 */
FLAGDNN_API flagdnnStatus_t
flagdnnSetConv2dFpropOperationDescriptorWithAsymmetricPadding(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t input,
    flagdnnTensorDescriptor_t filter,
    const int64_t pre_padding[2],
    const int64_t post_padding[2],
    const int64_t stride[2],
    const int64_t dilation[2],
    int64_t groups,
    flagdnnTensorDescriptor_t output);

FLAGDNN_API flagdnnStatus_t flagdnnCreateGraph(flagdnnGraph_t* graph);
FLAGDNN_API flagdnnStatus_t flagdnnDestroyGraph(flagdnnGraph_t graph);
FLAGDNN_API flagdnnStatus_t flagdnnSetGraphName(
    flagdnnGraph_t graph,
    const char* name);
FLAGDNN_API flagdnnStatus_t flagdnnGraphAddOperation(
    flagdnnGraph_t graph,
    flagdnnOperationDescriptor_t operation);
/* Validates graph topology and operation semantics without freezing it. */
FLAGDNN_API flagdnnStatus_t flagdnnValidateGraph(flagdnnGraph_t graph);
/* Validates and freezes the graph. */
FLAGDNN_API flagdnnStatus_t flagdnnFinalizeGraph(flagdnnGraph_t graph);
FLAGDNN_API flagdnnStatus_t flagdnnGetGraphOperationCount(
    flagdnnGraph_t graph,
    size_t* operation_count);

/*
 * Builds operations in stable dependency order; callers need not add producer
 * nodes before consumers. A virtual output may feed another operation through
 * the same UID, and its storage is part of executable workspace. Non-virtual
 * tensors are supplied through execute bindings.
 */
FLAGDNN_API flagdnnStatus_t flagdnnBuildExecutable(
    flagdnnHandle_t handle,
    flagdnnGraph_t graph,
    const flagdnnBuildOptions_t* options,
    flagdnnExecutable_t* executable);
FLAGDNN_API flagdnnStatus_t flagdnnDestroyExecutable(
    flagdnnExecutable_t executable);
FLAGDNN_API flagdnnStatus_t flagdnnGetExecutableOperationCount(
    flagdnnExecutable_t executable,
    size_t* operation_count);
FLAGDNN_API flagdnnStatus_t flagdnnGetExecutableWorkspaceSize(
    flagdnnExecutable_t executable,
    size_t* workspace_size);

/*
 * Enqueues work on caller_stream. This function does not compile, allocate,
 * or synchronize. Device buffers and workspace remain caller-owned.
 */
FLAGDNN_API flagdnnStatus_t flagdnnExecuteAsync(
    flagdnnExecutable_t executable,
    const flagdnnBinding_t bindings[],
    size_t binding_count,
    void* workspace,
    size_t workspace_size,
    flagdnnStream_t caller_stream);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* FLAGDNN_FLAGDNN_H_ */
