// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstddef>
#include <cstdint>
#include <new>

#include "aclnnop/aclnn_addcmul.h"
#include "aclnnop/aclnn_div.h"
#include "aclnnop/aclnn_eq_tensor.h"
#include "aclnnop/aclnn_fmod_tensor.h"
#include "aclnnop/aclnn_ge_tensor.h"
#include "aclnnop/aclnn_gt_tensor.h"
#include "aclnnop/aclnn_le_tensor.h"
#include "aclnnop/aclnn_logical_and.h"
#include "aclnnop/aclnn_logical_or.h"
#include "aclnnop/aclnn_lt_tensor.h"
#include "aclnnop/aclnn_maximum.h"
#include "aclnnop/aclnn_minimum.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnnop/aclnn_ne_tensor.h"
#include "aclnnop/aclnn_pow_tensor_tensor.h"
#include "aclnnop/aclnn_sigmoid_backward.h"
#include "aclnnop/aclnn_sub.h"
#include "common/oracle_common.h"

namespace flagdnn_test::ascend {
namespace {

constexpr uint64_t kMaxTensorRank = 8;

enum BinaryOperation : int32_t {
    kSub = 0,
    kMul = 1,
    kDiv = 2,
    kPow = 3,
    kMaximum = 4,
    kMinimum = 5,
    kRemainder = 6,
    kAddSquare = 7,
    kEqual = 8,
    kNotEqual = 9,
    kGreater = 10,
    kGreaterEqual = 11,
    kLess = 12,
    kLessEqual = 13,
    kLogicalAnd = 14,
    kLogicalOr = 15,
    kSigmoidBackward = 16,
};

struct PreparedBinary {
    int64_t x_shape[kMaxTensorRank] = {};
    int64_t x_strides[kMaxTensorRank] = {};
    int64_t y_shape[kMaxTensorRank] = {};
    int64_t y_strides[kMaxTensorRank] = {};
    int64_t output_shape[kMaxTensorRank] = {};
    int64_t output_strides[kMaxTensorRank] = {};
    uint64_t x_rank = 0;
    uint64_t y_rank = 0;
    uint64_t output_rank = 0;
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    aclTensor* output = nullptr;
    aclScalar* scalar = nullptr;
    float scalar_data = 1.0F;
    ExecutorWorkspace execution;
    ExecuteFunction execute = nullptr;
    const char* execute_stage = nullptr;
    aclrtStream stream = nullptr;
};

void CopyMetadata(
    int64_t* destination,
    const int64_t* source,
    uint64_t rank) noexcept {
    for (uint64_t index = 0; index < rank; ++index) {
        destination[index] = source[index];
    }
}

bool ValidMetadata(
    const int64_t* shape,
    const int64_t* strides,
    uint64_t rank) noexcept {
    return shape != nullptr && strides != nullptr && rank != 0 &&
        rank <= kMaxTensorRank;
}

void DestroyPreparedBinary(
    PreparedBinary* prepared,
    ErrorState* state) noexcept {
    DestroyExecutor(&prepared->execution, state);
    DestroyTensor(&prepared->x, "aclDestroyTensor(x)", state);
    DestroyTensor(&prepared->y, "aclDestroyTensor(y)", state);
    DestroyTensor(
        &prepared->output, "aclDestroyTensor(output)", state);
    DestroyScalar(&prepared->scalar, "aclDestroyScalar", state);
    FreeWorkspace(&prepared->execution, state);
}

int ConfigureExecutor(
    int32_t operation,
    PreparedBinary* prepared,
    ErrorState* state) noexcept {
    aclnnStatus status = ACL_SUCCESS;
    const char* workspace_stage = nullptr;
    switch (operation) {
        case kSub:
            status = aclnnSubGetWorkspaceSize(
                prepared->x,
                prepared->y,
                prepared->scalar,
                prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnSubGetWorkspaceSize";
            prepared->execute = aclnnSub;
            prepared->execute_stage = "aclnnSub";
            break;
        case kMul:
            status = aclnnMulGetWorkspaceSize(
                prepared->x,
                prepared->y,
                prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnMulGetWorkspaceSize";
            prepared->execute = aclnnMul;
            prepared->execute_stage = "aclnnMul";
            break;
        case kDiv:
            status = aclnnDivGetWorkspaceSize(
                prepared->x,
                prepared->y,
                prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnDivGetWorkspaceSize";
            prepared->execute = aclnnDiv;
            prepared->execute_stage = "aclnnDiv";
            break;
        case kPow:
            status = aclnnPowTensorTensorGetWorkspaceSize(
                prepared->x,
                prepared->y,
                prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnPowTensorTensorGetWorkspaceSize";
            prepared->execute = aclnnPowTensorTensor;
            prepared->execute_stage = "aclnnPowTensorTensor";
            break;
        case kMaximum:
            status = aclnnMaximumGetWorkspaceSize(
                prepared->x,
                prepared->y,
                prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnMaximumGetWorkspaceSize";
            prepared->execute = aclnnMaximum;
            prepared->execute_stage = "aclnnMaximum";
            break;
        case kMinimum:
            status = aclnnMinimumGetWorkspaceSize(
                prepared->x,
                prepared->y,
                prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnMinimumGetWorkspaceSize";
            prepared->execute = aclnnMinimum;
            prepared->execute_stage = "aclnnMinimum";
            break;
        case kRemainder:
            status = aclnnFmodTensorGetWorkspaceSize(
                prepared->x,
                prepared->y,
                prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnFmodTensorGetWorkspaceSize";
            prepared->execute = aclnnFmodTensor;
            prepared->execute_stage = "aclnnFmodTensor";
            break;
        case kAddSquare:
            status = aclnnAddcmulGetWorkspaceSize(
                prepared->x,
                prepared->y,
                prepared->y,
                prepared->scalar,
                prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnAddcmulGetWorkspaceSize";
            prepared->execute = aclnnAddcmul;
            prepared->execute_stage = "aclnnAddcmul";
            break;
        case kEqual:
            status = aclnnEqTensorGetWorkspaceSize(
                prepared->x,
                prepared->y,
                prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnEqTensorGetWorkspaceSize";
            prepared->execute = aclnnEqTensor;
            prepared->execute_stage = "aclnnEqTensor";
            break;
        case kNotEqual:
            status = aclnnNeTensorGetWorkspaceSize(
                prepared->x,
                prepared->y,
                prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnNeTensorGetWorkspaceSize";
            prepared->execute = aclnnNeTensor;
            prepared->execute_stage = "aclnnNeTensor";
            break;
        case kGreater:
            status = aclnnGtTensorGetWorkspaceSize(
                prepared->x,
                prepared->y,
                prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnGtTensorGetWorkspaceSize";
            prepared->execute = aclnnGtTensor;
            prepared->execute_stage = "aclnnGtTensor";
            break;
        case kGreaterEqual:
            status = aclnnGeTensorGetWorkspaceSize(
                prepared->x,
                prepared->y,
                prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnGeTensorGetWorkspaceSize";
            prepared->execute = aclnnGeTensor;
            prepared->execute_stage = "aclnnGeTensor";
            break;
        case kLess:
            status = aclnnLtTensorGetWorkspaceSize(
                prepared->x,
                prepared->y,
                prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnLtTensorGetWorkspaceSize";
            prepared->execute = aclnnLtTensor;
            prepared->execute_stage = "aclnnLtTensor";
            break;
        case kLessEqual:
            status = aclnnLeTensorGetWorkspaceSize(
                prepared->x,
                prepared->y,
                prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnLeTensorGetWorkspaceSize";
            prepared->execute = aclnnLeTensor;
            prepared->execute_stage = "aclnnLeTensor";
            break;
        case kLogicalAnd:
            status = aclnnLogicalAndGetWorkspaceSize(
                prepared->x,
                prepared->y,
                prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnLogicalAndGetWorkspaceSize";
            prepared->execute = aclnnLogicalAnd;
            prepared->execute_stage = "aclnnLogicalAnd";
            break;
        case kLogicalOr:
            status = aclnnLogicalOrGetWorkspaceSize(
                prepared->x,
                prepared->y,
                prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnLogicalOrGetWorkspaceSize";
            prepared->execute = aclnnLogicalOr;
            prepared->execute_stage = "aclnnLogicalOr";
            break;
        case kSigmoidBackward:
            status = aclnnSigmoidBackwardGetWorkspaceSize(
                prepared->x,
                prepared->y,
                prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnSigmoidBackwardGetWorkspaceSize";
            prepared->execute = aclnnSigmoidBackward;
            prepared->execute_stage = "aclnnSigmoidBackward";
            break;
        default:
            return ReturnFailure(
                state, "unsupported binary operation", -50);
    }
    RecordAclFailure(state, workspace_stage, status);
    return state->first_status;
}

}  // namespace

extern "C" int flagdnn_aclnn_binary_create(
    int32_t operation,
    const void* x_data,
    const int64_t* x_shape,
    const int64_t* x_strides,
    uint64_t x_rank,
    const void* y_data,
    const int64_t* y_shape,
    const int64_t* y_strides,
    uint64_t y_rank,
    void* output_data,
    const int64_t* output_shape,
    const int64_t* output_strides,
    uint64_t output_rank,
    int32_t dtype_code,
    int32_t output_dtype_code,
    double scalar_value,
    void* stream_handle,
    void** prepared_handle,
    char* error_buffer,
    size_t error_buffer_length) noexcept {
    ErrorState state{error_buffer, error_buffer_length, ACL_SUCCESS, false};
    ClearError(&state);
    if (prepared_handle == nullptr) {
        return ReturnFailure(
            &state, "prepared handle pointer is null", -51);
    }
    *prepared_handle = nullptr;
    if (x_data == nullptr || y_data == nullptr || output_data == nullptr) {
        return ReturnFailure(
            &state, "input/output device pointer is null", -52);
    }
    if (!ValidMetadata(x_shape, x_strides, x_rank) ||
        !ValidMetadata(y_shape, y_strides, y_rank) ||
        !ValidMetadata(output_shape, output_strides, output_rank)) {
        return ReturnFailure(&state, "invalid tensor metadata", -53);
    }
    if (stream_handle == nullptr) {
        return ReturnFailure(
            &state, "current stream pointer is null", -54);
    }
    aclDataType dtype = ACL_DT_UNDEFINED;
    aclDataType output_dtype = ACL_DT_UNDEFINED;
    if (!MapDataType(dtype_code, &dtype)) {
        return ReturnFailure(&state, "unsupported dtype code", -55);
    }
    if (!MapDataType(output_dtype_code, &output_dtype)) {
        return ReturnFailure(
            &state, "unsupported output dtype code", -62);
    }

    PreparedBinary* prepared = new (std::nothrow) PreparedBinary;
    if (prepared == nullptr) {
        return ReturnFailure(
            &state, "failed to allocate prepared binary operation", -56);
    }
    prepared->x_rank = x_rank;
    prepared->y_rank = y_rank;
    prepared->output_rank = output_rank;
    prepared->scalar_data = static_cast<float>(scalar_value);
    prepared->stream = reinterpret_cast<aclrtStream>(stream_handle);
    CopyMetadata(prepared->x_shape, x_shape, x_rank);
    CopyMetadata(prepared->x_strides, x_strides, x_rank);
    CopyMetadata(prepared->y_shape, y_shape, y_rank);
    CopyMetadata(prepared->y_strides, y_strides, y_rank);
    CopyMetadata(prepared->output_shape, output_shape, output_rank);
    CopyMetadata(prepared->output_strides, output_strides, output_rank);

    prepared->x = aclCreateTensor(
        prepared->x_shape,
        x_rank,
        dtype,
        prepared->x_strides,
        0,
        ACL_FORMAT_ND,
        prepared->x_shape,
        x_rank,
        const_cast<void*>(x_data));
    if (prepared->x == nullptr) {
        RecordWrapperFailure(&state, "aclCreateTensor(x)", -57);
    }
    if (state.first_status == ACL_SUCCESS) {
        prepared->y = aclCreateTensor(
            prepared->y_shape,
            y_rank,
            dtype,
            prepared->y_strides,
            0,
            ACL_FORMAT_ND,
            prepared->y_shape,
            y_rank,
            const_cast<void*>(y_data));
        if (prepared->y == nullptr) {
            RecordWrapperFailure(&state, "aclCreateTensor(y)", -58);
        }
    }
    if (state.first_status == ACL_SUCCESS) {
        prepared->output = aclCreateTensor(
            prepared->output_shape,
            output_rank,
            output_dtype,
            prepared->output_strides,
            0,
            ACL_FORMAT_ND,
            prepared->output_shape,
            output_rank,
            output_data);
        if (prepared->output == nullptr) {
            RecordWrapperFailure(
                &state, "aclCreateTensor(output)", -59);
        }
    }
    if (state.first_status == ACL_SUCCESS &&
        (operation == kSub || operation == kAddSquare)) {
        prepared->scalar = aclCreateScalar(
            &prepared->scalar_data, ACL_FLOAT);
        if (prepared->scalar == nullptr) {
            RecordWrapperFailure(
                &state, "aclCreateScalar(value)", -60);
        }
    }
    if (state.first_status == ACL_SUCCESS) {
        ConfigureExecutor(operation, prepared, &state);
    }
    MakeExecutorRepeatable(&prepared->execution, &state);
    AllocateWorkspace(&prepared->execution, &state);
    if (state.first_status != ACL_SUCCESS) {
        DestroyPreparedBinary(prepared, &state);
        delete prepared;
        return state.first_status;
    }
    *prepared_handle = prepared;
    return ACL_SUCCESS;
}

extern "C" int flagdnn_aclnn_binary_run(
    void* prepared_handle,
    char* error_buffer,
    size_t error_buffer_length) noexcept {
    ErrorState state{error_buffer, error_buffer_length, ACL_SUCCESS, false};
    ClearError(&state);
    if (prepared_handle == nullptr) {
        return ReturnFailure(
            &state, "prepared binary handle is null", -61);
    }
    PreparedBinary* prepared =
        static_cast<PreparedBinary*>(prepared_handle);
    const int status = prepared->execute(
        prepared->execution.workspace,
        prepared->execution.workspace_size,
        prepared->execution.executor,
        prepared->stream);
    RecordAclFailure(&state, prepared->execute_stage, status);
    return state.first_status;
}

extern "C" int flagdnn_aclnn_binary_destroy(
    void* prepared_handle,
    char* error_buffer,
    size_t error_buffer_length) noexcept {
    ErrorState state{error_buffer, error_buffer_length, ACL_SUCCESS, false};
    ClearError(&state);
    if (prepared_handle == nullptr) {
        return ACL_SUCCESS;
    }
    PreparedBinary* prepared =
        static_cast<PreparedBinary*>(prepared_handle);
    RecordAclFailure(
        &state,
        "aclrtSynchronizeStream",
        aclrtSynchronizeStream(prepared->stream));
    DestroyPreparedBinary(prepared, &state);
    delete prepared;
    return state.first_status;
}

}  // namespace flagdnn_test::ascend
