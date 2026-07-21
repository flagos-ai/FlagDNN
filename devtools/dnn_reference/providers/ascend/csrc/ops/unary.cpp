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

#include "aclnnop/aclnn_ceil.h"
#include "aclnnop/aclnn_cos.h"
#include "aclnnop/aclnn_erf.h"
#include "aclnnop/aclnn_elu.h"
#include "aclnnop/aclnn_exp.h"
#include "aclnnop/aclnn_floor.h"
#include "aclnnop/aclnn_gelu_v2.h"
#include "aclnnop/aclnn_leaky_relu.h"
#include "aclnnop/aclnn_log.h"
#include "aclnnop/aclnn_logical_not.h"
#include "aclnnop/aclnn_neg.h"
#include "aclnnop/aclnn_reciprocal.h"
#include "aclnnop/aclnn_relu.h"
#include "aclnnop/aclnn_rsqrt.h"
#include "aclnnop/aclnn_sigmoid.h"
#include "aclnnop/aclnn_sin.h"
#include "aclnnop/aclnn_sqrt.h"
#include "aclnnop/aclnn_softplus.h"
#include "aclnnop/aclnn_swish.h"
#include "aclnnop/aclnn_tan.h"
#include "aclnnop/aclnn_tanh.h"
#include "common/oracle_common.h"

namespace flagdnn_test::ascend {
namespace {

constexpr uint64_t kMaxTensorRank = 8;

enum UnaryOperation : int32_t {
    kNeg = 0,
    kSqrt = 1,
    kRsqrt = 2,
    kReciprocal = 3,
    kCeil = 4,
    kFloor = 5,
    kExp = 6,
    kLog = 7,
    kErf = 8,
    kSin = 9,
    kCos = 10,
    kTan = 11,
    kRelu = 12,
    kSigmoid = 13,
    kTanh = 14,
    kLogicalNot = 15,
    kLeakyRelu = 16,
    kElu = 17,
    kGelu = 18,
    kGeluApproxTanh = 19,
    kSwish = 20,
    kSoftplus = 21,
};

struct PreparedUnary {
    int64_t input_shape[kMaxTensorRank] = {};
    int64_t input_strides[kMaxTensorRank] = {};
    int64_t output_shape[kMaxTensorRank] = {};
    int64_t output_strides[kMaxTensorRank] = {};
    uint64_t input_rank = 0;
    uint64_t output_rank = 0;
    aclTensor* input = nullptr;
    aclTensor* output = nullptr;
    float scalar_data[3] = {};
    aclScalar* scalars[3] = {};
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

int ConfigureExecutor(
    int32_t operation,
    PreparedUnary* prepared,
    ErrorState* state) noexcept {
    aclnnStatus status = ACL_SUCCESS;
    const char* workspace_stage = nullptr;
    switch (operation) {
        case kNeg:
            status = aclnnNegGetWorkspaceSize(
                prepared->input, prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnNegGetWorkspaceSize";
            prepared->execute = aclnnNeg;
            prepared->execute_stage = "aclnnNeg";
            break;
        case kSqrt:
            status = aclnnSqrtGetWorkspaceSize(
                prepared->input, prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnSqrtGetWorkspaceSize";
            prepared->execute = aclnnSqrt;
            prepared->execute_stage = "aclnnSqrt";
            break;
        case kRsqrt:
            status = aclnnRsqrtGetWorkspaceSize(
                prepared->input, prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnRsqrtGetWorkspaceSize";
            prepared->execute = aclnnRsqrt;
            prepared->execute_stage = "aclnnRsqrt";
            break;
        case kReciprocal:
            status = aclnnReciprocalGetWorkspaceSize(
                prepared->input, prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnReciprocalGetWorkspaceSize";
            prepared->execute = aclnnReciprocal;
            prepared->execute_stage = "aclnnReciprocal";
            break;
        case kCeil:
            status = aclnnCeilGetWorkspaceSize(
                prepared->input, prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnCeilGetWorkspaceSize";
            prepared->execute = aclnnCeil;
            prepared->execute_stage = "aclnnCeil";
            break;
        case kFloor:
            status = aclnnFloorGetWorkspaceSize(
                prepared->input, prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnFloorGetWorkspaceSize";
            prepared->execute = aclnnFloor;
            prepared->execute_stage = "aclnnFloor";
            break;
        case kExp:
            status = aclnnExpGetWorkspaceSize(
                prepared->input, prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnExpGetWorkspaceSize";
            prepared->execute = aclnnExp;
            prepared->execute_stage = "aclnnExp";
            break;
        case kLog:
            status = aclnnLogGetWorkspaceSize(
                prepared->input, prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnLogGetWorkspaceSize";
            prepared->execute = aclnnLog;
            prepared->execute_stage = "aclnnLog";
            break;
        case kErf:
            status = aclnnErfGetWorkspaceSize(
                prepared->input, prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnErfGetWorkspaceSize";
            prepared->execute = aclnnErf;
            prepared->execute_stage = "aclnnErf";
            break;
        case kSin:
            status = aclnnSinGetWorkspaceSize(
                prepared->input, prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnSinGetWorkspaceSize";
            prepared->execute = aclnnSin;
            prepared->execute_stage = "aclnnSin";
            break;
        case kCos:
            status = aclnnCosGetWorkspaceSize(
                prepared->input, prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnCosGetWorkspaceSize";
            prepared->execute = aclnnCos;
            prepared->execute_stage = "aclnnCos";
            break;
        case kTan:
            status = aclnnTanGetWorkspaceSize(
                prepared->input, prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnTanGetWorkspaceSize";
            prepared->execute = aclnnTan;
            prepared->execute_stage = "aclnnTan";
            break;
        case kRelu:
            status = aclnnReluGetWorkspaceSize(
                prepared->input, prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnReluGetWorkspaceSize";
            prepared->execute = aclnnRelu;
            prepared->execute_stage = "aclnnRelu";
            break;
        case kSigmoid:
            status = aclnnSigmoidGetWorkspaceSize(
                prepared->input, prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnSigmoidGetWorkspaceSize";
            prepared->execute = aclnnSigmoid;
            prepared->execute_stage = "aclnnSigmoid";
            break;
        case kTanh:
            status = aclnnTanhGetWorkspaceSize(
                prepared->input, prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnTanhGetWorkspaceSize";
            prepared->execute = aclnnTanh;
            prepared->execute_stage = "aclnnTanh";
            break;
        case kLogicalNot:
            status = aclnnLogicalNotGetWorkspaceSize(
                prepared->input, prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnLogicalNotGetWorkspaceSize";
            prepared->execute = aclnnLogicalNot;
            prepared->execute_stage = "aclnnLogicalNot";
            break;
        case kLeakyRelu:
            status = aclnnLeakyReluGetWorkspaceSize(
                prepared->input,
                prepared->scalars[0],
                prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnLeakyReluGetWorkspaceSize";
            prepared->execute = aclnnLeakyRelu;
            prepared->execute_stage = "aclnnLeakyRelu";
            break;
        case kElu:
            status = aclnnEluGetWorkspaceSize(
                prepared->input,
                prepared->scalars[0],
                prepared->scalars[1],
                prepared->scalars[2],
                prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnEluGetWorkspaceSize";
            prepared->execute = aclnnElu;
            prepared->execute_stage = "aclnnElu";
            break;
        case kGelu:
        case kGeluApproxTanh:
            status = aclnnGeluV2GetWorkspaceSize(
                prepared->input,
                operation == kGeluApproxTanh ? 1 : 0,
                prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnGeluV2GetWorkspaceSize";
            prepared->execute = aclnnGeluV2;
            prepared->execute_stage = "aclnnGeluV2";
            break;
        case kSwish:
            status = aclnnSwishGetWorkspaceSize(
                prepared->input,
                prepared->scalars[0],
                prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnSwishGetWorkspaceSize";
            prepared->execute = aclnnSwish;
            prepared->execute_stage = "aclnnSwish";
            break;
        case kSoftplus:
            status = aclnnSoftplusGetWorkspaceSize(
                prepared->input,
                prepared->scalars[0],
                prepared->scalars[1],
                prepared->output,
                &prepared->execution.workspace_size,
                &prepared->execution.executor);
            workspace_stage = "aclnnSoftplusGetWorkspaceSize";
            prepared->execute = aclnnSoftplus;
            prepared->execute_stage = "aclnnSoftplus";
            break;
        default:
            return ReturnFailure(state, "unsupported unary operation", -40);
    }
    RecordAclFailure(state, workspace_stage, status);
    return state->first_status;
}

void DestroyPreparedUnary(
    PreparedUnary* prepared,
    ErrorState* state) noexcept {
    DestroyExecutor(&prepared->execution, state);
    DestroyTensor(&prepared->input, "aclDestroyTensor(x)", state);
    DestroyTensor(
        &prepared->output, "aclDestroyTensor(output)", state);
    for (aclScalar*& scalar : prepared->scalars) {
        DestroyScalar(&scalar, "aclDestroyScalar(parameter)", state);
    }
    FreeWorkspace(&prepared->execution, state);
}

}  // namespace

extern "C" int flagdnn_aclnn_unary_create(
    int32_t operation,
    const void* input_data,
    const int64_t* input_shape,
    const int64_t* input_strides,
    uint64_t input_rank,
    void* output_data,
    const int64_t* output_shape,
    const int64_t* output_strides,
    uint64_t output_rank,
    int32_t dtype_code,
    double parameter0,
    double parameter1,
    double parameter2,
    void* stream_handle,
    void** prepared_handle,
    char* error_buffer,
    size_t error_buffer_length) noexcept {
    ErrorState state{error_buffer, error_buffer_length, ACL_SUCCESS, false};
    ClearError(&state);
    if (prepared_handle == nullptr) {
        return ReturnFailure(&state, "prepared handle pointer is null", -41);
    }
    *prepared_handle = nullptr;
    if (input_data == nullptr || output_data == nullptr) {
        return ReturnFailure(
            &state, "input/output device pointer is null", -42);
    }
    if (!ValidMetadata(input_shape, input_strides, input_rank) ||
        !ValidMetadata(output_shape, output_strides, output_rank)) {
        return ReturnFailure(&state, "invalid tensor metadata", -43);
    }
    if (stream_handle == nullptr) {
        return ReturnFailure(&state, "current stream pointer is null", -44);
    }
    aclDataType dtype = ACL_DT_UNDEFINED;
    if (!MapDataType(dtype_code, &dtype)) {
        return ReturnFailure(&state, "unsupported dtype code", -45);
    }

    PreparedUnary* prepared = new (std::nothrow) PreparedUnary;
    if (prepared == nullptr) {
        return ReturnFailure(
            &state, "failed to allocate prepared unary operation", -46);
    }
    prepared->input_rank = input_rank;
    prepared->output_rank = output_rank;
    prepared->scalar_data[0] = static_cast<float>(parameter0);
    prepared->scalar_data[1] = static_cast<float>(parameter1);
    prepared->scalar_data[2] = static_cast<float>(parameter2);
    prepared->stream = reinterpret_cast<aclrtStream>(stream_handle);
    CopyMetadata(prepared->input_shape, input_shape, input_rank);
    CopyMetadata(prepared->input_strides, input_strides, input_rank);
    CopyMetadata(prepared->output_shape, output_shape, output_rank);
    CopyMetadata(prepared->output_strides, output_strides, output_rank);

    prepared->input = aclCreateTensor(
        prepared->input_shape,
        input_rank,
        dtype,
        prepared->input_strides,
        0,
        ACL_FORMAT_ND,
        prepared->input_shape,
        input_rank,
        const_cast<void*>(input_data));
    if (prepared->input == nullptr) {
        RecordWrapperFailure(&state, "aclCreateTensor(x)", -47);
    }
    if (state.first_status == ACL_SUCCESS) {
        prepared->output = aclCreateTensor(
            prepared->output_shape,
            output_rank,
            dtype,
            prepared->output_strides,
            0,
            ACL_FORMAT_ND,
            prepared->output_shape,
            output_rank,
            output_data);
        if (prepared->output == nullptr) {
            RecordWrapperFailure(
                &state, "aclCreateTensor(output)", -48);
        }
    }
    if (state.first_status == ACL_SUCCESS) {
        for (size_t index = 0;
             index < 3 && state.first_status == ACL_SUCCESS;
             ++index) {
            prepared->scalars[index] = aclCreateScalar(
                &prepared->scalar_data[index], ACL_FLOAT);
            if (prepared->scalars[index] == nullptr) {
                RecordWrapperFailure(
                    &state, "aclCreateScalar(parameter)", -51);
            }
        }
    }
    if (state.first_status == ACL_SUCCESS) {
        ConfigureExecutor(operation, prepared, &state);
    }
    MakeExecutorRepeatable(&prepared->execution, &state);
    AllocateWorkspace(&prepared->execution, &state);
    if (state.first_status != ACL_SUCCESS) {
        DestroyPreparedUnary(prepared, &state);
        delete prepared;
        return state.first_status;
    }
    *prepared_handle = prepared;
    return ACL_SUCCESS;
}

extern "C" int flagdnn_aclnn_unary_run(
    void* prepared_handle,
    char* error_buffer,
    size_t error_buffer_length) noexcept {
    ErrorState state{error_buffer, error_buffer_length, ACL_SUCCESS, false};
    ClearError(&state);
    if (prepared_handle == nullptr) {
        return ReturnFailure(&state, "prepared unary handle is null", -49);
    }
    PreparedUnary* prepared = static_cast<PreparedUnary*>(prepared_handle);
    const int status = prepared->execute(
        prepared->execution.workspace,
        prepared->execution.workspace_size,
        prepared->execution.executor,
        prepared->stream);
    RecordAclFailure(&state, prepared->execute_stage, status);
    return state.first_status;
}

extern "C" int flagdnn_aclnn_unary_destroy(
    void* prepared_handle,
    char* error_buffer,
    size_t error_buffer_length) noexcept {
    ErrorState state{error_buffer, error_buffer_length, ACL_SUCCESS, false};
    ClearError(&state);
    if (prepared_handle == nullptr) {
        return ReturnFailure(&state, "prepared unary handle is null", -50);
    }
    PreparedUnary* prepared = static_cast<PreparedUnary*>(prepared_handle);
    RecordAclFailure(
        &state,
        "aclrtSynchronizeStream",
        aclrtSynchronizeStream(prepared->stream));
    DestroyPreparedUnary(prepared, &state);
    delete prepared;
    return state.first_status;
}

}  // namespace flagdnn_test::ascend
