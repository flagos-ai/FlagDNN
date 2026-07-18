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

#include <cstring>
#include <string>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_abs.h"
#include "aclnnop/aclnn_add.h"

struct aclTensor {
    int kind;
    const int64_t* view_dims;
    uint64_t view_dim_num;
    aclDataType data_type;
    const int64_t* strides;
    int64_t storage_offset;
    aclFormat format;
    const int64_t* storage_dims;
    uint64_t storage_dim_num;
    void* data;
    bool valid;
};
struct aclScalar {
    float value;
    aclDataType data_type;
    bool valid;
};
struct aclOpExecutor {};

namespace {

enum FakeStatus {
    kSetRepeatable = 1,
    kMalloc = 2,
    kAdd = 3,
    kSynchronize = 4,
    kDestroyExecutor = 5,
    kDestroyX = 6,
    kDestroyY = 7,
    kDestroyOutput = 8,
    kDestroyScalar = 9,
    kFree = 10,
    kCreateX = 11,
    kCreateY = 12,
    kCreateOutput = 13,
    kCreateScalar = 14,
    kGetWorkspace = 15,
    kAbsGetWorkspace = 16,
    kAbs = 17,
};

int g_statuses[18] = {};
std::string g_events;
void* g_workspace = nullptr;
void* g_last_free = nullptr;
aclOpExecutor* g_unowned_executor = nullptr;
aclOpExecutor* g_executor = nullptr;
int g_executor_destroy_count = 0;
int g_operation = 0;
int g_next_tensor_kind = 0;
uint64_t g_workspace_size = 64;
void* g_last_add_workspace = nullptr;
uint64_t g_last_add_workspace_size = 0;
aclOpExecutor* g_last_add_executor = nullptr;
aclrtStream g_last_add_stream = nullptr;
void* g_last_abs_workspace = nullptr;
uint64_t g_last_abs_workspace_size = 0;
aclOpExecutor* g_last_abs_executor = nullptr;
aclrtStream g_last_abs_stream = nullptr;
aclrtStream g_last_execute_stream = nullptr;
aclrtStream g_last_synchronize_stream = nullptr;
bool g_wiring_valid = true;
std::string g_wiring_error;

constexpr uintptr_t kExpectedXData = 0x1010;
constexpr uintptr_t kExpectedYData = 0x2020;
constexpr uintptr_t kExpectedOutputData = 0x3030;
constexpr uintptr_t kExpectedStream = 0x4040;

void Record(const char* event) {
    if (!g_events.empty()) {
        g_events += ',';
    }
    g_events += event;
}

int Status(FakeStatus stage) {
    return g_statuses[stage];
}

void InvalidateWiring(const char* message) {
    if (g_wiring_valid) {
        g_wiring_valid = false;
        g_wiring_error = message;
    }
}

bool TensorMetadataIsValid(
    int kind,
    const int64_t* view_dims,
    uint64_t view_dim_num,
    aclDataType data_type,
    const int64_t* strides,
    int64_t storage_offset,
    aclFormat format,
    const int64_t* storage_dims,
    uint64_t storage_dim_num,
    void* data) {
    const uintptr_t expected_data = kind == 0 ? kExpectedXData :
        kind == 1 ? kExpectedYData : kExpectedOutputData;
    const bool valid = view_dims != nullptr && strides != nullptr &&
        storage_dims != nullptr && view_dim_num == 1 &&
        storage_dim_num == 1 && view_dims[0] == 1 && strides[0] == 1 &&
        storage_dims[0] == 1 && data_type == ACL_FLOAT &&
        storage_offset == 0 && format == ACL_FORMAT_ND &&
        data == reinterpret_cast<void*>(expected_data);
    if (!valid) {
        InvalidateWiring("invalid aclCreateTensor metadata or data pointer");
    }
    return valid;
}

bool ScalarIsValid(void* value, aclDataType data_type) {
    const bool valid = value != nullptr &&
        *static_cast<float*>(value) == 1.0F && data_type == ACL_FLOAT;
    if (!valid) {
        InvalidateWiring("invalid aclCreateScalar value or dtype");
    }
    return valid;
}

}  // namespace

extern "C" void fake_cann_reset() {
    delete g_unowned_executor;
    g_unowned_executor = nullptr;
    std::memset(g_statuses, 0, sizeof(g_statuses));
    g_events.clear();
    g_workspace = nullptr;
    g_last_free = nullptr;
    g_executor_destroy_count = 0;
    g_operation = 0;
    g_next_tensor_kind = 0;
    g_workspace_size = 64;
    g_executor = nullptr;
    g_last_add_workspace = nullptr;
    g_last_add_workspace_size = 0;
    g_last_add_executor = nullptr;
    g_last_add_stream = nullptr;
    g_last_abs_workspace = nullptr;
    g_last_abs_workspace_size = 0;
    g_last_abs_executor = nullptr;
    g_last_abs_stream = nullptr;
    g_last_execute_stream = nullptr;
    g_last_synchronize_stream = nullptr;
    g_wiring_valid = true;
    g_wiring_error.clear();
}

extern "C" void fake_cann_set_operation(int operation) {
    g_operation = operation;
    g_next_tensor_kind = 0;
}

extern "C" void fake_cann_set_status(int stage, int status) {
    if (stage > 0 && stage < static_cast<int>(sizeof(g_statuses) / sizeof(*g_statuses))) {
        g_statuses[stage] = status;
    }
}

extern "C" void fake_cann_set_workspace_size(uint64_t size) {
    g_workspace_size = size;
}

extern "C" const char* fake_cann_events() {
    return g_events.c_str();
}

extern "C" int fake_cann_count(const char* event) {
    if (event == nullptr || event[0] == '\0') {
        return 0;
    }
    int count = 0;
    size_t start = 0;
    while (start < g_events.size()) {
        const size_t end = g_events.find(',', start);
        const size_t length = (end == std::string::npos) ?
            g_events.size() - start : end - start;
        if (g_events.compare(start, length, event) == 0) {
            ++count;
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return count;
}

extern "C" void* fake_cann_last_free() {
    return g_last_free;
}

extern "C" void* fake_cann_workspace() {
    return g_workspace;
}

extern "C" int fake_cann_executor_destroy_count() {
    return g_executor_destroy_count;
}

extern "C" int fake_cann_wiring_is_valid() {
    return g_wiring_valid ? 1 : 0;
}

extern "C" const char* fake_cann_wiring_error() {
    return g_wiring_error.c_str();
}

extern "C" void* fake_cann_last_add_workspace() {
    return g_last_add_workspace;
}

extern "C" uint64_t fake_cann_last_add_workspace_size() {
    return g_last_add_workspace_size;
}

extern "C" void* fake_cann_last_add_executor() {
    return g_last_add_executor;
}

extern "C" void* fake_cann_last_add_stream() {
    return g_last_add_stream;
}

extern "C" void* fake_cann_last_abs_workspace() {
    return g_last_abs_workspace;
}

extern "C" uint64_t fake_cann_last_abs_workspace_size() {
    return g_last_abs_workspace_size;
}

extern "C" void* fake_cann_last_abs_executor() {
    return g_last_abs_executor;
}

extern "C" void* fake_cann_last_abs_stream() {
    return g_last_abs_stream;
}

extern "C" void* fake_cann_last_synchronize_stream() {
    return g_last_synchronize_stream;
}

extern "C" void* fake_cann_executor() {
    return g_executor;
}

extern "C" const char* aclGetRecentErrMsg() {
    return "fake CANN error";
}

extern "C" aclTensor* aclCreateTensor(
    const int64_t* view_dims,
    uint64_t view_dim_num,
    aclDataType data_type,
    const int64_t* strides,
    int64_t storage_offset,
    aclFormat format,
    const int64_t* storage_dims,
    uint64_t storage_dim_num,
    void* data) {
    const int position = g_next_tensor_kind++;
    const int kind = g_operation == 1 && position == 1 ? 2 : position;
    Record(kind == 0 ? "create_x" : kind == 1 ? "create_y" : "create_output");
    const bool valid = TensorMetadataIsValid(
        kind,
        view_dims,
        view_dim_num,
        data_type,
        strides,
        storage_offset,
        format,
        storage_dims,
        storage_dim_num,
        data);
    const FakeStatus stage = kind == 0 ? kCreateX :
        kind == 1 ? kCreateY : kCreateOutput;
    if (Status(stage) != ACL_SUCCESS) {
        return nullptr;
    }
    return new aclTensor{
        kind,
        view_dims,
        view_dim_num,
        data_type,
        strides,
        storage_offset,
        format,
        storage_dims,
        storage_dim_num,
        data,
        valid,
    };
}

extern "C" aclScalar* aclCreateScalar(void* value, aclDataType data_type) {
    Record("create_scalar");
    const bool valid = ScalarIsValid(value, data_type);
    if (Status(kCreateScalar) != ACL_SUCCESS) {
        return nullptr;
    }
    return new aclScalar{*static_cast<float*>(value), data_type, valid};
}

extern "C" aclnnStatus aclnnAddGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* y,
    const aclScalar* alpha,
    aclTensor* output,
    uint64_t* workspace_size, aclOpExecutor** executor) {
    Record("get_workspace");
    if (x == nullptr || y == nullptr || alpha == nullptr || output == nullptr ||
        !x->valid || !y->valid || !alpha->valid || !output->valid) {
        InvalidateWiring("aclnnAddGetWorkspaceSize received invalid metadata");
    }
    if (Status(kGetWorkspace) != ACL_SUCCESS) {
        *workspace_size = 0;
        *executor = nullptr;
        return Status(kGetWorkspace);
    }
    *workspace_size = g_workspace_size;
    *executor = new aclOpExecutor;
    g_unowned_executor = *executor;
    g_executor = *executor;
    return ACL_SUCCESS;
}

extern "C" aclnnStatus aclnnAbsGetWorkspaceSize(
    const aclTensor* self,
    aclTensor* out,
    uint64_t* workspace_size,
    aclOpExecutor** executor) {
    Record("get_abs_workspace");
    if (self == nullptr || out == nullptr || !self->valid || !out->valid ||
        self->kind != 0 || out->kind != 2) {
        InvalidateWiring(
            "aclnnAbsGetWorkspaceSize received invalid metadata");
    }
    if (Status(kAbsGetWorkspace) != ACL_SUCCESS) {
        *workspace_size = 0;
        *executor = nullptr;
        return Status(kAbsGetWorkspace);
    }
    *workspace_size = g_workspace_size;
    *executor = new aclOpExecutor;
    g_unowned_executor = *executor;
    g_executor = *executor;
    return ACL_SUCCESS;
}

extern "C" aclnnStatus aclSetAclOpExecutorRepeatable(aclOpExecutor*) {
    Record("set_repeatable");
    const aclnnStatus status = Status(kSetRepeatable);
    if (status == ACL_SUCCESS) {
        g_unowned_executor = nullptr;
    }
    return status;
}

extern "C" aclError aclrtMalloc(
    void** workspace, size_t size, aclrtMemMallocPolicy) {
    Record("malloc");
    if (size != g_workspace_size) {
        InvalidateWiring("aclrtMalloc received an unexpected workspace size");
    }
    if (Status(kMalloc) != ACL_SUCCESS) {
        *workspace = nullptr;
        return Status(kMalloc);
    }
    *workspace = new char[size];
    g_workspace = *workspace;
    return ACL_SUCCESS;
}

extern "C" aclnnStatus aclnnAdd(
    void* workspace,
    uint64_t workspace_size,
    aclOpExecutor* executor,
    aclrtStream stream) {
    Record("add");
    g_last_add_workspace = workspace;
    g_last_add_workspace_size = workspace_size;
    g_last_add_executor = executor;
    g_last_add_stream = stream;
    g_last_execute_stream = stream;
    const void* expected_workspace = g_workspace_size == 0 ? nullptr :
        g_workspace;
    if (workspace != expected_workspace || workspace_size != g_workspace_size ||
        executor != g_executor ||
        stream != reinterpret_cast<aclrtStream>(kExpectedStream)) {
        InvalidateWiring("aclnnAdd received incorrect workspace, executor, or stream");
    }
    return Status(kAdd);
}

extern "C" aclnnStatus aclnnAbs(
    void* workspace,
    uint64_t workspace_size,
    aclOpExecutor* executor,
    aclrtStream stream) {
    Record("abs");
    g_last_abs_workspace = workspace;
    g_last_abs_workspace_size = workspace_size;
    g_last_abs_executor = executor;
    g_last_abs_stream = stream;
    g_last_execute_stream = stream;
    const void* expected_workspace = g_workspace_size == 0 ? nullptr :
        g_workspace;
    if (workspace != expected_workspace || workspace_size != g_workspace_size ||
        executor != g_executor ||
        stream != reinterpret_cast<aclrtStream>(kExpectedStream)) {
        InvalidateWiring(
            "aclnnAbs received incorrect workspace, executor, or stream");
    }
    return Status(kAbs);
}

extern "C" aclError aclrtSynchronizeStream(aclrtStream stream) {
    Record("synchronize");
    g_last_synchronize_stream = stream;
    if (stream != reinterpret_cast<aclrtStream>(kExpectedStream) ||
        stream != g_last_execute_stream) {
        InvalidateWiring("aclrtSynchronizeStream received an incorrect stream");
    }
    return Status(kSynchronize);
}

extern "C" aclnnStatus aclDestroyAclOpExecutor(aclOpExecutor* executor) {
    Record("destroy_executor");
    ++g_executor_destroy_count;
    delete executor;
    return Status(kDestroyExecutor);
}

extern "C" aclnnStatus aclDestroyTensor(const aclTensor* tensor) {
    Record(tensor->kind == 0 ? "destroy_x" :
        tensor->kind == 1 ? "destroy_y" : "destroy_output");
    const aclnnStatus status = tensor->kind == 0 ? Status(kDestroyX) :
        tensor->kind == 1 ? Status(kDestroyY) : Status(kDestroyOutput);
    delete tensor;
    return status;
}

extern "C" aclnnStatus aclDestroyScalar(const aclScalar* scalar) {
    Record("destroy_scalar");
    delete scalar;
    return Status(kDestroyScalar);
}

extern "C" aclError aclrtFree(void* workspace) {
    Record("free");
    g_last_free = workspace;
    delete[] static_cast<char*>(workspace);
    return Status(kFree);
}
