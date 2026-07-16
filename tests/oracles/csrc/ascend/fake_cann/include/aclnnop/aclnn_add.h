#ifndef FLAGDNN_TEST_FAKE_CANN_ACLNN_ADD_H_
#define FLAGDNN_TEST_FAKE_CANN_ACLNN_ADD_H_

#include <stdint.h>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnAddGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* y,
    const aclScalar* alpha,
    aclTensor* output,
    uint64_t* workspace_size,
    aclOpExecutor** executor);
aclnnStatus aclnnAdd(
    void* workspace,
    uint64_t workspace_size,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // FLAGDNN_TEST_FAKE_CANN_ACLNN_ADD_H_
