#ifndef FLAGDNN_TEST_FAKE_CANN_ACLNN_ABS_H_
#define FLAGDNN_TEST_FAKE_CANN_ACLNN_ABS_H_

#include <stdint.h>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnAbsGetWorkspaceSize(
    const aclTensor* self,
    aclTensor* out,
    uint64_t* workspace_size,
    aclOpExecutor** executor);
aclnnStatus aclnnAbs(
    void* workspace,
    uint64_t workspace_size,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // FLAGDNN_TEST_FAKE_CANN_ACLNN_ABS_H_
