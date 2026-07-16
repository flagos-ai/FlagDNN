#ifndef FLAGDNN_TEST_FAKE_CANN_ACL_META_H_
#define FLAGDNN_TEST_FAKE_CANN_ACL_META_H_

#include <stdint.h>

#include "acl/acl.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclOpExecutor aclOpExecutor;
typedef int32_t aclnnStatus;

aclTensor* aclCreateTensor(
    const int64_t* view_dims,
    uint64_t view_dim_num,
    aclDataType data_type,
    const int64_t* strides,
    int64_t storage_offset,
    aclFormat format,
    const int64_t* storage_dims,
    uint64_t storage_dim_num,
    void* data);
aclScalar* aclCreateScalar(void* value, aclDataType data_type);
aclnnStatus aclDestroyTensor(const aclTensor* tensor);
aclnnStatus aclDestroyScalar(const aclScalar* scalar);
aclnnStatus aclSetAclOpExecutorRepeatable(aclOpExecutor* executor);
aclnnStatus aclDestroyAclOpExecutor(aclOpExecutor* executor);

#ifdef __cplusplus
}
#endif

#endif  // FLAGDNN_TEST_FAKE_CANN_ACL_META_H_
