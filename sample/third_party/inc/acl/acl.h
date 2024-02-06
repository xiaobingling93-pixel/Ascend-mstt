#ifndef INC_EXTERNAL_ACL_ACL_H_
#define INC_EXTERNAL_ACL_ACL_H_

#include <cstdint>
#include "stdlib.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum aclrtMemcpyKind {
 ACL_MEMCPY_HOST_TO_HOST,
 ACL_MEMCPY_HOST_TO_DEVICE,
 ACL_MEMCPY_DEVICE_TO_HOST,
 ACL_MEMCPY_DEVICE_TO_DEVICE,
} aclrtMemcpyKind;

typedef enum aclrtMemMallocPolicy {
 ACL_MEM_MALLOC_HUGE_FIRST,
 ACL_MEM_MALLOC_HUGE_ONLY,
 ACL_MEM_MALLOC_NORMAL_ONLY,
 ACL_MEM_MALLOC_HUGE_FIRST_P2P,
 ACL_MEM_MALLOC_HUGE_ONLY_P2P,
 ACL_MEM_MALLOC_NORMAL_ONLY_P2P,
} aclrtMemMallocPolicy;

typedef void* aclrtStream;
typedef void* aclrtContext;
typedef int aclError;

aclError aclInit(const char* configPath);
aclError aclrtSetDevice(int32_t deviceId);
aclError aclrtCreateContext(aclrtContext *context, int32_t deviceId);
aclError aclrtCreateStream(aclrtStream* stream);
aclError aclrtMallocHost(void** hostPtr, size_t size);
aclError aclrtMalloc(void** devPtr, size_t size, aclrtMemMallocPolicy policy);
aclError aclrtMemcpy(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind);
aclError aclrtFree(void* devPtr);
aclError aclrtFreeHost(void* hostPtr);
aclError aclrtDestroyStream(aclrtStream stream);
aclError aclrtDestroyContext(aclrtContext context);
aclError aclrtResetDevice(int32_t deviceId);
aclError aclFinalize();
aclError aclrtSynchronizeStream(aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_ACL_H_