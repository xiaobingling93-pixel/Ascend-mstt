#include <iostream>
#include "acl/acl.h"

#define ACL_ERROR_NONE 0

#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0);

extern "C" void raw_error_kernel_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *gmInput, uint8_t *gmOutput);

int main(void)
{
    CHECK_ACL(aclInit(nullptr));
    aclrtContext context;
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *gmInput = nullptr;
    uint8_t *gmOutput = nullptr;
    CHECK_ACL(aclrtMalloc((void**)&gmInput, 256, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&gmOutput, 256, ACL_MEM_MALLOC_HUGE_FIRST));

    uint64_t blockDim = 1UL;
    raw_error_kernel_do(blockDim, nullptr, stream, gmInput, gmOutput);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtFree(gmInput));
    CHECK_ACL(aclrtFree(gmOutput));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}