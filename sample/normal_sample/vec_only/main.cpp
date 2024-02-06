#include <iostream>
#include "acl/acl.h"

#define ACL_ERROR_NONE 0

#define CHECK_ACL(x)                                                        \
    do {                                                                    \
        aclError __ret = x;                                                 \
        if (__ret != ACL_ERROR_NONE) {                                      \
            printf("%s: %d aclError %d\n", __FILE__, __LINE__, __ret);      \
        }                                                                   \
    } while(0)

void prepareTensor(int16_t *ptr, size_t size)
{
    for (size_t i = 0; i < size / sizeof(int16_t); i++) {
        ptr[i] = i;
    }
}

void printTensor(int16_t *ptr, size_t size)
{
    size_t colNum = 8;
    for (size_t i = 0; i < size / colNum / sizeof(int16_t); i++) {
        for (size_t j = 0; j < colNum; j++) {
            printf("%hu ", ptr[colNum * i + j]);
        }
        printf("\n");
    }
}

extern void add_custom_do(uint32_t coreDim, void *l2ctrl, void *stream, uint8_t *x, uint8_t *y, uint8_t *z);
int32_t main(int32_t argc, char *argv[])
{
    size_t inputByteSize = 8 * 2048 * sizeof(int16_t);
    size_t outputByteSize = 8 * 2048 * sizeof(int16_t);
    uint32_t blockDim = 8;
    // AscendCL初始化
    CHECK_ACL(aclInit(nullptr));
    // 运行管理资源申请
    aclrtContext context;
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    // 分配Host内存
    int16_t *xHost = nullptr;
    int16_t *yHost = nullptr;
    int16_t *zHost = nullptr;
    CHECK_ACL(aclrtMallocHost((void**)(&xHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void**)(&yHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void**)(&zHost), outputByteSize));
    // 分配Device内存
    uint8_t *xDevice = nullptr;
    uint8_t *yDevice = nullptr;
    uint8_t *zDevice = nullptr;
    CHECK_ACL(aclrtMalloc((void**)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&yDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&zDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    // Host内存初始化
    prepareTensor(xHost, inputByteSize);
    prepareTensor(yHost, inputByteSize);
    CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(yDevice, inputByteSize, yHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    // 用内核调用符<<<>>>调用核函数完成指定的运算,add_custom_do中封装了<<<>>>调用
    add_custom_do(blockDim, nullptr, stream, xDevice, yDevice, zDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    // 将Device上的运算结果拷贝回Host
    CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    printTensor(zHost, 2 * 8 * 2); // 显示2行
    // 释放申请的资源
    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFree(zDevice));
    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(yHost));
    CHECK_ACL(aclrtFreeHost(zHost));
    // AscendCL去初始化
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}