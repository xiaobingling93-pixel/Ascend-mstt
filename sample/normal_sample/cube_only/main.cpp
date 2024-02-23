/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */

#include <chrono>
#include <iostream>
#include "assert.h"
#include "kernel_tiling/kernel_tiling.h"
#include "acl/acl.h"
extern void matmul_custom_do(
    uint32_t coreDim, void *l2ctrl, void *stream, uint8_t *param1, uint8_t *param2, uint8_t *param3, uint8_t *param4);

#define ACL_ERROR_NONE 0

#define CHECK_ACL(x)                                                   \
    do {                                                               \
        aclError __ret = x;                                            \
        if (__ret != ACL_ERROR_NONE) {                                 \
            printf("%s: %d aclError %d\n", __FILE__, __LINE__, __ret); \
        }                                                              \
    } while (0)

void printTensor(float *ptr, size_t size)
{
    size_t colNum = 8;
    for (size_t i = 0; i < size / colNum / sizeof(float); i++) {
        for (size_t j = 0; j < colNum; j++) {
            printf("%5.2f ", ptr[colNum * i + j]);
        }
        printf("\n");
    }
}

void fillValue(aclFloat16 *addr, size_t size)
{
    aclFloat16 val = aclFloatToFloat16(4.0f);
    for (size_t i = 0; i < size / sizeof(aclFloat16); i++) {
        addr[i] = val;
    }
}

void printAclFloat16(aclFloat16 *addr)
{
    for (int i = 0; i < 16; i++) {
        printf("%f ", aclFloat16ToFloat(addr[i]));
    }
}

void MakeTiling(uint32_t *addr, size_t size)
{
    assert(sizeof(TCubeTiling) <= size);
    TCubeTiling *tiling = (TCubeTiling *)addr;
    tiling->usedCoreNum = 16;
    tiling->M = 512;
    tiling->N = 1024;
    tiling->Ka = 512;
    tiling->Kb = 512;
    tiling->singleCoreM = 128;
    tiling->singleCoreN = 256;
    tiling->singleCoreK = 512;
    tiling->baseM = 128;
    tiling->baseN = 256;
    tiling->baseK = 64;
    tiling->depthA1 = 8;
    tiling->depthB1 = 8;
    tiling->stepM = 1;
    tiling->stepN = 1;
    tiling->isBias = 0;
    tiling->transLength = 131072;
    tiling->iterateOrder = 0;
    tiling->shareMode = 0;
    tiling->shareL1Size = 393216;
    tiling->shareL0CSize = 131072;
    tiling->shareUbSize = 0;
    tiling->batchM = 1;
    tiling->batchN = 1;
    tiling->singleBatchM = 1;
    tiling->singleBatchN = 1;
    tiling->stepKa = 8;
    tiling->stepKb = 8;
    tiling->dbL0A = 2;
    tiling->dbL0B = 2;
    tiling->dbL0C = 1;
    tiling->reserved = 0;
}

// y = matmul(xa, xb)
int32_t main(int32_t argc, char *argv[])
{
    size_t xaSize = 512 * 1024 * sizeof(aclFloat16);
    size_t xbSize = 512 * 1024 * sizeof(aclFloat16);
    size_t ySize = 512 * 1024 * sizeof(float);
    size_t tilingSize = 48 * sizeof(uint32_t);
    uint32_t blockDim = 8;

    CHECK_ACL(aclInit(nullptr));
    aclrtContext context;
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    aclFloat16 *xaHost;
    CHECK_ACL(aclrtMallocHost((void **)(&xaHost), xaSize));
    fillValue(xaHost, xaSize);

    aclFloat16 *xbHost;
    CHECK_ACL(aclrtMallocHost((void **)(&xbHost), xbSize));
    fillValue(xbHost, xbSize);

    uint32_t *tilingHost;
    CHECK_ACL(aclrtMallocHost((void **)(&tilingHost), tilingSize));
    MakeTiling(tilingHost, tilingSize);

    // 将host的输入同步到device
    uint8_t *xaDevice;
    uint8_t *xbDevice;
    uint8_t *tilingDevice;
    CHECK_ACL(aclrtMalloc((void **)&xaDevice, xaSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(xaDevice, xaSize, xaHost, xaSize, ACL_MEMCPY_HOST_TO_DEVICE));  // 准备 xa
    CHECK_ACL(aclrtMalloc((void **)&xbDevice, xbSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(xbDevice, xbSize, xbHost, xbSize, ACL_MEMCPY_HOST_TO_DEVICE));  // 准备 xb
    CHECK_ACL(aclrtMalloc((void **)&tilingDevice, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));  // 准备 tiling

    uint8_t *yDevice;
    CHECK_ACL(aclrtMalloc((void **)&yDevice, ySize, ACL_MEM_MALLOC_HUGE_FIRST));  // 准备 输出

    matmul_custom_do(blockDim, nullptr, stream, xaDevice, xbDevice, yDevice, tilingDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // 将device的输出同步到host
    float *yHost;
    CHECK_ACL(aclrtMallocHost((void **)(&yHost), ySize));
    CHECK_ACL(aclrtMemcpy(yHost, ySize, yDevice, ySize, ACL_MEMCPY_DEVICE_TO_HOST));
    printTensor(yHost, 4 * 8 * 4);

    // 释放资源
    CHECK_ACL(aclrtFree(xaDevice));
    CHECK_ACL(aclrtFree(xbDevice));
    CHECK_ACL(aclrtFree(tilingDevice));
    CHECK_ACL(aclrtFree(yDevice));

    CHECK_ACL(aclrtFreeHost(xaHost));
    CHECK_ACL(aclrtFreeHost(xbHost));
    CHECK_ACL(aclrtFreeHost(tilingHost));
    CHECK_ACL(aclrtFreeHost(yHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}