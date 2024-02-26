/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */

#include <cassert>
#include <iostream>
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
    // TCubeTiling该结构体参考kernel_tiling/kernel_tiling.h中的结构体定义
    // tiling_api.h中本身定义的结构与kernel_tiling.h
    TCubeTiling *tiling = (TCubeTiling *)addr;
    // 此处计算使用的核数
    tiling->usedCoreNum = 16;  // (M/singleCoreM)*(N/singleCoreN)*(K/singleCoreK)=4*4*1=16
    // 对于 xa 是[M, Ka]矩阵， xb 是[Kb, N]矩阵，此处数据需要与外部格式保持一致
    // 参考 AscendC算子开发文档
    // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha001/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0060.html
    // 中对 数据分块(Tiling) 部分的介绍
    tiling->M = 512;   //
    tiling->N = 1024;  //
    tiling->Ka = 512;  // Ka和Kb一般一样，只有pad的时候存在不一致，比如[1, 62]*[64, 2],这里64就是有pad的
    tiling->Kb = 512;    //
    tiling->isBias = 0;  // 是否有bias
    // 多核切分的tiling参数，用于度量单个核上处理的数据大小
    // xa在M轴上切分，分成多个singleCoreM；单核处理singleCoreM * singleCoreK大小数据
    // xb在N轴上切分，分成多个singleCoreN；单核处理singleCoreK * singleCoreN
    // 由于输入在M和N轴上切分了，输出
    tiling->singleCoreM = 128;
    tiling->singleCoreN = 256;
    tiling->singleCoreK = 512;  // 不建议对k进行切分，会导致累加，引起不确定计算
    // 核内切分的tiling参数，用于单个核内的最小计算单位
    tiling->baseM = 128;
    tiling->baseN = 256;
    tiling->baseK = 64;
    tiling->stepM = 1;
    tiling->stepN = 1;
    tiling->stepKa = 8;
    tiling->stepKb = 8;
    tiling->depthA1 = 8;  // 矩阵[baseM, baseK]的缓存数量
    tiling->depthB1 = 8;  // 矩阵[basek, baseN]的缓存数量
    // 其他参数
    tiling->iterateOrder = 0;           // 控制迭代的方向：0代表先M轴再N轴，1代表先N轴再M轴
    tiling->shareL1Size = 384 * 1024;   // 如存在多个matmul时，可以单独控制每个使用空间
    tiling->shareL0CSize = 128 * 1024;  // 如存在多个matmul时，可以单独控制每个使用空间
    tiling->shareUbSize = 0;            // 310P非分核时涉及
    tiling->transLength = 131072;       // 310P使用涉及格式转换时的额外空间长度
    // 下列是bmm中使用的batch参数，如果需要实现bmm，该结构体中还有其他tiling参数
    tiling->batchM = 1;  // 对于普通matmul，默认1
    tiling->batchN = 1;  // 对于普通matmul，默认1
    tiling->singleBatchM = 1;
    tiling->singleBatchN = 1;
    // 下面的db参数用于控制ping-pong
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