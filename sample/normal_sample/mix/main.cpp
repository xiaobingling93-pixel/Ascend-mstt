/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 */

#include <cassert>
#include <iostream>
#include "kernel_tiling/kernel_tiling.h"  // tiling结构体的依赖
#include "acl/acl.h"

extern void matmul_leakyrelu_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *ffts_addr, uint8_t *a,
    uint8_t *b, uint8_t *bias, uint8_t *c, uint8_t *workspace, uint8_t *tilingGm);

// 下面接口是libascendc_runtime.a中定义
extern "C" uint32_t GetAscendCoreSyncAddr(void **addr);

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

void fillValue(aclFloat16 *addr, size_t size, float value)
{
    aclFloat16 val = aclFloatToFloat16(value);
    for (size_t i = 0; i < size / sizeof(aclFloat16); i++) {
        addr[i] = val;
    }
}

void fillFloatValue(float *addr, size_t size, float value)
{
    for (size_t i = 0; i < size / sizeof(float); i++) {
        addr[i] = value;
    }
}

void MakeTiling(int32_t *addr, size_t size)
{
    assert(sizeof(TCubeTiling) <= size);
    // TCubeTiling该结构体在kernel_tiling/kernel_tiling.h中的结构体定义
    // tiling_api.h中本身定义的结构与kernel_tiling.h相近，通过GetTiling实现映射
    // TCubeTiling定义的可读性较好，可以直接理解，但使用tiling_api可以直接使能部分默认值
    // 考虑到工具本身需要体现对应用的细粒度控制，所以直接使用kernel_tiling.h中的结构
    TCubeTiling *tiling = (TCubeTiling *)addr;
    // 此处计算使用的核数
    tiling->usedCoreNum = 2;  // (M/singleCoreM)*(N/singleCoreN)*(K/singleCoreK)=2*1*1=2
    // 对于 xa 是[M, Ka]矩阵， xb 是[Kb, N]矩阵，此处数据需要与外部格式保持一致
    // 参考 AscendC算子开发文档
    // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha001/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0060.html
    // 中对 数据分块(Tiling) 部分的介绍
    tiling->M = 1024;
    tiling->N = 640;
    tiling->Ka = 256;  // Ka和Kb一般一样，只有pad的时候存在不一致，比如[1, 62]*[64, 2],这里64就是有pad的
    tiling->Kb = 256;
    tiling->isBias = 1;
    // 多核切分的tiling参数，用于度量单个核上处理的数据大小
    // xa在M轴上切分，分成多个singleCoreM；单核处理singleCoreM * singleCoreK大小数据
    // xb在N轴上切分，分成多个singleCoreN；单核处理singleCoreK * singleCoreN
    // 由于输入在M和N轴上切分了，输出singleCoreM * singleCoreN
    tiling->singleCoreM = 512;
    tiling->singleCoreN = 640;
    tiling->singleCoreK = 256;  // 不建议对k进行切分，会导致累加，引起不确定计算
    // 核内切分的tiling参数，用于单个核内的最小计算单位
    tiling->baseM = 256;
    tiling->baseN = 128;
    tiling->baseK = 64;
    tiling->stepM = 1;
    tiling->stepN = 1;
    tiling->stepKa = 4;
    tiling->stepKb = 1;
    // A1+B1的缓存数据需要小于等于shareL1Size大小
    tiling->depthA1 = 8;  // 矩阵[baseM, baseK]的缓存数量
    tiling->depthB1 = 2;  // 矩阵[basek, baseN]的缓存数量
    // 其他参数
    tiling->iterateOrder = 0;              // 控制迭代的方向：0代表先M轴再N轴，1代表先N轴再M轴
    tiling->shareL1Size = 294912;          // 如存在多个matmul时，可以单独控制每个使用空间 295424
                                           // 不小于(baseM*baseK*depthA1 + baseK+baseN*depthB1)*sizeof(half) = 294912
    tiling->shareL0CSize = 256 * 128 * 4;  // 如存在多个matmul时，可以单独控制每个使用空间
                                           // 不小于baseM*baseN*sizeof(float)
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

int32_t main(int32_t argc, char *argv[])
{
    size_t xaSize = 1024 * 256 * sizeof(aclFloat16);
    size_t xbSize = 256 * 640 * sizeof(aclFloat16);
    size_t biasSize = 640 * sizeof(float);
    size_t ySize = 1024 * 640 * sizeof(float);
    size_t workspaceSize = 16 * 1024 * 1024 * sizeof(float);  // AscendC::GetUserWorkspace中预留空间
    size_t tilingSize = 48 * sizeof(uint32_t);
    uint32_t blockDim = 1;

    CHECK_ACL(aclInit(nullptr));
    aclrtContext context;
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    aclFloat16 *xaHost;
    CHECK_ACL(aclrtMallocHost((void **)(&xaHost), xaSize));
    fillValue(xaHost, xaSize, 4.0f);

    aclFloat16 *xbHost;
    CHECK_ACL(aclrtMallocHost((void **)(&xbHost), xbSize));
    fillValue(xbHost, xbSize, 4.0f);

    float *biasHost;
    CHECK_ACL(aclrtMallocHost((void **)(&biasHost), biasSize));
    fillFloatValue(biasHost, biasSize, 0.0f);

    float *workspaceHost;
    CHECK_ACL(aclrtMallocHost((void **)(&workspaceHost), workspaceSize));
    fillFloatValue(workspaceHost, workspaceSize, 0.0f);

    int32_t *tilingHost;
    CHECK_ACL(aclrtMallocHost((void **)(&tilingHost), tilingSize));
    MakeTiling(tilingHost, tilingSize);

    // 将host的输入同步到device
    uint8_t *xaDevice;
    uint8_t *xbDevice;
    uint8_t *biasDevice;
    uint8_t *tilingDevice;
    uint8_t *workspaceDevice;
    CHECK_ACL(aclrtMalloc((void **)&xaDevice, xaSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(xaDevice, xaSize, xaHost, xaSize, ACL_MEMCPY_HOST_TO_DEVICE));  // 准备 xa
    CHECK_ACL(aclrtMalloc((void **)&xbDevice, xbSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(xbDevice, xbSize, xbHost, xbSize, ACL_MEMCPY_HOST_TO_DEVICE));  // 准备 xb
    CHECK_ACL(aclrtMalloc((void **)&biasDevice, biasSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(biasDevice, biasSize, biasHost, biasSize, ACL_MEMCPY_HOST_TO_DEVICE));  // 准备 bias
    CHECK_ACL(aclrtMalloc((void **)&tilingDevice, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));  // 准备 tiling
    CHECK_ACL(aclrtMalloc((void **)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(
        workspaceDevice, workspaceSize, workspaceHost, workspaceSize, ACL_MEMCPY_HOST_TO_DEVICE));  // 准备workspace

    uint8_t *yDevice;
    CHECK_ACL(aclrtMalloc((void **)&yDevice, ySize, ACL_MEM_MALLOC_HUGE_FIRST));  // 准备 输出

    void *addr;
    (void)GetAscendCoreSyncAddr(&addr);
    matmul_leakyrelu_custom_do(blockDim,
        nullptr,
        stream,
        (uint8_t *)addr,
        xaDevice,
        xbDevice,
        biasDevice,
        yDevice,
        workspaceDevice,
        tilingDevice);
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