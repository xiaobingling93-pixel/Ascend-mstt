/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

template <typename aType, typename bType, typename cType, typename biasType>
class MatmulLeakyKernel {
public:
    __aicore__ inline MatmulLeakyKernel(){};
    __aicore__ inline void Init(
        GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling, TPipe *pipe);
    __aicore__ inline void Process(TPipe *pipe);

    __aicore__ inline void MatmulCompute();
    __aicore__ inline void LeakyReluCompute();
    __aicore__ inline void CopyOut(uint32_t count);
    __aicore__ inline void CalcOffset(int32_t blockIdx, int32_t usedCoreNum, const TCubeTiling &tiling,
        int32_t &offsetA, int32_t &offsetB, int32_t &offsetC, int32_t &offsetBias);

    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, aType>, MatmulType<TPosition::GM, CubeFormat::ND, bType>,
        MatmulType<TPosition::VECIN, CubeFormat::ND, cType>, MatmulType<TPosition::GM, CubeFormat::ND, biasType>>
        matmulObj;

    GlobalTensor<aType> aGlobal;
    GlobalTensor<bType> bGlobal;
    GlobalTensor<cType> cGlobal;
    GlobalTensor<biasType> biasGlobal;
    LocalTensor<cType> reluOutLocal;
    TCubeTiling tiling;
    TQue<QuePosition::VECOUT, 1> reluOutQueue_;
};

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::Init(
    GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace, GM_ADDR tilingGM, TPipe *pipe)
{
    auto tempTilingGM = (__gm__ uint32_t *)tilingGM;
    auto tempTiling = (uint32_t *)&tiling;
    for (int32_t i = 0; i < sizeof(TCubeTiling) / sizeof(int32_t); ++i, ++tempTilingGM, ++tempTiling) {
        *tempTiling = *tempTilingGM;
    }
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), tiling.M * tiling.N);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), tiling.N);

    int32_t offsetA, offsetB, offsetC, offsetBias;
    CalcOffset(GetBlockIdx(), tiling.usedCoreNum, tiling, offsetA, offsetB, offsetC, offsetBias);
    aGlobal = aGlobal[offsetA];
    bGlobal = bGlobal[offsetB];
    cGlobal = cGlobal[offsetC];
    biasGlobal = biasGlobal[offsetBias];
    pipe->InitBuffer(reluOutQueue_, 1, tiling.baseM * tiling.baseN * sizeof(cType));
    SetSysWorkspace(workspace);
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::Process(TPipe *pipe)
{
    uint32_t computeRound = 0;

#ifdef CUSTOM_ASCEND310P
    TBuf<> tmpMMFormatUb;
    LocalTensor<uint8_t> mmformatUb;
    pipe->InitBuffer(tmpMMFormatUb, tiling.baseM * tiling.baseN * sizeof(cType));
    mmformatUb = tmpMMFormatUb.Get<uint8_t>(tiling.baseM * tiling.baseN * sizeof(cType));
    matmulObj.SetLocalWorkspace(mmformatUb);
#endif
    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);
    matmulObj.SetBias(biasGlobal);
    while (matmulObj.template Iterate<true>()) {
        MatmulCompute();
        LeakyReluCompute();
        CopyOut(computeRound);
        computeRound++;
    }
    matmulObj.End();
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::MatmulCompute()
{
    reluOutLocal = reluOutQueue_.AllocTensor<cType>();
    matmulObj.template GetTensorC<true>(reluOutLocal, false, true);
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::LeakyReluCompute()
{
    LeakyRelu(reluOutLocal, reluOutLocal, (cType)0.001, tiling.baseM * tiling.baseN);
    reluOutQueue_.EnQue(reluOutLocal);
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::CopyOut(uint32_t count)
{
    reluOutQueue_.DeQue<cType>();
    const uint32_t roundM = tiling.singleCoreM / tiling.baseM;
    const uint32_t roundN = tiling.singleCoreN / tiling.baseN;
    uint32_t startOffset = (count % roundM * tiling.baseM * tiling.N + count / roundM * tiling.baseN);
    DataCopyParams copyParam = {(uint16_t)tiling.baseM,
        (uint16_t)(tiling.baseN * sizeof(cType) / DEFAULT_C0_SIZE),
        0,
        (uint16_t)((tiling.N - tiling.baseN) * sizeof(cType) / DEFAULT_C0_SIZE)};
    DataCopy(cGlobal[startOffset], reluOutLocal, copyParam);
    reluOutQueue_.FreeTensor(reluOutLocal);
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::CalcOffset(int32_t blockIdx,
    int32_t usedCoreNum, const TCubeTiling &tiling, int32_t &offsetA, int32_t &offsetB, int32_t &offsetC,
    int32_t &offsetBias)
{
    auto mSingleBlocks = Ceil(tiling.M, tiling.singleCoreM);
    auto mCoreIndx = blockIdx % mSingleBlocks;
    auto nCoreIndx = blockIdx / mSingleBlocks;

    offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
    offsetB = nCoreIndx * tiling.singleCoreN;
    offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
    offsetBias = nCoreIndx * tiling.singleCoreN;
}

__aicore__ inline void matmul_leakyrelu_custom_inner(
    GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    MatmulLeakyKernel<half, half, float, float> matmulLeakyKernel;
    TPipe pipe;
    matmulLeakyKernel.Init(a, b, bias, c, workspace, tiling, &pipe);
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulLeakyKernel.matmulObj, &matmulLeakyKernel.tiling);
    matmulLeakyKernel.Process(&pipe);
}

// 基于CANN sample仓中的MatMulLeakyReluCustomSample进行改造
// https://gitee.com/ascend/samples/tree/master/operator/
// 如下内容是编译生成的build/auto_gen/ascendc_kernels_npu目录下
// auto_gen_matmul_leakyrelu_custom.cpp中代码的简化
// 具体简化内容：
// 1. 移除了cpu部分的代码
// 2. 由于本身不再需要修改核函数入口，故而不需要__global__宏的替代
// 3. 由于明确有workspace和tiling，故而移除了这些宏中的内容
__aicore__ inline GM_ADDR kfc_init(GM_ADDR ffts_addr, GM_ADDR workspace)
{
    GM_ADDR workspace_param;
    GM_ADDR workspace_usr;
    workspace_param = workspace;
    if (workspace_param == nullptr) {
        return workspace;
    }
    set_ffts_base_addr((uint64_t)ffts_addr);
    AscendC::SetSysWorkspaceForce(workspace_param);
    workspace_usr = AscendC::GetUserWorkspace(workspace_param);
#if defined(REGIST_MATMUL_OBJ)
    if constexpr (g_coreType == AscendC::AIC) {
        matmul::clearWorkspace(workspace_param);
    }
    if constexpr (g_coreType == AscendC::AIV) {
#ifdef MIX_N
        FftsCrossCoreSync(PIPE_MTE3, 0x501);
        WaitEvent(0x5);
#endif
    }
#endif
    return workspace_usr;
}

extern "C" __global__ __aicore__ void matmul_leakyrelu_custom(
    GM_ADDR ffts_addr, GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    workspace = kfc_init(ffts_addr, workspace);
    matmul_leakyrelu_custom_inner(a, b, bias, c, workspace, tiling);
}

// call of kernel function
void matmul_leakyrelu_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *ffts_addr, uint8_t *a,
    uint8_t *b, uint8_t *bias, uint8_t *c, uint8_t *workspace, uint8_t *tilingGm)
{
    matmul_leakyrelu_custom<<<blockDim, l2ctrl, stream>>>(ffts_addr, a, b, bias, c, workspace, tilingGm);
}
