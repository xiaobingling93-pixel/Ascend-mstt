#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;

constexpr int32_t BYTESIZE = 512;
constexpr int32_t NUM_DATA = 16;
constexpr int32_t CORE_OFFSET = 14;
constexpr int32_t LOOP_COUNT = 10;

extern "C" __global__ __aicore__ void out_of_bound_kernel(__gm__ uint8_t *gm)
{
    TPipe pipe;
    TBuf<QuePosition::VECCALC> xlm;
    GlobalTensor<half> xGm;
    pipe.InitBuffer(xlm, BYTESIZE);
    LocalTensor<half> xLm = xlm.Get<half>();
    for (int32_t i = 0; i < LOOP_COUNT; i++)
    {
        if (i == GetBlockIdx())
        {
            // 这里第22行CORE_OFFSET < NUM_DATA, 第23行多核写入GM时，写入的size大于偏移，导致出现内存踩踏
            // 解决方法时将CORE_OFFSET替换成NUM_DATA
            xGm.SetGlobalBuffer((__gm__ half *)gm + i * CORE_OFFSET, NUM_DATA);
            DataCopy(xGm, xLm, NUM_DATA);
        }
    }
}

extern "C" void out_of_bound_kernel_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *gm)
{
    out_of_bound_kernel<<<blockDim, l2ctrl, stream>>>(gm);
}
