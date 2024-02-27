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
    xGm.SetGlobalBuffer((__gm__ half *)gm + GetBlockIdx() * CORE_OFFSET, NUM_DATA);
    // 这里第17行CORE_OFFSET < NUM_DATA, 第21行多核写入GM时，写入的size大于偏移，导致出现内存踩踏
    // 以下是正确写法
    // xGm.SetGlobalBuffer((__gm__ half *)gm + GetBlockIdx() * NUM_DATA, NUM_DATA);
    DataCopy(xGm, xLm, NUM_DATA);
}

extern "C" void out_of_bound_kernel_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *gm)
{
    out_of_bound_kernel<<<blockDim, l2ctrl, stream>>>(gm);
}
