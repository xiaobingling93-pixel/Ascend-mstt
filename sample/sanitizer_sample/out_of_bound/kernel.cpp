#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;

constexpr int32_t BYTESIZE_EXAMPLE = 512;
constexpr int32_t NUM_DATA = 16;
constexpr int32_t CORE_OFFSET = 14;
constexpr int32_t LOOP_COUNT = 10;

extern "C" __global__ __aicore__ void kernel(__gm__ uint8_t *gm)
{
    TPipe pipe;
    TBuf<QuePosition::VECCALC> xlm;
    GlobalTensor<half> xGm;
    pipe.InitBuffer(xlm, BYTESIZE_EXAMPLE);
    LocalTensor<half> xLm = xlm.Get<half>();
    for (int32_t i = 0; i < LOOP_COUNT; i++)
    {
        if (i == GetBlockIdx())
        {
            xGm.SetGlobalBuffer((__gm__ half *)gm + i * CORE_OFFSET, NUM_DATA);
            DataCopy(xGm, xLm, NUM_DATA); // 这里CORE_OFFSET < NUM_DATA, 多核写入GM出现内存踩踏
        }
    }
}

extern "C" void kernel_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *gm)
{
    raw_error_kernel<<<blockDim, l2ctrl, stream>>>(gm);
}
