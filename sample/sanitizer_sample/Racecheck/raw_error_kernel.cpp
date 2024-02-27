#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;

constexpr int32_t BYTESIZE_EXAMPLE = 256;
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t NUM_DATA = BYTESIZE_EXAMPLE / sizeof(half);

extern "C" __global__ __aicore__ void raw_error_kernel(__gm__ uint8_t *gmInput, __gm__ uint8_t *gmOutput) {
    TPipe pipe;
    TQue<QuePosition::VECCALC, BUFFER_NUM> xQue;
    GlobalTensor<half> xInGm, xOutGm;
    pipe.InitBuffer(xQue, BUFFER_NUM, BYTESIZE_EXAMPLE);
    LocalTensor xLocal = xQue.AllocTensor<half>();
    xInGm.SetGlobalBuffer((__gm__ half*)gmInput, NUM_DATA);
    xOutGm.SetGlobalBuffer((__gm__ half*)gmOutput, NUM_DATA);
    DataCopy(xLocal, xInGm, NUM_DATA);
    // 17行为对UB进行写入，22行为对UB进行读，由于中间没有阻塞，UB上存在先写后读的竞争。解决方法为借助Que，先入队然后出队
    // xQue.EnQue(xLocal);
    // LocalTensor deQueLocal = xQue.DeQue<half>();
    // DataCopy(xOutGm, deQueLocal, NUM_DATA);
    DataCopy(xOutGm, xLocal, NUM_DATA);
}

extern "C" void raw_error_kernel_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *gmInput, uint8_t *gmOutput)
{
    raw_error_kernel<<<blockDim, l2ctrl, stream>>>(gmInput, gmOutput);
}
