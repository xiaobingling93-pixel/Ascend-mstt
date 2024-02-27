#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;

constexpr int32_t BYTESIZE_EXAMPLE = 256;
constexpr int32_t NUM_DATA = BYTESIZE_EXAMPLE / sizeof(half);

extern "C" __global__ __aicore__ void illegal_align_kernel(__gm__ uint8_t *gm) {
    TPipe pipe;
    TBuf<QuePosition::VECCALC> tbuf;
    pipe.InitBuffer(tbuf, BYTESIZE_EXAMPLE);
    LocalTensor<half> xLm = tbuf.Get<half>();

    GlobalTensor<half> xGm;
    xGm.SetGlobalBuffer((__gm__ half*)gm, NUM_DATA);

    // 对 UB 进行了错误的偏移导致 DataCopy 接口在对数据进行搬运时产生非对齐访问异常
    DataCopy(xGm, xLm[3], NUM_DATA);
    // 正确的用法如下，在操作 Local Tensor 时地址偏移量应为 32 字节对齐
    DataCopy(xGm, xLm[32], NUM_DATA);
}

extern "C" void illegal_align_kernel_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *gm)
{
    illegal_align_kernel<<<blockDim, l2ctrl, stream>>>(gm);
}
