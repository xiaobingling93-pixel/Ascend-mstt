#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;

constexpr int32_t BYTESIZE = 256;
constexpr int32_t BYTESIZE_LARGE = 512;
constexpr int32_t NUM_DATA = BYTESIZE / sizeof(half);
constexpr int32_t NUM_DATA_LARGE = BYTESIZE_LARGE / sizeof(half);

extern "C" __global__ __aicore__ void illegal_read_and_write_kernel(__gm__ uint8_t *gm)
{
    TPipe pipe;
    TBuf<QuePosition::VECCALC> xlm;
    GlobalTensor<half> xGm;
    pipe.InitBuffer(xlm, BYTESIZE_LARGE);
    LocalTensor<half> xLm = xlm.Get<half>();
    // 第18行给xGm分配了BYTESIZE字节的内存，但是第20、21行DataCopy搬运了BYTESIZE_LARGE字节的内存，
    // BYTESIZE_LARGE > BYTESIZE,导致对xGm的越界非法读写，解决方法是将第20、21行NUM_DATA_LARGE替换成NUM_DATA
    xGm.SetGlobalBuffer((__gm__ half *)gm, NUM_DATA);
    DataCopy(xLm, xGm, NUM_DATA_LARGE);
    DataCopy(xGm, xLm, NUM_DATA_LARGE);
}

extern "C" void illegal_read_and_write_kernel_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *gm)
{
    illegal_read_and_write_kernel<<<blockDim, l2ctrl, stream>>>(gm);
}
