#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;

constexpr int32_t BYTESIZE = 256;
constexpr int32_t BYTESIZE_LARGE = 512;
constexpr int32_t NUM_DATA = BYTESIZE_EXAMPLE / sizeof(half);
constexpr int32_t NUM_DATA_LARGE = BYTESIZE_LARGE / sizeof(half);

extern "C" __global__ __aicore__ void illegal_read_and_write_kernel(__gm__ uint8_t *gm)
{
    TPipe pipe;
    TBuf<QuePosition::VECCALC> xlm;
    GlobalTensor<half> xGm;
    pipe.InitBuffer(xlm, BYTESIZE_LARGE);
    LocalTensor<half> xLm = xlm.Get<half>();
    xGm.SetGlobalBuffer((__gm__ half *)gm, NUM_DATA);
    DataCopy(xLm, xGm, NUM_DATA_LARGE); // 这里NUM_DATA_LARGE > NUM_DATA, 发生了对GM的越界非法读
    DataCopy(xGm, xLm, NUM_DATA_LARGE); // 这里NUM_DATA_LARGE > NUM_DATA, 发生了对GM的越界非法写
}

extern "C" void illegal_read_and_write_kernel_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *gm)
{
    illegal_read_and_write_kernel<<<blockDim, l2ctrl, stream>>>(gm);
}
