import torch
import torch_npu
import ascend_custom_kernels_lib
from torch_npu.testing.testcase import TestCase, run_tests


class TestCustomAdd(TestCase):
    def test_add(self):
        # 由于kernel现在是静态tiling，所以此处尺寸需要匹配
        # 因为add是elementwise的，现有算子支持8*2048(详见kernel实现)，所以，小于这个应该都可以
        length = [8, 2048]
        x = torch.rand(length, device='cpu', dtype=torch.float16)
        y = torch.rand(length, device='cpu', dtype=torch.float16)

        x_npu = x.npu()
        y_npu = y.npu()
        x_npu.requires_grad = True
        y_npu.requires_grad = True
        output = ascend_custom_kernels_lib.my_add(x_npu, y_npu)
        # 反向能力验证
        output.backward(output)

        x.requires_grad = True
        y.requires_grad = True
        cpuout = torch.add(x, y)
        cpuout.backward(cpuout)

        self.assertRtolEqual(output, cpuout)
        self.assertRtolEqual(x_npu.grad, x.grad)
        self.assertRtolEqual(y_npu.grad, y.grad)


if __name__ == "__main__":
    run_tests()
