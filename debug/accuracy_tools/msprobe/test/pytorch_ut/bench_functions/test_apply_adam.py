import unittest
import torch

from msprobe.pytorch.bench_functions.apply_adam import npu_apply_adam


class TestNPUApplyAdam(unittest.TestCase):
        self.input0 = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        self.input1 = torch.tensor([[[[5.0, 6.0], [7.0, 8.0]]]])
        self.input2 = torch.tensor([[[[9.0, 10.0], [11.0, 12.0]]]])
        self.input3 = torch.tensor([[[[13.0, 14.0], [15.0, 16.0]]]])
        self.input4 = torch.tensor([[[[17.0, 18.0], [19.0, 20.0]]]])
        self.input5 = torch.tensor([[[[21.0, 22.0], [23.0, 24.0]]]])
        self.input6 = torch.tensor([[[[25.0, 26.0], [27.0, 28.0]]]])
        self.eps = 1e-5

    def test_npu_mish_positive(self):
        # 调用 npu_mish 函数
        result = npu_mish(self.input0, self.eps)
        
        # 预期的结果
        expected_result = torch.tensor([[[[0.8413, 0.9213], [0.9548, 0.9753]]]])
        
        # 使用 torch.allclose 进行近似比较
        self.assertTrue(torch.allclose(result[0], expected_result, atol=1e-4))

    def test_npu_mish_backward(self):
        # 创建一个需要梯度的张量
        input0 = self.input0.clone().requires_grad_(True)
        
        # 调用 npu_mish 函数
        result = npu_mish(input0, self.eps)
        
        # 计算梯度
        result[0].sum().backward()
        
        # 预期的梯度
        expected_grad = torch.tensor([[[[0.7311, 0.7311], [0.7311, 0.7311]]]])
        
        # 使用 torch.allclose 进行近似比较
        self.assertTrue(torch.allclose(input0.grad, expected_grad, atol=1e-4))