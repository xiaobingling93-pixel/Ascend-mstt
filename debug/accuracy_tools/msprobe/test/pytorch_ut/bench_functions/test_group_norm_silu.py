import unittest
import torch

from msprobe.pytorch.bench_functions.group_norm_silu import npu_group_norm_silu


class TestNPUGroupNormSILU(unittest.TestCase):
    def setUp(self):
        self.input0 = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        self.gama = torch.tensor([1.0])
        self.beta = torch.tensor([0.0])
        self.group = 1
        self.eps = 1e-5

    def test_npu_group_norm_silu_positive(self):
        # 调用 npu_group_norm_silu 函数
        result = npu_group_norm_silu(self.input0, self.gama, self.beta, self.group, self.eps)
        
        # 预期的结果
        expected_result = torch.tensor([[[[0.8413, 0.9213], [0.9548, 0.9753]]]])
        
        # 使用 torch.allclose 进行近似比较
        self.assertTrue(torch.allclose(result[0], expected_result, atol=1e-4))

    def test_npu_group_norm_silu_backward(self):
        # 创建一个需要梯度的张量
        input0 = self.input0.clone().requires_grad_(True)
        
        # 调用 npu_group_norm_silu 函数
        result = npu_group_norm_silu(input0, self.gama, self.beta, self.group, self.eps)
        
        # 计算梯度
        result[0].sum().backward()
        
        # 预期的梯度
        expected_grad = torch.tensor([[[[0.7311, 0.7311], [0.7311, 0.7311]]]])
        
        # 使用 torch.allclose 进行近似比较
        self.assertTrue(torch.allclose(input0.grad, expected_grad, atol=1e-4))
