import os
import unittest
import numpy as np
import mindspore as ms
from mindspore import Tensor, ops, nn, grad

from msprobe.mindspore import save, save_grad


class TestDumpFunctions(unittest.TestCase):
    def setUp(self):
        # 设置GRAPH模式确保算子融合
        ms.set_context(mode=ms.GRAPH_MODE, jit_config={'jit_level': "O2"}, device_target="Ascend")
        self.dump_dir = "./test_dump"
        os.makedirs(self.dump_dir, exist_ok=True)

    def tearDown(self):
        # 清理测试目录
        for root, _, files in os.walk(self.dump_dir):
            for f in files:
                os.remove(os.path.join(root, f))
        os.removedirs(self.dump_dir)

    def _assert_file_exists(self, path):
        self.assertTrue(os.path.exists(path), f"File {path} not found")

    def test_save_when_net_then_dump_data_in_dir(self):
        """测试正向保存+反向梯度保存的完整流程"""
        # Cell object to be differentiated
        class Net(nn.Cell):
            def construct(self, x, y, z):
                save(self.dump_dir, 'x', x)
                return x * y * z
        x = Tensor([1, 2], mindspore.float32)
        y = Tensor([-2, 3], mindspore.float32)
        z = Tensor([0, 3], mindspore.float32)
        net = Net()
        output = grad(net, grad_position=(1, 2))(x, y, z)


        # 验证dump文件生成
        step_dir = os.path.join(self.dump_dir, "step0", "rank0")
        self._assert_file_exists(os.path.join(step_dir, "x_float32_0.npy"))

    def test_save_grad_when_net_then_dump_in_dir(self):
        """测试控制流中的梯度保存"""
        class Net(nn.Cell):
            def construct(self, x, y, z):
                save_grad(self.dump_dir, 'z', z)
                return x * y * z
        x = Tensor([1, 2], mindspore.float32)
        y = Tensor([-2, 3], mindspore.float32)
        z = Tensor([0, 3], mindspore.float32)
        net = Net()
        output = grad(net, grad_position=(1, 2))(x, y, z)

        # 验证dump文件生成
        step_dir = os.path.join(self.dump_dir, "step0", "rank0")
        self._assert_file_exists(os.path.join(step_dir, "z_grad_float32_0.npy"))
