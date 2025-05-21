import os
import glob
import re
import shutil
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
        if os.path.exists(self.dump_dir):
            shutil.rmtree(self.dump_dir)

    def _assert_file_pattern_exists(self, dir_path, pattern):
        """
        检查目录下是否存在符合模式的文件
        :param pattern: 支持通配符的模式（如 'x_float32_*.npy'）
        """
        matches = glob.glob(os.path.join(dir_path, pattern))
        self.assertTrue(len(matches) > 0, 
                    f"No file matching {pattern} in {dir_path}. Found: {os.listdir(dir_path)}")

    def test_save_when_net_then_dump_data_in_dir(self):
        """测试正向保存+反向梯度保存的完整流程"""
        # Cell object to be differentiated
        class Net(nn.Cell):
            def construct(self, x, y, z):
                save("./test_dump", 'x', x)
                return x * y * z
        x = Tensor([1, 2], ms.float32)
        y = Tensor([-2, 3], ms.float32)
        z = Tensor([0, 3], ms.float32)
        net = Net()
        output = grad(net, grad_position=(1, 2))(x, y, z)


        # 验证dump文件生成
        step_dir = os.path.join("./test_dump", "step0", "rank0")
        self._assert_file_pattern_exists(step_dir, "x_float32_.*.npy")

    def test_save_grad_when_net_then_dump_in_dir(self):
        """测试梯度保存"""
        class Net(nn.Cell):
            def construct(self, x, y, z):
                z = save_grad("./test_dump", 'z', z)
                return x * y * z
        x = Tensor([1, 2], ms.float32)
        y = Tensor([-2, 3], ms.float32)
        z = Tensor([0, 3], ms.float32)
        net = Net()
        output = grad(net, grad_position=(1, 2))(x, y, z)

        # 验证dump文件生成
        step_dir = os.path.join("./test_dump", "step0", "rank0")
        self._assert_file_pattern_exists(step_dir, "z_grad_.*.npy")
