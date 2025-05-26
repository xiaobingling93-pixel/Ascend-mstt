import unittest
import os
import shutil
import torch
import torch.nn as nn
import mindspore
import mindspore.nn as mnn
from mindspore import Tensor
from msprobe.core import SingleSave
from msprobe.core import SingleComparator
from msprobe.core.common.file_utils import read_xlsx


# 固定随机性
torch.manual_seed(42)
mindspore.set_seed(42)


# 定义 PyTorch 简单网络
class SimpleTorchNet(nn.Module):
    def __init__(self):
        super(SimpleTorchNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        output = self.fc2(x)
        return x, output


# 定义 MindSpore 简单网络
class SimpleMindSporeNet(mnn.Cell):
    def __init__(self):
        super(SimpleMindSporeNet, self).__init__()
        self.fc1 = mnn.Dense(10, 5)
        self.fc2 = mnn.Dense(5, 1)

    def construct(self, x):
        x = self.fc1(x)
        x = mindspore.ops.relu(x)
        output = self.fc2(x)
        return x, output


class TestNetworkComparison(unittest.TestCase):
    def setUp(self):
        self.torch_dump_path = "./torch_dump"
        self.mindspore_dump_path = "./mindspore_dump"
        self.output_path = "./compare_output"
        self.num_test_cases = 5  # 随机测试用例数量        
    
    def tearDown(self):
        if os.path.exists(self.torch_dump_path):
            shutil.rmtree(self.torch_dump_path)
        if os.path.exists(self.mindspore_dump_path):
            shutil.rmtree(self.mindspore_dump_path)
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)

    def run_torch_network(self):
        net = SimpleTorchNet()
        saver = SingleSave(self.torch_dump_path, fmk="pytorch")
        
        for i in range(self.num_test_cases):
            # 为每个测试用例生成不同的随机输入
            input_tensor = torch.randn(1, 10)
            x, output = net(input_tensor)
            saver.save({"output1": x, "output2": output})
            saver.save({"output1": x})
            saver.step()  # 每个输入对应一个step

    def run_mindspore_network(self):
        net = SimpleMindSporeNet()
        SingleSave._instance = None  # 重置单例
        saver = SingleSave(self.mindspore_dump_path, fmk="mindspore")
        
        for i in range(self.num_test_cases):
            # 为每个测试用例生成不同的随机输入
            input_tensor = Tensor(mindspore.numpy.randn(1, 10))
            x, output = net(input_tensor)
            saver.save({"output1": x, "output2": output})
            saver.save({"output1": x})
            saver.step()  # 每个输入对应一个step

    def test_network_comparison(self):
        # 运行 PyTorch 网络并保存多组数据
        self.run_torch_network()

        # 运行 MindSpore 网络并保存多组数据
        self.run_mindspore_network()

        # 使用 SingleComparator 进行对比
        SingleComparator.compare(self.torch_dump_path, self.mindspore_dump_path, self.output_path)

        # 验证输出目录是否存在
        self.assertTrue(os.path.exists(self.output_path))
        
        output1_xlsx = read_xlsx(os.path.join(self.output_path, "output1.xlsx"))
        self.assertEqual(output1_xlsx.columns.tolist(), SingleComparator.result_header)
        self.assertEqual(len(output1_xlsx), 10)

        output2_xlsx = read_xlsx(os.path.join(self.output_path, "output2.xlsx"))
        self.assertEqual(output2_xlsx.columns.tolist(), SingleComparator.result_header)
        self.assertEqual(len(output2_xlsx), 5)