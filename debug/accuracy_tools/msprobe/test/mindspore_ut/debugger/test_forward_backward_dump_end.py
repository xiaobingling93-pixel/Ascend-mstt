import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import os
from mindspore import mint
import shutil
import json
from unittest import TestCase
import hashlib
from msprobe.core.common.file_utils import FileOpen

file_path = os.path.abspath(__file__)
directory = os.path.dirname(file_path)
config_json_path = os.path.join(directory, "config.json")

from msprobe.mindspore import PrecisionDebugger

def main():
    PrecisionDebugger._instance = None
    PrecisionDebugger.initialized = False
    debugger = PrecisionDebugger(config_json_path)
    num_classes = 10

    class SimplifiedAlexNet(nn.Cell):
        def __init__(self, num_classes=10, channel=3):
            super(SimplifiedAlexNet, self).__init__()
            # 第一层卷积
            self.conv1 = nn.Conv2d(channel, 96, 11, stride=4, pad_mode='same')
            self.relu1 = nn.ReLU()
            self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2)

            # 第二层卷积
            self.conv2 = nn.Conv2d(96, 256, 5, pad_mode='same')
            self.relu2 = nn.ReLU()

            # 全连接层
            self.flatten = nn.Flatten()
            self.fc1 = nn.Dense(13*13*256, num_classes)

        def construct(self, x):
            # 第一层卷积 + ReLU + MaxPool
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.max_pool2d(x)
            x = mint.add(x, 0.5)
            x = ops.mul(x, 1.2)
            debugger.forward_backward_dump_end()

            # 第二层卷积 + ReLU
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.max_pool2d(x)
            x = mint.add(x, 0.5)
            x = ops.mul(x, 1.2)

            # 展平 + 全连接层
            x = self.flatten(x)
            x = self.fc1(x)
            return x

    net = SimplifiedAlexNet(num_classes=num_classes)
    optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01)
    criterion = nn.MSELoss()

    def forward_fn(data, label):
        out = net(data)
        loss = criterion(out, label)
        return loss

    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters)

    def train_step(data, label):
        loss, grads = grad_fn(data, label)
        optimizer(grads)
        return loss

    batch_size = 1
    data = np.random.normal(1, 1, (batch_size, 3, 227, 227)).astype(np.float32)
    label = np.random.randint(0, num_classes, (batch_size,)).astype(np.int32)
    data = Tensor(data)
    label = Tensor(label)

    for i in range(3):
        debugger.start(net)
        loss = train_step(data, label)
        print(f"step: {i}, loss: {loss}")
        debugger.stop()
        debugger.step()

def save_dict_as_json(data, json_file_path):
    with FileOpen(json_file_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"字典已保存为json文件: {json_file_path}")

class TestDump(TestCase):
    def test_gradient_monitor_L2(self):
        output_path = os.path.join(directory, "output")
        if os.path.isfile(config_json_path):
            os.remove(config_json_path)
        if os.path.isdir(output_path):
            shutil.rmtree(output_path)

        config_dict = {
            "task": "statistics",
            "dump_path": output_path,
            "rank": [],
            "step": [],
            "level": "L1",
            "statistics": {
                "scope": [],
                "list":[],
                "data_mode": ["all"],
                "summary_mode": "statistics"
            },
        }
        save_dict_as_json(config_dict, config_json_path)
        main()

        #check
        target_keys = ["Primitive.conv2d.0.forward", "Primitive.relu.0.forward", "Primitive.max_pool.0.forward",
                       "Mint.add.0.forward", "Functional.mul.0.forward",
                       "Primitive.conv2d.0.backward", "Primitive.relu.0.backward", "Primitive.max_pool.0.backward",
                       "Mint.add.0.backward", "Functional.mul.0.backward",]
        for root, _, files in os.walk(output_path):
            for file in files:
                if file == 'dump.json':
                    dump_json_path = os.path.join(root, file)
                    with open(dump_json_path, 'r', encoding='utf-8') as file:
                        # 使用json.load()函数读取文件内容并转换为字典
                        data_dict = json.load(file)
                    data_dict = data_dict.get("data")
                    for key in target_keys:
                        self.assertTrue(key in data_dict, f"{key} not found in dump.json")
        if os.path.isfile(config_json_path):
            os.remove(config_json_path)
        if os.path.isdir(output_path):
            shutil.rmtree(output_path)