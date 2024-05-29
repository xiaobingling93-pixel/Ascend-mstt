import os
import random
import unittest
import hashlib
import torch
import numpy as np
import torch.nn as nn
from grad_tool.grad_monitor import GradientMonitor
from grad_tool.grad_comparator import GradComparator


def seed_all(seed=1234, mode=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode)

seed_all()

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

inputs = [torch.rand(10, 10) for _ in range(10)]
labels = [torch.randint(0, 5, (10,)) for _ in range(10)]


class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.linear(x)
        x2 = self.relu(x1)
        return x2


def test_grad_monitor():
    gm = GradientMonitor(os.path.join(base_dir, "resources/test_grad_monitor.yaml"))
    loss_fun = nn.CrossEntropyLoss()
    test_module = TestModule()
    nn.init.constant_(test_module.linear.weight, 1.0)
    nn.init.constant_(test_module.linear.bias, 1.0)
    gm.monitor(test_module)
    optimizer = torch.optim.SGD(test_module.parameters(), lr=1e-2)
    for input_data, label in zip(inputs, labels):
        output = test_module(input_data)
        loss = loss_fun(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return gm


def test_grad_monitor_1():
    gm = GradientMonitor(os.path.join(base_dir, "resources/test_save_grad.yaml"))
    loss_fun = nn.CrossEntropyLoss()
    test_module = TestModule()
    nn.init.constant_(test_module.linear.weight, 1.0)
    nn.init.constant_(test_module.linear.bias, 1.0)
    gm.monitor(test_module)
    optimizer = torch.optim.SGD(test_module.parameters(), lr=1e-2)
    for input_data, label in zip(inputs, labels):
        output = test_module(input_data)
        loss = loss_fun(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return gm


class TestGradMonitor(unittest.TestCase):
    def test_compare(self):
        gm1 = test_grad_monitor()
        gm2 = test_grad_monitor_1()
        compare_output_path = os.path.join(os.path.dirname(gm1._output_path), "grad_compare")
        GradComparator.compare_distributed(gm1._output_path, gm2._output_path, compare_output_path)
        items = os.listdir(compare_output_path)
        self.assertEqual(len(items), 1)
        with open(os.path.join(compare_output_path, items[0], "similarities.csv"), 'r') as f:
            data = f.read()
        self.assertEqual(hashlib.md5(data.encode("utf-8")).hexdigest(), "20441d98b8c8d14ee6f896ea29d01b14")
