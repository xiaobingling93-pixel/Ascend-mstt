import os
import shutil
import random
import unittest
import hashlib
import torch
import numpy as np
import torch.nn as nn
from msprobe.core.grad_probe.grad_compare import GradComparator
from msprobe.pytorch.grad_probe.grad_monitor import GradientMonitor
from msprobe.pytorch.pt_config import GradToolConfig

class config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

common_config_dict = {
    "rank": [],
    "step": [],
    "dump_path": "./grad_output"
}
common_config = config(common_config_dict)

task_config_dict = {
    "grad_level": "L1",
    "param_list": "",
    "bounds": [-1,0,1]
}
task_config = config(task_config_dict)

def seed_all(seed=1234, mode=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode)

seed_all()


inputs = [torch.rand(10, 10) for _ in range(10)]
labels = [torch.randint(0, 5, (10,)) for _ in range(10)]


class MockModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.linear(x)
        x2 = self.relu(x1)
        return x2


def get_grad_monitor():
    loss_fun = nn.CrossEntropyLoss()
    test_module = MockModule()
    nn.init.constant_(test_module.linear.weight, 1.0)
    nn.init.constant_(test_module.linear.bias, 1.0)
    optimizer = torch.optim.SGD(test_module.parameters(), lr=1e-2)

    gm = GradientMonitor(common_config, task_config)
    gm.monitor(test_module)

    for input_data, label in zip(inputs, labels):
        output = test_module(input_data)
        loss = loss_fun(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return gm


class TestGradMonitor(unittest.TestCase):
    def test_compare(self):
        gm = get_grad_monitor()
        compare_output_path = os.path.join(gm.output_path, "grad_compare")
        GradComparator.compare_distributed(gm.output_path, gm.output_path,
                                           compare_output_path)
        items = os.listdir(compare_output_path)
        self.assertEqual(len(items), 1)
        with open(os.path.join(compare_output_path, items[0], "similarities.csv"), 'r') as f:
            data = f.read()
        self.assertEqual(hashlib.md5(data.encode("utf-8")).hexdigest(), "138910fa9a4607d0adf6ff05e3753ed2")
        shutil.rmtree(gm.output_path)

