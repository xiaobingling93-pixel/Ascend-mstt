import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import unittest
from unittest.mock import patch
from msprobe.core.common_config import CommonConfig
from msprobe.core.debugger.precision_debugger import BasePrecisionDebugger
from msprobe.pytorch.pt_config import StatisticsConfig
from msprobe.pytorch.debugger.precision_debugger import PrecisionDebugger
from msprobe.core.common.file_utils import load_json
import shutil

# 生成随机分类数据
X = torch.randn(100, 2)
y = ((X[:, 0] + X[:, 1]) > 0).float().reshape(-1, 1)

# 创建数据加载器
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10)

# 定义单层神经网络
class SingleLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x)


class MultiStartDebugger:
    debugger = None
    dump_path = None
    hooked_model = []

    @classmethod
    def init(cls, dump_path):
        cls.dump_path = dump_path
        json_config = {
            "task": "statistics",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [],
            "level": "L1",
            "async_dump": False
        }

        common_config = CommonConfig(json_config)
        task_config = StatisticsConfig(json_config)
        with patch.object(BasePrecisionDebugger, "_parse_config_path", return_value=(common_config, task_config)):
            cls.debugger = PrecisionDebugger(task="statistics", level="L0", dump_path=dump_path, step=["2-3"])
    
    @classmethod
    def debugger_start(cls, model, tag):
        cls.debugger.service.first_start = True if model not in cls.hooked_model else False
        cls.debugger.service.config.dump_path = os.path.join(cls.dump_path, tag)
        cls.debugger.start(model=model)
        if not cls.debugger.service.first_start and model not in cls.hooked_model:
            cls.hooked_model.append(model)

    @classmethod
    def debugger_stop(cls):
        cls.debugger.stop()
        cls.debugger.service.reset_status()

    @classmethod
    def debugger_step(cls):
        cls.debugger.step()


class TestPTDebuggerStart(unittest.TestCase):
    def test_debugger_multiple_start(self):
        dump_path = "./test_debugger_multiple_start_dump"
        
        model1 = SingleLayerNet()
        model2 = SingleLayerNet()
        MultiStartDebugger.init(dump_path)

        for batch_X, batch_y in dataloader:
            MultiStartDebugger.debugger_start(model=model1, tag="model1")
            output1 = model1(batch_X)
            MultiStartDebugger.debugger_stop()

            MultiStartDebugger.debugger_start(model=model2, tag="model2")
            output2 = model2(batch_X)
            MultiStartDebugger.debugger_stop()
            MultiStartDebugger.debugger_step()
        
        model1_dump_path = os.path.join(dump_path, "model1")
        self.assertTrue(os.path.exists(model1_dump_path))
        self.assertEqual(len(os.listdir(model1_dump_path)), 2)
        model1_construct_json = load_json(os.path.join(model1_dump_path, "step2", "rank", "construct.json"))
        self.assertEqual(len(model1_construct_json), 1)

        model2_dump_path = os.path.join(dump_path, "model2")
        self.assertTrue(os.path.exists(model2_dump_path))
        self.assertEqual(len(os.listdir(model2_dump_path)), 2)
        model2_construct_json = load_json(os.path.join(model2_dump_path, "step2", "rank", "construct.json"))
        self.assertEqual(len(model2_construct_json), 1)
        
        shutil.rmtree(dump_path)

