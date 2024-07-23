import unittest

import torch.nn as nn
from msprobe.pytorch import PrecisionDebugger
from msprobe.pytorch.functional.dump_module import module_dump, module_count


class TestDumpModule(unittest.TestCase):
    def setUp(self):
        self.module = nn.Linear(in_features=8, out_features=4)

    def test_module_dump(self):
        PrecisionDebugger(dump_path="./dump")
        module_dump(self.module, "TestModule")
        self.assertTrue("TestModule" in module_count)
