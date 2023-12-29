import unittest
import torch
import os
import shutil
from api_accuracy_checker.common.base_api import BaseAPIInfo


class TestBaseAPI(unittest.TestCase):
    def setUp(self):
        if os.path.exists('./forward'):
            shutil.rmtree('./forward')
        os.makedirs('./forward', mode=0o755)
        self.api = BaseAPIInfo("test_api", "./", "forward")

    def test_analyze_element(self):
        element = [torch.tensor(1), torch.rand(2), 3]
        self.api.analyze_element(element)
        self.assertTrue(os.path.exists("./forward/test_api.0.pt"))
        self.assertTrue(os.path.exists("./forward/test_api.1.pt"))
        self.assertFalse(os.path.exists("./forward/test_api.2.pt"))

    def test_analyze_tensor(self):
        tensor = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True)
        self.api.analyze_tensor(tensor)
        self.assertTrue(os.path.exists("./forward/test_api.0.pt"))

    def tearDown(self):
        if os.path.exists('./forward'):
            shutil.rmtree('./forward')
