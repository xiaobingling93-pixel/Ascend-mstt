import unittest
import torch
from msprobe.pytorch.api_accuracy_checker.precision_standard.standard_config import StandardConfig

class TestStandardConfig(unittest.TestCase):
    def test_get_small_value(self):
        # 测试已定义的数据类型
        self.assertEqual(StandardConfig.get_small_valuel(torch.float16), 2**-10)
        self.assertEqual(StandardConfig.get_small_valuel(torch.bfloat16), 2**-10)
        self.assertEqual(StandardConfig.get_small_valuel(torch.float32), 2**-20)
        
        # 测试未定义的数据类型（应返回默认值）
        self.assertEqual(StandardConfig.get_small_valuel(torch.int32), 2**-20)

    def test_get_small_value_atol(self):
        # 测试已定义的数据类型
        self.assertEqual(StandardConfig.get_small_value_atol(torch.float16), 2**-16)
        self.assertEqual(StandardConfig.get_small_value_atol(torch.bfloat16), 2**-16)
        self.assertEqual(StandardConfig.get_small_value_atol(torch.float32), 2**-30)
        
        # 测试未定义的数据类型（应返回默认值）
        self.assertEqual(StandardConfig.get_small_value_atol(torch.int32), 2**-30)

    def test_get_rtol(self):
        # 测试已定义的数据类型
        self.assertEqual(StandardConfig.get_rtol(torch.float16), 2**-10)
        self.assertEqual(StandardConfig.get_rtol(torch.bfloat16), 2**-8)
        self.assertEqual(StandardConfig.get_rtol(torch.float32), 2**-20)
        
        # 测试未定义的数据类型（应返回默认值）
        self.assertEqual(StandardConfig.get_rtol(torch.int32), 2**-20)

if __name__ == '__main__':
    unittest.main()