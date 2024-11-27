#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2024-2024. Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import unittest

from msprobe.core.overflow_check.checker import AnomalyDetector
from msprobe.core.overflow_check.level import OverflowLevel


class TestAnomalyDetector(unittest.TestCase):
    def setUp(self):
        """初始化测试数据"""
        self.dump_data = {
            # 场景 1: 输入正常，输出异常
            "Torch.add.0.forward": {
                "input_args": [
                    {
                        "type": "torch.Tensor",
                        "dtype": "torch.float32",
                        "shape": [100, 40],
                        "Max": 1.223,
                        "Min": -1.386,
                        "Mean": -0.0448,
                        "Norm": 26.5,
                        "requires_grad": True,
                    }
                ],
                "input_kwargs": {},
                "output": [
                    {
                        "type": "torch.Tensor",
                        "dtype": "torch.float32",
                        "shape": [100, 40],
                        "Max": float("nan"),
                        "Min": float("-inf"),
                        "Mean": float("nan"),
                        "Norm": float("inf"),
                        "requires_grad": False,
                    }
                ],
            },
            # 场景 2: 输入异常，输出异常
            "Torch.mul.0.forward": {
                "input_args": [
                    {
                        "type": "torch.Tensor",
                        "dtype": "torch.float32",
                        "shape": [100, 40],
                        "Max": float("inf"),
                        "Min": -1.386,
                        "Mean": float("nan"),
                        "Norm": float("nan"),
                        "requires_grad": True,
                    }
                ],
                "input_kwargs": {},
                "output": [
                    {
                        "type": "torch.Tensor",
                        "dtype": "torch.float32",
                        "shape": [100, 40],
                        "Max": float("nan"),
                        "Min": float("-inf"),
                        "Mean": float("nan"),
                        "Norm": float("inf"),
                        "requires_grad": False,
                    }
                ],
            },
            # 场景 3: 输入异常，输出正常
            "Torch.div.0.forward": {
                "input_args": [
                    {
                        "type": "torch.Tensor",
                        "dtype": "torch.float32",
                        "shape": [100, 40],
                        "Max": float("inf"),
                        "Min": -1.386,
                        "Mean": float("nan"),
                        "Norm": float("nan"),
                        "requires_grad": True,
                    }
                ],
                "input_kwargs": {},
                "output": [
                    {
                        "type": "torch.Tensor",
                        "dtype": "torch.float32",
                        "shape": [100, 40],
                        "Max": 2.0,
                        "Min": -1.0,
                        "Mean": 0.5,
                        "Norm": 50.0,
                        "requires_grad": False,
                    }
                ],
            },
            # 场景 4: 数值突变
            "Torch.matmul.0.forward": {
                "input_args": [
                    {
                        "type": "torch.Tensor",
                        "dtype": "torch.float32",
                        "shape": [100, 40],
                        "Max": 1.0,
                        "Min": -1.0,
                        "Mean": 0.0,
                        "Norm": 10.0,
                        "requires_grad": True,
                    }
                ],
                "input_kwargs": {},
                "output": [
                    {
                        "type": "torch.Tensor",
                        "dtype": "torch.float32",
                        "shape": [100, 40],
                        "Max": 1e6,
                        "Min": -1e6,
                        "Mean": 0.0,
                        "Norm": 1e10,
                        "requires_grad": False,
                    }
                ],
            },
        }
        self.detector = AnomalyDetector(self.dump_data)

    def test_analyze(self):
        """测试 analyze 方法，确保场景检测正确分类"""
        self.detector.analyze()

        # 检查每个场景是否正确分类
        self.assertTrue(self.detector.has_overflow("Torch.add.0.forward"))  # 输入正常，输出异常
        self.assertEqual(
            self.detector.get_overflow_level("Torch.add.0.forward"),
            OverflowLevel.CRITICAL,
        )

        self.assertTrue(self.detector.has_overflow("Torch.mul.0.forward"))  # 输入异常，输出异常
        self.assertEqual(
            self.detector.get_overflow_level("Torch.mul.0.forward"),
            OverflowLevel.HIGH,
        )

        self.assertTrue(self.detector.has_overflow("Torch.div.0.forward"))  # 输入异常，输出正常
        self.assertEqual(
            self.detector.get_overflow_level("Torch.div.0.forward"),
            OverflowLevel.MEDIUM,
        )

        self.assertTrue(self.detector.has_overflow("Torch.matmul.0.forward"))  # 数值突变
        self.assertEqual(
            self.detector.get_overflow_level("Torch.matmul.0.forward"),
            OverflowLevel.HIGH,
        )

    def test_filter(self):
        """测试 filter 方法，确保 Torch.empty 被正确过滤"""
        self.dump_data["Torch.empty.0.forward"] = {
            "input_args": [
                {
                    "type": "torch.Tensor",
                    "dtype": "torch.float32",
                    "shape": [
                        100,
                        40
                    ],
                    "Max": 1.2230066061019897,
                    "Min": -1.3862265348434448,
                    "Mean": -0.044829513877630234,
                    "Norm": 26.499610900878906,
                    "requires_grad": True
                }
            ],
            "input_kwargs": {},
            "output": [
                {
                    "type": "torch.Tensor",
                    "dtype": "torch.float32",
                    "shape": [100, 40],
                    "Max": float("inf"),
                    "Min": float("-inf"),
                    "Mean": float("nan"),
                    "Norm": float("nan"),
                    "requires_grad": False,
                }
            ],
        }

        new_detector = AnomalyDetector(self.dump_data)

        new_detector.analyze().filter()
        self.assertFalse(new_detector.has_overflow("Torch.empty.0.forward"))  # 被过滤
        self.assertTrue(new_detector.has_overflow("Torch.add.0.forward"))  # 未过滤

    def test_statistics(self):
        """测试统计信息输出"""
        self.detector.analyze().filter()
        stats = self.detector.get_statistics()
        self.assertIn("critical_apis", stats)
        self.assertIn("high_priority_apis", stats)
        self.assertIn("medium_priority_apis", stats)
        self.assertIn("anomaly_details", stats)

    def test_overflow_result(self):
        """测试 overflow_result 方法"""
        self.detector.analyze()
        results = self.detector.overflow_result()

        # 验证结果是否包含异常
        self.assertIn("Torch.add.0.forward", results)
        self.assertIn("Torch.mul.0.forward", results)

    def test_has_overflow(self):
        """测试 has_overflow 方法"""
        self.detector.analyze()
        self.assertTrue(self.detector.has_overflow("Torch.add.0.forward"))
        self.assertFalse(self.detector.has_overflow("Non.existent.api"))

    def test_get_overflow_level(self):
        """测试 get_overflow_level 方法"""
        self.detector.analyze()
        level = self.detector.get_overflow_level("Torch.add.0.forward")
        self.assertEqual(level, OverflowLevel.CRITICAL)

        # 测试不存在的 API
        self.assertIsNone(self.detector.get_overflow_level("Non.existent.api"))

    def test_chain_calls(self):
        """测试链式调用"""
        self.detector.analyze().filter()
        stats = self.detector.get_statistics()

        # 验证链式调用的结果
        self.assertIn("anomaly_details", stats)


if __name__ == "__main__":
    unittest.main()
