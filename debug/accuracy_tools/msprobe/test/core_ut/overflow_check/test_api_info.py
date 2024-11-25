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
from msprobe.core.common.const import Const
from msprobe.core.overflow_check.api_info import APIInfo


class TestAPIInfo(unittest.TestCase):

    def setUp(self):
        self.api_name = "Functional.linear.1.forward"
        self.input_args = [
            {
                "type": "torch.Tensor",
                "dtype": "torch.float32",
                "shape": [
                    40,
                    10
                ],
                "Max": 0.3156644105911255,
                "Min": -0.3159552812576294,
                "Mean": -0.007069610990583897,
                "Norm": 3.8414149284362793,
                "requires_grad": True
            },
            {
                "type": "torch.Tensor",
                "dtype": "torch.float32",
                "shape": [
                    40
                ],
                "Max": 0.2751258611679077,
                "Min": -0.29283690452575684,
                "Mean": -0.01155175268650055,
                "Norm": 1.0337861776351929,
                "requires_grad": True
            }
        ]
        self.input_kwargs = {
            "bias": {"type": "torch.Tensor", "Norm": 0.2}
        }
        self.output_data = [
            {"type": "torch.Tensor", "Norm": 0.8}
        ]

    def test_init(self):
        api_info = APIInfo(
            api_name=self.api_name,
            input_args=self.input_args,
            input_kwargs=self.input_kwargs,
            output_data=self.output_data
        )
        self.assertEqual(api_info.api_name, self.api_name)
        self.assertEqual(api_info.input_args, self.input_args)
        self.assertEqual(api_info.input_kwargs, self.input_kwargs)
        self.assertEqual(api_info.output_data, self.output_data)

    def test_extract_torch_api(self):
        torch_api = APIInfo.extract_torch_api("Functional.linear.1.backward")
        self.assertEqual(torch_api, "functional.linear")

        # Case with single part
        torch_api = APIInfo.extract_torch_api("torch")
        self.assertEqual(torch_api, "torch")

        # Case with empty string
        torch_api = APIInfo.extract_torch_api("")
        self.assertEqual(torch_api, "")



if __name__ == "__main__":
    unittest.main()
