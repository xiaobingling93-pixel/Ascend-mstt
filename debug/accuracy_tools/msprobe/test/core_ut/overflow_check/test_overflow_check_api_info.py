#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
# `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------


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
