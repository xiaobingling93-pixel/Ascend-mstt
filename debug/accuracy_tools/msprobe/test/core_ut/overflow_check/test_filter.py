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
from unittest.mock import MagicMock, patch

from msprobe.core.overflow_check.api_info import APIInfo
from msprobe.core.overflow_check.filter import IgnoreFilter, Rule, IgnoreItem


class TestIgnoreFilter(unittest.TestCase):

    def setUp(self):
        self.mock_rule_path = "./mock_ignore_rules.yaml"
        self.filter = IgnoreFilter()

    @patch("msprobe.core.common.file_utils.load_yaml")
    def test_load_rules(self, mock_load_yaml):
        mock_load_yaml.return_value = {
            "ignore_nan_inf": [
                {
                    "api_name": "distributed.reduce_scatter",
                    "description": "Combines reduction and scatter operations. The output tensor may contain "
                                   "uninitialized data before the reduce_scatter call, but it will be overwritten "
                                   "with the reduced and scattered data from all processes.",
                    "input_ignore": [
                        {"index": 0}
                    ]
                }
            ]
        }

        self.filter._load_rules(self.mock_rule_path)
        self.assertIn("distributed.reduce_scatter", self.filter.rules)
        rule = self.filter.rules["distributed.reduce_scatter"]
        self.assertEqual(rule.api_name, "distributed.reduce_scatter")
        self.assertTrue(rule.input_ignore.has_index(0))

    def test_has_api_rule(self):
        self.filter.rules = {"distributed.reduce_scatter": Rule("distributed.reduce_scatter")}
        self.assertTrue(self.filter.has_api_rule("distributed.reduce_scatter"))
        self.assertFalse(self.filter.has_api_rule("torch.mul"))

    def test_apply_filter(self):
        api_info = APIInfo(
            api_name="torch.empty.0.forward",
            input_args=[{"Max": "nan"}],
            input_kwargs={},
            output_data=[]
        )
        rule = Rule(
            api_name="torch.empty",
            output_ignore=[{"index": 0}]
        )
        rule.match = MagicMock(return_value=True)
        self.filter.rules = {"torch.empty": rule}
        self.assertTrue(self.filter.apply_filter(api_info))

    def test_apply_filter_no_rule(self):
        api_info = APIInfo(
            api_name="torch.mul",
            input_args=[{"Max": "inf"}],
            input_kwargs={},
            output_data=[]
        )
        self.filter.rules = {"torch.empty_like": Rule("torch.empty_like")}
        self.assertFalse(self.filter.apply_filter(api_info))


class TestRule(unittest.TestCase):

    def setUp(self):
        self.rule = Rule(
            api_name="distributed.recv",
            desc="Test description",
            input_ignore=[
                {"index": 0},
                {"name": "tensor"}
            ],
            output_ignore=[{"index": 1}]
        )

    def test_verify_field(self):
        self.assertTrue(self.rule.verify_field())
        self.rule.api_name = ""
        self.assertFalse(self.rule.verify_field())

    def test_match(self):
        api_info = APIInfo(
            api_name="distributed.recv",
            input_args=[{"Max": "nan"}, {"Max": 0}],
            input_kwargs={
                "tensor": {"Max": "nan"}
            },
            output_data=[{"Max": 1}, {"Max": "inf"}]
        )
        self.assertTrue(self.rule.match(api_info))

    def test_match_no_ignore(self):
        api_info = APIInfo(
            api_name="torch.add",
            input_args=[{"Max": 0}],
            input_kwargs={},
            output_data=[{"Max": 1}]
        )
        self.assertFalse(self.rule.match(api_info))


class TestIgnoreItem(unittest.TestCase):

    def setUp(self):
        self.item = IgnoreItem()

    def test_add_index(self):
        self.item.add_index(0)
        self.assertIn(0, self.item.index)

    def test_add_name(self):
        self.item.add_name("bias")
        self.assertIn("bias", self.item.name)

    def test_has_index(self):
        self.item.add_index(0)
        self.assertTrue(self.item.has_index(0))
        self.assertFalse(self.item.has_index(1))

    def test_has_name(self):
        self.item.add_name("bias")
        self.assertTrue(self.item.has_name("bias"))
        self.assertFalse(self.item.has_name("weight"))


if __name__ == "__main__":
    unittest.main()
