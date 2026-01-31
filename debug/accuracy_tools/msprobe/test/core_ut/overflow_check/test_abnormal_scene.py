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

from msprobe.core.overflow_check.abnormal_scene import AnomalyScene, InputNormalOutputAnomalyScene, \
    InputAnomalyOutputAnomalyScene, InputAnomalyOutputNormalScene, NumericalMutationScene
from msprobe.core.overflow_check.api_info import APIInfo
from msprobe.core.overflow_check.level import OverflowLevel


class TestAnomalyScene(unittest.TestCase):

    def setUp(self):
        self.api_info = APIInfo(
            api_name="torch.add",
            input_args=[{"type": "torch.Tensor", "Max": "nan"}],
            input_kwargs={"bias": {"type": "torch.Tensor", "Max": "-inf"}},
            output_data=[{"type": "torch.Tensor", "Norm": 0.8}]
        )
        self.anomaly_scene = InputAnomalyOutputNormalScene(self.api_info)

    def test_get_details(self):
        details = self.anomaly_scene.get_details()
        self.assertEqual(details["api_name"], self.api_info.api_name)
        self.assertEqual(details["scene_type"], "InputAnomalyOutputNormalScene")
        self.assertEqual(details["input_args_anomaly_indices"], [0])
        self.assertEqual(details["input_kwargs_anomaly_keys"], ["bias"])
        self.assertEqual(details["output_anomaly_indices"], [])


class TestInputNormalOutputAnomalyScene(unittest.TestCase):

    def setUp(self):
        self.api_info = APIInfo(
            api_name="torch.mul",
            input_args=[{"type": "torch.Tensor", "Max": 0.2}],
            input_kwargs={},
            output_data=[{"type": "torch.Tensor", "Max": "nan"}]
        )
        self.scene = InputNormalOutputAnomalyScene(self.api_info)

    def test_rank(self):
        self.assertEqual(self.scene.rank, OverflowLevel.CRITICAL)

    def test_matches(self):
        self.assertTrue(self.scene.matches())


class TestInputAnomalyOutputAnomalyScene(unittest.TestCase):

    def setUp(self):
        self.api_info = APIInfo(
            api_name="torch.div",
            input_args=[{"type": "torch.Tensor", "Max": "nan"}],
            input_kwargs={},
            output_data=[{"type": "torch.Tensor", "Max": "nan"}]
        )
        self.scene = InputAnomalyOutputAnomalyScene(self.api_info)

    def test_rank(self):
        self.assertEqual(self.scene.rank, OverflowLevel.HIGH)

    def test_matches(self):
        self.assertTrue(self.scene.matches())


class TestInputAnomalyOutputNormalScene(unittest.TestCase):

    def setUp(self):
        self.api_info = APIInfo(
            api_name="torch.relu",
            input_args=[{"type": "torch.Tensor", "Max": "nan"}],
            input_kwargs={},
            output_data=[{"type": "torch.Tensor", "Max": 0.8}]
        )
        self.scene = InputAnomalyOutputNormalScene(self.api_info)

    def test_rank(self):
        self.assertEqual(self.scene.rank, OverflowLevel.MEDIUM)

    def test_matches(self):
        self.assertTrue(self.scene.matches())

    def test_input_kwargs_matches(self):
        api_info = APIInfo(
            api_name="torch.linear",
            input_args=[],
            input_kwargs={
                "input1":{
                    "type": "torch.Tensor",
                    "Min": "nan",
                    "Max": 1.245486,
                }
            },
            output_data=[{"type": "torch.Tensor", "Norm": 0.8}]
        )
        scene = InputAnomalyOutputNormalScene(api_info)
        self.assertTrue(scene.matches())


class TestNumericalMutationScene(unittest.TestCase):

    def setUp(self):
        self.api_info = APIInfo(
            api_name="torch.exp",
            input_args=[{"type": "torch.Tensor", "Norm": 1.0}],
            input_kwargs={},
            output_data=[{"type": "torch.Tensor", "Norm": 200000.0}]
        )
        self.scene = NumericalMutationScene(self.api_info, threshold=100000.0)

    def test_rank(self):
        self.assertEqual(self.scene.rank, OverflowLevel.HIGH)

    def test_matches(self):
        self.assertTrue(self.scene.matches())

    def test_get_details(self):
        details = self.scene.get_details()
        self.assertEqual(details["api_name"], self.api_info.api_name)
        self.assertEqual(details["threshold"], 100000.0)
        self.assertTrue(details["scale_change_detected"])


if __name__ == "__main__":
    unittest.main()
