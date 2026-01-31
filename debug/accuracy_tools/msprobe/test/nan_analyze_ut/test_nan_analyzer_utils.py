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
from unittest.mock import patch

from msprobe.nan_analyze.utils import (FileCache, is_communication_op, is_ignore_op, check_item_anomaly,
                                       analyze_anomaly_in_group)
from msprobe.nan_analyze.graph import CommunicationNode, DataNode
from test_nan_analyzer import DumpDataBuilder


json_dict = {chr(no): {f'test_{chr(no)}_{i}': [f'content_{j}' for j in range(10)] for i in range(10)} for no in range(ord('a'), ord('z') + 1)}


def mocked_load_json(json_path):
    return json_dict.get(json_path)


class MockedMemory:
    def __init__(self):
        self.available = 100000


def mocked_virtual_memory():
    return MockedMemory()


class TestFileCache(unittest.TestCase):
    def test_load_json(self):
        with patch('msprobe.nan_analyze.utils.load_json', mocked_load_json), \
             patch('psutil.virtual_memory', mocked_virtual_memory):
            cache = FileCache()
            self.assertFalse('a' in cache._cache)
            a = cache.load_json('a')
            self.assertTrue('a' in cache._cache)
            self.assertTrue('test_a_5' in a)

    def test_clean_up(self):
        with patch('msprobe.nan_analyze.utils.load_json', mocked_load_json), \
                patch('psutil.virtual_memory', mocked_virtual_memory):
            cache = FileCache()
            for _ in range(100):
                cache.load_json('a')
            for i, no in enumerate(range(ord('a'), ord('g'))):
                cache.load_json(chr(no))
                self.assertEqual('b' in cache._cache, 0 < i < 3)
            self.assertTrue('a' in cache._cache)

class TestUtils(unittest.TestCase):
    def test_is_communication_op(self):
        self.assertTrue(is_communication_op('Distributed.broadcast.0.forward'))
        self.assertFalse(is_communication_op('Torch.operator.1.forward'))

    def test_is_ignore_op(self):
        self.assertTrue(is_ignore_op('Torch.empty.1.forward'))
        self.assertFalse(is_ignore_op('Torch.operator.1.forward'))

    def test_check_item_anomaly(self):
        self.assertTrue(check_item_anomaly(DumpDataBuilder.gen_data(False, type='compute')['output']))
        self.assertFalse(check_item_anomaly(DumpDataBuilder.gen_data(True, type='compute')['output']))

    def test_analyze_anomaly_in_group(self):
        name = 'broadcast'
        data_type = 'p2g_src'
        src = 0
        dst = 0
        ranks = [0, 1, 2, 3]
        op_name = f'Distributed.{name}.0.forward'
        data = DumpDataBuilder.gen_data(False, name=name, type=data_type, src=src, dst=dst, ranks=ranks,
                                        is_input_normal=True, is_output_normal=False)
        node_id = f'0.{op_name}'
        node = CommunicationNode(node_id, 0, DataNode(op_name, 0, data))
        anomalies = analyze_anomaly_in_group([node])
        self.assertEqual(anomalies[0].op_name, op_name)