#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
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

import os.path
import unittest
from unittest.mock import patch
import argparse

from msprobe.nan_analyze.analyzer import _nan_analyze_parser, NanAnalyzer


class DumpDataBuilder:
    def __init__(self):
        self.nodes = {}
        self.layer = {}
        
    @staticmethod    
    def gen_data(is_normal, **kwargs):
        def gen_single_data(normal):
            return {
                'type': 'torch.Tensor',
                'dtype': 'torch.float32',
                'shape': [
                    2,
                    1024
                    ],
                'Max': 2.0 if normal else 'inf',
                'Min': 1.0 if normal else '-inf',
                'Mean': 1.5 if normal else 'nan',
                'Norm': 2.236 if normal else 'nan',
                'requires_grad': False
            }

        def gen_int(value):
            return {
                'type': 'int',
                'value': value
            }

        def gen_process_group(ranks):
            return {
                'type': 'ProcessGroup',
                'group_ranks': ranks
            }

        data_type = kwargs.get('type')
        if data_type == 'compute':
            return {
                'input_args': [gen_single_data(True)],
                'input_kwargs': {},
                'output': [gen_single_data(is_normal)]
            }
        if data_type == 'p2p_src':
            return {
                'input_args': [gen_single_data(kwargs.get('is_input_normal'))],
                'input_kwargs': {'dst': gen_int(kwargs.get('dst'))},
                'output': [gen_single_data(kwargs.get('is_output_normal'))]
            }
        if data_type == 'p2p_dst':
            return {
                'input_args': [gen_single_data(kwargs.get('is_input_normal'))],
                'input_kwargs': {'src': gen_int(kwargs.get('src'))},
                'output': [gen_single_data(kwargs.get('is_output_normal'))]
            }
        if data_type == 'p2g_src':
            return {
                'input_args': [gen_single_data(kwargs.get('is_input_normal')), gen_int(kwargs.get('src'))],
                'input_kwargs': {'group': gen_process_group(kwargs.get('ranks'))},
                'output': [gen_single_data(kwargs.get('is_output_normal'))]
            }
        if data_type == 'p2g_dst':
            return {
                'input_args': [gen_single_data(kwargs.get('is_input_normal')), gen_int(kwargs.get('dst'))],
                'input_kwargs': {'group': gen_process_group(kwargs.get('ranks'))},
                'output': [gen_single_data(kwargs.get('is_output_normal'))]
            }
        if data_type == 'link':
            return {
                'input_args': [gen_single_data(kwargs.get('is_input_normal'))],
                'input_kwargs': {},
                'output': [gen_single_data(kwargs.get('is_output_normal'))]
            }
    
    def add_node(self, is_normal, **kwargs):
        name = kwargs.get("name", 'operator')
        layer = self.layer.get(name, 0)
        if kwargs.get('type') == 'compute':
            node_name = f'Torch.operator.{layer}.forward'
        else:
            node_name = f'Distributed.{name}.{layer}.forward'
        self.nodes[node_name] = self.gen_data(is_normal, **kwargs)
        self.layer[name] = layer + 1
        return self
    
    def build(self):
        return self.nodes


rank_order_dict = {
    # (name, type, src, dst, ranks)
    0: [(0, 'compute', 0, 0, []),
        (0, 'compute', 0, 0, []),
        (0, 'compute', 0, 0, []),
        (0, 'compute', 0, 0, []),
        (0, 'compute', 0, 0, []),
        ('send', 'p2p_src', 0, 1, []),
        ('recv', 'p2p_dst', 1, 0, []),
        (0, 'compute', 0, 0, []),
        ('broadcast', 'p2g_src', 0, 0, [0, 1, 2, 3]),
        (0, 'compute', 0, 0, []),
        ('all_gather', 'link', 0, 0, []),
        (0, 'compute', 0, 0, []),
        ('gather', 'p2g_dst', 0, 0, [0, 1, 2, 3]),
        (0, 'compute', 0, 0, [])],
    1: [('recv', 'p2p_dst', 0, 0, []),
        (0, 'compute', 0, 0, []),
        (0, 'compute', 0, 0, []),
        ('send', 'p2p_src', 0, 2, []),
        ('recv', 'p2p_dst', 2, 0, []),
        (0, 'compute', 0, 0, []),
        (0, 'compute', 0, 0, []),
        ('send', 'p2p_src', 0, 0, []),
        ('broadcast', 'p2g_src', 0, 0, [0, 1, 2, 3]),
        (0, 'compute', 0, 0, []),
        ('all_gather', 'link', 0, 0, []),
        (0, 'compute', 0, 0, []),
        ('gather', 'p2g_dst', 0, 0, [0, 1, 2, 3]),
        (0, 'compute', 0, 0, [])],
    2: [('recv', 'p2p_dst', 1, 0, []),
        (0, 'compute', 0, 0, []),
        (0, 'compute', 0, 0, []),
        ('send', 'p2p_src', 0, 3, []),
        ('recv', 'p2p_dst', 3, 0, []),
        (0, 'compute', 0, 0, []),
        (0, 'compute', 0, 0, []),
        ('send', 'p2p_src', 0, 1, []),
        ('broadcast', 'p2g_src', 0, 0, [0, 1, 2, 3]),
        (0, 'compute', 0, 0, []),
        ('all_gather', 'link', 0, 0, []),
        (0, 'compute', 0, 0, []),
        ('gather', 'p2g_dst', 0, 0, [0, 1, 2, 3]),
        (0, 'compute', 0, 0, [])],
    3: [('recv', 'p2p_dst', 2, 0, []),
        (0, 'compute', 0, 0, []),
        (0, 'compute', 0, 0, []),
        (0, 'compute', 0, 0, []),
        ('send', 'p2p_src', 0, 2, []),
        ('broadcast', 'p2g_src', 0, 0, [0, 1, 2, 3]),
        (0, 'compute', 0, 0, []),
        ('all_gather', 'link', 0, 0, []),
        (0, 'compute', 0, 0, []),
        ('gather', 'p2g_dst', 0, 0, [0, 1, 2, 3]),
        (0, 'compute', 0, 0, []),
        (0, 'compute', 0, 0, []),
        (0, 'compute', 0, 0, []),
        (0, 'compute', 0, 0, [])]
}


def do_nothing(*args, **kwargs):
    return


def gen_normal_dump_json(rank):
    builder = DumpDataBuilder()
    for name, data_type, src, dst, ranks in rank_order_dict[rank]:
        builder = builder.add_node(True, name=name, type=data_type, src=src, dst=dst, ranks=ranks,
                                   is_input_normal=True, is_output_normal=True)
    return {'task': 'statistics',
            'level': 'mix',
            'dump_data_dir': None,
            'data': builder.build()
           }


def gen_pre_anomaly_dump_json(rank):
    builder = DumpDataBuilder()
    for i, (name, data_type, src, dst, ranks) in enumerate(rank_order_dict[rank]):
        is_normal = True
        if i == rank and i in [0, 1]:
            is_normal = False
        builder = builder.add_node(is_normal, name=name, type=data_type, src=src, dst=dst, ranks=ranks,
                                   is_input_normal=True, is_output_normal=True)
    return {'task': 'statistics',
            'level': 'mix',
            'dump_data_dir': None,
            'data': builder.build()
            }


def gen_anomaly_dump_json(rank):
    builder = DumpDataBuilder()
    start = 999
    for i, (name, data_type, src, dst, ranks) in enumerate(rank_order_dict[rank]):
        is_normal = True
        is_input_normal = True
        is_output_normal = True
        if rank == 0:
            if i == 7:
                is_normal = False
            elif i == 8:
                is_input_normal = False
                is_output_normal = False
        else:
            if name == 'broadcast':
                start = i
                is_output_normal = False
            elif i > start:
                is_normal = False
                is_input_normal = False
                is_output_normal = False
        builder = builder.add_node(is_normal, name=name, type=data_type, src=src, dst=dst, ranks=ranks,
                                   is_input_normal=is_input_normal, is_output_normal=is_output_normal)
    return {'task': 'statistics',
            'level': 'mix',
            'dump_data_dir': None,
            'data': builder.build()
            }


def gen_after_anomaly_dump_json(rank):
    builder = DumpDataBuilder()
    for i, (name, data_type, src, dst, ranks) in enumerate(rank_order_dict[rank]):
        is_normal = (rank != 2 or i != 13) and (rank != 3 or i != 11)
        builder = builder.add_node(is_normal, name=name, type=data_type, src=src, dst=dst, ranks=ranks,
                                   is_input_normal=True, is_output_normal=True)
    return {'task': 'statistics',
            'level': 'mix',
            'dump_data_dir': None,
            'data': builder.build()
            }


json_dict = {os.path.join('./step0', f'rank{i if i > 0 else ""}', 'construct.json'): {} for i in range(4)}


def gen_stack_json(rank):
    return {f'0': [list(json_dict[os.path.join('./step0', f'rank{rank if rank > 0 else ""}', 'dump.json')]['data'].keys()),
                   ['File /root/example.py, line 10, in test_fcn, \\n test(tensor)']]}


class mock_time:
    _uni_value = 1

    @staticmethod
    def set_uni_value(var):
        mock_time._uni_value = var

    @staticmethod
    def time_ns():
        return mock_time._uni_value



class MockedFileCache:
    def load_json(self, file_path):
        return json_dict[file_path]


class TestAnalyzer(unittest.TestCase):
    def setUp(self):
        self.output = {}
        self.input_path = './step0'
        self.output_path = './output'
        with patch('os.listdir', return_value=['rank', 'rank1', 'rank2', 'rank3', 'rank_others']), \
            patch('msprobe.nan_analyze.utils.check_file_or_directory_path', do_nothing), \
            patch('msprobe.nan_analyze.analyzer.FileCache', MockedFileCache):
            self.analyzer = NanAnalyzer(self.input_path, self.output_path)

    def mocked_save_json(self, file, content, indent):
        self.output[file] = content

    def test_nan_analyze_parser(self):
        args = [
            '-i', '/path/to/input',
            '-o', '/path/to/output',
        ]

        parser = argparse.ArgumentParser()
        _nan_analyze_parser(parser)
        parsed_args = parser.parse_args(args)
        self.assertEqual(parsed_args.input_path, '/path/to/input')
        self.assertEqual(parsed_args.output_path, '/path/to/output')

    def test_normal(self):
        json_dict.update({os.path.join('./step0', f'rank{i if i > 0 else ""}', 'dump.json'): gen_normal_dump_json(i) for i in range(4)})
        json_dict.update({os.path.join('./step0', f'rank{i if i > 0 else ""}', 'stack.json'): gen_stack_json(i) for i in range(4)})
        with patch('os.path.exists', return_value=True), \
             patch('msprobe.nan_analyze.analyzer.logger.info', print), \
             patch('msprobe.nan_analyze.analyzer.logger.warning', print), \
             patch('msprobe.nan_analyze.analyzer.FileCache', MockedFileCache):
            self.analyzer.analyze()
            self.assertFalse(bool(self.output))

    def test_pre_anomaly(self):
        json_dict.update({os.path.join('./step0', f'rank{i if i > 0 else ""}', 'dump.json'): gen_pre_anomaly_dump_json(i) for i in range(4)})
        json_dict.update({os.path.join('./step0', f'rank{i if i > 0 else ""}', 'stack.json'): gen_stack_json(i) for i in range(4)})
        with patch('os.path.exists', return_value=True), \
             patch('msprobe.nan_analyze.analyzer.save_json', self.mocked_save_json), \
             patch('msprobe.nan_analyze.analyzer.logger.info', print), \
             patch('msprobe.nan_analyze.analyzer.logger.warning', print), \
             patch('msprobe.nan_analyze.analyzer.FileCache', MockedFileCache), \
             patch('msprobe.nan_analyze.graph.FileCache', MockedFileCache), \
             patch('msprobe.nan_analyze.analyzer.time', mock_time):
            mock_time.set_uni_value(1)
            self.analyzer.analyze()
            res_json = self.output.get(os.path.join('./output', 'anomaly_analyze_1.json'))
            self.assertTrue(bool(res_json))
            self.assertEqual('Torch.operator.0.forward', res_json['rank_0'][0]['op_name'])

    def test_anomaly(self):
        json_dict.update({os.path.join('./step0', f'rank{i if i > 0 else ""}', 'dump.json'): gen_anomaly_dump_json(i) for i in range(4)})
        json_dict.update({os.path.join('./step0', f'rank{i if i > 0 else ""}', 'stack.json'): gen_stack_json(i) for i in range(4)})
        with patch('os.path.exists', return_value=True), \
             patch('msprobe.nan_analyze.analyzer.save_json', self.mocked_save_json), \
             patch('msprobe.nan_analyze.analyzer.logger.info', print), \
             patch('msprobe.nan_analyze.analyzer.logger.warning', print), \
             patch('msprobe.nan_analyze.analyzer.FileCache', MockedFileCache), \
             patch('msprobe.nan_analyze.graph.FileCache', MockedFileCache), \
             patch('msprobe.nan_analyze.analyzer.time', mock_time):
            mock_time.set_uni_value(2)
            self.analyzer.analyze()
            res_json = self.output.get(os.path.join('./output', 'anomaly_analyze_2.json'))
            self.assertTrue(bool(res_json))
            self.assertEqual('Torch.operator.5.forward', res_json['rank_0'][0]['op_name'])

    def test_after_anomaly(self):
        json_dict.update({os.path.join('./step0', f'rank{i if i > 0 else ""}', 'dump.json'): gen_after_anomaly_dump_json(i) for i in range(4)})
        json_dict.update({os.path.join('./step0', f'rank{i if i > 0 else ""}', 'stack.json'): gen_stack_json(i) for i in range(4)})
        with patch('os.path.exists', return_value=True), \
             patch('msprobe.nan_analyze.analyzer.save_json', self.mocked_save_json), \
             patch('msprobe.nan_analyze.analyzer.logger.info', print), \
             patch('msprobe.nan_analyze.analyzer.logger.warning', print), \
             patch('msprobe.nan_analyze.analyzer.FileCache', MockedFileCache), \
             patch('msprobe.nan_analyze.graph.FileCache', MockedFileCache), \
             patch('msprobe.nan_analyze.analyzer.time', mock_time):
            mock_time.set_uni_value(3)
            self.analyzer.analyze()
            res_json = self.output.get(os.path.join('./output', 'anomaly_analyze_3.json'))
            self.assertTrue(bool(res_json))
            self.assertEqual(res_json['rank_2'][0]['op_name'], 'Torch.operator.6.forward')
            self.assertEqual(res_json['rank_3'][0]['op_name'], 'Torch.operator.6.forward')


