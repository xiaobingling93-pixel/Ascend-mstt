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

import unittest
import os
from unittest.mock import patch

from msprobe.nan_analyze.graph import CommunicationNode, DataNode
from msprobe.nan_analyze.utils import RankPath

from msprobe.core.common.exceptions import MsprobeException
from test_nan_analyzer import DumpDataBuilder, gen_normal_dump_json, MockedFileCache, json_dict, gen_stack_json, do_nothing


dump_json = {i: gen_normal_dump_json(i) for i in range(4)}


class TestCommunicationNode(unittest.TestCase):
    def test_add_next(self):
        op_name_0 = 'Distributed.send.0.forward'
        op_name_1 = 'Distributed.recv.0.forward'
        comm_node_0 = CommunicationNode(f'0.{op_name_0}', 0, DataNode(op_name_0, 0, dump_json[0]['data'][op_name_0]))
        comm_node_1 = CommunicationNode(f'0.{op_name_1}', 0, DataNode(op_name_1, 0, dump_json[0]['data'][op_name_1]))
        comm_node_0.add_next(comm_node_1)
        self.assertEqual(comm_node_0.layer + 1, comm_node_1.layer)
        self.assertTrue(comm_node_0 is comm_node_1.pre_node)
        self.assertTrue(comm_node_1.node_id in comm_node_0.next_nodes)

    def test_add_link(self):
        op_name = 'Distributed.all_gather.0.forward'
        comm_node_0 = CommunicationNode(f'0.{op_name}', 0, DataNode(op_name, 0, dump_json[0]['data'][op_name]))
        comm_node_1 = CommunicationNode(f'1.{op_name}', 1, DataNode(op_name, 1, dump_json[1]['data'][op_name]))
        comm_node_0.add_link(comm_node_1)
        self.assertEqual(comm_node_0.layer, comm_node_1.layer)
        self.assertTrue(comm_node_0.node_id in comm_node_1.link_nodes)
        self.assertTrue(comm_node_1.node_id in comm_node_0.link_nodes)

    def test_add_dst(self):
        op_name = 'Distributed.broadcast.0.forward'
        comm_node_0 = CommunicationNode(f'0.{op_name}', 0, DataNode(op_name, 0, dump_json[0]['data'][op_name]))
        comm_node_1 = CommunicationNode(f'2.{op_name}', 2, DataNode(op_name, 2, dump_json[2]['data'][op_name]))
        comm_node_0.add_dst(comm_node_1)
        self.assertEqual(comm_node_0.layer, comm_node_1.layer)
        self.assertTrue(comm_node_0.node_id in comm_node_1.src_nodes)
        self.assertTrue(comm_node_1.node_id in comm_node_0.dst_nodes)

    def test_delete(self):
        op_name = 'Distributed.broadcast.0.forward'
        comm_node_0 = CommunicationNode(f'0.{op_name}', 0, DataNode(op_name, 0, dump_json[0]['data'][op_name]))
        comm_node_1 = CommunicationNode(f'2.{op_name}', 2, DataNode(op_name, 2, dump_json[2]['data'][op_name]))
        op_name = 'Distributed.recv.0.forward'
        comm_node_2 = CommunicationNode(f'0.{op_name}', 0, DataNode(op_name, 0, dump_json[0]['data'][op_name]))
        comm_node_2.add_next(comm_node_0)
        comm_node_0.add_dst(comm_node_1)
        comm_node_0.delete()
        self.assertFalse(comm_node_1.src_nodes)
        self.assertFalse(comm_node_2.next_nodes)

    def test_has_nan_inf(self):
        op_name = 'Distributed.broadcast.0.forward'
        comm_node_0 = CommunicationNode(f'0.{op_name}', 0, DataNode(op_name, 0, dump_json[0]['data'][op_name]))
        self.assertFalse(comm_node_0.has_nan_inf())

    def test_input_has_nan_inf(self):
        op_name = 'Distributed.broadcast.0.forward'
        comm_node_0 = CommunicationNode(f'0.{op_name}', 0, DataNode(op_name, 0, dump_json[0]['data'][op_name]))
        self.assertFalse(comm_node_0.input_has_nan_inf())

    def test_find_connected_nodes(self):
        op_name = 'Distributed.broadcast.0.forward'
        comm_node_0 = CommunicationNode(f'0.{op_name}', 0, DataNode(op_name, 0, dump_json[0]['data'][op_name]))
        comm_node_1 = CommunicationNode(f'1.{op_name}', 1, DataNode(op_name, 1, dump_json[1]['data'][op_name]))
        comm_node_2 = CommunicationNode(f'2.{op_name}', 2, DataNode(op_name, 2, dump_json[2]['data'][op_name]))
        comm_node_3 = CommunicationNode(f'3.{op_name}', 3, DataNode(op_name, 3, dump_json[3]['data'][op_name]))
        comm_node_0.add_dst(comm_node_1)
        comm_node_0.add_dst(comm_node_2)
        comm_node_0.add_dst(comm_node_3)
        conn_info = comm_node_0.find_connected_nodes()
        self.assertEqual(conn_info['ranks'], {0, 1, 2, 3})
        self.assertEqual(conn_info['api'], 'Distributed.broadcast')
        self.assertEqual(conn_info['type'], 'dst')

    def test_resolve_type(self):
        op_name = 'Distributed.broadcast.0.forward'
        comm_node_0 = CommunicationNode(f'0.{op_name}', 0, DataNode(op_name, 0, dump_json[0]['data'][op_name]))
        comm_node_1 = CommunicationNode(f'1.{op_name}', 1, DataNode(op_name, 1, dump_json[1]['data'][op_name]))
        self.assertEqual(comm_node_0.type, 'src')
        self.assertEqual(comm_node_1.type, 'dst')

        op_name = 'Distributed.all_gather.0.forward'
        comm_node_2 = CommunicationNode(f'0.{op_name}', 0, DataNode(op_name, 0, dump_json[0]['data'][op_name]))
        self.assertEqual(comm_node_2.type, 'link')


class TestDataNode(unittest.TestCase):
    def setUp(self):
        json_dict.update({os.path.join('./step0', f'rank{i if i > 0 else ""}', 'dump.json'): gen_normal_dump_json(i) for i in range(4)})
        json_dict.update({os.path.join('./step0', f'rank{i if i > 0 else ""}', 'stack.json'): gen_stack_json(i) for i in range(4)})
        json_dict[os.path.join('./step0', 'rank', 'construct.json')] = {
            'Torch.operator.1.forward': 'Module.module.test_model.forward.0',
            'Module.module.test_model.forward.0': 'Module.module.parent_model.forward.0',
            'Module.module.parent_model.forward.0': 'Module.module.root_model.forward.0',
            'Module.module.root_model.forward.0': None
        }

    def test_find_stack(self):
        with patch('msprobe.nan_analyze.graph.FileCache', MockedFileCache):
            op_name = 'Torch.operator.1.forward'
            data_node = DataNode(op_name, 0, dump_json[0]['data'][op_name])
            stack_info = data_node.find_stack(json_dict[os.path.join('./step0', 'rank', 'stack.json')])
            self.assertEqual(stack_info[0], 'File /root/example.py, line 10, in test_fcn, \\n test(tensor)')
            with self.assertRaises(MsprobeException) as context:
                data_node.find_stack({op_name: 'blabla'})
                self.assertEqual(context.exception.code, 4)

    def test_find_complete_construct(self):
        with patch('msprobe.nan_analyze.graph.FileCache', MockedFileCache):
            op_name = 'Torch.operator.1.forward'
            construct = DataNode.find_complete_construct(json_dict[os.path.join('./step0', 'rank', 'construct.json')],
                                                         op_name)
            self.assertEqual(len(construct), 4)
            self.assertEqual(construct[0], 'Module.module.root_model.forward.0')

    def test_is_anomaly(self):
        data_node_0 = DataNode('Torch.operator.1.forward', 0, DumpDataBuilder.gen_data(False, type='compute'))
        data_node_1 = DataNode('Torch.operator.1.forward', 0, DumpDataBuilder.gen_data(True, type='compute'))
        self.assertTrue(data_node_0.is_anomaly())
        self.assertFalse(data_node_1.is_anomaly())

    def test_gen_node_info(self):
        with patch('msprobe.nan_analyze.graph.FileCache', MockedFileCache), \
             patch('msprobe.nan_analyze.utils.check_file_or_directory_path', do_nothing):
            op_name = 'Torch.operator.1.forward'
            data_node = DataNode(op_name, 0, dump_json[0]['data'][op_name])
            node_info = data_node.gen_node_info(RankPath(0, os.path.join('./step0', 'rank', 'dump.json'),
                                                          os.path.join('./step0', 'rank', 'construct.json'),
                                                          os.path.join('./step0', 'rank', 'stack.json')))
            data_info = node_info['data_info']
            self.assertEqual(data_info['input_args'][0]['Max'], 2.0)
            stack_info = node_info['stack_info']
            self.assertEqual(stack_info[0], 'File /root/example.py, line 10, in test_fcn, \\n test(tensor)')