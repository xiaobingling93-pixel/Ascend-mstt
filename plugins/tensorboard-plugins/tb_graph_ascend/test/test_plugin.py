# Copyright (c) 2025, Huawei Technologies.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import sys
import os
sys.path.append(os.path.pardir)
from unittest.mock import MagicMock, patch
import unittest
from server.plugin import GraphsPlugin


class TestGraphsPlugin(unittest.TestCase):
    
    @property
    def plugin(self):
        """提供对受保护实例属性 _plugin 的只读访问"""
        return self._plugin

    @property
    def instance(self):
        """提供对受保护实例属性 _instance 的只读访问"""
        return self._instance
    
    def setUp(self):
        """在每个测试之前初始化环境"""
        fake_context = MagicMock()
        
        # 创建 GraphsPlugin 实例并传递 context
        self.plugin = GraphsPlugin(context=fake_context) 
        self.plugin._current_file_path = ""  # 初始化文件路径
        self.plugin.batch_id = '-1'  # 设置为 -1 来触发 _process_subnode 中的判断逻辑
        self.plugin.step_id = '-1'   # 设置为 -1 来触发 _process_subnode 中的判断逻辑

        self.plugin._current_file_data = {
            "npu": {
                "node": {
                    "npu_node_1": {
                        "matched_node_link": []
                    }
                }
            },
            "bench": {
                "node": {
                    "bench_node_1": {
                        "matched_node_link": []
                    }
                }
            },
            "match": [],
            'task': 'md5'
        }
        
        self.app = Flask(__name__)
        self.app.debug = True
        self.client = self.app.test_client()

        # 创建模拟的 data_provider
        mock_data_provider = MagicMock()
        mock_data_provider.some_method.return_value = "some_value"  # 根据需要设置模拟的方法

        # 创建一个模拟的 context，并将 mock_data_provider 赋值给它
        context = MagicMock()
        context.data_provider = mock_data_provider

        # 使用 context 创建 GraphsPlugin 实例
        self._instance = GraphsPlugin(context=context)

    def test_get_all_node_name_with_valid_batch_and_step(self):
        # 模拟 request.args
        mock_request = MagicMock()
        mock_request.args.get.return_value = '0'  # 模拟 batch=0 和 step=0
        
        # 构造 json_data
        json_data = {
            'npu': {
                'root': 'root_node',
                'node': {
                    'root_node': {
                        'micro_step_id': 0,
                        'subnodes': ['subnode1', 'subnode2']
                    },
                    'subnode1': {'micro_step_id': 0},
                    'subnode2': {'micro_step_id': 0},
                }
            },
            'bench': {
                'node': {
                    'bench1': {},
                    'bench2': {}
                }
            }
        }

        # 调用 get_all_nodeName 方法
        npu_ids, bench_ids = self.plugin.get_all_node_names(json_data, mock_request)

        # 验证返回的 npu_ids 和 bench_ids
        self.assertEqual(npu_ids, ['subnode1', 'subnode2'])
        self.assertEqual(bench_ids, ['bench1', 'bench2'])

    def test_dfs_collect_nodes_with_valid_batch_and_step(self):
        # 模拟 request.args
        mock_request = MagicMock()
        side_effect_dict = {
            'batch': '0',
            'step': '0'
        }

        # 设置 mock_request.args.get 的 side_effect
        mock_request.args.get.side_effect = side_effect_dict.get
        
        # 构造 json_data
        json_data = {
            'npu': {
                'node': {
                    'node1': {'micro_step_id': 0, 'step_id': 0, 'subnodes': []},
                    'node2': {'micro_step_id': 0, 'step_id': 0, 'subnodes': []},
                    'node3': {'micro_step_id': 1, 'step_id': 1, 'subnodes': []},
                    'node4': {'micro_step_id': 0, 'step_id': 1, 'subnodes': ['subnode1']}
                }
            }
        }

        # 调用 dfs_collect_nodes 方法
        all_node_names = self.plugin.dfs_collect_nodes(json_data, mock_request)

        # 验证返回的 all_node_names
        self.assertEqual(all_node_names, ['node1', 'node2'])

    def test_group_precision_set_with_even_number_of_elements(self):
        # 测试正常的输入（偶数个元素）
        precision_set = [1, 2, 3, 4, 5, 6]
        expected_result = [[1, 2], [3, 4], [5, 6]]
        
        # 调用 group_precision_set 方法
        result = self.plugin.group_precision_set(precision_set)
        
        # 验证结果是否正确
        self.assertEqual(result, expected_result)

    def test_group_precision_set_with_odd_number_of_elements(self):
        # 测试输入长度为奇数的情况
        precision_set = [1, 2, 3]
        
        # 验证是否抛出 ValueError 异常
        with self.assertRaises(ValueError) as context:
            self.plugin.group_precision_set(precision_set)
        
        self.assertEqual(str(context.exception), 'The number of elements in precision_set is not even')

    def test_process_data_with_md5_mismatch(self):
        # 测试 md5 不匹配的情况
        data = []
        data_set = {
            'npu_keys': ['npu_key_1'],
            'bench_keys': ['bench_key_1'],
            'input_data': {'npu_key_1': [1, 2, 3], 'md5': 'abcd'},
            'output_data': {'bench_key_1': [1, 2, 3], 'md5': 'efgh'},
            'precision_index': 0,
            'file_path': 'test_path',
            'data_type': 'test_type',
        }
        npu_node = 'test_node'

        # 调用方法
        result = self.plugin.process_data(data, data_set, npu_node)

        # 验证结果
        self.assertEqual(result, 0)
        self.assertEqual(data_set['precision_index'], 0)

    def test_should_update_node_with_valid_batch_and_step(self):
        # 模拟 json_data 和 subnode_id_data
        subnode_id_data = {'micro_step_id': '-1', 'step_id': '-1', 'matched_node_link': ['N___subnode_1']}
        subgraph = {'node': {}}
        json_data = {
            'StepList': ['0', '1', '2']  # 测试 StepList 数据
        }
        
        prefix = 'N___'
        subnode_id = 'subnode_1'
        
        # 调用 _process_subnode 方法
        self.plugin._process_subnode(subgraph, prefix, subnode_id, subnode_id_data, json_data)
        
        # 验证 subnode_id 是否更新
        self.assertIn(prefix + subnode_id, subgraph['node'])
        self.assertEqual(subgraph['node'][prefix + subnode_id], subnode_id_data)

    def mock_json_get(self, *args):
        """ 模拟 json_get 方法，返回不同层级的数据 """
        if len(args) == 4 and args[1] == "node":
            # 返回节点的 matched_node_link 数据
            return self.plugin._current_file_data[args[0]][args[1]].get(args[2], {}).get('matched_node_link', [])
        return None

    def test_should_update_node_with_invalid_batch_or_step(self):
        # 测试 batch_id 和 step_id 为无效值时不会更新
        self.plugin.batch_id = '-1'
        self.plugin.step_id = '-1'
        
        subnode_id_data = {'micro_step_id': '1', 'step_id': '1', 'matched_node_link': []}
        subgraph = {'node': {}}
        json_data = {
            'StepList': ['0', '1', '2']
        }
        
        prefix = 'B___'
        subnode_id = 'subnode_1'
        
        # 调用 _process_subnode 方法
        self.plugin._process_subnode(subgraph, prefix, subnode_id, subnode_id_data, json_data)
        
        # 验证 subnode_id 是否被更新
        self.assertIn(prefix + subnode_id, subgraph['node'])
        self.assertEqual(subgraph['node'][prefix + subnode_id], subnode_id_data)
    
    def test_update_matched_node_links(self):
        subnode_id_data = {
            'matched_node_link': ['link_1', 'link_2']
        }
        prefix = 'B___'
        
        # 调用 _update_matched_node_links 方法
        self.plugin._update_matched_node_links(subnode_id_data, prefix)
        
        # 验证 matched_node_link 是否被正确更新
        self.assertEqual(subnode_id_data['matched_node_link'], ['N___link_1', 'N___link_2'])
    
    def test_no_update_matched_node_links(self):
        subnode_id_data = {
            'matched_node_link': ['link_1', 'link_2']
        }
        prefix = 'N___'
        
        # 模拟常量 SETS
        constants = MagicMock()
        constants.SETS = {
            'bench': ('bench', 'B___', 'N___'),
            'npu': ('npu', 'N___', 'B___'),
            'B___': ('bench', 'N___'),
            'N___': ('npu', 'B___')
        }
        
        # 不更新第一个 matched_node_link
        subnode_id_data['matched_node_link'][0] = 'prefixlink_1'
        
        # 调用 _update_matched_node_links 方法
        self.plugin._update_matched_node_links(subnode_id_data, prefix)
        
        # 验证 linked node 是否正确更新
        self.assertEqual(subnode_id_data['matched_node_link'], ['B___prefixlink_1', 'B___link_2'])
    
    @patch('os.walk')  # 模拟 os.walk
    def test_get_run_dirs(self, mock_os_walk):
        """测试 _get_run_dirs 方法"""
        
        # 设置模拟返回的文件夹和文件
        fake_logdir = os.path.join(os.getcwd(), "fake", "logdir")  # 使用绝对路径
        mock_os_walk.return_value = [(fake_logdir, [], ["run1_tag1.vis", "run2_tag2.vis"])]
        
        # 设置文件大小返回值
        with patch('os.path.getsize', return_value=500):  # 模拟文件小于限制
            run_tag_pairs = self.plugin._get_run_dirs()
        
        # 验证返回的 run_tag_pairs
        # 使用 os.path.normpath 来确保路径在不同操作系统上被标准化
        expected_run_tag_pairs = [
            (os.path.normpath(os.path.join(fake_logdir)), 'run1_tag1'),
            (os.path.normpath(os.path.join(fake_logdir)), 'run2_tag2')
        ]
        
        self.assertEqual(run_tag_pairs, expected_run_tag_pairs)

    @patch('os.path.getsize')  # 模拟 os.path.getsize
    def test_get_run_dirs_with_large_file(self, mock_getsize):
        """测试 _get_run_dirs 方法，当文件超过大小限制时"""
        
        # 模拟一个文件大于最大限制
        mock_getsize.return_value = 2000 * 1024 * 1024  # 文件超过 1GB
        
        # 使用 os.path.join 来构建路径，确保兼容 Windows 和 Linux
        fake_logdir = os.path.join("fake", "logdir")
        large_file = "large_file.vis"
        
        with patch('os.walk', return_value=[(fake_logdir, [], [large_file])]):
            run_tag_pairs = self.plugin._get_run_dirs()
        
        # 验证文件被跳过，不会返回任何文件
        self.assertEqual(run_tag_pairs, [])  # 文件被跳过，不会返回任何文件

    def test_convert_to_protobuf_format(self):
        """测试 _convert_to_protobuf_format 方法"""
        # 模拟节点数据
        subgraph = {
            'node': {
                'npu_node_1': {
                    'id': 'op_1',
                    'node_type': 1,
                    'matched_node_link': ['bench_node_1'],
                    'data': {
                        'precision_index': 10,
                        'other_data': 'value'
                    },
                    'input_data': {},
                    'output_data': {},
                    'suggestions': {},
                    'subnodes': [],
                    'stack_info': 'stack_1'
                }
            }
        }

        # 调用方法
        protobuf_format = self.plugin._convert_to_protobuf_format(subgraph)

        # 验证 protobuf 格式是否正确
        self.assertIn('node {', protobuf_format)
        self.assertIn('name: "npu_node_1"', protobuf_format)
        self.assertIn('op: "op_1"', protobuf_format)
        self.assertIn('precision_index: 10', protobuf_format)
        self.assertIn('isLeaf: true', protobuf_format)

    @patch('json.load')  # 模拟 json.load
    def test_read_json_file_invalid(self, mock_json_load):
        """测试 _read_json_file 方法，当文件无效时"""
        # 设置模拟的文件路径
        mock_file_path = os.path.join("fake", "file.vis")  # 使用 os.path.join 来构造路径

        # 模拟 json.load 抛出异常
        mock_json_load.side_effect = Exception("Invalid JSON")
        
        # 使用模拟的路径读取文件
        result = self.plugin._read_json_file(mock_file_path)
        
        # 验证返回值是 None，并且日志中有错误消息
        self.assertIsNone(result)

    @patch('os.path.exists', return_value=False)
    def test_load_json_file_not_found(self, mock_exists):
        """测试 _load_json_file 方法，当文件不存在时"""
        
        # 使用 os.path.join 来确保路径的兼容性
        mock_file_path = os.path.join("fake", "file.vis")
        mock_tag = "tag1"
        
        # 调用方法
        result = self.plugin._load_json_file(mock_file_path, mock_tag)
        
        # 验证返回值是否为 None
        self.assertIsNone(result)
        
        # 验证 _current_file_path 是否为空
        self.assertEqual(self.plugin._current_file_path, "")

    def test_get_input_output_data(self):
        npu_node_data = {
            'input_data': {'input_1': 'data1', 'input_2': 'data2'},
            'output_data': {'output_1': 'data3'}
        }
        bench_node_data = {
            'input_data': {'input_1': 'dataA', 'input_2': 'dataB'},
            'output_data': {'output_1': 'dataC', 'output_2': 'dataD'}
        }
        
        # 调用方法
        (
            npu_input_data, 
            bench_input_data, 
            npu_output_data, 
            bench_output_data
        ) = self.instance.get_input_output_data(npu_node_data, bench_node_data)

        # 验证返回结果
        self.assertEqual(npu_input_data, {'input_1': 'data1', 'input_2': 'data2'})
        self.assertEqual(bench_input_data, {'input_1': 'dataA', 'input_2': 'dataB'})
        self.assertEqual(npu_output_data, {'output_1': 'data3'})
        self.assertEqual(bench_output_data, {'output_1': 'dataC', 'output_2': 'dataD'})

    def test_calculate_min_length(self):
        # 模拟输入输出数据
        npu_input_data = {'input_1': 'data1', 'input_2': 'data2'}
        bench_input_data = {'input_1': 'dataA', 'input_2': 'dataB', 'input_3': 'dataC'}
        npu_output_data = {'output_1': 'data3'}
        bench_output_data = {'output_1': 'dataC', 'output_2': 'dataD'}

        # 调用方法
        input_min_length, output_min_length = self.instance.calculate_min_length(
            npu_input_data, 
            bench_input_data, 
            npu_output_data, 
            bench_output_data
        )

        # 验证返回值
        self.assertEqual(input_min_length, 2)  # 最小输入数据长度
        self.assertEqual(output_min_length, 1)  # 最小输出数据长度

    def test_get_keys(self):
        # 模拟 npu 和 bench 数据
        npu_data = {
            'input_data': {'input_1': 'data1', 'input_2': 'data2'},
            'output_data': {'output_1': 'data3'}
        }
        bench_data = {
            'input_data': {'input_1': 'dataA', 'input_2': 'dataB'},
            'output_data': {'output_1': 'dataC', 'output_2': 'dataD'}
        }

        # 调用方法
        npu_keys, bench_keys = self.instance.get_keys(npu_data, bench_data)

        # 验证返回值
        self.assertEqual(npu_keys, ['input_data', 'output_data'])
        self.assertEqual(bench_keys, ['input_data', 'output_data'])
    
    def test_calculate_diff_and_relative_error(self):
        # 模拟 npu_data 和 bench_data
        npu_data = {
            'Max': 10.0,
            'Min': 1.0,
            'Mean': 5.5,
            'Norm': 8.0
        }
        bench_data = {
            'Max': 12.0,
            'Min': 2.0,
            'Mean': 5.0,
            'Norm': 7.0
        }
        
        # 调用方法
        results = self.instance.calculate_diff_and_relative_error(npu_data, bench_data)
        
        # 验证返回结果
        self.assertEqual(results['npu_Max'], 10.0)
        self.assertEqual(results['bench_Max'], 12.0)
        self.assertEqual(results['Max_relative_err'], "16.666667%")

        self.assertEqual(results['npu_Min'], 1.0)
        self.assertEqual(results['bench_Min'], 2.0)
        self.assertEqual(results['Min_relative_err'], "50.000000%")

        self.assertEqual(results['npu_Mean'], 5.5)
        self.assertEqual(results['bench_Mean'], 5.0)
        self.assertEqual(results['Mean_relative_err'], "10.000000%")

        self.assertEqual(results['npu_Norm'], 8.0)
        self.assertEqual(results['bench_Norm'], 7.0)
        self.assertEqual(results['Norm_relative_err'], "14.285714%")

    def test_calculate_diff_and_relative_error_with_zero_bench_value(self):
        # 模拟 npu_data 和 bench_data，其中 bench_data 有零值
        npu_data = {
            'Max': 10.0,
            'Min': 1.0,
            'Mean': 5.5,
            'Norm': 8.0
        }
        bench_data = {
            'Max': 12.0,
            'Min': 0.0,  # bench Min 为 0，应该触发 "N/A" 处理
            'Mean': 5.0,
            'Norm': 0.0  # bench Norm 为 0，应该触发 "N/A" 处理
        }
        
        # 调用方法
        results = self.instance.calculate_diff_and_relative_error(npu_data, bench_data)
        
        # 验证返回结果
        self.assertEqual(results['Min_relative_err'], "N/A")
        self.assertEqual(results['Norm_relative_err'], "N/A")

if __name__ == '__main__':
    unittest.main()