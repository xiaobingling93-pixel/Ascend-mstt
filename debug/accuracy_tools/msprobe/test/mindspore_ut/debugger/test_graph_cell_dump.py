# Copyright (c) 2025-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
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
import os
import re
import unittest
from unittest.mock import MagicMock, patch

import mindspore as ms
from mindspore import ops

from msprobe.core.common.const import Const as CoreConst
from msprobe.mindspore.dump.cell_dump_process import cell_construct_wrapper
from msprobe.mindspore.dump.cell_dump_process import rename_filename, sort_filenames, del_same_file
from msprobe.mindspore.dump.cell_dump_process import check_relation


class TestCellWrapperProcess(unittest.TestCase):

    @patch('msprobe.mindspore.dump.cell_dump_process.ops.is_tensor')
    @patch('msprobe.mindspore.dump.cell_dump_process.td')
    @patch('msprobe.mindspore.dump.cell_dump_process.td_in')
    def test_cell_construct_wrapper(self, mock_td_in, mock_td, mock_istensor):

        # Mock the TensorDump operations
        mock_td.return_value = MagicMock()
        mock_td_in.return_value = MagicMock()
        mock_istensor.return_value = False

        # Create a mock cell with necessary attributes
        mock_cell = MagicMock()
        mock_cell.data_mode = "all"
        mock_cell.dump_path = "mock_dump_path"
        mock_cell.cell_prefix = "mock_cell_prefix"

        # Define a mock function to wrap
        def mock_func(*args, **kwargs):
            return args

        # Wrap the mock function using cell_construct_wrapper
        wrapped_func = cell_construct_wrapper(mock_func, mock_cell)

        # Create mock inputs
        mock_input = ms.Tensor([1, 2, 3])
        mock_args = (mock_input,)

        # Call the wrapped function
        wrapped_func(mock_cell, *mock_args)

        # Verify that the TensorDump operations were not called
        mock_td_in.assert_not_called()
        mock_td.assert_not_called()


class TestSortFilenames(unittest.TestCase):

    @patch('os.listdir')
    def test_sort_filenames(self, mock_listdir):
        # Mock the list of filenames returned by os.listdir
        mock_listdir.return_value = [
            'Cell.network._backbone.model.LlamaModel.backward.0.input.0_float16_177.npy',
            'Cell.network._backbone.model.LlamaModel.forward.0.input.0_in_int32_1.npy',
            'Cell.network._backbone.model.LlamaModel.forward.0.output.10_float16_165.npy',
            'Cell.network._backbone.model.norm_out.LlamaRMSNorm.backward.0.input.0_float16_178.npy'
        ]

        # Mock the CoreConst values
        CoreConst.REPLACEMENT_CHARACTER = '_'
        CoreConst.NUMPY_SUFFIX = '.npy'

        # Expected sorted filenames
        expected_sorted_filenames = [
            'Cell.network._backbone.model.LlamaModel.forward.0.input.0_in_int32_1.npy',
            'Cell.network._backbone.model.LlamaModel.forward.0.output.10_float16_165.npy',
            'Cell.network._backbone.model.LlamaModel.backward.0.input.0_float16_177.npy',
            'Cell.network._backbone.model.norm_out.LlamaRMSNorm.backward.0.input.0_float16_178.npy'
        ]

        # Call the function
        sorted_filenames = sort_filenames('/mock/path')

        # Assert the filenames are sorted correctly
        self.assertEqual(sorted_filenames, expected_sorted_filenames)


class TestCheckRelation(unittest.TestCase):

    def setUp(self):
        CoreConst.SEP = '.'
        global KEY_LAYERS
        KEY_LAYERS = "layers"

    def test_direct_parent_child_relation(self):
        self.assertTrue(check_relation("network._backbone", "network"))
        self.assertTrue(check_relation("network._backbone.model", "network._backbone"))

    def test_no_relation(self):
        self.assertFalse(check_relation("network._backbone", "network.loss"))
        self.assertFalse(check_relation("network._backbone.model", "network.loss"))

    def test_layer_pattern_relation(self):
        self.assertTrue(check_relation("network.model.layers.0", "network.model"))
        self.assertTrue(check_relation("network._backbone.model.layers.1", "network._backbone.model"))

    def test_edge_cases(self):
        self.assertFalse(check_relation("", "network"))
        self.assertFalse(check_relation("network.layer1", ""))
        self.assertFalse(check_relation("", ""))