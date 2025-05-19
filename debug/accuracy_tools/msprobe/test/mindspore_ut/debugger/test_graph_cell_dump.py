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
from msprobe.mindspore.dump.cell_dump_process import generate_file_path
from msprobe.mindspore.dump.cell_dump_process import partial_func, clip_gradient
from msprobe.mindspore.dump.cell_dump_process import cell_construct_wrapper
from msprobe.mindspore.dump.cell_dump_process import rename_filename, sort_filenames, del_same_file
from msprobe.mindspore.dump.cell_dump_process import check_relation


class TestGenerateFilePath(unittest.TestCase):
    def setUp(self):
        self.dump_path = "/path"
        self.cell_prefix = "Cell.network._backbone.LlamaForCausalLM"
        self.suffix = "forward"
        self.io_type = "input"
        self.index = 0

    def test_generate_file_path(self):
        expected_path = os.path.join(
            self.dump_path,
            "{step}",
            "{rank}",
            CoreConst.DUMP_TENSOR_DATA,
            CoreConst.SEP.join([self.cell_prefix, self.suffix, self.io_type, str(self.index)])
        )
        result = generate_file_path(self.dump_path, self.cell_prefix, self.suffix, self.io_type, self.index)
        self.assertEqual(result, expected_path)


class TestPartialFunc(unittest.TestCase):

    @patch('msprobe.mindspore.dump.cell_dump_process.CoreConst')
    @patch('msprobe.mindspore.dump.cell_dump_process.td')
    @patch('msprobe.mindspore.dump.cell_dump_process.td_in')
    @patch('msprobe.mindspore.dump.cell_dump_process.generate_file_path')
    @patch('msprobe.mindspore.dump.cell_dump_process.ops.depend')
    def test_clip_gradient_output(self, mock_depend, mock_generate_file_path, mock_td_in, mock_td, mock_CoreConst):
        mock_CoreConst.OUTPUT = "output"
        mock_CoreConst.BACKWARD = "backward"
        mock_generate_file_path.return_value = "mock_path"
        mock_td.return_value = "temp_tensor"
        mock_depend.return_value = "dependent_tensor"

        result = clip_gradient("dump_path", "cell_prefix", 0, "output", "dx")

        mock_generate_file_path.assert_called_with("dump_path", "cell_prefix", "backward", "output", 0)
        mock_td.assert_called_with("mock_path", "dx")
        mock_depend.assert_called_with("dx", "temp_tensor")
        self.assertEqual(result, "dependent_tensor")

    @patch('msprobe.mindspore.dump.cell_dump_process.CoreConst')
    @patch('msprobe.mindspore.dump.cell_dump_process.td')
    @patch('msprobe.mindspore.dump.cell_dump_process.td_in')
    @patch('msprobe.mindspore.dump.cell_dump_process.generate_file_path')
    @patch('msprobe.mindspore.dump.cell_dump_process.ops.depend')
    def test_clip_gradient_input(self, mock_depend, mock_generate_file_path, mock_td_in, mock_td, mock_CoreConst):
        mock_CoreConst.INPUT = "input"
        mock_CoreConst.BACKWARD = "backward"
        mock_generate_file_path.return_value = "mock_path"
        mock_td_in.return_value = "temp_tensor"
        mock_depend.return_value = "dependent_tensor"

        result = clip_gradient("dump_path", "cell_prefix", 0, "input", "dx")

        mock_generate_file_path.assert_called_with("dump_path", "cell_prefix", "backward", "input", 0)
        mock_td_in.assert_called_with("mock_path", "dx")
        mock_depend.assert_called_with("dx", "temp_tensor")
        self.assertEqual(result, "dependent_tensor")

    def test_partial_func(self):
        def mock_func(dump_path, cell_prefix, index, io_type, *args, **kwargs):
            return dump_path, cell_prefix, index, io_type, args, kwargs

        new_func = partial_func(mock_func, "dump_path", "cell_prefix", 0, "io_type")
        result = new_func("arg1", "arg2", kwarg1="value1")

        self.assertEqual(result, ("dump_path", "cell_prefix", 0, "io_type", ("arg1", "arg2"), {'kwarg1': 'value1'}))


class TestCellWrapperProcess(unittest.TestCase):

    @patch('msprobe.mindspore.dump.cell_dump_process.generate_file_path')
    @patch('msprobe.mindspore.dump.cell_dump_process.td')
    @patch('msprobe.mindspore.dump.cell_dump_process.td_in')
    def test_cell_construct_wrapper(self, mock_td_in, mock_td, mock_generate_file_path):
        # Mock the generate_file_path function
        mock_generate_file_path.return_value = "mock_path"

        # Mock the TensorDump operations
        mock_td.return_value = MagicMock()
        mock_td_in.return_value = MagicMock()

        # Create a mock cell with necessary attributes
        mock_cell = MagicMock()
        mock_cell.data_mode = "all"
        mock_cell.dump_path = "mock_dump_path"
        mock_cell.cell_prefix = "mock_cell_prefix"
        mock_cell.input_clips = [MagicMock() for _ in range(50)]
        mock_cell.output_clips = [MagicMock() for _ in range(50)]

        # Define a mock function to wrap
        def mock_func(*args, **kwargs):
            return args

        # Wrap the mock function using cell_construct_wrapper
        wrapped_func = cell_construct_wrapper(mock_func, mock_cell)

        # Create mock inputs
        mock_input = ms.Tensor([1, 2, 3])
        mock_args = (mock_input,)

        # Call the wrapped function
        result = wrapped_func(mock_cell, *mock_args)

        # Check if the result is as expected
        self.assertEqual(result, mock_args)

        # Verify that the TensorDump operations were called
        mock_td_in.assert_called()
        mock_td.assert_called()

    @patch('msprobe.mindspore.dump.cell_dump_process.generate_file_path')
    @patch('msprobe.mindspore.dump.cell_dump_process.td')
    @patch('msprobe.mindspore.dump.cell_dump_process.td_in')
    def test_cell_construct_wrapper_with_tuple_output(self, mock_td_in, mock_td, mock_generate_file_path):
        # Mock the generate_file_path function
        mock_generate_file_path.return_value = "mock_path"

        # Mock the TensorDump operations
        mock_td.return_value = MagicMock()
        mock_td_in.return_value = MagicMock()

        # Create a mock cell with necessary attributes
        mock_cell = MagicMock()
        mock_cell.data_mode = "all"
        mock_cell.dump_path = "mock_dump_path"
        mock_cell.cell_prefix = "mock_cell_prefix"
        mock_cell.input_clips = [MagicMock() for _ in range(50)]
        mock_cell.output_clips = [MagicMock() for _ in range(50)]

        # Define a mock function to wrap
        def mock_func(*args, **kwargs):
            return (args[0], args[0])

        # Wrap the mock function using cell_construct_wrapper
        wrapped_func = cell_construct_wrapper(mock_func, mock_cell)

        # Create mock inputs
        mock_input = ms.Tensor([1, 2, 3])
        mock_args = (mock_input,)

        # Call the wrapped function
        result = wrapped_func(mock_cell, *mock_args)

        # Check if the result is as expected
        self.assertEqual(result, (mock_input, mock_input))

        # Verify that the TensorDump operations were called
        mock_td_in.assert_called()
        mock_td.assert_called()


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


class TestRenameFilename(unittest.TestCase):

    @patch('msprobe.mindspore.dump.cell_dump_process.sort_filenames')
    @patch('msprobe.mindspore.dump.cell_dump_process.del_same_file')
    @patch('msprobe.mindspore.dump.cell_dump_process.os.rename')
    def test_rename_filename(self, mock_rename, mock_del_same_file, mock_sort_filenames):
        # Mock the constants
        CoreConst.REPLACEMENT_CHARACTER = '_'
        CoreConst.FORWARD_PATTERN = '.forward.'
        CoreConst.BACKWARD_PATTERN = '.backward.'
        CoreConst.SEP = '.'

        # Mock the filenames
        mock_sort_filenames.return_value = [
            "Cell.learning_rate.CosineWithWarmUpLR.forward.input.0_int32_101.npy",
            "Cell.learning_rate.CosineWithWarmUpLR.forward.output.0_float32_102.npy",
            "Cell.loss_scaling_manager.DynamicLossScaleUpdateCell.backward.input.0_float32_103.npy",
            "Cell.loss_scaling_manager.DynamicLossScaleUpdateCell.backward.input.1_bool_104.npy",
            "Cell.loss_scaling_manager.DynamicLossScaleUpdateCell.backward.output.1_bool_105.npy",
            "Cell.learning_rate.CosineWithWarmUpLR.forward.input.0_int32_111.npy",
            "Cell.learning_rate.CosineWithWarmUpLR.forward.output.0_float32_112.npy",
        ]
        mock_del_same_file.return_value = [mock_sort_filenames.return_value]

        # Call the function
        rename_filename('/mock/path')

        # Check if os.rename was called with the correct arguments
        mock_rename.assert_any_call(
            '/mock/path/Cell.learning_rate.CosineWithWarmUpLR.forward.input.0_int32_101.npy',
            '/mock/path/Cell_learning_rate_CosineWithWarmUpLR.forward.0.input_0_int32_101.npy'
        )
        mock_rename.assert_any_call(
            '/mock/path/Cell.learning_rate.CosineWithWarmUpLR.forward.output.0_float32_102.npy',
            '/mock/path/Cell_learning_rate_CosineWithWarmUpLR.forward.0.output_0_float32_102.npy'
        )
        mock_rename.assert_any_call(
            '/mock/path/Cell.loss_scaling_manager.DynamicLossScaleUpdateCell.backward.input.0_float32_103.npy',
            '/mock/path/Cell_loss_scaling_manager_DynamicLossScaleUpdateCell.backward.0.input_0_float32_103.npy'
        )
        mock_rename.assert_any_call(
            '/mock/path/Cell.loss_scaling_manager.DynamicLossScaleUpdateCell.backward.input.1_bool_104.npy',
            '/mock/path/Cell_loss_scaling_manager_DynamicLossScaleUpdateCell.backward.0.input_1_bool_104.npy'
        )
        mock_rename.assert_any_call(
            '/mock/path/Cell.loss_scaling_manager.DynamicLossScaleUpdateCell.backward.output.1_bool_105.npy',
            '/mock/path/Cell_loss_scaling_manager_DynamicLossScaleUpdateCell.backward.0.output_1_bool_105.npy'
        )
        mock_rename.assert_any_call(
            '/mock/path/Cell.learning_rate.CosineWithWarmUpLR.forward.input.0_int32_111.npy',
            '/mock/path/Cell_learning_rate_CosineWithWarmUpLR.forward.1.input_0_int32_111.npy'
        )
        mock_rename.assert_any_call(
            '/mock/path/Cell.learning_rate.CosineWithWarmUpLR.forward.output.0_float32_112.npy',
            '/mock/path/Cell_learning_rate_CosineWithWarmUpLR.forward.1.output_0_float32_112.npy'
        )

        # Mock the filenames
        mock_sort_filenames.return_value = []
        mock_del_same_file.return_value = []

        # Call the function
        rename_filename('/mock/path')

        # Check if os.rename was not called
        mock_rename.assert_not_called()


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

    def test_no_layer_pattern_relation(self):
        self.assertFalse(check_relation("network.model.layers.0", "network.loss"))
        self.assertFalse(check_relation("network._backbone.model.layers.1", "network._backbone.model.layers"))

    def test_edge_cases(self):
        self.assertFalse(check_relation("", "network"))
        self.assertFalse(check_relation("network.layer1", ""))
        self.assertFalse(check_relation("", ""))