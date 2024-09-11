import unittest
from unittest.mock import patch, MagicMock
import os

import numpy as np
from msprobe.core.common.log import logger
from msprobe.core.data_dump.data_processor.base import ModuleForwardInputsOutputs, ModuleBackwardInputsOutputs, \
    TensorStatInfo, BaseDataProcessor


class TestModuleForwardInputsOutputs(unittest.TestCase):

    @patch('msprobe.core.common.utils.convert_tuple')
    def test_args_tuple(self, mock_convert_tuple):
        mock_convert_tuple.return_value = (1, 2, 3)
        module = ModuleForwardInputsOutputs(args=(1, 2, 3), kwargs=None, output=None)
        self.assertEqual(module.args_tuple, (1, 2, 3))

    @patch('msprobe.core.common.utils.convert_tuple')
    def test_output_tuple(self, mock_convert_tuple):
        mock_convert_tuple.return_value = (4, 5, 6)
        module = ModuleForwardInputsOutputs(args=None, kwargs=None, output=(4, 5, 6))
        self.assertEqual(module.output_tuple, (4, 5, 6))

    def test_concat_args_and_kwargs(self):
        module = ModuleForwardInputsOutputs(args=(1, 2), kwargs={'a': 3, 'b': 4}, output=None)
        self.assertEqual(module.concat_args_and_kwargs(), (1, 2, 3, 4))


class TestModuleBackwardInputsOutputs(unittest.TestCase):

    @patch('msprobe.core.common.utils.convert_tuple')
    def test_grad_input_tuple(self, mock_convert_tuple):
        mock_convert_tuple.return_value = (1, 2, 3)
        module = ModuleBackwardInputsOutputs(grad_output=None, grad_input=(1, 2, 3))
        self.assertEqual(module.grad_input_tuple, (1, 2, 3))

    @patch('msprobe.core.common.utils.convert_tuple')
    def test_grad_output_tuple(self, mock_convert_tuple):
        mock_convert_tuple.return_value = (4, 5, 6)
        module = ModuleBackwardInputsOutputs(grad_output=(4, 5, 6), grad_input=None)
        self.assertEqual(module.grad_output_tuple, (4, 5, 6))


class TestTensorStatInfo(unittest.TestCase):

    def test_tensor_stat_info(self):
        tensor_info = TensorStatInfo(max_val=10, min_val=1, mean_val=5, norm_val=3)
        self.assertEqual(tensor_info.max, 10)
        self.assertEqual(tensor_info.min, 1)
        self.assertEqual(tensor_info.mean, 5)
        self.assertEqual(tensor_info.norm, 3)


class TestBaseDataProcessor(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        self.data_writer = MagicMock()
        self.processor = BaseDataProcessor(self.config, self.data_writer)
        self.data_writer.dump_tensor_data_dir = "./dump_data"
        self.processor.current_api_or_module_name = "test_api"
        self.processor.api_data_category = "input"
    
    @patch('inspect.stack')
    def test_analyze_api_call_stack(self, mock_stack):
        mock_stack.return_value = [
            (None, 'file0.py', 0, 'function0', ['code line 0'], None),
            (None, 'file1.py', 10, 'function1', ['code line 1'], None),
            (None, 'file2.py', 20, 'function2', ['code line 2'], None),
            (None, 'file3.py', 30, 'function3', ['code line 3'], None),
            (None, 'file4.py', 40, 'function4', ['code line 4'], None),
            (None, 'file5.py', 50, 'function5', ['code line 5'], None),
            (None, 'file6.py', 60, 'function6', ['code line 6'], None),
            (None, 'file7.py', 70, 'function7', ['code line 7'], None),
        ]
        result = BaseDataProcessor.analyze_api_call_stack('test_stack')
        expected_output = {
            'test_stack': [
                'File file5.py, line 50, in function5, \n code line 5', 
                'File file6.py, line 60, in function6, \n code line 6', 
                'File file7.py, line 70, in function7, \n code line 7',
            ]
        }
        self.assertEqual(result, expected_output)

    def test_convert_numpy_to_builtin(self):
        self.assertEqual(BaseDataProcessor._convert_numpy_to_builtin(np.int32(5)), (5, 'int32'))
        self.assertEqual(BaseDataProcessor._convert_numpy_to_builtin(np.float64(3.14)), (3.14, 'float64'))
        self.assertEqual(BaseDataProcessor._convert_numpy_to_builtin(np.bool_(True)), (True, 'bool_'))
        self.assertEqual(BaseDataProcessor._convert_numpy_to_builtin(np.str_('test')), ('test', 'str_'))
        self.assertEqual(BaseDataProcessor._convert_numpy_to_builtin(5), (5, ''))

    def test_analyze_builtin(self):
        result = self.processor._analyze_builtin(slice(1, 10, 2))
        expected = {'type': 'slice', 'value': [1, 10, 2]}
        self.assertEqual(result, expected)

        result = self.processor._analyze_builtin(slice(1, np.int64(10), np.int64(2)))
        expected = {'type': 'slice', 'value': [1, 10, 2]}
        self.assertEqual(result, expected)

        result = self.processor._analyze_builtin(...)
        expected = {'type': 'ellipsis', 'value': "..."}
        self.assertEqual(result, expected)

        result = self.processor._analyze_builtin(1)
        expected = {'type': 'int', 'value': 1}
        self.assertEqual(result, expected)

    def test_analyze_numpy(self):
        result = BaseDataProcessor._analyze_numpy(5, 'int32')
        self.assertEqual(result, {'type': 'int32', 'value': 5})

    def test_get_special_types(self):
        self.assertIn(int, BaseDataProcessor.get_special_types())

    def test_recursive_apply_transform(self):
        transform = lambda x, _: x * 2
        self.assertEqual(BaseDataProcessor.recursive_apply_transform(2, transform), 4)
        self.assertEqual(BaseDataProcessor.recursive_apply_transform([1, 2], transform), [2, 4])
        self.assertEqual(BaseDataProcessor.recursive_apply_transform((1, 2), transform), (2, 4))
        self.assertEqual(BaseDataProcessor.recursive_apply_transform({'a': 1}, transform), {'a': 2})

    @patch.object(logger, 'warning')
    def test_recursive_apply_transform_with_warning(self, mock_logger):
        transform = lambda x, _: x * 2
        BaseDataProcessor.recursive_apply_transform({1, 2, 3}, transform)
        mock_logger.assert_called_with(f"Data type {type({1, 2, 3})} is not supported.")

    def test_if_return_forward_new_output(self):
        self.processor._return_forward_new_output = True
        self.assertTrue(self.processor.if_return_forward_new_output())

    def test_get_forward_new_output(self):
        self.processor._return_forward_new_output = True
        self.processor._forward_new_output = "new_output"
        self.assertEqual(self.processor.get_forward_new_output(), "new_output")
        self.assertFalse(self.processor._return_forward_new_output)

    def test_update_iter(self):
        self.processor.update_iter(5)
        self.assertEqual(self.processor.current_iter, 5)

    def test_update_api_or_module_name(self):
        self.processor.update_api_or_module_name("new_api")
        self.assertEqual(self.processor.current_api_or_module_name, "new_api")

    def test_is_dump_for_data_mode(self):
        self.config.data_mode = ["all"]
        self.assertTrue(self.processor.is_dump_for_data_mode("forward", "input"))
        self.config.data_mode = ["forward"]
        self.assertTrue(self.processor.is_dump_for_data_mode("forward", "input"))
        self.config.data_mode = ["input"]
        self.assertTrue(self.processor.is_dump_for_data_mode("forward", "input"))
        self.config.data_mode = ["backward"]
        self.assertFalse(self.processor.is_dump_for_data_mode("forward", "input"))

    @patch.object(BaseDataProcessor, 'analyze_element')
    def test_analyze_forward(self, mock_analyze_element):
        mock_analyze_element.side_effect = lambda args: args
        module_io = ModuleForwardInputsOutputs(args=(1, 2), kwargs={'a': 3}, output=(4, 5))
        self.config.data_mode = ["all"]
        result = self.processor.analyze_forward("test_forward", None, module_io)
        expected = {
            "test_forward": {
                "input_args": (1, 2),
                "input_kwargs": {'a': 3},
                "output": (4, 5)
            }
        }
        self.assertEqual(result, expected)

    @patch.object(BaseDataProcessor, 'analyze_element')
    def test_analyze_pre_forward_inplace(self, mock_analyze_element):
        mock_analyze_element.side_effect = lambda args: args
        module_io = ModuleForwardInputsOutputs(args=(1, 2), kwargs={'a': 3}, output=None)
        self.config.data_mode = ["all"]
        result = self.processor.analyze_pre_forward_inplace("test_pre_forward", module_io)
        expected = {
            "test_pre_forward": {
                "input_args": (1, 2),
                "input_kwargs": {'a': 3}
            }
        }
        self.assertEqual(result, expected)

    @patch.object(BaseDataProcessor, 'analyze_element')
    def test_analyze_forward_inplace(self, mock_analyze_element):
        mock_analyze_element.side_effect = lambda args: args
        module_io = ModuleForwardInputsOutputs(args=(1, 2), kwargs={'a': 3}, output=None)
        self.config.data_mode = ["all"]
        result = self.processor.analyze_forward_inplace("test_forward_inplace", module_io)
        expected = {
            "test_forward_inplace": {
                "output": (1, 2, 3)
            }
        }
        self.assertEqual(result, expected)

    @patch.object(BaseDataProcessor, 'analyze_element')
    def test_analyze_backward(self, mock_analyze_element):
        mock_analyze_element.side_effect = lambda args: args
        module_io = ModuleBackwardInputsOutputs(grad_input=(1, 2), grad_output=(3, 4))
        self.config.data_mode = ["all"]
        result = self.processor.analyze_backward("test_backward", None, module_io)
        expected = {
            "test_backward": {
                "input": (1, 2),
                "output": (3, 4)
            }
        }
        self.assertEqual(result, expected)

    def test_get_save_file_path(self):
        self.config.framework = "pytorch"
        result = self.processor.get_save_file_path("suffix")
        expected_file_name = "test_api.input.suffix.pt"
        expected_file_path = os.path.join(self.data_writer.dump_tensor_data_dir, expected_file_name)
        self.assertEqual(result, (expected_file_name, expected_file_path))
