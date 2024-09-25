import unittest
from unittest.mock import MagicMock, patch
# import torch

from msprobe.core.common.const import Const
from msprobe.core.common.log import logger
from msprobe.pytorch.function_factory import npu_custom_functions
from msprobe.pytorch.hook_module.wrap_npu_custom import NpuOPTemplate


try:
    import torch_npu
except ImportError:
    logger.warning("Failing to import torch_npu.")


class TestNpuOPTemplate(unittest.TestCase):

    def setUp(self):
        # Create an instance of NpuOPTemplate with a mock hook
        self.mock_hook = MagicMock()
        self.template = NpuOPTemplate("test_op", self.mock_hook)

    @patch('torch.ops.npu.test_op')
    @patch('torch_npu._C._VariableFunctionsClass.test_op')
    def test_forward_with_hook(self, mock_variable_func, mock_ops):
        # Setup mocks
        mock_ops.return_value = "output_from_ops"
        mock_variable_func.return_value = "output_from_variable_funcs"

        # Test with default device (NPU)
        result = self.template.forward(1, 2, key='value')
        self.assertEqual(result, "output_from_variable_funcs")
        mock_variable_func.assert_called_once_with(1, 2, key='value')

    def test_forward_without_hook(self):
        # Test when need_hook is False
        self.template.need_hook = False
        self.template.op_name_ = "test_op"
        npu_custom_functions["test_op"] = MagicMock(return_value="custom_output")

        result = self.template.forward(1, 2, key='value')
        self.assertEqual(result, "custom_output")
        npu_custom_functions["test_op"].assert_called_once_with(1, 2, key='value')

    def test_forward_invalid_op(self):
        # Test when op_name_ is not in npu_custom_functions
        self.template.need_hook = False
        self.template.op_name_ = "invalid_op"

        with self.assertRaises(Exception) as context:
            self.template.forward(1, 2)
        self.assertEqual(str(context.exception), 'There is not bench function invalid_op')

    @patch('cuda_func_mapping.get')
    def test_forward_cuda_device(self, mock_cuda_mapping):
        # Test CUDA device case
        self.template.need_hook = False
        self.template.device = Const.CUDA_LOWERCASE
        self.template.op_name_ = "test_op"
        mock_cuda_mapping.return_value = "mapped_op"
        npu_custom_functions["mapped_op"] = MagicMock(return_value="cuda_output")

        result = self.template.forward(1, 2)
        self.assertEqual(result, "cuda_output")
        npu_custom_functions["mapped_op"].assert_called_once_with(1, 2)
