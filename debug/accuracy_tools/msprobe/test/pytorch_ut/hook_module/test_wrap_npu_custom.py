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
    logger.info("Failing to import torch_npu.")


class TestNpuOPTemplate(unittest.TestCase):

    def setUp(self):
        self.mock_hook = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock(), None))
        self.template = NpuOPTemplate("test_op", self.mock_hook)

    @patch('torch.ops.npu.test_op')
    @patch('torch_npu._C._VariableFunctionsClass.test_op')
    def test_forward_without_hook(self, mock_variable_func, mock_ops):
        self.template.need_hook = False
        npu_custom_functions["test_op"] = MagicMock(return_value="output_from_custom")
        
        result = self.template.forward(1, 2, key='value')
        self.assertEqual(result, "output_from_custom")
        mock_ops.assert_not_called()
        mock_variable_func.assert_not_called()

    @patch('torch.ops.npu.test_op')
    @patch('torch_npu._C._VariableFunctionsClass.test_op')
    def test_forward_with_unknown_op(self, mock_variable_func, mock_ops):
        self.template.op_name_ = "unknown_op"
        
        with self.assertRaises(Exception) as context:
            self.template.forward(1, 2)
        self.assertEqual(str(context.exception), 'There is not bench function unknown_op')

    @patch('torch.ops.npu.test_op')
    @patch('torch_npu._C._VariableFunctionsClass.test_op')
    def test_forward_with_cuda_device(self, mock_variable_func, mock_ops):
        self.template.device = Const.CUDA_LOWERCASE
        npu_custom_functions["test_op"] = MagicMock(return_value="output_from_cuda")
        
        result = self.template.forward(1, 2)
        self.assertEqual(result, "output_from_cuda")
        mock_ops.assert_called_once_with(1, 2)
        mock_variable_func.assert_not_called()

unittest.main()
