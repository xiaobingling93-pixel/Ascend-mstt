# import unittest
# from unittest.mock import MagicMock, patch
# # import torch

# from msprobe.core.common.const import Const
# from msprobe.core.common.log import logger
# from msprobe.pytorch.function_factory import npu_custom_functions
# from msprobe.pytorch.hook_module.wrap_npu_custom import NpuOPTemplate


# try:
#     import torch_npu
# except ImportError:
#     logger.warning("Failing to import torch_npu.")


# class TestNpuOPTemplate(unittest.TestCase):

#     def setUp(self):
#         # Create an instance of NpuOPTemplate with a mock hook
#         self.mock_hook = MagicMock()
#         self.template = NpuOPTemplate("test_op", self.mock_hook)

#     @patch('torch.ops.npu.test_op')
#     @patch('torch_npu._C._VariableFunctionsClass.test_op')
#     def test_forward_with_hook(self, mock_variable_func, mock_ops):
#         # Setup mocks
#         mock_ops.return_value = "output_from_ops"
#         mock_variable_func.return_value = "output_from_variable_funcs"

#         # Test with default device (NPU)
#         result = self.template.forward(1, 2, key='value')
#         self.assertEqual(result, "output_from_variable_funcs")
#         mock_variable_func.assert_called_once_with(1, 2, key='value')



# unittest.main()
