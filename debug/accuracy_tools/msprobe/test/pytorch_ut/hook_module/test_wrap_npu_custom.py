import unittest
from unittest.mock import MagicMock, patch

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
        self.template = NpuOPTemplate("sum", self.mock_hook)

    def test_init(self):
        self.assertEqual(self.template.op_name_, "sum")
        self.assertEqual(self.template.prefix_op_name_, f"NPU{Const.SEP}sum{Const.SEP}")
        self.assertTrue(self.template.need_hook)
        self.assertEqual(self.template.device, Const.CPU_LOWERCASE)

    @patch('torch.ops.npu.sum')
    def test_forward_without_hook(self, mock_npu_sum):
        self.template.need_hook = False
        npu_custom_functions["sum"] = MagicMock(return_value="output_from_custom")

        result = self.template.forward(1, 2, key='value')
        self.assertEqual(result, "output_from_custom")
        mock_npu_sum.assert_not_called()

    @patch('torch.ops.npu.sum')
    def test_forward_with_hook(self, mock_npu_sum):
        self.template.need_hook = True
        mock_npu_sum.return_value = "output_from_npu"

        result = self.template.forward(1, 2, key='value')
        self.assertEqual(result, "output_from_npu")
        mock_npu_sum.assert_called_once_with(1, 2, key='value')
