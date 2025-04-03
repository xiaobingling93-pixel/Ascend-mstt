import unittest
from unittest.mock import patch, MagicMock

import torch
import numpy as np

from msprobe.core.common.utils import CompareException
from msprobe.core.common.file_utils import FileCheckConst
from msprobe.pytorch.compare.utils import read_pt_data


class TestReadPtData(unittest.TestCase):

    @patch('msprobe.pytorch.compare.utils.load_pt')
    @patch('msprobe.pytorch.compare.utils.FileChecker')
    @patch('os.path.join', return_value='/fake/path/to/file.pt')
    def test_read_pt_data(self, mock_os, mock_file_checker, mock_load_pt):
        mock_file_checker.return_value.common_check.return_value = '/fake/path/to/file.pt'

        mock_tensor = MagicMock()
        mock_tensor.detach.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor
        mock_tensor.dtype = torch.bfloat16
        mock_tensor.numpy.return_value = np.array([1.0, 2.0, 3.0])
        mock_load_pt.return_value = mock_tensor

        result = read_pt_data('/fake/dir', 'file_name.pt')

        mock_file_checker.assert_called_once_with('/fake/path/to/file.pt', FileCheckConst.FILE, FileCheckConst.READ_ABLE, FileCheckConst.PT_SUFFIX, False)
        mock_load_pt.assert_called_once_with('/fake/path/to/file.pt', to_cpu=True)
        mock_tensor.to.assert_called_once_with(torch.float32)
        self.assertTrue(np.array_equal(result, np.array([1.0, 2.0, 3.0])))

    @patch('os.path.join', return_value='/fake/path/to/file.pt')
    @patch('msprobe.pytorch.compare.utils.FileChecker')
    @patch('msprobe.pytorch.compare.utils.load_pt')
    def test_read_real_data_pt_exception(self, mock_load_pt, mock_file_checker, mock_os):
        mock_file_checker.return_value.common_check.return_value = '/fake/path/to/file.pt'

        mock_load_pt.side_effect = RuntimeError("Test Error")

        with self.assertRaises(CompareException):
            read_pt_data('/fake/dir', 'file_name.pt')
