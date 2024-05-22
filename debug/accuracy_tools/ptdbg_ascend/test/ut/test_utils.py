import unittest
import torch
import pytest
import ptdbg_ascend.common.utils as utils

from ptdbg_ascend.common.utils import CompareException, get_md5_for_tensor
from ptdbg_ascend.common.file_check_util import FileCheckException


class TestUtilsMethods(unittest.TestCase):
    def test_check_file_or_directory_path_1(self):
        file = "list"
        with pytest.raises(FileCheckException) as error:
            utils.check_file_or_directory_path(file)
        self.assertEqual(error.value.code, FileCheckException.INVALID_PATH_ERROR)

    def test_check_file_or_directory_path_2(self):
        file = "/list/dir"
        with pytest.raises(FileCheckException) as error:
            utils.check_file_or_directory_path(file)
        self.assertEqual(error.value.code, FileCheckException.INVALID_PATH_ERROR)

    def test_check_file_size_1(self):
        file = "/list/dir"
        with pytest.raises(CompareException) as error:
            utils.check_file_size(file, 100)
        self.assertEqual(error.value.code, CompareException.INVALID_FILE_ERROR)

    def test_check_file_size_2(self):
        file = "../run_ut.py"
        with pytest.raises(CompareException) as error:
            utils.check_file_size(file, 0)
        self.assertEqual(error.value.code, CompareException.INVALID_FILE_ERROR)


    def test_get_md5_for_tensor(self):
        data = [[1, 2], [3, 4]]
        x_data = torch.tensor(data)
        md5_value = get_md5_for_tensor(x_data)
        self.assertEqual(md5_value, '9c692600')
