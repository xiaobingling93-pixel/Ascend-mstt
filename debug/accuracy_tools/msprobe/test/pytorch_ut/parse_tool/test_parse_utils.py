import unittest
import os
import shutil
from unittest.mock import patch
from pathlib import Path
from collections import namedtuple
from rich.panel import Panel

import numpy as np

from msprobe.pytorch.parse_tool.lib.utils import Util
from msprobe.pytorch.parse_tool.lib.parse_exception import ParseException
from msprobe.pytorch.parse_tool.lib.config import Const


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.util = Util()
        self.base_dir = './base_dir'
        os.makedirs(self.base_dir, exist_ok=True)
        for i in range(3):
            os.makedirs(os.path.join(self.base_dir, f'subdir{i}'), exist_ok=True)
            (Path(self.base_dir) / f'file{i}.txt').touch()
        self.npy_file_dir = './npy_file_dir'
        os.makedirs(self.npy_file_dir, exist_ok=True)
        for i in range(3):
            (Path(self.npy_file_dir) / f'file{i}.npy').touch()
        self.empty_dir = './empty_dir'
        os.makedirs(self.empty_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)
        if os.path.exists(self.npy_file_dir):
            shutil.rmtree(self.npy_file_dir)
        if os.path.exists(self.empty_dir):
            shutil.rmtree(self.empty_dir)

    def test_path_strip(self):
        path = "\'\"./path\"\'"
        res = self.util.path_strip(path)

        self.assertEqual(res, './path')

    @patch('msprobe.pytorch.parse_tool.lib.utils.check_path_owner_consistent', return_value=None)
    @patch('msprobe.pytorch.parse_tool.lib.utils.check_other_user_writable', return_value=None)
    @patch('msprobe.pytorch.parse_tool.lib.utils.check_path_executable', return_value=None)
    def test_check_executable_file(self, mock_check_path_owner_consistent, mock_check_other_user_writable, mock_check_path_executable):
        path = './test_file'
        self.util.check_executable_file(path)

        mock_check_path_owner_consistent.assert_called_once()
        mock_check_other_user_writable.assert_called_once()
        mock_check_path_executable.assert_called_once()

    def test_get_subdir_count(self):
        res = self.util.get_subdir_count(self.base_dir)

        self.assertEqual(res, 3)

    def test_get_subfiles_count(self):
        res = self.util.get_subfiles_count(self.base_dir)

        self.assertEqual(res, 3)

    def test_get_sorted_subdirectories_names(self):
        res = self.util.get_sorted_subdirectories_names(self.base_dir)

        self.assertEqual(res, ['subdir0', 'subdir1', 'subdir2'])

    def test_get_sorted_files_name(self):
        res = self.util.get_sorted_files_names(self.base_dir)

        self.assertEqual(res, ['file0.txt', 'file1.txt', 'file2.txt'])

    def test_check_npy_files_valid_in_dir(self):
        res = self.util.check_npy_files_valid_in_dir(self.npy_file_dir)

        self.assertTrue(res)

    def test_check_npy_files_valid_in_dir_false(self):
        (Path(self.npy_file_dir) / f'file4.pt').touch()
        res = self.util.check_npy_files_valid_in_dir(self.npy_file_dir)

        self.assertFalse(res)        

    def test_get_md5_for_numpy(self):
        obj = np.array([1, 2, 3, 4, 5])
        res = self.util.get_md5_for_numpy(obj)

        self.assertEqual(res, 'baa24928')

    def test_deal_with_dir_or_file_inconsistency(self):
        with self.assertRaises(ParseException):
            path = './inconsistency'
            self.util.deal_with_dir_or_file_inconsistency(path)

    def test_deal_with_value_if_has_zero(self):
        data = np.array([1, 0, 0, 1, 0, 1], dtype=np.half)
        res = self.util.deal_with_value_if_has_zero(data)

        self.assertTrue(np.all(res != 0))

    def test_dir_contains_only(self):
        res1 = self.util.dir_contains_only(self.base_dir, 'npy')
        res2 = self.util.dir_contains_only(self.npy_file_dir, 'npy')

        self.assertFalse(res1)
        self.assertTrue(res2)

    def test_change_filemode_safe(self):
        test_path = './test/path'
        res = self.util.change_filemode_safe(test_path)

        self.assertIsNone(res)

    def test_execute_command(self):
        res = self.util.execute_command('pwd')

        self.assertEqual(res, 0)

    def test_execute_command_error(self):
        res = self.util.execute_command(None)
        
        self.assertEqual(res, -1)

    @patch('msprobe.pytorch.parse_tool.lib.utils.Panel')
    def test_print_panel_none(self, mock_panel):
        mock_panel.return_value = None
        res = self.util.print_panel('test content')

        self.assertIsNone(res)

    @patch('msprobe.pytorch.parse_tool.lib.utils.Panel')
    def test_print_panel_with_fit(self, mock_panel):
        self.util.print_panel('test content')

        mock_panel.fit.assert_called_once_with('test content', title='')

    @patch('msprobe.pytorch.parse_tool.lib.utils.Util.print')
    def test_print_panel_with_none_fit(self, mock_print):
        self.util.print_panel('test content', fit=False)

        mock_print.assert_called_once()

    @patch('msprobe.pytorch.parse_tool.lib.utils.subprocess.run')
    def test_check_msaccucmp_fail(self, mock_run):
        mock_run.returncode.return_value = 1
        
        with self.assertRaises(ParseException):
            self.util.check_msaccucmp('./msaccucmp.py')

    def test_check_msaccucmp_with_wrong_file(self):
        with self.assertRaises(ParseException):
            self.util.check_msaccucmp('./aerfaew')

    @patch('msprobe.pytorch.parse_tool.lib.utils.Util.npy_info')
    def test_gen_npy_info_txt(self, mock_npu_info):
        mock_npu_info.return_value = (1, 1, 1, 1, 1)
        source_data = 1
        res = self.util.gen_npy_info_txt(source_data)

        self.assertEqual(res, '[Shape: 1] [Dtype: 1] [Max: 1] [Min: 1] [Mean: 1]')

    @patch('msprobe.pytorch.parse_tool.lib.utils.Util.list_file_with_pattern')
    def test_list_convert_files(self, mock_list_file_with_pattern):
        self.util.list_convert_files('./mypath')

        mock_list_file_with_pattern.assert_called_once()

    @patch('msprobe.pytorch.parse_tool.lib.utils.Util.list_file_with_pattern')
    def test_list_numpy_files(self, mock_list_file_with_pattern):
        self.util.list_numpy_files('./mypath')

        mock_list_file_with_pattern.assert_called_once()

    def test_check_path_valid(self):
        with patch('msprobe.pytorch.parse_tool.lib.utils.check_file_or_directory_path', return_value=None):
            res = self.util.check_path_valid(self.base_dir)

            self.assertTrue(res)

    def test_check_path_valid_fail(self):
        with self.assertRaises(ParseException):
            self.util.check_path_valid('non_existent_path')

    def test_check_files_in_path(self):
        with self.assertRaises(ParseException):
            self.util.check_files_in_path(self.empty_dir)

    def test_npy_info(self):
        var = np.array([1, 2, 3, 4, 5])
        res = self.util.npy_info(var)
        npu_info_res = namedtuple('npu_info_res', ['shape', 'dtype', 'max', 'min', 'mean'])

        self.assertEqual(res, npu_info_res(shape=(5,), dtype=np.int64, max=5, min=1, mean=3))

    def test_npy_info_fail_with_none_nparray(self):
        with self.assertRaises(ParseException):
            self.util.npy_info(1)

    def test_npy_info_fail_with_none_object(self):
        with self.assertRaises(ParseException):
            var = np.array([1, 2, 3, 4, 5], dtype=object)
            self.util.npy_info(var)
            
    def test_npy_info_fail_with_size_0(self):
        with self.assertRaises(ParseException):
            var = np.empty((0,))
            self.util.npy_info(var)

    def test_list_file_with_pattern(self):
        with patch('msprobe.pytorch.parse_tool.lib.utils.Util.check_path_valid', return_value=True), \
            patch('msprobe.pytorch.parse_tool.lib.utils.check_file_or_directory_path', return_value=None):
            res = self.util.list_file_with_pattern(self.npy_file_dir, Const.NUMPY_PATTERN, '', self.util._gen_numpy_file_info)
            
            self.assertEqual(len(res), 3)

    def test_check_file_path_format_with_dir(self):
        with self.assertRaises(ParseException):
            self.util.check_file_path_format(self.base_dir, Const.PKL_SUFFIX)

    def test_check_file_path_format_with_file(self):
        with self.assertRaises(ParseException):
            self.util.check_file_path_format(self.npy_file_dir + '.file1.npy', Const.PKL_SUFFIX)

    def test_check_str_param(self):
        with self.assertRaises(ParseException):
            param = 'a' * 256
            self.util.check_str_param(param)

    def test_check_str_param(self):
        with self.assertRaises(ParseException):
            self.util.check_str_param('faworf9 823*(A#&./)')

    def test_is_subdir_count_equal(self):
        self.assertFalse(self.util.is_subdir_count_equal(self.base_dir, self.npy_file_dir))

    def test_check_positive(self):
        with self.assertRaises(ParseException):
            self.util.check_positive(-1)

if __name__ == '__main__':
    unittest.main()
