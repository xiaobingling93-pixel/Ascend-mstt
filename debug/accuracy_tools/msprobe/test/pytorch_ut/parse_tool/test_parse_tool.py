import unittest
from unittest.mock import patch
import argparse

from msprobe.pytorch.parse_tool.lib.parse_tool import ParseTool


class TestParseTool(unittest.TestCase):
    def setUp(self):
        self.parse_tool = ParseTool()
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "-m", "--my_dump_path", dest="my_dump_path", default=None,
            help="<Required> my dump path, the data compared with golden data",
            required=True
        )
        self.parser.add_argument(
            "-g", "--golden_dump_path", dest="golden_dump_path", default=None,
            help="<Required> the golden dump path",
            required=True
        )
        self.parser.add_argument(
            "-out", "--output_path", dest="output_path", default=None,
            help="<Optional> the output path",
            required=False
        )
        self.parser.add_argument(
            "-cmp_path", "--msaccucmp_path", dest="msaccucmp_path", default=None,
            help="<Optional> the msaccucmp.py file path",
            required=False
        )

    @patch('msprobe.pytorch.parse_tool.lib.parse_tool.create_directory')
    def test_prepare(self, mock_create_directory):
        self.parse_tool.prepare()

        mock_create_directory.assert_called_once()

    @patch('msprobe.pytorch.parse_tool.lib.parse_tool.Compare.npu_vs_npu_compare', return_value=None)
    def test_do_vector_compare(self, mock_npu_vs_npu_compare):
        with patch('msprobe.pytorch.parse_tool.lib.parse_tool.Util.check_path_valid'), \
            patch('msprobe.pytorch.parse_tool.lib.parse_tool.Util.check_executable_file'), \
            patch('msprobe.pytorch.parse_tool.lib.parse_tool.os.path.isdir', return_value=True):
            args = self.parser.parse_args(['-m', 'my_dump_path', '-g', 'golden_dump_path', '-out', 'output_path', '-cmp_path', 'msaccucmp_path'])
            self.parse_tool.do_vector_compare(args)

            mock_npu_vs_npu_compare.assert_called_once()

    @patch('msprobe.pytorch.parse_tool.lib.parse_tool.Compare.convert_dump_to_npy', return_value=None)
    def test_do_convert_dump(self, mock_convert_dump_to_npy):
        with patch('msprobe.pytorch.parse_tool.lib.parse_tool.Util.check_path_valid'), \
            patch('msprobe.pytorch.parse_tool.lib.parse_tool.Util.check_executable_file'), \
            patch('msprobe.pytorch.parse_tool.lib.parse_tool.Util.check_files_in_path'):
            self.parse_tool.do_convert_dump(['-n', 'file_name/file_path', '-out', 'output_path'])

            mock_convert_dump_to_npy.assert_called_once()

    @patch('msprobe.pytorch.parse_tool.lib.parse_tool.Visualization.print_npy_data', return_value=None)
    def test_do_print_data(self, mock_print_npy_data):
        self.parse_tool.do_print_data(['-n', 'file_path'])

        mock_print_npy_data.assert_called_once()

    @patch('msprobe.pytorch.parse_tool.lib.parse_tool.Visualization.parse_pkl', return_value=None)
    def test_do_parse_pkl(self, mock_parse_pkl):
        self.parse_tool.do_parse_pkl(['-f', 'pkl_path', '-n', 'api_name'])

        mock_parse_pkl.assert_called_once()

    @patch('msprobe.pytorch.parse_tool.lib.parse_tool.Compare.compare_data', return_value=None)
    def test_do_compare_data(self, mock_compare_data):
        with patch('msprobe.pytorch.parse_tool.lib.parse_tool.Util.check_positive'), \
            patch('msprobe.pytorch.parse_tool.lib.parse_tool.Util.check_path_valid', return_value=True), \
            patch('msprobe.pytorch.parse_tool.lib.parse_tool.Util.check_file_path_format'):
            self.parse_tool.do_compare_data(['-m', 'my_data*.npy', '-g', 'golden*.npy'])

            mock_compare_data.assert_called_once()

    @patch('msprobe.pytorch.parse_tool.lib.parse_tool.Compare.compare_converted_dir', return_value=None)
    def test_do_compare_converted_dir(self, mock_compare_converted_dir):
        args = self.parser.parse_args(['-m', 'my_dump_path', '-g', 'golden_dump_path', '-out', 'output_path', '-cmp_path', 'msaccucmp_path'])
        self.parse_tool.do_compare_converted_dir(args)

        mock_compare_converted_dir.assert_called_once()

    @patch('msprobe.pytorch.parse_tool.lib.parse_tool.Compare.convert_api_dir_to_npy', return_value=None)
    def test_do_convert_api_dir(self, mock_convert_api_dir_to_npy):
        with patch('msprobe.pytorch.parse_tool.lib.parse_tool.Util.check_path_valid', return_value=True), \
            patch('msprobe.pytorch.parse_tool.lib.parse_tool.Util.check_executable_file'), \
            patch('msprobe.pytorch.parse_tool.lib.parse_tool.Util.check_files_in_path'):
            self.parse_tool.do_convert_api_dir(['-m', 'my_dump_path'])

            mock_convert_api_dir_to_npy.assert_called_once()
