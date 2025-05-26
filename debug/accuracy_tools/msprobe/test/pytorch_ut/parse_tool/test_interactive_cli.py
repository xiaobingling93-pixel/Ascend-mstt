import unittest
from unittest.mock import patch

from msprobe.pytorch.parse_tool.lib.interactive_cli import InteractiveCli


class TestInteractiveCli(unittest.TestCase):
    def setUp(self):
        self.interactive_cli = InteractiveCli()

    def test_parse_argv(self):
        res1 = self.interactive_cli._parse_argv("not None")
        res2 = self.interactive_cli._parse_argv("not None", "insert")
        res3 = self.interactive_cli._parse_argv("not None", "not")
        res4 = self.interactive_cli._parse_argv("")
        res5 = self.interactive_cli._parse_argv("", "no_insert")
        res6 = self.interactive_cli._parse_argv("-h return", "no_insert")

        self.assertEqual(res1, ["not", "None"])
        self.assertEqual(res2, ["insert", "not", "None"])
        self.assertEqual(res3, ["not", "None"])
        self.assertEqual(res4, [])
        self.assertEqual(res5, [])
        self.assertEqual(res6, ["-h", "return"])

    @patch('msprobe.pytorch.parse_tool.lib.interactive_cli.ParseTool.prepare', return_value=None)
    def test_prepare(self, mock_prepare):
        self.interactive_cli.prepare()
        mock_prepare.assert_called_once()

    def test_default(self, command='rm'):
        res = self.interactive_cli.default(command)
        self.assertIsNone(res)

    @patch('msprobe.pytorch.parse_tool.lib.interactive_cli.ParseTool.do_compare_converted_dir')
    @patch('msprobe.pytorch.parse_tool.lib.interactive_cli.ParseTool.do_vector_compare')
    def test_do_vc(self, mock_do_vector_compare, mock_do_compare_converted_dir):
        with (patch('msprobe.pytorch.parse_tool.lib.interactive_cli.Util.check_path_valid'),
              patch('msprobe.pytorch.parse_tool.lib.interactive_cli.Util.check_files_in_path')):
            with patch('msprobe.pytorch.parse_tool.lib.interactive_cli.Util.dir_contains_only',
                       return_value=False):
                self.interactive_cli.do_vc(
                    '-m my_dump_path -g golden_dump_path -out output_path -cmp_path msaccucmp_path')
                mock_do_vector_compare.assert_called_once()

            with patch('msprobe.pytorch.parse_tool.lib.interactive_cli.Util.dir_contains_only',
                       return_value=True):
                self.interactive_cli.do_vc(
                    '-m my_dump_path -g golden_dump_path -out output_path -cmp_path msaccucmp_path')
                mock_do_compare_converted_dir.assert_called_once()

    @patch('msprobe.pytorch.parse_tool.lib.interactive_cli.ParseTool.do_convert_dump', return_value=None)
    def test_do_dc(self, mock_do_convert_dump):
        self.interactive_cli.do_dc('-n file_name/file_path -f format -out output_path')
        mock_do_convert_dump.assert_called_once()

    @patch('msprobe.pytorch.parse_tool.lib.interactive_cli.ParseTool.do_print_data', return_value=None)
    def test_do_pt(self, mock_do_print_data):
        self.interactive_cli.do_pt('-n file_path')
        mock_do_print_data.assert_called_once()

    @patch('msprobe.pytorch.parse_tool.lib.interactive_cli.ParseTool.do_parse_pkl', return_value=None)
    def test_do_pk(self, mock_do_parse_pkl):
        self.interactive_cli.do_pk('-f pkl_path -n api_name')
        mock_do_parse_pkl.assert_called_once()

    @patch('msprobe.pytorch.parse_tool.lib.interactive_cli.ParseTool.do_compare_data', return_value=None)
    def test_do_cn(self, mock_do_comapre_data):
        self.interactive_cli.do_cn('-m my_data*.npy -g golden*.npu -p num -al atol -rl rtol')
        mock_do_comapre_data.assert_called_once()

    @patch('msprobe.pytorch.parse_tool.lib.interactive_cli.ParseTool.do_convert_api_dir', return_value=None)
    def test_do_cad(self, mock_do_convert_api_dir):
        self.interactive_cli.do_cad('-m my_dump_path -out output_path -asc msaccucmp_path')
        mock_do_convert_api_dir.assert_called_once()
