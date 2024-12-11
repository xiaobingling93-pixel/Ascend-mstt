# coding=utf-8
import json
import os
import shutil
import unittest
import threading
import pandas as pd
import multiprocessing
from unittest.mock import patch
from msprobe.core.compare.merge_result.merge_result import check_compare_result_name, reorder_path, get_result_path, \
    get_dump_mode, check_index_dump_mode_consistent, extract_api_full_name, search_api_index_result, \
    table_value_check, result_process, handle_multi_process, generate_result_df, generate_merge_result, df_merge, \
    merge_result
from msprobe.core.common.const import FileCheckConst, Const, CompareConst


base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_merge_result')


class TestUtilsMethods(unittest.TestCase):
    def setUp(self):
        os.makedirs(base_dir, mode=0o750, exist_ok=True)

        # 设置测试数据
        self.api_list = ['api1', 'api2']
        self.compare_index_list = ['index1', 'index2']

        # 模拟的 DataFrame
        self.result_df = pd.DataFrame({
            CompareConst.NPU_NAME: ['api1', 'api2'],
            'index1': [100, 200],
            'index2': [300, 400]
        })

        # 模拟的 compare_index_dict
        self.compare_index_dict = {}

        # 模拟的 DataFrame
        self.result_df_1 = pd.DataFrame({
            CompareConst.NPU_NAME: ['api1', 'api2'],
            'index1': [100, 200],
            'index2': [300, 400]
        })
        self.result_df_2 = pd.DataFrame({
            CompareConst.NPU_NAME: ['api1', 'api2'],
            'index1': [150, 250],
            'index2': [350, 450]
        })

        # 设置测试数据
        self.compare_result_path_list = [
            "/path/to/compare_result_rank1-rank.xlsx",
            "/path/to/compare_result_rank2-rank.xlsx"
        ]

    def tearDown(self):
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)

    def test_check_compare_result_name_multi_rank_pattern(self):
        # 测试多卡模式
        valid_name = "compare_result_rank1-rank1_20240101010101.xlsx"

        result = check_compare_result_name(valid_name)
        self.assertTrue(result)

    @patch('msprobe.core.compare.merge_result.merge_result.logger')
    def test_single_rank_pattern_single_rank_pattern(self, mock_logger):
        # 测试单卡模式
        valid_name = "compare_result_rank-rank_20240101010101.xlsx"

        result = check_compare_result_name(valid_name)
        self.assertFalse(result)
        mock_logger.warning.assert_called_once_with("Single rank compare result do not need to be merged.")

    def test_reorder_path(self):
        paths = [
            "/path/to/compare_result_rank3-rank3_20240101010101.xlsx",
            "/path/to/compare_result_rank1-rank1_20240101010101.xlsx",
            "/path/to/compare_result_rank2-rank2_20240101010101.xlsx",
        ]
        expected_order = [
            "/path/to/compare_result_rank1-rank1_20240101010101.xlsx",
            "/path/to/compare_result_rank2-rank2_20240101010101.xlsx",
            "/path/to/compare_result_rank3-rank3_20240101010101.xlsx",
        ]
        result = reorder_path(paths)
        self.assertEqual(result, expected_order)

    @patch("os.listdir")
    @patch("os.path.join")
    @patch("msprobe.core.compare.merge_result.merge_result.check_compare_result_name")
    @patch("msprobe.core.compare.merge_result.merge_result.FileChecker")
    @patch("msprobe.core.compare.merge_result.merge_result.reorder_path")
    def test_get_result_path_valid_files(self, mock_reorder_path, mock_file_checker, mock_check_name, mock_join,
                                         mock_listdir):
        mock_listdir.return_value = [
            "/path/to/compare_result_rank2-rank2_20240101010101.xlsx",
            "/path/to/compare_result_rank1-rank1_20240101010101.xlsx",
            "/path/to/compare_result_rank3-rank3_20240101010101.xlsx"
        ]
        mock_join.side_effect = lambda dir, name: f"{dir}/{name}"
        mock_check_name.return_value = True
        mock_file_checker.return_value.common_check.side_effect = lambda: True
        mock_reorder_path.return_value = [
            "/mock_dir/path/to/compare_result_rank1-rank1_20240101010101.xlsx",
            "/mock_dir/path/to/compare_result_rank2-rank2_20240101010101.xlsx",
            "/mock_dir/path/to/compare_result_rank3-rank3_20240101010101.xlsx"
        ]

        input_dir = "/mock_dir"
        result = get_result_path(input_dir)

        expected_result = [
            "/mock_dir/path/to/compare_result_rank1-rank1_20240101010101.xlsx",
            "/mock_dir/path/to/compare_result_rank2-rank2_20240101010101.xlsx",
            "/mock_dir/path/to/compare_result_rank3-rank3_20240101010101.xlsx"
        ]
        self.assertEqual(result, expected_result)
        mock_file_checker.assert_called()
        mock_reorder_path.assert_called_once()

    def test_get_dump_mode_all_mode(self):
        header = CompareConst.COMPARE_RESULT_HEADER
        result_df = pd.DataFrame(columns=header)

        result = get_dump_mode(result_df, rank_num=1)
        self.assertEqual(result, Const.ALL)

    def test_get_dump_mode_summary_mode(self):
        header = CompareConst.SUMMARY_COMPARE_RESULT_HEADER
        result_df = pd.DataFrame(columns=header)

        result = get_dump_mode(result_df, rank_num=2)
        self.assertEqual(result, Const.SUMMARY)

    def test_get_dump_mode_md5_mode(self):
        header = CompareConst.MD5_COMPARE_RESULT_HEADER
        result_df = pd.DataFrame(columns=header)

        result = get_dump_mode(result_df, rank_num=3)
        self.assertEqual(result, Const.MD5)

    @patch("msprobe.core.compare.merge_result.merge_result.logger")
    def test_check_index_dump_mode_consistent_md5(self, mock_logger):
        result = check_index_dump_mode_consistent(["index1", "index2"], Const.MD5, rank_num=1)

        self.assertEqual(result, [])
        mock_logger.warning.assert_called_once_with(
            "Rank1 compare result is 'md5' dump task and does not support merging result, please "
            "check! The compare result will not be shown in merged result."
        )

    def test_check_index_dump_mode_consistent_valid_compare_index_subset(self):
        compare_index_list = ["Cosine", "MaxAbsErr"]
        result = check_index_dump_mode_consistent(compare_index_list, Const.ALL, rank_num=2)

        self.assertEqual(result, compare_index_list)

    def test_extract_api_full_name_all_apis_found(self):
        api_list = ["api1", "api2"]
        result_df = pd.DataFrame({
            CompareConst.NPU_NAME: ["api1_full_name", "api2_full_name", "api3_full_name"]
        })
        rank_num = 1

        result = extract_api_full_name(api_list, result_df, rank_num)
        expected = ["api1_full_name", "api2_full_name"]
        self.assertEqual(result, expected)

    @patch("msprobe.core.compare.merge_result.merge_result.table_value_check")
    @patch("msprobe.core.compare.merge_result.merge_result.extract_api_full_name")
    def test_search_api_index_result(self, mock_extract_api_full_name, mock_table_value_check):
        mock_extract_api_full_name.return_value = self.api_list
        mock_table_value_check.return_value = None
        result = search_api_index_result(
            self.api_list,
            self.compare_index_list,
            self.result_df,
            1,  # rank_num
            self.compare_index_dict
        )

        expected_result = {
            'index1': {
                'api1': {1: 100},
                'api2': {1: 200},
            },
            'index2': {
                'api1': {1: 300},
                'api2': {1: 400},
            }
        }

        self.assertEqual(result, expected_result)
        mock_table_value_check.assert_any_call('api1')
        mock_table_value_check.assert_any_call('api2')
        mock_extract_api_full_name.assert_called_with(self.api_list, self.result_df, 1)

    @patch("msprobe.core.compare.merge_result.merge_result.table_value_is_valid")
    def test_table_value_check_invalid_value(self, mock_table_value_is_valid):
        mock_table_value_is_valid.return_value = False
        value = "invalid_value"

        with self.assertRaises(RuntimeError) as context:
            table_value_check(value)

        self.assertEqual(
            str(context.exception),
            f"Malicious value [{value}] is not allowed to be written into the merged xlsx."
        )
        mock_table_value_is_valid.assert_called_once_with(value)

    @patch('msprobe.core.compare.merge_result.merge_result.read_xlsx')
    @patch('msprobe.core.compare.merge_result.merge_result.get_dump_mode')
    @patch('msprobe.core.compare.merge_result.merge_result.check_index_dump_mode_consistent')
    @patch('msprobe.core.compare.merge_result.merge_result.search_api_index_result')
    @patch('msprobe.core.compare.merge_result.merge_result.logger')
    def test_result_process(self, mock_logger, mock_search_api_index_result, mock_check_index_dump_mode_consistent,
                            mock_get_dump_mode, mock_read_xlsx):

        mock_read_xlsx.side_effect = [self.result_df_1, self.result_df_2]
        mock_get_dump_mode.side_effect = ["mode1", "mode1"]
        mock_check_index_dump_mode_consistent.return_value = self.compare_index_list
        mock_search_api_index_result.return_value = {
            "index1":
                {"api1": {1: 100}, "api2": {1: 200}},
            "index2":
                {"api1": {1: 300}, "api2": {1: 400}}
        }

        compare_index_dict_list, rank_num_list = result_process(self.compare_result_path_list, self.api_list,
                                                                self.compare_index_list)

        self.assertEqual(len(compare_index_dict_list), 2)
        self.assertEqual(len(rank_num_list), 2)
        self.assertEqual(rank_num_list, [1, 2])

        mock_logger.info.assert_any_call("Parsing rank1 compare result...")
        mock_logger.warning.assert_not_called()

        # 验证返回的字典结构是否正确
        expected_dict = {
            "index1": {"api1": {1: 100}, "api2": {1: 200}},
            "index2": {"api1": {1: 300}, "api2": {1: 400}},
        }
        self.assertEqual(compare_index_dict_list[0], expected_dict)
        self.assertEqual(compare_index_dict_list[1], expected_dict)

    def test_generate_result_df_valid_input(self):
        api_index_dict = {
            "api_full_name1": {"rank1": 100},
            "api_full_name2": {"rank1": 200},
        }
        header = ["API Full Name", "rank1"]

        result_df = generate_result_df(api_index_dict, header)

        expected_data = [
            ["api_full_name1", 100],
            ["api_full_name2", 200],
        ]
        expected_df = pd.DataFrame(expected_data, columns=header, dtype="object")
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_df_merge_multiple_dataframes(self):
        df1 = pd.DataFrame({CompareConst.NPU_NAME: ["api1", "api2"], "rank1": [100, 200]})
        df2 = pd.DataFrame({CompareConst.NPU_NAME: ["api2", "api3"], "rank2": [150, 250]})
        df3 = pd.DataFrame({CompareConst.NPU_NAME: ["api1", "api3"], "rank3": [120, 300]})

        all_result_df_list = [[df1], [df2], [df3]]

        result = df_merge(all_result_df_list)

        expected_df = pd.DataFrame({
            CompareConst.NPU_NAME: ["api1", "api2", "api3"],
            "rank1": [100, 200, None],
            "rank2": [None, 150, 250],
            "rank3": [120, None, 300]
        })

        self.assertEqual(len(result), 1)
        pd.testing.assert_frame_equal(result[0], expected_df)
