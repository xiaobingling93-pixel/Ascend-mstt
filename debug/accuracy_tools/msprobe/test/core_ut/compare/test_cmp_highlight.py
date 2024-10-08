# coding=utf-8
import unittest
import os
import shutil
import pandas as pd
import numpy as np
import openpyxl
import multiprocessing
from collections import namedtuple
from msprobe.core.compare.highlight import HighlightCheck, CheckMaxRelativeDiff, highlight_rows_xlsx, csv_value_is_valid
from msprobe.core.compare.acc_compare import Comparator
from msprobe.core.common.const import CompareConst
from msprobe.core.common.utils import CompareException


base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_acc_compare_data')


def generate_result_xlsx(base_dir):
    data_path = os.path.join(base_dir, 'target_result.xlsx')
    data = [['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
             'torch.float32', 'torch.float32', [2, 2], [2, 2],
             '', '', '', '', '', 1, 1, 1, 1, 1, 1, 1, 1, 'Yes', '', '-1']
            ]
    columns = CompareConst.COMPARE_RESULT_HEADER + ['Data_name']
    result_df = pd.DataFrame(data, columns=columns)
    result_df.to_excel(data_path, index=False)


def compare_excel_files_with_highlight(file1, file2):
    # 加载工作簿
    wb1 = openpyxl.load_workbook(file1)
    wb2 = openpyxl.load_workbook(file2)

    # 比较每个工作表
    if len(wb1.sheetnames) != len(wb2.sheetnames):
        return False

    for sheet_name in wb1.sheetnames:
        if sheet_name not in wb2.sheetnames:
            return False
        ws1 = wb1[sheet_name]
        ws2 = wb2[sheet_name]

        # 比较单元格内容和高亮
        for row in ws1.iter_rows():
            for cell in row:
                other_cell = ws2[cell.coordinate]

                # 比较单元格的值
                if cell.value != other_cell.value:
                    return False

                # 比较单元格的高亮
                if cell.fill != other_cell.fill:  # 比较填充样式
                    return False

    return True


class TestUtilsMethods(unittest.TestCase):

    def setUp(self):
        os.makedirs(base_dir, mode=0o750, exist_ok=True)

    def tearDown(self):
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)

    def test_CheckMaxRelativeDiff_1(self):
        ColorColumns = namedtuple('ColorColumns', ['red', 'yellow'])

        red_lines, yellow_lines = [], []
        color_columns = ColorColumns(red=red_lines, yellow=yellow_lines)

        api_in = {'Max diff': 0, 'Bench max': 1}
        api_out = {'Max diff': 0.6, 'Bench max': 1}
        num = 1
        info = (api_in, api_out, num)
        CheckMaxRelativeDiff().apply(info, color_columns)
        red_lines, yellow_lines = [1], []
        target_color_columns = ColorColumns(red=red_lines, yellow=yellow_lines)
        self.assertEqual(color_columns, target_color_columns)

    def test_CheckMaxRelativeDiff_2(self):
        ColorColumns = namedtuple('ColorColumns', ['red', 'yellow'])

        red_lines, yellow_lines = [], []
        color_columns = ColorColumns(red=red_lines, yellow=yellow_lines)

        api_in = {'Max diff': 0.001, 'Bench max': 1}
        api_out = {'Max diff': 0.2, 'Bench max': 1}
        num = 1
        info = (api_in, api_out, num)
        CheckMaxRelativeDiff().apply(info, color_columns)
        red_lines, yellow_lines = [], [1]
        target_color_columns = ColorColumns(red=red_lines, yellow=yellow_lines)
        self.assertEqual(color_columns, target_color_columns)

    def test_CheckMaxRelativeDiff_3(self):
        ColorColumns = namedtuple('ColorColumns', ['red', 'yellow'])

        red_lines, yellow_lines = [], []
        color_columns = ColorColumns(red=red_lines, yellow=yellow_lines)

        api_in = {'Max diff': 0.001, 'Bench max': np.nan}
        api_out = {'Max diff': 0.2, 'Bench max': 1}
        num = 1
        info = (api_in, api_out, num)
        result = CheckMaxRelativeDiff().apply(info, color_columns)
        self.assertEqual(result, None)

    def test_highlight_rows_xlsx_1(self):
        data = [['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
                 'torch.float32', 'torch.float32', [2, 2], [2, 2],
                 '', '', '', '', '', 1, 1, 1, 1, 1, 1, 1, 1, 'Yes', '', '-1']
                ]
        columns = CompareConst.COMPARE_RESULT_HEADER + ['Data_name']
        result_df = pd.DataFrame(data, columns=columns)
        highlight_dict = {'red_rows': 0}
        file_path = os.path.join(base_dir, 'result.xlsx')
        highlight_rows_xlsx(result_df, highlight_dict, file_path)
        generate_result_xlsx(base_dir)
        self.assertTrue(compare_excel_files_with_highlight(file_path, os.path.join(base_dir, 'target_result.xlsx')))

    def test_highlight_rows_xlsx_2(self):
        data = [['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
                 'torch.float32', 'torch.float32', [2, 2], [2, 2],
                 '', '', '', '', '', 1, 1, 1, 1, 1, 1, 1, 1, 'Yes', '', '-1']
                ]
        columns = CompareConst.COMPARE_RESULT_HEADER + ['=Data_name']
        result_df = pd.DataFrame(data, columns=columns)
        highlight_dict = {}
        file_path = base_dir
        with self.assertRaises(RuntimeError) as context:
            highlight_rows_xlsx(result_df, highlight_dict, file_path)
        self.assertIn("Malicious value", str(context.exception))

    def test_highlight_rows_xlsx_3(self):
        data = [['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
                 '=torch.float32', 'torch.float32', [2, 2], [2, 2],
                 '', '', '', '', '', 1, 1, 1, 1, 1, 1, 1, 1, 'Yes', '', '-1']
                ]
        columns = CompareConst.COMPARE_RESULT_HEADER + ['Data_name']
        result_df = pd.DataFrame(data, columns=columns)
        highlight_dict = {}
        file_path = base_dir
        with self.assertRaises(RuntimeError) as context:
            highlight_rows_xlsx(result_df, highlight_dict, file_path)
        self.assertIn("Malicious value", str(context.exception))

    def test_csv_value_is_valid_1(self):
        result = csv_value_is_valid(1)
        self.assertTrue(result)

    def test_csv_value_is_valid_2(self):
        result = csv_value_is_valid("-1.00")
        self.assertTrue(result)

        result = csv_value_is_valid("+1.00")
        self.assertTrue(result)

    def test_csv_value_is_valid_3(self):
        result = csv_value_is_valid("=1.00")
        self.assertTrue(result)

