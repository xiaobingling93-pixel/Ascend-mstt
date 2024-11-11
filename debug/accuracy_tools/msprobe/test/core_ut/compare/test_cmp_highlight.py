# coding=utf-8
import unittest
import os
import shutil
import pandas as pd
import numpy as np
import openpyxl
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
from collections import namedtuple
from msprobe.core.compare.highlight import CheckMaxRelativeDiff, highlight_rows_xlsx, csv_value_is_valid, \
    add_highlight_row_info, update_highlight_err_msg
from msprobe.core.common.const import CompareConst, Const


base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_highlight')


def generate_result_xlsx(base_dir):
    data_path = os.path.join(base_dir, 'target_result.xlsx')
    data = [['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
             'torch.float32', 'torch.float32', [2, 2], [2, 2],
             '', '', '', '', '', 1, 1, 1, 1, 1, 1, 1, 1, 'Yes', '', '-1']
            ]
    columns = CompareConst.COMPARE_RESULT_HEADER + ['Data_name']
    result_df = pd.DataFrame(data, columns=columns)
    result_df.to_excel(data_path, index=False, sheet_name='Sheet')
    wb = load_workbook(data_path)
    ws = wb.active
    red_fill = PatternFill(start_color=CompareConst.RED, end_color=CompareConst.RED, fill_type='solid')
    for row_index, row in enumerate(ws.iter_rows()):
        if row_index == 0:
            continue
        for cell in row:
            cell.fill = red_fill
    wb.save(data_path)


def compare_excel_files_with_highlight(file1, file2):
    wb1 = openpyxl.load_workbook(file1)
    wb2 = openpyxl.load_workbook(file2)

    if len(wb1.sheetnames) != len(wb2.sheetnames):
        return False

    for sheet_name in wb1.sheetnames:
        if sheet_name not in wb2.sheetnames:
            return False
        ws1 = wb1[sheet_name]
        ws2 = wb2[sheet_name]

        for row_index, row in enumerate(ws1.iter_rows()):
            if row_index == 0:
                continue
            for cell in row:
                other_cell = ws2[cell.coordinate]

                if cell.value != other_cell.value:
                    return False
                if cell.fill.start_color.index != other_cell.fill.start_color.index:
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

        api_in = {6: 0, 18: 1}
        api_out = {6: 0.6, 18: 1}
        num = 1
        info = (api_in, api_out, num)
        CheckMaxRelativeDiff().apply(info, color_columns, dump_mode=Const.SUMMARY)
        red_lines, yellow_lines = [1], []
        target_color_columns = ColorColumns(red=red_lines, yellow=yellow_lines)
        self.assertEqual(color_columns, target_color_columns)

    def test_CheckMaxRelativeDiff_2(self):
        ColorColumns = namedtuple('ColorColumns', ['red', 'yellow'])

        red_lines, yellow_lines = [], []
        color_columns = ColorColumns(red=red_lines, yellow=yellow_lines)

        api_in = {6: 0.001, 18: 1}
        api_out = {6: 0.2, 18: 1}
        num = 1
        info = (api_in, api_out, num)
        CheckMaxRelativeDiff().apply(info, color_columns, dump_mode=Const.SUMMARY)
        red_lines, yellow_lines = [], [1]
        target_color_columns = ColorColumns(red=red_lines, yellow=yellow_lines)
        self.assertEqual(color_columns, target_color_columns)

    def test_CheckMaxRelativeDiff_3(self):
        ColorColumns = namedtuple('ColorColumns', ['red', 'yellow'])

        red_lines, yellow_lines = [], []
        color_columns = ColorColumns(red=red_lines, yellow=yellow_lines)

        api_in = {6: 0.001, 18: np.nan}
        api_out = {6: 0.2, 18: 1}
        num = 1
        info = (api_in, api_out, num)
        result = CheckMaxRelativeDiff().apply(info, color_columns, dump_mode=Const.SUMMARY)
        self.assertEqual(result, None)

    def test_highlight_rows_xlsx_1(self):
        data = [['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
                 'torch.float32', 'torch.float32', [2, 2], [2, 2],
                 '', '', '', '', '', 1, 1, 1, 1, 1, 1, 1, 1, 'Yes', '', '-1']
                ]
        columns = CompareConst.COMPARE_RESULT_HEADER + ['Data_name']
        result_df = pd.DataFrame(data, columns=columns)
        highlight_dict = {'red_rows': [0]}
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
                 '', '', '', '', '', 1, 1, 1, 1, 1, 1, 1, 1, 'Yes', '', '-1'],
                ['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
                 '=torch.float32', 'torch.float32', [2, 2], [2, 2],
                 '', '', '', '', '', 1, 1, 1, 1, 1, 1, 1, 1, 'Yes', '', '-1']
                ]
        columns = CompareConst.COMPARE_RESULT_HEADER + ['Data_name']
        result_df = pd.DataFrame(data, columns=columns)
        highlight_dict = {'red_rows': [], 'yellow_rows': []}
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
        self.assertFalse(result)

    def test_add_highlight_row_info_existing(self):
        color_list = [(1, ["a", "b"]), (5, ["c"])]
        num = 5
        highlight_err_msg = "highlight"
        add_highlight_row_info(color_list, num, highlight_err_msg)
        self.assertEqual(color_list, [(1, ["a", "b"]), (5, ["c", "highlight"])])

    def test_add_highlight_row_info_new(self):
        color_list = [(1, ["a", "b"]), (5, ["c"])]
        num = 6
        highlight_err_msg = "highlight"
        add_highlight_row_info(color_list, num, highlight_err_msg)
        self.assertEqual(color_list, [(1, ["a", "b"]), (5, ["c"]), (6, ["highlight"])])

    def test_update_highlight_err_msg(self):
        data = [['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
                 'torch.float32', 'torch.float32', [2, 2], [2, 2],
                 '', '', '', '', '', 1, 1, 1, 1, 1, 1, 1, 1, 'Yes', '', '-1'],
                ['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
                 'torch.float32', 'torch.float32', [2, 2], [2, 2],
                 '', '', '', '', '', 1, 1, 1, 1, 1, 1, 1, 1, 'Yes', '', '-1']
                ]
        columns = CompareConst.COMPARE_RESULT_HEADER + ['Data_name']
        result_df = pd.DataFrame(data, columns=columns)
        highlight_dict = {
            'red_rows': set(1),
            'yellow_rows': {1, 2},
            'red_lines': [(1, ['a', 'b'])],
            'yellow_lines': [(1, ['c']), (2, ['d'])]
        }
        update_highlight_err_msg(result_df, highlight_dict)

        t_data = [['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
                   'torch.float32', 'torch.float32', [2, 2], [2, 2],
                   '', '', '', '', '', 1, 1, 1, 1, 1, 1, 1, 1, 'Yes', 'a\nb', '-1'],
                  ['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
                   'torch.float32', 'torch.float32', [2, 2], [2, 2],
                   '', '', '', '', '', 1, 1, 1, 1, 1, 1, 1, 1, 'Yes', 'd', '-1']
                  ]
        target_result_df = pd.DataFrame(t_data, columns=columns)
        self.assertEqual(result_df, target_result_df)

    def test_update_highlight_err_msg(self):
        data = [
            ['err_msg1'],
            ['err_msg2']
        ]
        columns = ['Err_message']
        result_df = pd.DataFrame(data, columns=columns)
        highlight_dict = {
            'red_rows': set(1),
            'yellow_rows': {1, 2},
            'red_lines': [(1, ['a', 'b'])],
            'yellow_lines': [(1, ['c']), (2, ['d'])]
        }
        result = update_highlight_err_msg(result_df, highlight_dict)
        self.assertEqual(result, None)
