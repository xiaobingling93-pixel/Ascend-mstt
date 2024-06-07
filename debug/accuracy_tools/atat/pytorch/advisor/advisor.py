#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2022-2024. Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import os
import pandas as pd

from .advisor_result import AdvisorResult
from .advisor_const import AdvisorConst
from ...core.utils import CompareException, CompareConst, Const, print_info_log, print_warn_log, print_error_log
from ..common.file_check import FileChecker, FileCheckConst


class Advisor:
    """
    Class for generate advisor
    """

    def __init__(self, input_file, out_path=""):
        self.input_file = os.path.realpath(input_file)
        self.out_path = os.path.realpath(out_path)

    def _parse_input_file(self):
        try:
            df = pd.read_csv(self.input_file, on_bad_lines='skip')
        except OSError as os_err:
            print_error_log('Failed to parse the input file %s. %s'
                            % (self.input_file, str(os_err)))
            raise CompareException(CompareException.PARSE_FILE_ERROR) from os_err
        data_columns = df.columns.values
        if {CompareConst.ACCURACY, CompareConst.NPU_NAME}.issubset(data_columns):
            self.file_type = Const.ALL
        elif {CompareConst.RESULT, CompareConst.NPU_MD5}.issubset(data_columns):
            self.file_type = Const.MD5
        elif {CompareConst.MAX_DIFF, CompareConst.RESULT}.issubset(data_columns):
            self.file_type = Const.SUMMARY
        else:
            print_error_log('Compare result file does not meet the required conditions.')
            raise CompareException(CompareException.INVALID_FILE_ERROR)
        df.reset_index(inplace=True)
        # The value of index is consistent with the line number of csv, csv file first line is 2
        df.iloc[:, 0] += 2
        return df

    def _check_path_vaild(self):
        input_file_checker = FileChecker(self.input_file, FileCheckConst.FILE, FileCheckConst.READ_ABLE,
                                         FileCheckConst.CSV_SUFFIX)
        input_file_checker.common_check()
        out_path_checker = FileChecker(self.out_path, FileCheckConst.DIR, FileCheckConst.WRITE_ABLE)
        out_path_checker.common_check()

    def gen_advisor_message(self, node_name):
        if AdvisorConst.FORWARD in node_name:
            if AdvisorConst.INPUT in node_name:
                message = AdvisorConst.FORWARD_INPUT_SUGGEST
            else:
                message = AdvisorConst.FORWARD_OUTPUT_SUGGEST
                message = self.deterministic_advisor(message, node_name)
        else:
            if AdvisorConst.INPUT in node_name:
                message = AdvisorConst.BACKWARD_INPUT_SUGGEST
            else:
                message = AdvisorConst.BACKWARD_OUTPUT_SUGGEST
                message = self.deterministic_advisor(message, node_name)
        message = self.batch_norm_advisor(message, node_name)
        return message

    @staticmethod
    def deterministic_advisor(message, node_name):
        for api_name in AdvisorConst.NEED_DETERMINISTIC_API:
            if api_name in node_name:
                return AdvisorConst.DETERMINISTIC_SUGGEST
        return message

    @staticmethod
    def batch_norm_advisor(message, node_name):
        if AdvisorConst.FUNC_BATCH_NORM in node_name and AdvisorConst.FORWARD_INPUT_1 in node_name:
            message = AdvisorConst.BATCH_NORM_SUGGEST
        return message

    def analyze_unmatched(self, analyze_data):
        if self.file_type == Const.ALL:
            accuracy_unmatched = analyze_data[analyze_data[CompareConst.ACCURACY] == CompareConst.ACCURACY_CHECK_UNMATCH]
        else:
            accuracy_unmatched = analyze_data[(analyze_data[CompareConst.NPU_SHAPE] == CompareConst.NAN) | 
                                              (analyze_data[CompareConst.BENCH_SHAPE] == CompareConst.NAN)]
        num_unmatch = len(accuracy_unmatched)
        if num_unmatch != 0:
            for i in range(len(accuracy_unmatched)):
                item = accuracy_unmatched.iloc[i]
                print_warn_log("The tensor name matches but the shape or dtype does not match: {}"
                            .format(item[CompareConst.NPU_NAME]))

    def gen_advisor_result(self, pd_data):
        first_failing_data = pd_data.iloc[0]
        node_name = first_failing_data[CompareConst.NPU_NAME]
        index = first_failing_data['index']
        message = self.gen_advisor_message(node_name)
        print_warn_log("Find %s accuracy not reached, the line is %s" % (node_name, index))
        result = AdvisorResult(node_name, index, message)
        return result

    def analysis(self):
        self._check_path_vaild()
        analyze_data = self._parse_input_file()
        print_info_log("Start analyzing the comparison result: %s" % self.input_file)
        self.analyze_unmatched(analyze_data)
        if self.file_type == Const.ALL:
            failing_data = analyze_data[analyze_data[CompareConst.ACCURACY] == CompareConst.ACCURACY_CHECK_NO]
        elif self.file_type == Const.MD5:
            failing_data = analyze_data[analyze_data[CompareConst.RESULT] == CompareConst.DIFF]
        elif self.file_type == Const.SUMMARY:
            failing_data = analyze_data[analyze_data[CompareConst.RESULT] == CompareConst.WARNING]
        if failing_data.empty:
            print_info_log("All data from api input/output accuracy reached")
            result = AdvisorResult(AdvisorConst.NO_ERROR_API, AdvisorConst.NO_ERROR_API, AdvisorConst.NO_ERR_SUGGEST)
        else:
            result = self.gen_advisor_result(failing_data)
        message_list = result.print_advisor_log()
        result.gen_summary_file(self.out_path, message_list)
