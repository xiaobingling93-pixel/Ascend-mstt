#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import difflib
import os
import shutil
import unittest
from ptdbg_ascend.advisor.advisor import Advisor
import pandas


class TestAdvisor(unittest.TestCase):
    def setUp(self) -> None:
        os.makedirs("test_result/output", exist_ok=True)
        self.output_path = os.path.abspath("test_result/output")
        self.has_error = False

    def tearDown(self) -> None:
        shutil.rmtree("test_result/", ignore_errors=True)

    def test_advisor_summary_file(self):
        input_data = pandas.read_csv("resources/compare/compare_result_20230703104808.csv")
        advisor = Advisor(input_data, self.output_path)
        advisor.analysis()
        filenames = os.listdir(self.output_path)
        for filename in filenames:
            filename = os.path.join(self.output_path, filename)
            self.result_check("resources/compare/advisor.txt", filename)
        self.assertFalse(self.has_error)

    def result_check(self, standard_file, output_file):
        with open(standard_file, 'r', encoding='utf-8') as st_file:
            standard_content = st_file.read().splitlines()
        with open(output_file, 'r', encoding='utf-8') as out_file:
            output_content = out_file.read().splitlines()
        result = list(difflib.unified_diff(standard_content, output_content, n=0))
        if result:
            print('\n\n-------------------------------------------------------------------------', flush=True)
            print(f'[ERROR] {output_file.replace(self.output_path, "")} advisor summary are inconsistent.',
                  flush=True)
            print('\n'.join(result), flush=True)
            print('-------------------------------------------------------------------------', flush=True)
            self.has_error = True
