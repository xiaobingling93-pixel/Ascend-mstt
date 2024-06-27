#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import os
import shutil
import unittest
from ptdbg_ascend.advisor.advisor import Advisor
from ptdbg_ascend.common.file_check_util import FileCheckException
from ptdbg_ascend.common.utils import CompareException
import pandas


class TestAdvisor(unittest.TestCase):
    def setUp(self) -> None:
        os.makedirs("test_result/output", mode=0o700, exist_ok=True)
        self.output_path = os.path.abspath("test_result/output")

    def tearDown(self) -> None:
        shutil.rmtree("test_result/", ignore_errors=True)

    def test_analysis_when_csv_is_valid(self):
        input_data = pandas.read_csv("resources/compare/compare_result_20230703104808.csv")
        advisor = Advisor(input_data, self.output_path)
        advisor.analysis()
        filenames = os.listdir(self.output_path)
        self.assertEqual(len(filenames), 1)
