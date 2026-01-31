# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
# `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------


import unittest

from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams


class TestHandlerParams(unittest.TestCase):
    def test_handler_params(self):
        self.assertIsNone(HandlerParams.args)
        self.assertIsNone(HandlerParams.kwargs)
        self.assertIsNone(HandlerParams.index)
        self.assertIsNone(HandlerParams.original_result)
        self.assertIsNone(HandlerParams.fuzzed_result)
        self.assertTrue(HandlerParams.is_consistent)
        self.assertIsNone(HandlerParams.fuzzed_value)
        self.assertIsNone(HandlerParams.original_func)
