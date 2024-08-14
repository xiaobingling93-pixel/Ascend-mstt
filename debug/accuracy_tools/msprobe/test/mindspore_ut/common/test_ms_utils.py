#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2024-2024. Huawei Technologies Co., Ltd. All rights reserved.
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
import unittest

from msprobe.mindspore.common.utils import MsprobeStep


class TestMsprobeStep(unittest.TestCase):
    def setUp(self):
        class Debugger:
            def __init__(self):
                self.start_called = False
                self.stop_called = False
                self.step_called = False
                self.stop_called_first = False

            def start(self):
                self.start_called = True

            def stop(self):
                self.stop_called = True

            def step(self):
                if self.stop_called:
                    self.stop_called_first = True
                self.step_called = True
        debugger = Debugger()
        self.msprobe_step = MsprobeStep(debugger)

    def test_on_train_step_begin(self):
        self.msprobe_step.on_train_step_begin("run_context")
        self.assertTrue(self.msprobe_step.debugger.start_called)
        self.assertFalse(self.msprobe_step.debugger.stop_called)
        self.assertFalse(self.msprobe_step.debugger.step_called)

    def test_on_train_step_end(self):
        self.msprobe_step.on_train_step_end("run_context")
        self.assertFalse(self.msprobe_step.debugger.start_called)
        self.assertTrue(self.msprobe_step.debugger.stop_called)
        self.assertTrue(self.msprobe_step.debugger.step_called)
        self.assertTrue(self.msprobe_step.debugger.stop_called_first)
