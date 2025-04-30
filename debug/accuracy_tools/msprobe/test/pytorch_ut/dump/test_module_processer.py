# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
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

import unittest
from unittest.mock import MagicMock

import torch

from msprobe.core.data_dump.scope import ModuleRangeScope
from msprobe.pytorch.dump.module_dump.module_processer import ModuleProcesser


class TestModuleProcesser(unittest.TestCase):

    def setUp(self):
        self.mock_tensor = MagicMock(spec=torch.Tensor)
        self.mock_scope = MagicMock()
        self.processor = ModuleProcesser(self.mock_scope)

    def test_scope_is_module_range_scope(self):
        scope = ModuleRangeScope([], [])
        processor = ModuleProcesser(scope)
        self.assertEqual(processor.scope, scope)

    def test_scope_is_not_module_range_scope(self):
        scope = "not a ModuleRangeScope"
        processor = ModuleProcesser(scope)
        self.assertIsNone(processor.scope)

    def test_set_and_get_calls_number(self):
        ModuleProcesser.reset_module_stats()
        test = ModuleProcesser(None)
        self.assertEqual(test.module_count, {})
        module_name = "nope"
        test.set_and_get_calls_number(module_name)
        self.assertEqual(test.module_count["nope"], 0)

        ModuleProcesser.reset_module_stats()
