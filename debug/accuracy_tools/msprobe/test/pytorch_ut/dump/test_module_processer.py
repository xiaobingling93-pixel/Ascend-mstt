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

import threading

import unittest
from unittest.mock import patch, MagicMock
import torch

import msprobe.pytorch.dump.module_dump.module_processer as mp
from msprobe.core.data_dump.scope import ModuleRangeScope
from msprobe.pytorch.dump.module_dump.module_processer import (
    ModuleProcesser,
    replace_checkpoint,
    checkpoint_without_early_stop,
    wrap_megatron_deallocate
)

ori_checkpoint = torch.utils.checkpoint.checkpoint


class TestWrapper(unittest.TestCase):
    def setUp(self):
        torch.utils.checkpoint.checkpoint = ori_checkpoint

    def test_replace_checkpoint_for_torch_version_above_2(self):
        mp.torch_version_above_or_equal_2 = True
        with patch('msprobe.pytorch.dump.module_dump.module_processer.checkpoint_without_early_stop') as mock_obj:
            replace_checkpoint()
            self.assertEqual(torch.utils.checkpoint.checkpoint, mock_obj)

    def test_replace_checkpoint_for_torch_version_below_2(self):
        mp.torch_version_above_or_equal_2 = False
        replace_checkpoint()
        self.assertEqual(torch.utils.checkpoint.checkpoint, ori_checkpoint)

    def test_checkpoint_without_early_stop(self):
        mock_checkpoint = MagicMock(return_value="test_result")

        with patch('msprobe.pytorch.dump.module_dump.module_processer.set_checkpoint_early_stop', MagicMock()), \
                patch('msprobe.pytorch.dump.module_dump.module_processer.origin_checkpoint', mock_checkpoint):
            result = checkpoint_without_early_stop("input")
            mock_checkpoint.assert_called_once_with("input")
            self.assertEqual(result, "test_result")

    def test_wrap_megatron_deallocate(self):
        mock_func = MagicMock(return_value="output_test")
        wrapped = wrap_megatron_deallocate(mock_func)

        mock_tensor = MagicMock(spec=torch.Tensor)
        mock_tensor._base = True
        mock_tensor.device = "cpu"
        mock_tensor.dtype = torch.float32
        mock_tensor.clone.return_value = "cloned"

        result = wrapped(mock_tensor, deallocate_pipeline_outputs=True)
        mock_tensor.clone.assert_called_once()
        self.assertEqual(mock_tensor.data.shape, (1,))
        self.assertEqual(result, "output_test")
        mock_func.assert_called_once_with("cloned", True)

        result = wrapped("normal_input", False)
        self.assertEqual(result, "output_test")
        mock_func.assert_called_with("normal_input", False)


class TestModuleProcesser(unittest.TestCase):
    def setUp(self):
        ModuleProcesser.module_count = {}
        ModuleProcesser.module_stack = {}
        ModuleProcesser.module_node = {}
        ModuleProcesser.api_parent_node = {}

        self.scope = ModuleRangeScope([], [])
        self.mock_scope = MagicMock()

    @patch('msprobe.pytorch.dump.module_dump.module_processer.wrap_setup_input_output_hook')
    @patch('msprobe.pytorch.dump.module_dump.module_processer.replace_checkpoint')
    def test_init_with_valid_scope(self, mock_replace, mock_wrap):
        processor = ModuleProcesser(self.scope)
        self.assertEqual(processor.scope, self.scope)
        mock_wrap.assert_called_once()
        mock_replace.assert_called_once()

    @patch('msprobe.pytorch.dump.module_dump.module_processer.logger.info_on_rank_0')
    def test_init_without_megatron(self, mock_log):
        ModuleProcesser(self.scope)
        mock_log.assert_called_with("No megatron find.")

    def test_set_and_get_calls_number(self):
        count = ModuleProcesser.set_and_get_calls_number("test_module")
        self.assertEqual(count, 0)

        count = ModuleProcesser.set_and_get_calls_number("test_module")
        self.assertEqual(count, 1)

    def test_has_register_backward_hook(self):
        module1 = torch.nn.Linear(10, 10)
        self.assertFalse(ModuleProcesser.has_register_backward_hook(module1))

        module2 = MagicMock()
        module2._backward_hooks = [1, 2, 3]
        module2._is_full_backward_hook = False
        self.assertTrue(ModuleProcesser.has_register_backward_hook(module2))

        module2._is_full_backward_hook = True
        self.assertFalse(ModuleProcesser.has_register_backward_hook(module2))

    def test_get_modules_and_names_with_model_list(self):
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        mock_model1.named_modules.return_value = [("layer1", "obj1"), ("layer2", "obj2")]
        mock_model2.named_modules.return_value = [("layer3", "obj3")]

        result = ModuleProcesser.get_modules_and_names(
            [mock_model1, mock_model2],
            recursive=True,
            module_names=["model1", "model2"]
        )
        self.assertEqual(result, {
            "0": [("layer1", "obj1"), ("layer2", "obj2")],
            "1": [("layer3", "obj3")]
        })

    def test_get_modules_and_names_with_model_tuple(self):
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        mock_model1.named_modules.return_value = [("layer1", "obj1")]
        mock_model2.named_modules.return_value = [("layer2", "obj2")]

        result = ModuleProcesser.get_modules_and_names(
            (mock_model1, mock_model2),
            recursive=True,
            module_names=["model1", "model2"]
        )
        self.assertEqual(result, {
            "0": [("layer1", "obj1")],
            "1": [("layer2", "obj2")]
        })

    def test_get_modules_and_names_with_single_recursive(self):
        mock_model = MagicMock()
        mock_model.named_modules.return_value = [("layer1", "obj1")]

        result = ModuleProcesser.get_modules_and_names(
            mock_model,
            recursive=True,
            module_names=["single_model"]
        )
        self.assertEqual(result, {
            "-1": [("layer1", "obj1")]
        })

    def test_get_modules_and_names_with_single_non_recursive(self):
        mock_model = MagicMock()
        result = ModuleProcesser.get_modules_and_names(
            mock_model,
            recursive=False,
            module_names=["single_model"]
        )
        self.assertEqual(result, {
            "-1": [("single_model", mock_model)]
        })

    def test_get_modules_and_names_invalid_case(self):
        result = ModuleProcesser.get_modules_and_names(
            [MagicMock(), MagicMock()],
            recursive=False,
            module_names=["only_one_name"]
        )
        self.assertEqual(result, {})

        result = ModuleProcesser.get_modules_and_names(
            MagicMock(),
            recursive=False,
            module_names=["name1", "name2"]
        )
        self.assertEqual(result, {})

    def test_reset_module_stats(self):
        ModuleProcesser.module_count = {"test": 1}
        ModuleProcesser.module_stack = ["layer1"]
        ModuleProcesser.api_parent_node = "parent"
        ModuleProcesser.module_node = {"key": "value"}
        ModuleProcesser.module_bw_hook_kernels = {"hook": "data"}
        ModuleProcesser.enable_module_dump = True

        ModuleProcesser.reset_module_stats()

        self.assertEqual(ModuleProcesser.module_count, {})
        self.assertEqual(ModuleProcesser.module_stack, {})
        self.assertEqual(ModuleProcesser.api_parent_node, {})
        self.assertEqual(ModuleProcesser.module_node, {})
        self.assertEqual(ModuleProcesser.module_bw_hook_kernels, {})
        self.assertFalse(ModuleProcesser.enable_module_dump)

    def test_set_construct_info_in_pre_hook_with_stack(self):
        processor = ModuleProcesser(self.mock_scope)
        ModuleProcesser.module_stack[threading.get_ident()] = ["parent_module"]
        processor.scope = self.mock_scope

        processor.set_construct_info_in_pre_hook("current_module")

        self.assertEqual(ModuleProcesser.module_node["current_module"], "parent_module")
        self.assertEqual(
            ModuleProcesser.module_stack[threading.get_ident()],
            ["parent_module", "current_module"]
        )
        self.assertEqual(ModuleProcesser.api_parent_node[threading.get_ident()], "current_module")
        self.mock_scope.begin_module.assert_called_once_with("current_module")

    def test_set_construct_info_in_pre_hook_empty_stack(self):
        processor = ModuleProcesser(self.mock_scope)
        processor.scope = self.mock_scope
        processor.set_construct_info_in_pre_hook("root_module")

        self.assertIsNone(ModuleProcesser.module_node["root_module"])
        self.assertEqual(ModuleProcesser.module_stack[threading.get_ident()], ["root_module"])
        self.assertEqual(ModuleProcesser.api_parent_node[threading.get_ident()], "root_module")

    def test_set_construct_info_in_hook_with_forward(self):
        mp.torch_version_above_or_equal_2 = True
        processor = ModuleProcesser(self.mock_scope)
        ModuleProcesser.module_stack = {threading.get_ident(): ["parent", "current"]}
        processor.scope = self.mock_scope

        processor.set_construct_info_in_hook("current")

        self.assertEqual(ModuleProcesser.module_stack[threading.get_ident()], ["parent"])
        self.assertEqual(ModuleProcesser.api_parent_node[threading.get_ident()], "parent")
        self.mock_scope.end_module.assert_called_once_with("current")

    def test_set_construct_info_in_hook_with_backward(self):
        mp.torch_version_above_or_equal_2 = False
        processor = ModuleProcesser(self.mock_scope)
        processor.scope = self.mock_scope

        processor.set_construct_info_in_hook("backward_module", is_forward=False)

        self.assertEqual(ModuleProcesser.api_parent_node[threading.get_ident()], "backward_module")
        self.mock_scope.begin_module.assert_called_once_with("backward_module")

    def test_set_construct_info_in_hook_empty_stack(self):
        mp.torch_version_above_or_equal_2 = True
        processor = ModuleProcesser(self.mock_scope)

        processor.set_construct_info_in_hook("module")

        self.assertEqual(ModuleProcesser.api_parent_node, {threading.get_ident(): None})


if __name__ == "__main__":
    unittest.main()
