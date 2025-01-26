# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
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
# limitations under the License. language governing permissions and
# limitations under the License.

import unittest
from collections import defaultdict
from unittest.mock import MagicMock, patch

from mindspore import nn, ops

from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.utils import Const
from msprobe.core.data_dump.scope import BaseScope
from msprobe.mindspore.cell_processor import CellProcessor
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.common.utils import register_backward_hook_functions
from msprobe.mindspore.dump.hook_cell.api_registry import ApiRegistry, api_register
from msprobe.mindspore.dump.hook_cell.hook_cell import HOOKCell
from msprobe.mindspore.dump.jit_dump import JitDump
from msprobe.mindspore.service import Service


class TestService(unittest.TestCase):
    def setUp(self):
        self.config_mock = MagicMock()
        self.config_mock.level_ori = Const.LEVEL_L0
        self.config_mock.dump_path = "/tmp/dump"
        self.config_mock.step = []
        self.config_mock.rank = []
        self.config_mock.task = Const.TENSOR
        self.config_mock.framework = Const.MS_FRAMEWORK
        self.config_mock.list = []
        self.config_mock.scope = []
        self.service = Service(self.config_mock)
        self.service.model = MagicMock(spec=nn.Cell)
        self.service.data_collector = MagicMock()
        self.service.primitive_hook_service = MagicMock()

    def tearDown(self) -> None:
        api_register.api_set_ori_func()

    def test_init(self):
        self.assertEqual(self.service.config.level, "L0")
        self.assertFalse(self.service.switch)
        self.assertFalse(self.service.should_stop_service)
        self.assertFalse(self.service.start_call)
        self.assertTrue(self.service.first_start)

    def test_check_model_valid_with_valid_cell(self):
        model = nn.Cell()
        model_list = [model]
        self.assertEqual(self.service.check_model_valid(model), model)
        self.assertEqual(self.service.check_model_valid(model_list), model_list)

    def test_check_model_valid_with_invalid_type(self):
        model = nn.Cell()
        with self.assertRaises(MsprobeException):
            self.service.check_model_valid("not a cell")
        with self.assertRaises(MsprobeException):
            self.service.check_model_valid(["not a cell", model])

    def test_update_primitive_counters(self):
        self.service.primitive_counters = {}
        self.service.update_primitive_counters("conv2d")
        self.assertEqual(self.service.primitive_counters["conv2d"], 0)
        self.service.update_primitive_counters("conv2d")
        self.assertEqual(self.service.primitive_counters["conv2d"], 1)

    @patch('msprobe.mindspore.service.create_directory')
    def test_create_dirs(self, mock_create_directory):
        self.service.current_iter = 1
        self.service.current_rank = 0
        self.service.data_collector.tasks_need_tensor_data = [Const.TENSOR]
        self.service.create_dirs()
        expected_calls = [
            ("/tmp/dump"),
            ("/tmp/dump/step1/rank0"),
            "/tmp/dump/step1/rank0/dump_tensor_data"
        ]
        mock_create_directory.assert_has_calls(
            [unittest.mock.call(path) for path in expected_calls], any_order=True)

        self.service.data_collector.update_dump_paths.assert_called_once_with(
            "/tmp/dump/step1/rank0/dump.json",
            "/tmp/dump/step1/rank0/stack.json",
            "/tmp/dump/step1/rank0/construct.json",
            "/tmp/dump/step1/rank0/dump_tensor_data",
            None
        )
        self.service.data_collector.initialize_json_file.assert_called_once_with(
            framework=Const.MS_FRAMEWORK
        )

    @patch.object(Service, 'need_end_service', return_value=False)
    def test_start_stop_cycle(self, mock_need_end_service):
        self.service.model = nn.Cell()
        with patch.object(self.service, 'register_cell_hook') as mock_register_hook:
            self.should_stop_service = False
            self.service.start(self.service.model)
            self.assertTrue(self.service.switch)
            self.service.stop()
            self.assertFalse(self.service.switch)
            mock_register_hook.assert_called_once()
            mock_need_end_service.assert_called_once()

    def test_should_execute_hook_return_false(self):
        cell = MagicMock()
        self.service.switch = False
        self.assertFalse(self.service.should_execute_hook("Module", cell, True))
        self.assertFalse(self.service.should_execute_hook("api", cell, True))

        self.service.switch = True
        cell.forward_data_collected = False
        self.assertFalse(self.service.should_execute_hook("api", cell, False))

        self.service.inner_switch = True
        self.assertFalse(self.service.should_execute_hook("Module", cell, True))

        self.service.inner_switch = False
        self.service.data_collector = None
        self.assertFalse(self.service.should_execute_hook("Module", cell, True))

    def test_should_execute_hook_return_true(self):
        cell = MagicMock()
        self.service.switch = True
        self.service.inner_switch = False
        self.service.data_collector = MagicMock()
        self.service.data_collector.data_processor = MagicMock()
        self.service.data_collector.data_processor.is_terminated = False
        self.assertTrue(self.service.should_execute_hook("Module", cell, True))

        cell.forward_data_collected = True
        self.assertTrue(self.service.should_execute_hook("api", cell, False))

    def test_need_end_service_with_high_step(self):
        self.service.config.step = [1, 2, 3]
        self.service.current_iter = 4
        self.assertTrue(self.service.need_end_service())

    def test_need_end_service_with_low_step(self):
        self.service.config.step = [1, 2, 3]
        self.service.current_iter = 2
        self.service.data_collector.data_processor.is_terminated = False
        self.assertFalse(self.service.need_end_service())

    def test_start_with_termination_condition(self):
        self.service.config.step = [1, 2, 3]
        self.service.current_iter = 4
        self.service.start()
        self.assertFalse(self.service.switch)
        self.assertTrue(self.service.should_stop_service)
        self.assertFalse(self.service.primitive_switch)

    @patch('msprobe.mindspore.service.print_tools_ends_info')
    @patch.object(Service, 'need_end_service', return_value=True)
    def test_start_with_end_service(self, mock_need_end_service, mock_print_tools_ends_info):
        self.service.start(self.service.model)
        mock_need_end_service.assert_called_once()
        mock_print_tools_ends_info.assert_called_once()
        self.assertFalse(self.service.switch)
        self.assertTrue(self.service.should_stop_service)

    @patch.object(Service, 'need_end_service', return_value=False)
    @patch.object(logger, 'info')
    @patch.object(Service, 'register_cell_hook')
    @patch.object(Service, 'register_primitive_hook')
    @patch.object(Service, 'create_dirs')
    @patch('msprobe.mindspore.service.get_rank_if_initialized', return_value=0)
    def test_start_first_time(self, mock_get_rank, mock_create_dirs, mock_register_primitive_hook,
                              mock_register_cell_hook, mock_logger, mock_need_end_service):
        self.service.first_start = True
        self.service.should_stop_service = False
        self.service.start(self.service.model)
        mock_get_rank.assert_called_once()
        mock_register_cell_hook.assert_called_once()
        mock_register_primitive_hook.assert_called_once()
        mock_need_end_service.assert_called_once()
        mock_create_dirs.assert_called_once()
        self.assertFalse(self.service.first_start)
        self.assertTrue(self.service.switch)
        self.assertTrue(self.service.primitive_switch)
        mock_logger.assert_called_with(f"Dump data will be saved in {self.service.dump_iter_dir}.")

    @patch.object(Service, 'register_primitive_hook')
    @patch.object(Service, 'register_cell_hook')
    @patch.object(Service, 'need_end_service', return_value=False)
    @patch.object(JitDump, 'set_config')
    @patch.object(JitDump, 'set_data_collector')
    @patch.object(ApiRegistry, 'api_set_hook_func')
    def test_start_with_jit_dump_enabled(self, mock_api_set_hook_func, mock_set_data_collector,
                                         mock_set_config, mock_need_end_service, mock_register_cell_hook,
                                         mock_register_primitive_hook):
        self.service.config.level = Const.LEVEL_MIX
        self.service.first_start = True
        self.service.should_stop_service = False
        self.service.start(self.service.model)
        mock_set_config.assert_called_with(self.service.config)
        mock_set_data_collector.assert_called_with(self.service.data_collector)
        mock_api_set_hook_func.assert_called_once()
        mock_need_end_service.assert_called_once()
        mock_register_cell_hook.assert_called_once()
        mock_register_primitive_hook.assert_called_once()
        self.assertTrue(JitDump.jit_dump_switch)

    def test_step_updates(self):
        CellProcessor.cell_count = {"test_api": 1}
        HOOKCell.cell_count = {"test_api": 1}
        JitDump.jit_count = {"test_api": 1}
        self.service.primitive_hook_service.primitive_counters = {"test_api": 1}
        self.service.current_iter = 0
        self.service.step()
        self.assertEqual(self.service.current_iter, 1)
        self.service.data_collector.update_iter.assert_called_once_with(1)
        self.service.data_collector.data_writer.reset_cache.assert_called_once()
        self.assertEqual(JitDump.jit_count, defaultdict(int))
        self.assertEqual((self.service.primitive_hook_service.primitive_counters), {})

    @patch.object(Service, 'should_execute_hook')
    def test_build_forward_and_backward_hooks(self, mock_should_execute_hook):
        mock_should_execute_hook.return_value = True
        self.service.data_collector = MagicMock()
        self.service.data_collector.update_api_or_module_name = MagicMock()
        self.service.data_collector.forward_data_collect = MagicMock()
        self.service.data_collector.if_return_forward_new_output = MagicMock(return_value=False)
        self.service.data_collector.backward_data_collect = MagicMock()

        mock_cell = MagicMock()
        mock_cell.mindstudio_reserved_name = "TestCell"
        mock_input = (MagicMock(),)
        mock_output = MagicMock()

        _, forward_hook, backward_hook, _ = self.service.build_hook(BaseScope.Module_Type_Module, "TestHook")

        forward_hook(mock_cell, mock_input, mock_output)
        self.service.data_collector.update_api_or_module_name.assert_called_with('TestCell')
        self.service.data_collector.forward_data_collect.assert_called()

        self.service.data_collector.reset_mock()

        mock_grad_input = (MagicMock(),)
        mock_grad_output = MagicMock()

        backward_hook(mock_cell, mock_grad_input, mock_grad_output)
        self.service.data_collector.update_api_or_module_name.assert_called_with('TestHookbackward.0')
        self.service.data_collector.backward_data_collect.assert_called()

    def test_register_primitive_hook(self):
        self.service.config.level = Const.LEVEL_MIX
        primitive_attr = ops.Add()
        primitive_name = "primitive_api"
        cell_mock = MagicMock()
        cell_mock.primitive_api = primitive_attr
        primitive_combined_name = primitive_name + Const.SEP + primitive_attr.__class__.__name__
        self.service.model.cells_and_names.return_value = [("cell_name", cell_mock)]
        self.service.register_primitive_hook()
        self.assertTrue(hasattr(primitive_attr.__class__, '__call__'))
        self.assertEqual(self.service.primitive_hook_service.wrap_primitive.call_args[0][1],
                         primitive_combined_name)

    @patch.object(ApiRegistry, 'initialize_hook')
    @patch.object(ApiRegistry, 'api_set_hook_func')
    @patch("msprobe.mindspore.service.logger.info")
    def test_register_hook_new_with_level_mix(self, mock_logger, mock_api_set_hook_func, mock_initialize_hook):
        self.service.config.level = Const.LEVEL_MIX
        self.service.register_api_hook()
        self.service.register_cell_hook()
        mock_logger.assert_called_with(f"The cell {self.service.config.task} hook function "
                                       "is successfully mounted to the model.")
        mock_api_set_hook_func.assert_called()
        mock_initialize_hook.assert_called()

    @patch.object(CellProcessor, 'node_hook')
    def test_register_hook_new_with_level_l0(self, mock_node_hook):
        global register_backward_hook_functions
        self.service.config.level = Const.LEVEL_L0
        cell_mock = MagicMock()
        self.service.model.cells_and_names.return_value = [("cell_name", cell_mock)]
        register_backward_hook_functions["pre"] = cell_mock.register_backward_pre_hook
        register_backward_hook_functions["full"] = cell_mock.register_backward_hook
        self.service.register_cell_hook()
        cell_mock.register_forward_hook.assert_called()
        cell_mock.register_backward_hook.assert_called()
        mock_node_hook.assert_called()
        register_backward_hook_functions = {}

    def test_register_hook_new_without_model_raises_exception(self):
        self.service.config.level = Const.LEVEL_L0
        self.service.model = None
        with self.assertRaises(MsprobeException):
            self.service.register_cell_hook()
