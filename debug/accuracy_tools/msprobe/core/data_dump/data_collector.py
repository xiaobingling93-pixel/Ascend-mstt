# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
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

import os

from msprobe.core.data_dump.scope import build_scope, ListScope
from msprobe.core.data_dump.json_writer import DataWriter
from msprobe.core.common.log import logger
from msprobe.core.common.const import Const
from msprobe.core.data_dump.data_processor.factory import DataProcessorFactory


def build_data_collector(config):
    return DataCollector(config)


class DataCollector:
    multi_output_apis = ["_sort_", "npu_flash_attention"]
    tasks_need_tensor_data = [Const.OVERFLOW_CHECK, Const.TENSOR, Const.FREE_BENCHMARK]
    level_without_construct = [Const.LEVEL_L1, Const.LEVEL_L2]

    def __init__(self, config):
        self.config = config
        self.data_writer = DataWriter()
        self.data_processor = DataProcessorFactory.create_processor(self.config, self.data_writer)
        self.module_processor = DataProcessorFactory.get_module_processor(self.config.framework)
        self.module_count = {}
        if self.config.task == Const.FREE_BENCHMARK:
            self.scope = build_scope(ListScope, self.config.scope, self.config.list)
        else:
            self.scope = build_scope(None, self.config.scope, self.config.list)

    def __del__(self):
        self.write_json()

    @property
    def dump_data_dir(self):
        return self.data_writer.dump_tensor_data_dir

    @property
    def dump_file_path(self):
        return self.data_writer.dump_file_path

    @staticmethod
    def check_scope_and_pid(scope, name, pid):
        return (not scope or scope.check(name)) and pid == os.getpid()

    @staticmethod
    def is_inplace(module):
        return getattr(module, "op_is_inplace", False)

    def if_return_forward_new_output(self):
        return self.data_processor.if_return_forward_new_output()

    def get_forward_new_output(self):
        return self.data_processor.get_forward_new_output()

    def update_api_or_module_name(self, api_or_module_name):
        self.data_processor.update_api_or_module_name(api_or_module_name)

    def write_json(self):
        self.data_writer.write_json()

    def update_data(self, name, data_info):
        msg = f"msprobe is collecting data on {name}."
        if self.config.task == Const.OVERFLOW_CHECK:
            if self.data_processor.has_overflow:
                msg += " Overflow detected."
                logger.warning(msg)
                self.data_writer.update_data(data_info)
            return
        logger.debug(msg)
        self.data_writer.update_data(data_info)

    def pre_forward_data_collect(self, name, module, pid, module_input_output):
        backward_name = name.replace(Const.FORWARD, Const.BACKWARD)
        if self.check_scope_and_pid(self.scope, backward_name, pid):
            self.data_processor.analyze_pre_forward(backward_name, module, module_input_output)
        if not self.is_inplace(module) or not self.check_scope_and_pid(self.scope, name, pid):
            return
        logger.info(f"API {name} is inplace.")
        data_info = self.data_processor.analyze_pre_forward_inplace(name, module_input_output)
        self.handle_data(name, data_info, flush=self.data_processor.is_terminated)

    def forward_data_collect(self, name, module, pid, module_input_output):
        self.update_construct(name)
        if not self.check_scope_and_pid(self.scope, name, pid):
            return

        if not self.is_inplace(module):
            data_info = self.data_processor.analyze_forward(name, module, module_input_output)
        else:
            data_info = self.data_processor.analyze_forward_inplace(name, module_input_output)
        if self.config.level == "L2":
            return
        self.data_writer.update_stack(self.data_processor.analyze_api_call_stack(name))
        self.handle_data(name, data_info, flush=self.data_processor.is_terminated)

    def backward_data_collect(self, name, module, pid, module_input_output):
        self.update_construct(name)
        if not self.check_scope_and_pid(self.scope, name, pid):
            return

        data_info = self.data_processor.analyze_backward(name, module, module_input_output)
        self.handle_data(name, data_info, flush=self.data_processor.is_terminated)

    def backward_input_data_collect(self, name, module, pid, module_input_output):
        self.update_construct(name)
        if not self.check_scope_and_pid(self.scope, name, pid):
            return

        data_info = self.data_processor.analyze_backward_input(name, module, module_input_output)
        self.handle_data(name, data_info)

    def backward_output_data_collect(self, name, module, pid, module_input_output):
        self.update_construct(name)
        if not self.check_scope_and_pid(self.scope, name, pid):
            return

        data_info = self.data_processor.analyze_backward_output(name, module, module_input_output)
        self.handle_data(name, data_info)

    def update_construct(self, name):
        if self.config.level not in DataCollector.level_without_construct:
            self.data_writer.update_construct({name: self.module_processor.api_parent_node})
            self.data_writer.update_construct(self.module_processor.module_node)

    def handle_data(self, name, data_info, flush=False):
        if data_info:
            self.update_data(name, data_info)
        if not flush:
            self.data_writer.flush_data_periodically()
        else:
            self.write_json()

    def update_dump_paths(self, *args):
        self.data_writer.update_dump_paths(*args)
        self.data_writer.initialize_json_file(task=self.config.task, level=self.config.level)

    def update_iter(self, current_iter):
        self.data_processor.update_iter(current_iter)
