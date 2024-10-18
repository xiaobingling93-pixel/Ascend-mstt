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

from msprobe.core.grad_probe.constant import GradConst
from msprobe.mindspore.grad_probe.global_context import grad_context
from msprobe.mindspore.grad_probe.grad_analyzer import csv_generator
from msprobe.mindspore.grad_probe.hook import hook_optimizer


class GradientMonitor:

    def __init__(self, common_dict, task_config):
        config = {}
        config[GradConst.OUTPUT_PATH] = common_dict.dump_path
        config[GradConst.STEP] = common_dict.step
        config[GradConst.RANK] = common_dict.rank
        config[GradConst.PARAM_LIST] = task_config.param_list
        config[GradConst.LEVEL] = task_config.grad_level
        config[GradConst.BOUNDS] = task_config.bounds
        self.config = config
        grad_context.init_context(self.config)

    @staticmethod
    def monitor(opt):
        csv_generator.init(grad_context)
        hook_optimizer(opt)

    @staticmethod
    def stop():
        csv_generator.stop()
