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
