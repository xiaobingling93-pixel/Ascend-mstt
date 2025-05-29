# Copyright (c) 2025-2025, Huawei Technologies Co., Ltd.
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

import mindspore as ms
from mindspore import hal, ops, Tensor
from mindspore.ops.primitive import _run_op

from msprobe.core.common.const import Const as CoreConst
from msprobe.mindspore.common.const import Const
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.debugger.debugger_config import DebuggerConfig
import msprobe.mindspore.dump.cell_dump_process as cellDumperWithDumpGradient
import msprobe.mindspore.dump.cell_dump_with_insert_gradient as cellDumperWithInsertGradient
from msprobe.mindspore.runtime import Runtime

tensordump_flag = True
try:
    from mindspore._c_expression import _tensordump_set_step
except ImportError:
    tensordump_flag = False


class GraphModeCellDump:
    def __init__(self, config: DebuggerConfig, model, strict=True):
        self.net = model
        self.white_list = []
        self.black_list = []
        self.execution_mode = config.execution_mode
        self.dump_path = config.dump_path if config.dump_path else "./"
        self.rank = config.rank
        self.step = config.step
        self.scope = config.scope
        self.list = config.list
        self.data_mode = config.data_mode
        self.file_format = config.file_format
        self.check_config(strict)
        self.set_step()

    @staticmethod
    def step():
        hal.synchronize()
        temp_tensor = ms.Tensor([1], dtype=ms.float32)
        step_flag = "<tensordump-update-step>"
        _run_op(ops.TensorDump(), "TensorDump", (step_flag, temp_tensor))
        ops.tensordump(step_flag, temp_tensor)

    def check_config(self, strict):
        if not self.net:
            raise Exception("The model is empty and cell dump is not enabled.")

        if strict:
            if self.rank:
                raise Exception("In graph mode, cell dump does not currently support specifying rank.")
            if self.scope:
                raise Exception("In graph mode, cell dump does not currently support specifying scope.")
            if self.list:
                raise Exception("In graph mode, cell dump does not currently support specifying list.")
            if len(self.data_mode) != 1 or self.data_mode[0] not in Const.GRAPH_CELL_DUMP_DATA_MODE_LIST:
                raise Exception("In graph mode and cell dump, data_mode must be one of all, forword, backword.")
            if self.file_format != []:
                logger.warning("In graph mode, cell dump does not currently support specifying file_format."
                               " The file will be stored in npy format.")
        else:
            self.rank = []
            self.scope = []
            self.list = []
            self.file_format = []
            if len(self.data_mode) != 1 or self.data_mode[0] not in Const.GRAPH_CELL_DUMP_DATA_MODE_LIST:
                self.data_mode = [CoreConst.ALL]

        return True

    def set_step(self):
        if tensordump_flag:
            _tensordump_set_step(self.step)
        else:
            raise Exception(
                "Importing _tensordump_set_step failed, "
                "please use the latest version package of MindSpore."
            )

    def handle(self):
        os.environ['MS_JIT_MODULES'] = 'msprobe'

        if Runtime.run_mode == Const.PYNATIVE_GRAPH_MODE:
            dump_path = os.path.join(self.dump_path, Const.GRAPH_MODE)
        else:
            dump_path = self.dump_path

        cell_dumper = cellDumperWithDumpGradient

        if self.execution_mode == Const.PYNATIVE_MODE:
            enable_dump_gradient = hasattr(ops, 'DumpGradient')
            if hasattr(ops, 'DumpGradient'):
                try:
                    ops.DumpGradient()('grad.npy', Tensor([0], dtype=ms.float32), 'in')
                except Exception:
                    enable_dump_gradient = False
                    logger.warning('the DumpGradient operator failed to execute.')
            if not enable_dump_gradient:
                cell_dumper = cellDumperWithInsertGradient

        cell_dumper.start(
            net=self.net,
            dump_path=dump_path,
            data_mode=self.data_mode[0]
        )
