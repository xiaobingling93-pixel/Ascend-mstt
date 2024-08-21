# Copyright 2024 Huawei Technologies Co., Ltd
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
# limitations under the License.
# ============================================================================

import mindspore as ms

from msprobe.core.common.exceptions import DistributedNotInitializedError
from msprobe.core.common.file_check import path_len_exceeds_limit
from msprobe.core.common.utils import save_npy
from msprobe.core.common.log import logger


def get_rank_if_initialized():
    if ms.communication.GlobalComm.INITED:
        return ms.communication.get_rank()
    else:
        raise DistributedNotInitializedError("mindspore distributed environment is not initialized")


def convert_bf16_to_fp32(tensor):
    if tensor.dtype == ms.bfloat16:
        tensor = tensor.to(ms.float32)
    return tensor


def save_tensor_as_npy(tensor, file_path):
    if not path_len_exceeds_limit(file_path):
        tensor = convert_bf16_to_fp32(tensor)
        saved_tensor = tensor.asnumpy()
        save_npy(saved_tensor, file_path)
    else:
        logger.warning(f'The file path {file_path} length exceeds limit.')


class MsprobeStep(ms.train.Callback):

    def __init__(self, debugger):
        super(MsprobeStep, self).__init__()
        self.debugger = debugger

    def on_train_step_begin(self, run_context):
        self.debugger.start()

    def on_train_step_end(self, run_context):
        self.debugger.stop()
        self.debugger.step()
