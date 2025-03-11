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

try:
    from msprobe.lib import _msprobe_c
    os.environ["MS_HOOK_ENABLE"] = "on"
    dir_path = os.path.dirname(os.path.abspath(_msprobe_c.__file__))
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    # 更新环境变量
    os.environ["LD_LIBRARY_PATH"] = f"{current_ld_path}:{dir_path}"
except ImportError:
    from .common.log import logger
    logger.info("Module _msprobe_c has not been installed. L2-Dump may not work normally.")

from msprobe.mindspore.debugger.precision_debugger import PrecisionDebugger
from msprobe.mindspore.common.utils import seed_all, MsprobeStep, MsprobeInitStep
from msprobe.mindspore.monitor.module_hook import TrainerMon
