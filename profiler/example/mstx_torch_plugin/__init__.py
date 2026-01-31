# -------------------------------------------------------------------------
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#    http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import sys
import logging
from .mstx_torch_plugin import apply_mstx_patch

logger = logging.getLogger()
requirements_module_list = ['torch', 'torch_npu']

enable_mstx_torch = True
for module_name in requirements_module_list:
    if module_name not in sys.modules:
        enable_mstx_torch = False
        logger.error(f"mstx_torch_plugin not enabled, please ensure that {module_name} has been installed.")

if enable_mstx_torch:
    apply_mstx_patch()
