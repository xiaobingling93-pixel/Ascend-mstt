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
import time
import sys
import re
from datetime import timezone, timedelta
from functools import wraps
from torch import distributed as dist

from msprobe.core.common.const import MonitorConst
from msprobe.core.common.log import logger

FILE_MAX_SIZE = 10 * 1024 * 1024 * 1024
FILE_NAME_MAX_LENGTH = 255
DIRECTORY_MAX_LENGTH = 4096
FILE_NAME_VALID_PATTERN = r"^[a-zA-Z0-9_.:/-]+$"

beijing_tz = timezone(timedelta(hours=8))


class MsgConst:
    """
    Class for log messages const
    """
    SPECIAL_CHAR = ["\n", "\r", "\u007F", "\b", "\f", "\t", "\u000B", "%08", "%0a", "%0b", "%0c", "%0d", "%7f"]


def filter_special_chars(func):
    @wraps(func)
    def func_level(msg):
        for char in MsgConst.SPECIAL_CHAR:
            msg = msg.replace(char, '_')
        return func(msg)

    return func_level


def get_param_struct(param):
    if isinstance(param, tuple):
        return f"tuple[{len(param)}]"
    if isinstance(param, list):
        return f"list[{len(param)}]"
    return "tensor"


def validate_ops(ops):
    if not isinstance(ops, list):
        raise TypeError("ops should be a list")
    if not ops:
        raise TypeError(f"specify ops to calculate metrics. Optional ops: {MonitorConst.OP_LIST}")

    valid_ops = []
    for op in ops:
        if op not in MonitorConst.OP_LIST:
            raise ValueError(f"op {op} is not supported. Optional ops: {MonitorConst.OP_LIST}")
        else:
            valid_ops.append(op)
    return valid_ops


def validate_ranks(ranks):
    world_size = dist.get_world_size()
    if not isinstance(ranks, list):
        raise TypeError("module_ranks should be a list")
    for rank in ranks:
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise TypeError(f"element in module_ranks should be a int, get {type(rank)}")
        if rank < 0 or rank >= world_size:
            logger.warning(f"rank {rank} is beyond world size [0, {world_size - 1}] and will be ignored")


def validate_targets(targets):
    if not isinstance(targets, dict):
        raise TypeError('targets in config.json should be a dict')
    for module_name, field in targets.items():
        if not isinstance(module_name, str):
            raise TypeError('key of targets should be module_name[str] in config.json')
        if not isinstance(field, dict):
            raise TypeError('values of targets should be cared filed e.g. {"input": "tensor"} in config.json')

def validate_print_struct(print_struct):
    if not isinstance(print_struct, bool):
        raise TypeError("print_struct should be a bool")
    
def validate_ur_distribution(ur_distribution):
    if not isinstance(ur_distribution, bool):
        raise TypeError('ur_distribution should be a bool')

def validate_xy_distribution(xy_distribution):
    if not isinstance(xy_distribution, bool):
        raise TypeError('xy_distribution should be a bool')

def validate_wg_distribution(wg_distribution):
    if not isinstance(wg_distribution, bool):
        raise TypeError('wg_distribution should be a bool')

def validate_mg_distribution(mg_distribution):
    if not isinstance(mg_distribution, bool):
        raise TypeError('mg_distribution should be a bool')

def validate_cc_distribution(cc_distribution):
    if not isinstance(cc_distribution, dict):
        raise TypeError('cc_distribution should be a dictionary')
    for key, value in cc_distribution.items():
        if key == 'enable':
            if not isinstance(value, bool):
                raise TypeError('cc_distribution enable should be a bool')
        elif key == 'cc_codeline':
            if not isinstance(value, list):
                raise TypeError('cc_distribution cc_codeline should be a list')
        elif key == 'cc_pre_hook':
            if not isinstance(value, bool):
                raise TypeError('cc_distribution cc_pre_hook should be a bool')
        elif key == 'cc_log_only':
            if not isinstance(value, bool):
                raise TypeError('cc_distribution cc_log_only should be a bool')
        else:
            raise TypeError(f'{key} of cc_distribution is not supported.')

def validate_alert(alert):
    if not isinstance(alert, dict):
        raise TypeError('alert should be a dictionary')
    for key, value in alert.items():
        if key == 'rules':
            if len(value) == 0:
                continue
            elif value[0]['rule_name'] not in MonitorConst.RULE_NAME:
                raise TypeError(f"{value[0]['rule_name']} is not supported") 
            elif not isinstance(value[0]['args']['threshold'], float) or value[0]['args']['threshold'] < 0:
                raise TypeError('threshold must be float and not less than 0')
        else:
            if len(value) > 0 and value['recipient'] not in ['database', 'email']:
                raise TypeError('recipient must be database or email')

def validate_config(config):
    config['ops'] = validate_ops(config.get('ops', []))

    eps = config.get('eps', 1e-8)
    if not isinstance(eps, float):
        raise TypeError("eps should be a float")

    ranks = config.get("module_ranks", [])
    validate_ranks(ranks)

    targets = config.get("targets", {})
    validate_targets(targets)

    print_struct = config.get('print_struct', False)
    validate_print_struct(print_struct)

    ur_distribution = config.get('ur_distribution', False)
    validate_ur_distribution(ur_distribution)

    xy_distribution = config.get('xy_distribution', False)
    validate_xy_distribution(xy_distribution)

    wg_distribution = config.get('wg_distribution', False)
    validate_wg_distribution(wg_distribution)

    mg_distribution = config.get('mg_distribution', False)
    validate_mg_distribution(mg_distribution)

    cc_distribution = config.get('cc_distribution', {})
    validate_cc_distribution(cc_distribution)

    alert = config.get('alert', {})
    validate_alert(alert)
