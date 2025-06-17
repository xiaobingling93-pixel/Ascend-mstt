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
import os
import re
from datetime import datetime
from mindspore import dtype as mstype, Tensor

from msprobe.mindspore.monitor.features import FUNC_MAP
from msprobe.core.common.const import MonitorConst
from msprobe.core.common.utils import is_int
from msprobe.core.common.log import logger
from msprobe.core.common.file_utils import check_file_or_directory_path


def get_single_metrics(op_list, tag, tensor, eps=1e-8, output=None):
    if output is None:
        output = {}
    if tag not in output:
        output[tag] = {}
    for op in op_list:
        func = FUNC_MAP.get(op)
        if op == "zeros":
            statistic = func(tensor, eps)
        else:
            statistic = func(tensor)
        if hasattr(statistic, "dtype") and statistic.dtype == mstype.bfloat16:
            statistic = float(statistic)
            statistic = Tensor(statistic)
        if isinstance(statistic, Tensor):
            output[tag][op] = statistic.astype(mstype.float32)
        else:
            output[tag][op] = statistic


def get_metrics(op_list, tag2tensor, eps, output=None):
    if output is None:
        output = {}
    for tag, tensor in tag2tensor.items():
        if tag not in output:
            output[tag] = {}
        get_single_metrics(op_list, tag, tensor, eps, output)
    return output


def get_summary_writer_tag_name(module_or_param_name: str, tag: str, rank):
    if rank is None:
        return f"{module_or_param_name}/{tag}"
    else:
        return f"{module_or_param_name}/rank{rank}/{tag}"


def step_accumulates_one(context, micro_batch_number):
    """
    :param context: ModuleHookContext
    :param micro_batch_number: mbs of training model.
    :return:
    """
    context.micro_step += 1
    if context.micro_step == micro_batch_number:
        context.micro_step = 0
        context.step += 1


def is_skip_step(step, start_step, step_interval, has_collect_times=0, collect_times=1e8):
    """
    If current step less than start_step or not reach step_interval, skip current step.
    :param step: current training step, int
    :param start_step: int
    :param step_interval: int
    :return: whether skip or not, bool
    """
    return step < start_step or (step - start_step) % step_interval != 0 or has_collect_times >= collect_times


def validate_ops(ops):
    if not isinstance(ops, list):
        raise TypeError("ops should be a list")
    valid_ops = []
    for op in ops:
        if op not in MonitorConst.OP_LIST:
            logger.warning(f"op {op} is not supported. Optional ops: {MonitorConst.OP_LIST}")
            continue
        valid_ops.append(op)
    if not valid_ops:
        default_op = MonitorConst.OP_LIST[0]
        valid_ops.append(default_op)
        logger.info(f"There is no valid ops, default op {default_op} is used")
    # 增加默认shape和dtype参数
    if "shape" not in valid_ops:
        valid_ops.append("shape")
    if "dtype" not in valid_ops:
        valid_ops.append("dtype")
    return valid_ops


def validate_ranks(ranks):
    if not isinstance(ranks, list):
        raise TypeError("module_ranks should be a list")
    for rank in ranks:
        if not isinstance(rank, int):
            raise TypeError(f"element in module_ranks should be a int, get {type(rank)}")


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


def validate_param_distribution(param_distribution):
    if not isinstance(param_distribution, bool):
        raise TypeError('param_distribution should be a bool')


def validate_cc_distribution(cc_distribution):
    if not isinstance(cc_distribution, dict):
        raise TypeError('cc_distribution should be a dictionary')
    expected_keys = {
        'enable': bool,
        'cc_codeline': list,
        'cc_pre_hook': bool,
        'cc_log_only': bool
    }
    for key, value in cc_distribution.items():
        if key in expected_keys:
            if not isinstance(value, expected_keys[key]):
                raise TypeError(f'cc_distribution {key} should be a {expected_keys[key].__name__}')
        else:
            raise TypeError(f'{key} of cc_distribution is not supported.')


def validate_alert(alert):
    if not isinstance(alert, dict):
        raise TypeError('alert should be a dictionary')
    rules = alert.get('rules')
    if rules and isinstance(rules, list):
        for rule in rules:
            rule_name = rule.get("rule_name")
            if rule_name and rule_name not in MonitorConst.RULE_NAME:
                raise TypeError(f"{rule_name} is not supported")
            args = rule.get("args")
            if args and isinstance(args, dict):
                threshold = args.get("threshold")
                if not isinstance(threshold, (float, int)) or threshold < 0:
                    raise TypeError('threshold must be float and not less than 0')
    dump = alert.get('dump')
    if dump and not isinstance(dump, bool):
        raise TypeError('dump must be bool.')


def validate_step_count_per_record(step_count_per_record):
    if not is_int(step_count_per_record):
        raise TypeError('step_count_per_record must be int.')
    if step_count_per_record < 1:
        raise ValueError("step_count_per_record must greater than 0")
    if step_count_per_record > 1e6:
        raise ValueError("step_count_per_record must smaller than 1e6")


def validate_start_step(start_step):
    if not is_int(start_step):
        raise TypeError('start_step must be int.')
    if start_step < 0:
        raise ValueError("start_step must greater than 0")
    if start_step > 1e8:
        raise ValueError("start_step must smaller than 1e8")


def validate_step_interval(step_interval):
    if not is_int(step_interval):
        raise TypeError('step_interval must be int.')
    if step_interval < 1:
        raise ValueError("step_interval must greater than 1")
    if step_interval > 1e8:
        raise ValueError("step_interval must smaller than 1e8")


def validate_collect_times(collect_times):
    if not is_int(collect_times):
        raise TypeError('collect_times must be int.')
    if collect_times < 1:
        raise ValueError("collect_times must greater than 1")


def validate_dynamic_on(dynamic_on):
    if not isinstance(dynamic_on, bool):
        raise TypeError('dynamic_on should be a bool')


def validate_monitor_mbs_grad(monitor_mbs_grad):
    if not isinstance(monitor_mbs_grad, bool):
        logger.warning(f'monitor_mbs_grad should be a bool, actual value is {monitor_mbs_grad}.')
        return False
    return monitor_mbs_grad


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

    param_distribution = config.get('param_distribution', False)
    validate_param_distribution(param_distribution)

    cc_distribution = config.get('cc_distribution', {})
    validate_cc_distribution(cc_distribution)

    alert = config.get('alert', {})
    validate_alert(alert)

    step_count_per_record = config.get('step_count_per_record', 1)
    validate_step_count_per_record(step_count_per_record)

    start_step = config.get('start_step', 0)
    validate_start_step(start_step)

    step_interval = config.get('step_interval', 1)
    validate_step_interval(step_interval)

    collect_times = config.get('collect_times', int(1e8))
    validate_collect_times(collect_times)

    config["monitor_mbs_grad"] = validate_monitor_mbs_grad(config.get('monitor_mbs_grad', False))

    dynamic_on = config.get('dynamic_on', False)
    validate_dynamic_on(dynamic_on)

    if not targets:
        if xy_distribution:
            config["all_xy"] = True
        config["targets"] = {"": {}}
        config["is_select"] = False
    else:
        config["is_select"] = True


def time_str2time_digit(time_str):
    time_format = '%b%d_%H-%M-%S'
    try:
        time_digit = datetime.strptime(time_str, time_format)
    except Exception as e:
        raise RuntimeError(f"illegal timestamp: {time_str}, timestamp should be prefix \
                           of existing output dirpath, like 'Dec03_21-34-40'.") from e
    return time_digit


def get_target_output_dir(monitor_path, time_start, time_end):
    check_file_or_directory_path(monitor_path, isdir=True)
    time_start = time_str2time_digit(time_start) if time_start is not None else time_start
    time_end = time_str2time_digit(time_end) if time_end is not None else time_end
    if time_start and time_end and time_start > time_end:
        raise ValueError(f"time_start({time_start}) greater than time_end({time_end})")
    result = {}
    for dirname in os.listdir(monitor_path):
        match = re.match(MonitorConst.OUTPUT_DIR_PATTERN, dirname)
        if not match:
            continue
        time_tag = match.group(1)
        rank = match.group(2)
        target_time = time_str2time_digit(time_tag)
        start_ok = time_start is None or target_time >= time_start
        end_ok = time_end is None or target_time <= time_end
        if start_ok and end_ok:
            result[rank] = os.path.join(monitor_path, dirname)
    return result
