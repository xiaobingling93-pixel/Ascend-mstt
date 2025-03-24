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
import uuid
from collections import defaultdict
from datetime import datetime

import pytz
import mindspore as ms
from mindspore import Tensor, mint
from mindspore import nn, _no_grad
from mindspore.communication import get_rank

from msprobe.core.common.log import logger
from msprobe.core.common.const import MonitorConst
from msprobe.core.common.file_utils import load_json, save_json
from msprobe.mindspore.monitor.utils import get_summary_writer_tag_name, validate_config, step_accumulates_one, \
    is_skip_step, get_metrics, get_single_metrics, get_target_output_dir
from msprobe.mindspore.monitor.module_spec_verifier import validate_config_spec
from msprobe.mindspore.monitor.anomaly_detect import AnomalyScanner, AnomalyDataFactory, \
    CSVWriterWithAD, BaseWriterWithAD, WriterInput
from msprobe.mindspore.monitor.distributed.wrap_distributed import api_register, create_hooks, op_aggregate, \
    get_process_group

FORMAT_MAPPING = {
    MonitorConst.CSV: CSVWriterWithAD,
    MonitorConst.API: BaseWriterWithAD
}


def get_output_base_dir():
    return os.getenv(MonitorConst.MONITOR_OUTPUT_DIR, MonitorConst.DEFAULT_MONITOR_OUTPUT_DIR)


def get_param_struct(param):
    res = {}
    if isinstance(param, (tuple, list)):
        res['config'] = f'{type(param).__name__}[{len(param)}]'
        for i, x in enumerate(param):
            res[i] = f'size={tuple(x.shape)}, dtype={x.dtype}' if isinstance(x, Tensor) else f'{type(x)}'
    elif isinstance(param, Tensor):
        res['config'] = 'tensor'
        res['tensor'] = f'size={tuple(param.shape)}, dtype={param.dtype}'
    else:
        res['config'] = f'{type(param)}'
        logger.warning(f'Not support type({type(param)}) now, please check the type of param {param}')
    return res


def param_is_not_tensor_parallel_duplicate(param, tp_group):
    return (hasattr(param, 'tensor_model_parallel') and param.tensor_model_parallel) or (
            mint.distributed.get_rank(group=tp_group) == 0
    )


def param_is_data_parallel_duplicate(dp_group):
    return mint.distributed.get_rank(group=dp_group) != 0


def squash_param_name(param_name):
    for pattern in ['layers?\.(.*)', 'embeddings?\.(.*)', 'final.*', 'output.*', 'norm.*']:
        match = re.findall(pattern, param_name)
        if match:
            return match[0]
    return param_name


# Used For Module Forward & Backward Collect
class ModuleHookContext:
    def __init__(self, module_name) -> None:
        self.step = 0
        self.micro_step = 0
        self.actv = defaultdict(dict)
        self.actvgrad = []
        self.module_name = module_name
        self.struct = {}
        self.format_by_arg = {}
        self.verified = False
        self.focused_in_col = 0
        self.focused_out_col = 0
        self.ignore_in = False  # no need to care when no key 'input' or 'input_grad' found

    def set_format_by_arg(self, key_name: str, target_config: dict):
        cared = target_config.get(self.module_name, self.struct)
        if key_name in cared:
            if isinstance(cared[key_name], dict):
                # current cared is self.struct
                config = cared[key_name].get('config')
                self.format_by_arg[key_name] = config
            else:
                # current cared is target_config[self.module_name]
                self.format_by_arg[key_name] = cared[key_name]
        elif key_name in ['input', 'input_grad']:
            self.ignore_in = True

    def reset(self):
        self.actv.clear()
        self.actvgrad.clear()


start_step = 0


# Used For Optimizer Weight Grad & M/V Collect
class OptimizerContext:
    def __init__(self) -> None:
        self.step = start_step
        self.param_mg_direction = defaultdict(float)
        self.param_adam_update = defaultdict()
        self.param_adam_ratio = defaultdict()
        self.param_weight_grad = defaultdict()
        self.param_exp_avg = defaultdict()
        self.exp_avg_metric = {}
        self.param_exp_avg_sq = defaultdict()
        self.exp_avg_sq_metric = {}
        self.metric_dict = {}
        self.param_metric = {}

    def reset(self) -> None:
        self.param_mg_direction.clear()
        self.param_adam_update.clear()
        self.param_adam_ratio.clear()
        self.param_weight_grad.clear()
        self.param_exp_avg.clear()
        self.exp_avg_metric.clear()
        self.param_exp_avg_sq.clear()
        self.exp_avg_sq_metric.clear()
        self.metric_dict.clear()
        self.param_metric.clear()


# Used For Weight Grad Collect
class GradContext:
    def __init__(self) -> None:
        self.pre = {}
        self.post = {}
        self.acc_metric = {}
        self.acc = {}
        self.actv = {}

    def reset(self):
        self.pre.clear()
        self.post.clear()
        self.acc_metric.clear()
        self.acc.clear()
        self.actv.clear()


class CommunicationContext:
    def __init__(self) -> None:
        self.data = {}

    @staticmethod
    def _agg(data):
        aggregated_data = {}
        for tag, op2tensorlist in data.items():
            aggregated_data[tag] = {}
            for op, tensorlist in op2tensorlist.items():
                aggregated_data[tag][op] = op_aggregate(op, tensorlist)
        return aggregated_data

    def reset(self):
        self.data = {}

    def aggregate(self):
        self.data = self._agg(self.data)


class TrainerMon:
    def __init__(self, config_file_path, process_group=None, params_have_main_grad=True) -> None:
        # TYPE1: 只在这里初始化的变量, 不会随着训练中途config配置改变而重置
        self.config_file_path = config_file_path
        self.process_group = process_group
        self.params_have_main_grad = params_have_main_grad
        self.config_timestamp = 0  # 后面有校验时间戳, 首次监控无需为了更新config文件时间戳而去改, 可通过dynamic_on开关直接打开
        self.config = load_json(config_file_path)
        validate_config(self.config)

        local_tz = pytz.timezone("Asia/Shanghai")  # 根据需要调整为目标时区
        cur_time = datetime.now(local_tz).strftime('%b%d_%H-%M-%S')
        self.unique_id = str(uuid.uuid4())[:8]
        self.output_base_dir = get_output_base_dir()
        time_tags = self.config.get("append_output", [])
        try:
            self.rank = get_rank()
            if time_tags:
                output_append_dirs = get_target_output_dir(self.output_base_dir, time_tags[0], time_tags[1])
                if str(self.rank) in output_append_dirs:
                    self.tensorboard_dir = output_append_dirs[str(self.rank)]
                    logger.info(f"Append rank({self.rank}) result to {self.tensorboard_dir}")
            else:
                self.tensorboard_dir = os.path.join(self.output_base_dir,
                                                    f"{cur_time}-rank{self.rank}-{self.unique_id}")
        except Exception as e:
            self.rank = 0
            self.tensorboard_dir = os.path.join(self.output_base_dir, f"{cur_time}-rank{self.rank}-{self.unique_id}")

        self.pp_stage = 0
        self.group_mates = [0]

        # TYPE2: 只会在set_monitor()主调中赋值的变量
        self.model = None
        self.vpp = False
        self.dp_group = None
        self.tp_group = None
        self.micro_batch_number = 1

        # TYPE3: 会随着训练中途config配置更新或监控状态改变而重置的变量
        self.module_fwd_hook_context_by_module = defaultdict(ModuleHookContext)
        self.module_bwd_hook_context_by_module = defaultdict(ModuleHookContext)
        self.optimizer_context = defaultdict(OptimizerContext)
        self.cc_context = defaultdict(CommunicationContext)
        self.grad_context = GradContext()
        self.handles = defaultdict(list)
        self.param2name = defaultdict(str)
        self.name2index = defaultdict()
        self.name2indices = defaultdict()
        self.name2param = {}
        self.duplicate_param = {}
        self.name2tag = {}
        self.param_name_call_id = {}
        self.call_id = 0
        self.module_struct = defaultdict(dict)
        self.grad_accs = []
        self.weight_hooked = False
        self.optimizer_hooked = False
        self.param_registered = False
        self.struct_printed = False

        # 动静态区分
        self.dynamic_enable = os.getenv("DYNAMIC_MONITOR", 'False').lower() == 'true'
        if self.dynamic_enable:
            logger.warning(f"DYNAMIC_MONITOR is set, "
                           f"please make sure you have 'dynamic_on' and 'collect_times' in {self.config_file_path}")
            self.monitoring = False
        else:
            self.set_config()
            # 静态且collect_times>0时在第0步self.monitoring就可以True, 动态默认在下一步开启
            if self.collect_times > 0:
                self.monitoring = True

    def set_config(self):
        self.start_step = self.config.get("start_step", 0)
        self.collect_times = self.config.get("collect_times", 100000000)  # 默认大值, 目的是一直采集
        self.step_interval = self.config.get("step_interval", 1)
        self.has_collect_times = 0  # 重设采集计数器
        self.print_struct = self.config.get("print_struct", False)
        self.targets = self.config.get("targets", None)
        self.is_select = self.config.get("is_select", False)
        self.module_rank_list = self.config.get("module_ranks", [])
        self.format = self.config.get('format', MonitorConst.CSV)  # only csv supported in mindspore
        self.eps = self.config.get('eps', 1e-8)
        self.ops = self.config.get('ops', [])  # monitor mean/max/norm/min/nan...
        self.ndigits = self.config.get('ndigits', 6)
        self.all_xy = self.config.get('all_xy', False)
        self.xy_distribution = self.config.get('xy_distribution', False)
        self.forward_only = self.config.get('forward_only', False)
        self.backward_only = self.config.get('backward_only', False)
        self.ur_distribution = self.config.get('ur_distribution', False)  # vector and ratio vector of adam
        self.mv_distribution = self.config.get("mv_distribution", False)  # m/v of adam
        self.wg_distribution = self.config.get("wg_distribution", False)
        self.param_distribution = self.config.get("param_distribution", False)
        self.mg_direction = self.config.get('mg_direction', False)  # main grad direction
        self.cc_distribution = self.config.get("cc_distribution", {})  # communication ops
        if not self.cc_distribution.get('enable', False):
            self.cc_log_only = False
        else:
            self.cc_codeline = self.cc_distribution.get('cc_codeline', [])
            self.cc_log_only = self.cc_distribution.get('cc_log_only', False)
            self.cc_logged_stack = defaultdict(set)
            self.cc_pre_hook = self.cc_distribution.get('cc_pre_hook', False)
        self.common_info()

        # 初始化AnomalyData工厂
        alert_setting = self.config.get('alert', {"rules": []})
        self.alert_rules = AnomalyScanner.load_rules(alert_setting["rules"])
        self.anomaly_data_factory = None
        if alert_setting.get('dump', False):
            self.anomaly_data_factory = AnomalyDataFactory(self.rank, self.pp_stage, self.group_mates)

        # 初始化writer, 创建输出目录
        if self.format not in FORMAT_MAPPING:
            logger.error(f"Unsupported format: {self.format}, use default format: {MonitorConst.CSV}")
            self.format = MonitorConst.CSV
        writer = FORMAT_MAPPING[self.format]
        self.step_count_per_record = self.config.get('step_count_per_record', 1)
        self.summary_writer = writer(
            WriterInput(
                self.tensorboard_dir,
                self.alert_rules,
                self.unique_id,
                self.anomaly_data_factory,
                self.ndigits,
                self.step_count_per_record
            )
        )

    def common_info(self):
        if not self.xy_distribution:
            logger.info("> module input/output input_grad/output_grad is not monitored. ")
        if self.forward_only:
            logger.info("> only module forward is monitored. ")
        if not self.ur_distribution:
            logger.info("> update vector and ratio vector of adam is not monitored. ")
        if not self.mv_distribution:
            logger.info("> momentum and variance of adam is not monitored. ")
        if not self.wg_distribution:
            logger.info("> weight grad of specified module is not monitored. ")
        if not self.mg_direction:
            logger.info('> grad and momentum direction will not be compared.')
        if not self.cc_distribution.get('enable', False):
            logger.info("> cc operator is not monitored.")

    def set_monitor(
            self,
            model,
            optimizer,
            grad_acc_steps=1,
            tp_group=None,
            dp_group=None,
            start_iteration=0
    ):
        global start_step
        start_step = start_iteration
        self.micro_batch_number = grad_acc_steps
        self.dp_group = dp_group
        self.tp_group = tp_group
        self.hook_step_final(optimizer)
        if not isinstance(model, list):
            model = [model]
        self.model = model
        if len(model) > 1:
            self.vpp = True
            logger.info('vpp enabled')
        if not self.dynamic_enable:
            self.register_hooks(optimizer)

    def hook_step_final(self, optimizer):
        def step_final_hook(optimizer, *args, **kwargs):
            context = self.optimizer_context[optimizer]
            # 静态在第0步就可以保存, 动态在第0步不可以, 因为动态设计的就是重置后下一步开启, 第0步的self.monitoring还是False
            if self.monitoring:
                module_rank_valid = self.is_target_rank()
                step_condition = (context.step >= self.start_step and (
                            context.step - self.start_step) % self.step_interval == 0)
                if module_rank_valid and step_condition:
                    self.has_collect_times += 1
                    self.write_xy_tb(context.step)
                    self.write_grad_tb(context.step)
                    self.write_mv_tb(context)
                    self.write_param_tb(context)

                    if context.metric_dict:
                        self.summary_writer.write_metrics(self.ops, context.metric_dict, context.step, 'other')
                    context.metric_dict.clear()

                    self.summary_writer.clear_anomalies()
                    self.call_id = 0
                    self.param_name_call_id.clear()

                    if self.has_collect_times >= self.collect_times:
                        self._remove_all_hooks_final(optimizer)

            context.step += 1
            self.dynamic_monitor(optimizer)

        optimizer.register_forward_hook(step_final_hook)
        return

    def dynamic_monitor(self, optimizer):
        """
        If dynamic monitor enabled and config.json updated,
        remove hooks and register new hooks according to new configuration.
        """
        context = self.optimizer_context[optimizer]
        if not self.dynamic_enable:
            return
        try:
            # 如果文件时间戳没变, 可以不读取节省时间
            config_timestamp = os.path.getmtime(self.config_file_path)
            if config_timestamp == self.config_timestamp:
                return
            # 更新config文件最新修改时间戳
            self.config_timestamp = config_timestamp
            config = load_json(self.config_file_path)
        except Exception as e:
            logger.error(f"get config.json wrong because {e}, not updated, please check!!!")
            return

        if config.get("dynamic_on", False):
            try:
                validate_config(config)
                self.config = config
                self.set_config()
                self.start_step = context.step  # 动态启停时不受原start_step影响，永远从下一步开始
                logger.warning(f"config is updated at step{context.step - 1}, "
                               f"will start new hook at step{context.step}.")
            except Exception as e:
                logger.error(f"set config wrong because {e}, not updated, please check!!!")
                return

            self._remove_all_hooks()
            self.register_hooks(optimizer)

    def register_hooks(self, optimizer):
        self._register_param_name()
        self.hook_modules()
        self.hook_optimizer(optimizer)
        self._patch_grad_sync()
        if self.cc_distribution.get('enable', False):
            self.handles['cc'] = api_register.initialize_hook(*create_hooks(context=self.cc_context, monitor=self))
            api_register.redirect_api()
        self.monitoring = True

    def hook_modules(self):
        if not self.is_target_rank():
            return
        module_in_all_stage = [key for key in self.targets.keys() if MonitorConst.NAME_SEP not in key]

        for key in module_in_all_stage:
            struct = self.targets.pop(key)
            self.targets.update(
                {f'{vpp_stage}{MonitorConst.NAME_SEP}{key}': struct for vpp_stage in range(len(self.model))})

        hooked_count = 0
        for vpp_stage, model_chunk in enumerate(self.model):
            if not isinstance(model_chunk, nn.Cell):
                logger.info("Target Model is not Cell")
                continue
            vpp_stage = f'{vpp_stage}{MonitorConst.NAME_SEP}'
            targets = [x for x, _ in model_chunk.cells_and_names()] if self.print_struct else self.targets.keys()
            hooked_count += self._hook_module(targets, model_chunk, vpp_stage)
        logger.info(f"> {hooked_count} modules are monitored.")

    def hook_optimizer(self, optimizer):
        def optimizer_pre_hook_function(opt, grad_names, gradients):
            context = self.optimizer_context[opt]
            if is_skip_step(context.step, self.start_step, self.step_interval, self.has_collect_times,
                            self.collect_times):
                return
            gradient_list = gradients[0] if isinstance(gradients, tuple) else gradients
            is_select = self.is_select
            for idx, grad in enumerate(gradient_list):
                grad_name = grad_names[idx]
                if is_select and grad_name not in self.targets:
                    continue
                get_single_metrics(self.ops, grad_name, grad, context.param_weight_grad)

            if self.mv_distribution:
                # fetch mean
                for param in m_list:
                    name = param.name
                    if is_select and name not in self.targets:
                        continue
                    get_single_metrics(self.ops, name, param, context.exp_avg_metric)
                # fetch variance
                for param in v_list:
                    name = param.name
                    if is_select and name not in self.targets:
                        continue
                    get_single_metrics(self.ops, name, param, context.exp_avg_sq_metric)
            if self.param_distribution:
                for param in param_list:
                    get_single_metrics(self.ops, param.name, param, context.param_metric)
            self.generate_wgrad_metrics()
            metric_dict = {}
            for cc in self.cc_context.values():
                cc.aggregate()
                metric_dict.update(cc.data)
                cc.reset()

            if not metric_dict:
                return
            context.metric_dict = metric_dict
            return

        def optimizer_pre_hook_wrapper(func, grad_names):
            def wrapper(opt, gradients):
                return func(opt, grad_names, gradients)
            return wrapper

        if self.optimizer_hooked or not self.is_target_rank():
            return

        m_list = []
        v_list = []
        param_list = []
        grad_names = []
        for param in optimizer.get_parameters():
            if MonitorConst.EXP_AVG_SQ in param.name:
                v_list.append(param)
            elif MonitorConst.EXP_AVG in param.name:
                m_list.append(param)
            elif param.name in ['global_step', 'learning_rate']:
                pass
            else:
                param_list.append(param)
                grad_names.append(param.name)

        handle = optimizer.register_forward_pre_hook(
            optimizer_pre_hook_wrapper(optimizer_pre_hook_function, grad_names))
        self.handles['optimizer'].append(handle)
        self.optimizer_hooked = True
        return

    def generate_wgrad_metrics(self):
        if not self.wg_distribution:
            return {}, {}

        if self.weight_hooked:
            try:
                get_metrics(self.ops, self.grad_context.acc, self.eps, self.grad_context.acc_metric)
            except Exception as e:
                logger.warning(f"An error occurred while generating wgrad pre metrics")
                return {}, {}

        grad_dict = {}
        for param, name in self.param2name.items():
            if self.duplicate_param.get(name, False):
                continue
            grad = param.main_grad if self.params_have_main_grad else param.grad
            if grad is None:
                logger.warning(f"grad is None: {name}, maybe something wrong happened.")
                continue
            tag = self.name2tag.get(name, {}).get(MonitorConst.POST_GRAD)
            self._register_param_call_id("hook_optimizer", tag)
            grad_dict[tag] = grad
        try:
            get_metrics(self.ops, grad_dict, self.eps, self.grad_context.post)
        except Exception as e:
            logger.warning(f"An error occurred while generating wgrad post metrics")
            return {}, {}
        return self.grad_context.post, self.grad_context.pre

    def write_xy_tb(self, step):
        if not self.xy_distribution:
            return
        for _, fwd_context in self.module_fwd_hook_context_by_module.items():
            if len(fwd_context.actv) == 0:
                continue
            self.summary_writer.write_metrics(self.ops, fwd_context.actv, step, 'actv')
            fwd_context.actv.clear()
        if self.grad_context.actv:
            self.summary_writer.write_metrics(self.ops, self.grad_context.actv, step, 'actv_grad')

    def write_param_tb(self, opt_context):
        if not self.param_distribution:
            return
        self.summary_writer.write_metrics(self.ops, opt_context.param_metric, opt_context.step, 'param')

    def write_mv_tb(self, opt_context):
        if not self.mv_distribution:
            return
        self.summary_writer.write_metrics(self.ops, opt_context.exp_avg_metric, opt_context.step, 'exp_avg')
        self.summary_writer.write_metrics(self.ops, opt_context.exp_avg_sq_metric, opt_context.step, 'exp_avg_sq')

    def write_grad_tb(self, step):
        if not self.wg_distribution:
            return

        self.summary_writer.write_metrics(self.ops, self.grad_context.acc_metric, step, 'grad_unreduced')
        self.summary_writer.write_metrics(self.ops, self.grad_context.post, step, 'grad_reduced')

    def is_target_rank(self):
        if self.module_rank_list and (self.rank not in self.module_rank_list):
            return False
        return True

    def build_tbtag_tensor_map(self, module_name, tag, tensor):
        metrics = {}
        key = get_summary_writer_tag_name(module_name, tag, str(self.rank))
        if isinstance(tensor, Tensor):
            self._register_param_call_id("_hook_module", key)
            metrics[key] = tensor
        return metrics

    def _register_param_name(self):
        for vpp_stage, model_chunk in enumerate(self.model):
            prefix = f'{vpp_stage}{MonitorConst.NAME_SEP}'
            self._register_chunk(model_chunk, prefix)

    def _register_chunk(self, model_chunk, prefix):
        index = 0
        for param in model_chunk.get_parameters():
            param_name = param.name
            if not param.requires_grad:
                continue
            if self._is_target_param(param_name, param, prefix):
                name = prefix + squash_param_name(param_name)
                if name in self.param2name.values():
                    name = prefix + param_name
                self.param2name[param] = name
                self.name2param[name] = param
                self.name2index[name] = index

                if self.tp_group and not param_is_not_tensor_parallel_duplicate(param, self.tp_group):
                    self.duplicate_param[name] = True
                if self.dp_group and param_is_data_parallel_duplicate(self.dp_group):
                    self.duplicate_param[name] = True
                self.name2tag[name] = {
                    MonitorConst.PRE_GRAD: get_summary_writer_tag_name(name, MonitorConst.PRE_GRAD, self.rank),
                    MonitorConst.POST_GRAD: get_summary_writer_tag_name(name, MonitorConst.POST_GRAD, self.rank)
                }
                index += 1

    def _hook_module(self, target_names, module, vpp_stage=''):
        if not isinstance(module, nn.Cell):
            # nothing to hook
            return 0

        def fwd_hook_fun(module, module_input, module_output, name):
            if module not in self.module_fwd_hook_context_by_module:
                self.module_fwd_hook_context_by_module[module] = ModuleHookContext(name)
            context: ModuleHookContext = self.module_fwd_hook_context_by_module[module]
            if not context.struct:
                context.struct = {
                    MonitorConst.ACTV_IN: get_param_struct(module_input),
                    MonitorConst.ACTV_OUT: get_param_struct(module_output)
                }
            if self.print_struct:
                self.module_struct[context.module_name].update(context.struct)
                return
            if not module.training:
                return
            if is_skip_step(context.step, self.start_step, self.step_interval, self.has_collect_times,
                            self.collect_times):
                step_accumulates_one(context, self.micro_batch_number)
                return
            if not context.format_by_arg:
                context.set_format_by_arg(MonitorConst.ACTV_IN, self.targets)
                context.set_format_by_arg(MonitorConst.ACTV_OUT, self.targets)
            if not context.format_by_arg:
                return
            if not context.verified:
                if not context.ignore_in:
                    context.focused_in_col = validate_config_spec(context.format_by_arg[MonitorConst.ACTV_IN],
                                                                  module_input, context.module_name,
                                                                  MonitorConst.ACTV_IN)
                context.focused_out_col = validate_config_spec(context.format_by_arg[MonitorConst.ACTV_OUT],
                                                               module_output, context.module_name,
                                                               MonitorConst.ACTV_OUT)
                context.verified = True

            tbtag_tensor_map = {}
            if not context.ignore_in:
                cared_input = module_input if context.focused_in_col is None else module_input[context.focused_in_col]
                tbtag_tensor_map.update(
                    self.build_tbtag_tensor_map(f'{context.module_name}_{context.micro_step}', MonitorConst.ACTV_IN,
                                                cared_input))
            cared_output = module_output if context.focused_out_col is None else module_output[context.focused_out_col]
            tbtag_tensor_map.update(
                self.build_tbtag_tensor_map(f'{context.module_name}_{context.micro_step}', MonitorConst.ACTV_OUT,
                                            cared_output))
            try:
                get_metrics(self.ops, tbtag_tensor_map, self.eps, context.actv)
            except Exception as e:
                logger.warning(f"An error occurred while generating forward activation metrics")

            step_accumulates_one(context, self.micro_batch_number)
            return

        def bwd_hook_fun(module, input_grad, output_grad):
            context: ModuleHookContext = self.module_bwd_hook_context_by_module[module]
            if not context.struct:
                context.struct = {
                    MonitorConst.ACTVGRAD_IN: get_param_struct(input_grad),
                    MonitorConst.ACTVGRAD_OUT: get_param_struct(output_grad)
                }
            if self.print_struct:
                self.module_struct[context.module_name].update(context.struct)
                return

            if is_skip_step(context.step, self.start_step, self.step_interval, self.has_collect_times,
                            self.collect_times):
                step_accumulates_one(context, self.micro_batch_number)
                return

            if not context.format_by_arg:
                context.set_format_by_arg(MonitorConst.ACTVGRAD_IN, self.targets)
                context.set_format_by_arg(MonitorConst.ACTVGRAD_OUT, self.targets)
            if not context.format_by_arg:
                return
            if not context.verified:
                if not context.ignore_in:
                    context.focused_in_col = validate_config_spec(context.format_by_arg[MonitorConst.ACTVGRAD_IN],
                                                                  input_grad, context.module_name,
                                                                  MonitorConst.ACTVGRAD_IN)
                context.focused_out_col = validate_config_spec(context.format_by_arg[MonitorConst.ACTVGRAD_OUT],
                                                               output_grad, context.module_name,
                                                               MonitorConst.ACTVGRAD_OUT)
                context.verified = True

            tbtag_tensor_map = {}
            if not context.ignore_in:
                cared_input_grad = input_grad if context.focused_in_col is None else input_grad[context.focused_in_col]
                tbtag_tensor_map.update(
                    self.build_tbtag_tensor_map(
                        f'{context.module_name}_{context.micro_step}', MonitorConst.ACTVGRAD_IN, cared_input_grad))
            cared_output_grad = output_grad if context.focused_out_col is None else output_grad[context.focused_out_col]
            tbtag_tensor_map.update(
                self.build_tbtag_tensor_map(f'{context.module_name}_{context.micro_step}', MonitorConst.ACTVGRAD_OUT,
                                            cared_output_grad))

            if context.micro_step == 0 and context.actvgrad:
                logger.warning(f"actvgrad context of {context.module_name} is not empty when first micro_step, "
                               f"maybe something wrong happened. Now clear it.")
                context.actvgrad.clear()
            try:
                get_metrics(self.ops, tbtag_tensor_map, self.eps, self.grad_context.actv)
            except Exception as e:
                logger.warning(f"An error occurred while generating backward activation metrics: {e}")

            step_accumulates_one(context, self.micro_batch_number)
            return

        def fwd_hook_fun_wrapper(fwd_hook_fun, name):
            def wrapper(module, module_input, module_output):
                return fwd_hook_fun(module, module_input, module_output, name)
            return wrapper

        if self.backward_only and self.forward_only:
            logger.warning('not enable backward_only and forward_only simultaneously')
        hooked_count = 0
        if self.xy_distribution or self.print_struct:
            for module_name, submodule in module.cells_and_names():
                name = self._is_target_module(module_name, target_names, vpp_stage)
                if not name:
                    continue
                if not self.backward_only:
                    handle = submodule.register_forward_hook(fwd_hook_fun_wrapper(fwd_hook_fun, name=name))
                    self.handles['xy'].append(handle)
                if not self.forward_only:
                    handle = submodule.register_backward_hook(bwd_hook_fun)
                    self.handles['xy'].append(handle)
                    self.module_bwd_hook_context_by_module[submodule] = ModuleHookContext(name)
                logger.info(f"> {name} is monitored successfully")
                hooked_count += 1
        return hooked_count

    def _patch_grad_sync(self):
        if not self.wg_distribution:
            return
        self._hook_weights()

    def _hook_weights(self):
        context = self.grad_context

        @_no_grad()
        def param_hook(grad, context_dict, param, key):
            param.micro_step += 1
            self._register_param_call_id("param_hook", key)
            if param.micro_step == self.micro_batch_number:
                param.micro_step = 0
                context_dict[key] = grad

        def param_hook_wrapper(param_hook, context_dict, param, key):
            def wrapper(grad):
                return param_hook(grad, context_dict, param, key)
            return wrapper

        for param, name in self.param2name.items():
            key = get_summary_writer_tag_name(name, 'acc_grad', self.rank)
            setattr(param, 'micro_step', 0)
            handle = param.register_hook(param_hook_wrapper(param_hook, context_dict=context.acc, param=param, key=key))
            self.handles['wgrads'].append(handle)
        self.weight_hooked = True

    def _is_target_param(self, param_name, param, prefix):
        if not self.targets:
            return True
        squash_name = prefix + squash_param_name(param_name)
        name = prefix + param_name
        for target in self.targets.keys():
            if param_name.startswith(target) or squash_name.startswith(target) or name.startswith(target):
                setattr(param, "zero_out_wgrad", True)
                return True
        return False

    def _is_target_module(self, module_name, targets, vpp_stage):
        if self.all_xy or self.print_struct:
            return vpp_stage + squash_param_name(module_name)
        for pattern in [
            vpp_stage + squash_param_name(module_name),
            vpp_stage + module_name,
        ]:
            if pattern in targets:
                return pattern
        return ""

    def _register_param_call_id(self, hook_name: str, key: str):
        """
        :param hook_name:
        :param key: str, '0:relu_0/output_grad'
        :return:
        """
        logger.debug(f"{hook_name} {key}: {self.call_id}")
        self.param_name_call_id[key] = self.call_id
        self.call_id += 1

    def _remove_all_hooks(self):
        # 清空hook handle
        for handle in self.handles['xy']:
            handle.remove()
        self.handles['xy'].clear()
        # 清空对应context缓存
        for _, fwd_context in self.module_fwd_hook_context_by_module.items():
            fwd_context.reset()
        for _, bwd_context in self.module_bwd_hook_context_by_module.items():
            bwd_context.reset()
        self.grad_context.reset()  # 权重梯度和激活值梯度都在这

        for handle in self.handles['wgrads']:
            handle.remove()
        self.handles['wgrads'].clear()
        self.weight_hooked = False

        if self.optimizer_hooked:
            for handle in self.handles['optimizer']:
                handle.remove()
            self.handles['optimizer'].clear()
        for _, context in self.optimizer_context.items():
            context.reset()
        self.optimizer_hooked = False

        for handle in self.handles['cc']:
            handle.remove()
        self.handles['cc'].clear()
        api_register.restore_api()
        for _, context in self.cc_context.items():
            context.reset()

        # 清空节点缓存
        self.param2name.clear()
        self.name2index.clear()
        self.name2indices.clear()
        self.name2param.clear()
        self.duplicate_param.clear()
        self.name2tag.clear()
        self.module_struct.clear()
        self.grad_accs.clear()

        # 关闭采集状态
        self.monitoring = False

    def _remove_all_hooks_final(self, optimizer):
        if self.dynamic_enable:
            # 结束后自动重置dynamic_on为False等待用户手动开启
            try:
                config = load_json(self.config_file_path)
                config['dynamic_on'] = False
                save_json(self.config_file_path, config, indent=2)
                config_timestamp = os.path.getmtime(self.config_file_path)
                self.config_timestamp = config_timestamp
                logger.info(
                    "Finish monitor, set config'dynamic_on=False, will restart by set it to True and update config")
            except Exception as e:
                logger.warning(f"Finish monitor, set config'dynamic_on=False fail because {e}, please check!!!")
        logger.info("Finish monitor")
        self._remove_all_hooks()
