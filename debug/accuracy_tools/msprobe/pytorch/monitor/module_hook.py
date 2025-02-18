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
import json
import os
import uuid
from collections import defaultdict
from datetime import datetime
from functools import partial

import pytz
import torch
import torch.distributed as dist
from torch.utils.hooks import BackwardHook

from msprobe.core.common.const import MonitorConst, Const
from msprobe.core.common.file_utils import load_json, save_json
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.common.utils import is_recomputation
from msprobe.pytorch.monitor.anomaly_analyse import AnomalyDataWriter
from msprobe.pytorch.monitor.anomaly_detect import AnomalyScanner, SummaryWriterWithAD, AnomalyDataFactory, \
    CSVWriterWithAD, BaseWriterWithAD, WriterInput
from msprobe.pytorch.monitor.distributed.wrap_distributed import api_register, create_hooks, op_aggregate, \
    get_process_group
from msprobe.pytorch.monitor.features import get_sign_matches
from msprobe.pytorch.monitor.module_metric import get_metrics, get_summary_writer_tag_name, \
    TensorMetrics, squash_param_name
from msprobe.pytorch.monitor.module_spec_verifier import validate_config_spec
from msprobe.pytorch.monitor.optimizer_collect import OptimizerMonFactory
from msprobe.pytorch.monitor.utils import get_param_struct, validate_config, validate_ops, \
    get_output_base_dir, get_target_output_dir
from msprobe.pytorch.monitor.visualizer import HeatmapVisualizer

torch_version_above_or_equal_2 = torch.__version__.split('+')[0] >= '2.0'
if not torch_version_above_or_equal_2:
    raise ValueError("monitor require torch>=2.0")


FORMAT_MAPPING = {
    MonitorConst.TENSORBOARD: SummaryWriterWithAD,
    MonitorConst.CSV: CSVWriterWithAD,
    MonitorConst.API: BaseWriterWithAD
}


def param_is_not_tensor_parallel_duplicate(param, tp_group):
    return (hasattr(param, 'tensor_model_parallel') and param.tensor_model_parallel) or (
            torch.distributed.get_rank(group=tp_group) == 0
    )


def param_is_data_parallel_duplicate(dp_group):
    return torch.distributed.get_rank(group=dp_group) != 0


class ModuleHookContext:
    def __init__(self, module_name) -> None:
        self.micro_step = 0
        self.actv = defaultdict(dict)
        self.actvgrad = []
        self.module_name = module_name
        self.struct = {}
        self.format_by_arg = {}
        self.verified = False
        self.focused_in_col = 0
        self.focused_out_col = 0

    def set_format_by_arg(self, key_name: str, target_config: dict):
        """ 按照监控对象配置format_by_arg
        1) module_name 在 target 中配置监控对象
        2) module_name 未在 targets 中配置，且 all_xy 全量监控
        3) module_name 未在 targets 中配置，且 all_xy 未全量监控

        :param key_name: str, one of [input, output, input_grad, output_grad]
        :param target_config: target obj in config json.
        :return:
        """
        cared = target_config.get(self.module_name, self.struct)
        if key_name in cared:
            target_module_config = cared[key_name]
            if isinstance(target_module_config, dict):
                # current cared is self.struct, monitor all data for module_name
                self.format_by_arg[key_name] = target_module_config.get('config')
            elif isinstance(target_module_config, str):
                # current cared is target_config[self.module_name]
                self.format_by_arg[key_name] = target_module_config
            else:
                logger.warning_on_rank_0(f"target module config error, result maybe empty."
                                         f"module_name: {self.module_name}, key_name: {key_name}")
                self.format_by_arg[key_name] = None
        else:
            self.format_by_arg[key_name] = self.struct.get(key_name).get('config')

    def reset(self):
        self.actv.clear()
        self.actvgrad.clear()


start_step = 0


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

    def reset(self):
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


class TrainerMon:
    tensor_metrics = TensorMetrics()

    def __init__(self, config_file_path, process_group=None, params_have_main_grad=True) -> None:
        # TYPE1: 只在这里初始化的变量, 不会随着训练中途config配置改变而重置
        self.config_file_path = config_file_path
        self.process_group = get_process_group(process_group)
        self.params_have_main_grad = params_have_main_grad
        self.update_heatmap_visualizer = defaultdict(HeatmapVisualizer)
        self.ratio_heatmap_visualizer = defaultdict(HeatmapVisualizer)
        self.origin_step_func = None
        self.origin_start_grad_sync = None
        self.config_timestamp = 0  # 后面有校验时间戳, 首次监控无需为了更新config文件时间戳而去改, 可通过dynamic_on开关直接打开
        self.config = load_json(config_file_path)
        validate_config(self.config)

        self.squash_name = self.config.get('squash_name', True)  # 不允许修改防止前后名字对不上
        local_tz = pytz.timezone("Asia/Shanghai")  # 根据需要调整为目标时区
        cur_time = datetime.now(local_tz).strftime('%b%d_%H-%M-%S')
        self.unique_id = str(uuid.uuid4())[:8]
        self.output_base_dir = get_output_base_dir()
        time_tags = self.config.get("append_output", [])
        if dist.is_initialized():
            self.rank = dist.get_rank()
            if time_tags:
                output_append_dirs = get_target_output_dir(self.output_base_dir, time_tags[0], time_tags[1])
                if str(self.rank) in output_append_dirs:
                    self.tensorboard_dir = output_append_dirs[str(self.rank)]
                    logger.info(f"append rank({self.rank}) result to {self.tensorboard_dir}")
            else:
                self.tensorboard_dir = os.path.join(self.output_base_dir,
                                                    f"{cur_time}-rank{self.rank}-{self.unique_id}")
            self.pp_stage = dist.get_group_rank(self.process_group, self.rank)
            self.group_mates = dist.get_process_group_ranks(self.process_group)
        else:
            self.rank = 0
            self.tensorboard_dir = os.path.join(self.output_base_dir, f"{cur_time}-rank{self.rank}-{self.unique_id}")
            self.pp_stage = 0
            self.group_mates = [0]

        # TYPE2: 只会在set_monitor()主调中赋值的变量
        self.model = None
        self.vpp = False
        self.dp_group = None
        self.tp_group = None
        self.enable_megatron = False
        self.micro_batch_number = 1
        self.optimizer_class = None
        self.optimizer_mon = None

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

    def __del__(self):
        if hasattr(self, "summary_writer"):
            self.summary_writer.close()

    @property
    def ops(self):
        return self._ops

    @ops.setter
    def ops(self, value):
        self._ops = validate_ops(value)

    @staticmethod
    def has_register_backward_hook(module_name, module):
        if hasattr(module, '_backward_hooks') and \
                len(module._backward_hooks) > 0 and \
                module._is_full_backward_hook is False:
            logger.warning(
                f"The {module_name} has registered deprecated register_backward_hook,"
                f"which may cause abnormal data dump. The backward input/output for this module will be skipped."
            )
            return True
        return False

    @staticmethod
    def generate_cc_metrics(cc_name, cc_tensor):
        metrics = defaultdict(dict)
        rank = dist.get_rank() if dist.is_initialized() else None
        for op, tag2tensor in cc_tensor.data.items():
            for tag, tensor in tag2tensor.items():
                key = get_summary_writer_tag_name(cc_name, tag, rank)
                metrics[op].update({key: tensor})
        cc_tensor.reset()
        return metrics

    def set_config(self):
        logger.info(f"current config: {self.config}")
        self.start_step = self.config.get("start_step", 0)
        self.collect_times = self.config.get("collect_times", 100000000)  # 默认大值, 目的是一直采集
        self.step_interval = self.config.get("step_interval", 1)
        self.has_collect_times = 0  # 重设采集计数器
        self.print_struct = self.config.get("print_struct", False)
        self.module_rank_list = self.config.get("module_ranks", [])
        self.format = self.config.get('format', MonitorConst.CSV)
        self.eps = self.config.get('eps', 1e-8)
        self.ops = self.config.get('ops', [])
        self.ndigits = self.config.get('ndigits', 6)
        self.all_xy = self.config.get('all_xy', False)
        self.xy_distribution = self.config.get('xy_distribution', False)
        self.forward_only = self.config.get('forward_only', False)
        self.backward_only = self.config.get('backward_only', False)
        self.ur_distribution = self.config.get('ur_distribution', False)
        self.mv_distribution = self.config.get("mv_distribution", False)
        self.wg_distribution = self.config.get("wg_distribution", False)
        self.param_distribution = self.config.get("param_distribution", False)
        self.mg_direction = self.config.get('mg_direction', False)
        self.cc_distribution = self.config.get("cc_distribution", {})

        if not self.cc_distribution.get('enable', False):
            self.cc_log_only = False
        else:
            self.cc_codeline = self.cc_distribution.get('cc_codeline', [])
            self.cc_log_only = self.cc_distribution.get('cc_log_only', False)
            self.cc_logged_stack = defaultdict(set)
            self.cc_pre_hook = self.cc_distribution.get('cc_pre_hook', False)
            self.handles['cc'] = api_register.initialize_hook(*create_hooks(context=self.cc_context, monitor=self))
            api_register.redirect_api()

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

        if self.ur_distribution and self.format != 'tensorboard':
            logger.error("can only set ur_distribution when format is 'tensorboard', cancel ur_distribution")
            self.ur_distribution = False

        writer = FORMAT_MAPPING[self.format]
        self.step_count_per_record = self.config.get('step_count_per_record', 1)

        if (self.rank in self.module_rank_list) or len(self.module_rank_list) == 0:
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
            # 初始化anomaly detected文件目录
            if self.anomaly_data_factory:
                self.anomaly_data_writer = AnomalyDataWriter(os.path.join(self.output_base_dir, "anomaly_detected"),
                                                             self.rank)
                self.anomaly_data_writer.init_detected_json()

    def adhoc_check(self, target_tensor: torch.tensor, module_name: str, tensor_name: str, rank_list, ops_list):
        rank = None
        if dist.is_initialized():
            rank = dist.get_rank()
            if (rank not in rank_list) and len(rank_list) != 0:
                return
        self.tensor_metrics.stat_insert(target_tensor, ops_list, module_name, tensor_name, rank)

    def build_tbtag_tensor_map(self, module_name, tag, tensor):
        key = get_summary_writer_tag_name(module_name, tag, self.rank)
        self._register_param_call_id("_hook_module", key)
        return {key: tensor}

    def common_info(self):
        if not self.xy_distribution:
            logger.info_on_rank_0("> module input/output input_grad/output_grad is not monitored. ")
        if self.forward_only:
            logger.info_on_rank_0("> only module forward is monitored. ")
        if not self.ur_distribution:
            logger.info_on_rank_0("> update vector and ratio vector of adam is not monitored. ")
        if not self.mv_distribution:
            logger.info_on_rank_0("> momentum and variance of adam is not monitored. ")
        if not self.wg_distribution:
            logger.info_on_rank_0("> weight grad of specified module is not monitored. ")
        if not self.mg_direction:
            logger.info_on_rank_0('> grad and momentum direction will not be compared.')
        if not self.cc_distribution.get('enable', False):
            logger.info_on_rank_0("> cc operator is not monitored.")

    def hook_modules(self):
        if self.module_rank_list and (self.rank not in self.module_rank_list):
            return

        targets = self.config['targets']
        module_in_all_stage = [key for key in targets.keys() if MonitorConst.NAME_SEP not in key]
        for key in module_in_all_stage:
            struct = targets.pop(key)
            targets.update({f'{vpp_stage}{MonitorConst.NAME_SEP}{key}': struct for vpp_stage in range(len(self.model))})

        hooked_count = 0
        for vpp_stage, model_chunk in enumerate(self.model):
            vpp_stage = f'{vpp_stage}{MonitorConst.NAME_SEP}'
            targets = [x for x, _ in model_chunk.named_modules()] if self.print_struct else self.config[
                'targets'].keys()
            hooked_count += self._hook_module(targets, model_chunk, vpp_stage)

        logger.info_on_rank_0(f"> {hooked_count} modules are monitored.")

        def clone_if_tensor(args):
            if isinstance(args, tuple):
                return tuple([clone_if_tensor(arg) for arg in args])
            elif isinstance(args, torch.Tensor):
                return args.clone()
            else:
                return args

        @torch.no_grad
        def wrap_hook_setup(setup):
            def wrapped_setup(*args, **kwargs):
                args = setup(*args, **kwargs)
                args = clone_if_tensor(args)
                return args

            return wrapped_setup

        BackwardHook.setup_output_hook = wrap_hook_setup(BackwardHook.setup_output_hook)

        return

    def generate_param_metrics(self, opt_context):
        if not self.param_distribution:
            return
        get_metrics(self.ops, self.name2param, self.eps, opt_context.param_metric)

    def generate_mv_metrics(self, opt_context):
        if not self.mv_distribution:
            return
        opt_context.exp_avg_metric = {}
        opt_context.exp_avg_sq_metric = {}
        m_tag_tensor_map = self.generate_param_map(MonitorConst.EXP_AVG, opt_context.param_exp_avg)
        v_tag_tensor_map = self.generate_param_map(MonitorConst.EXP_AVG_SQ, opt_context.param_exp_avg_sq)
        get_metrics(self.ops, m_tag_tensor_map, self.eps, opt_context.exp_avg_metric)
        get_metrics(self.ops, v_tag_tensor_map, self.eps, opt_context.exp_avg_sq_metric)

    def generate_wgrad_metrics(self):
        if not self.wg_distribution:
            return {}, {}

        if self.weight_hooked:
            get_metrics(self.ops, self.grad_context.acc, self.eps, self.grad_context.acc_metric)

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

        get_metrics(self.ops, grad_dict, self.eps, self.grad_context.post)
        unreduced_grad = self.grad_context.acc_metric if self.weight_hooked else self.grad_context.pre
        return self.grad_context.post, unreduced_grad

    def set_monitor(
            self,
            model,
            grad_acc_steps=1,
            optimizer=None,
            tp_group=None,
            dp_group=None,
            start_iteration=0
    ):
        """External interface"""
        global start_step
        start_step = start_iteration
        logger.info(f'grad acc steps {grad_acc_steps}')
        self.micro_batch_number = grad_acc_steps
        self.dp_group = dp_group
        self.tp_group = tp_group
        self.optimizer_mon, self.optimizer_class = OptimizerMonFactory.create_optimizer_mon(optimizer)
        self.hook_step_final(optimizer)
        if not isinstance(model, list):
            model = [model]
        self.model = model
        if len(model) > 1:
            self.vpp = True
            self._smallest_rank_print('vpp enabled')
        if not self.dynamic_enable:
            self.register_hooks(optimizer)

    def register_hooks(self, optimizer):
        self._register_param_name()
        self.hook_optimizer(optimizer)
        self._patch_grad_sync()
        self.hook_modules()
        self.monitoring = True

    def generate_param_map(self, tag, param_tensor):
        metrics = {}
        for name in self.param2name.values():
            key = get_summary_writer_tag_name(name, tag, self.rank)
            self._register_param_call_id("optimizer_pre_step_hook", key)
            if name not in param_tensor or param_tensor[name] is None:
                continue
            metrics[key] = param_tensor[name]
        return metrics

    def generate_xy_metrics(self):
        actv = {}
        for fwd_context in self.module_fwd_hook_context_by_module.values():
            actv.update(fwd_context.actv)

        actv_grad = self.grad_context.actv

        return actv, actv_grad

    def reload_xy(self, xy_distribution=False):
        logger.warning("reload_xy() is deprecated and will be removed in a future version. "
                       "Use DYNAMIC_MONITOR instead.")
        self.xy_distribution = xy_distribution

        for handle in self.handles['xy']:
            handle.remove()
        self.handles['xy'].clear()
        self.hook_modules()
        for _, fwd_context in self.module_fwd_hook_context_by_module.items():
            fwd_context.actv.clear()

    def write_adhoc_check(self, step):
        self.tensor_metrics.flush(self.summary_writer)

    def write_xy_tb(self, step):
        if not self.xy_distribution:
            return
        for _, fwd_context in self.module_fwd_hook_context_by_module.items():
            if len(fwd_context.actv) == 0:
                continue
            self.summary_writer.write_metrics(self.ops, fwd_context.actv, step, MonitorConst.ACTV)
            fwd_context.actv.clear()
        if self.grad_context.actv:
            self.summary_writer.write_metrics(self.ops, self.grad_context.actv, step, MonitorConst.ACTVGRAD)

    def write_param_tb(self, opt_context):
        if not self.param_distribution:
            return
        self.summary_writer.write_metrics(self.ops, opt_context.param_metric, opt_context.step, MonitorConst.PARAM)

    def write_mv_tb(self, opt_context):
        if not self.mv_distribution:
            return
        self.summary_writer.write_metrics(self.ops, opt_context.exp_avg_metric, 
                                          opt_context.step, MonitorConst.EXP_AVG)
        self.summary_writer.write_metrics(self.ops, opt_context.exp_avg_sq_metric, 
                                          opt_context.step, MonitorConst.EXP_AVG_SQ)

    def write_grad_tb(self, step):
        if not self.wg_distribution:
            return

        if self.enable_megatron:
            self.summary_writer.write_metrics(self.ops, self.grad_context.pre, step, 'grad_unreduced')
        else:
            self.summary_writer.write_metrics(self.ops, self.grad_context.acc_metric, step, 'grad_unreduced')
        self.summary_writer.write_metrics(self.ops, self.grad_context.post, step, 'grad_reduced')

    def hook_optimizer(self, optimizer=None):
        # in DDP by default use params_have_main_grad
        def optimizer_pre_step_hook(optimizer, args, kwargs):
            context = self.optimizer_context[optimizer]

            if (self.print_struct and not all(value == {} for value in self.module_struct.values())
                    and not self.struct_printed):
                self._save_module_struct()
                if not self.cc_log_only:
                    raise Exception("exit after first monitor step when print model struct")
            if self.cc_log_only and context.step > 0:
                self._smallest_rank_print("> Used communication ops and corresponding stack")
                self._smallest_rank_print(
                    json.dumps({k: [i.split(';') for i in v] for k, v in self.cc_logged_stack.items()}))
                raise Exception("exit after first step when print cc stack")

            # skip generate metrics
            if context.step < self.start_step or (context.step - self.start_step) % self.step_interval != 0:
                return
            if MonitorConst.DEEPSPEED_ZERO_OPT_FILTER in self.optimizer_class:  # use deepspeed with zero1/2/3
                if not self.name2indices:
                    self.name2indices = self.optimizer_mon.get_param_index(self.param2name, self.name2index, optimizer)
                mv_result = self.optimizer_mon.fetch_mv(self, optimizer, self.param2name, self.name2indices)
                self.param2name = mv_result.grad
            else:
                mv_result = self.optimizer_mon.fetch_mv(self, optimizer, self.param2name)
            context.param_exp_avg = mv_result.exp_avg
            context.param_exp_avg_sq = mv_result.exp_avg_sq
            context.param_adam_update = mv_result.update
            context.param_adam_ratio = mv_result.ratio

            self.generate_wgrad_metrics()
            self.generate_mv_metrics(context)
            self.generate_param_metrics(context)

            tbtag_tensor_map = {}
            if self.mg_direction:
                for param, name in self.param2name.items():
                    grad = param.main_grad if self.params_have_main_grad else param.grad
                    if grad is None:
                        logger.warning(f"grad is None: {name}, maybe something wrong happened.")
                        continue
                    if context.step == 0:
                        same_direction_ratio = torch.tensor(1.)
                    else:
                        same_direction_ratio = get_sign_matches(grad, context.param_exp_avg[name])
                    context.param_mg_direction[name] = same_direction_ratio
                tbtag_tensor_map.update(self.generate_param_map('mg_direction', context.param_mg_direction))

            metric_dict = {}
            get_metrics(self.ops, tbtag_tensor_map, self.eps, metric_dict)
            for cc in self.cc_context.values():
                cc.aggregate()
                metric_dict.update(cc.data)
                cc.reset()

            if not metric_dict:
                return
            context.metric_dict = metric_dict
            return

        def patch_step(func, optimizer):
            def wrapper(*args, **kwargs):
                optimizer_pre_step_hook(optimizer, args, kwargs)
                out = func(*args, **kwargs)
                return out

            return wrapper

        if self.optimizer_hooked:
            return

        optimizer.__class__.step = patch_step(optimizer.__class__.step, optimizer)

        self.optimizer_hooked = True
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
                logger.warning(f"config is updated at step{context.step - 1}, "
                               f"will start new hook at step{context.step}.")
            except Exception as e:
                logger.error(f"set config wrong because {e}, not updated, please check!!!")
                return

            self._remove_all_hooks(optimizer)
            self.register_hooks(optimizer)

    def hook_step_final(self, optimizer):
        def step_final_hook(optimizer, args, kwargs):
            context = self.optimizer_context[optimizer]
            rank = dist.get_rank() if dist.is_initialized() else None
            # 静态在第0步就可以保存, 动态在第0步不可以, 因为动态设计的就是重置后下一步开启, 第0步的self.monitoring还是False
            if self.monitoring:
                module_rank_valid = not self.module_rank_list or (
                            dist.is_initialized() and dist.get_rank() in self.module_rank_list)
                step_condition = (context.step >= self.start_step and (
                            context.step - self.start_step) % self.step_interval == 0)
                if module_rank_valid and step_condition:
                    self.has_collect_times += 1

                    if self.anomaly_data_factory:
                        self.anomaly_data_factory.set_call_id(self.param_name_call_id)
                    self.write_xy_tb(context.step)
                    self.write_grad_tb(context.step)
                    self.write_mv_tb(context)
                    self.write_param_tb(context)
                    self.write_adhoc_check(context.step)

                    if self.ur_distribution:
                        for param_name, _ in context.param_adam_update.items():
                            self.update_heatmap_visualizer[param_name].visualize(
                                get_summary_writer_tag_name(param_name, 'adam_update', rank), context.step,
                                self.summary_writer)
                        for param_name, _ in context.param_adam_ratio.items():
                            self.ratio_heatmap_visualizer[param_name].visualize(
                                get_summary_writer_tag_name(param_name, 'adam_ratio', rank), context.step,
                                self.summary_writer)

                    if context.metric_dict:
                        self.summary_writer.write_metrics(self.ops, context.metric_dict, context.step, 'other')
                    context.metric_dict.clear()

                    if self.anomaly_data_factory:
                        self.anomaly_data_writer.write_detected_json(self.summary_writer.get_anomalies())
                    self.summary_writer.clear_anomalies()
                    self.call_id = 0
                    self.param_name_call_id.clear()

                    if self.has_collect_times >= self.collect_times:
                        self._remove_all_hooks_final(optimizer)

            context.step += 1
            self.dynamic_monitor(optimizer)

        def patch_step(func, optimizer):
            def wrapper(*args, **kwargs):
                out = func(*args, **kwargs)
                step_final_hook(optimizer, args, kwargs)
                return out
            return wrapper

        optimizer.__class__.step = patch_step(optimizer.__class__.step, optimizer)
        self.origin_step_func = optimizer.__class__.step

        return

    def _remove_all_hooks(self, optimizer):
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

        if self.origin_start_grad_sync:  # megatron
            try:
                from megatron.core.distributed.param_and_grad_buffer import Bucket
                Bucket.start_grad_sync = self.origin_start_grad_sync
                logger.info("remove Bucket start_grad_sync")
            except ImportError:
                pass
            try:
                from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup
                _ParamAndGradBucketGroup.start_grad_sync = self.origin_start_grad_sync
                logger.info("remove _ParamAndGradBucketGroup start_grad_sync")
            except ImportError:
                pass
        else:  # not megatron
            for handle in self.handles['wgrads']:
                handle.remove()
            self.handles['wgrads'].clear()
            self.weight_hooked = False

        if self.optimizer_hooked:
            optimizer.__class__.step = self.origin_step_func

        for _, context in self.optimizer_context.items():
            context.reset()
        self.optimizer_hooked = False

        for handle in self.handles['cc']:
            handle.remove()
        self.handles['cc'].clear()
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
        self._remove_all_hooks(optimizer)

    def _smallest_rank_print(self, msg):
        if dist.is_initialized():
            if self.module_rank_list:
                if dist.get_rank() == min(self.module_rank_list):
                    logger.info(msg)
            else:
                if dist.get_rank() == 0:
                    logger.info(msg)
        else:
            logger.info(msg)

    def _save_module_struct(self):
        save_module_struct = (not dist.is_initialized()
                              or (self.module_rank_list and dist.get_rank() == min(self.module_rank_list))
                              or (not self.module_rank_list and dist.get_rank() == 0))

        if save_module_struct:
            module_struct_file = os.path.realpath(os.path.join(get_output_base_dir(), 'module_struct.json'))
            save_json(module_struct_file, self.module_struct, indent=2)
            logger.info(f"> save module struct to {module_struct_file}")
        self.struct_printed = True

    def _is_target_param(self, param_name, param, prefix):
        name = prefix + param_name
        squash_name = prefix + squash_param_name(param_name, self.squash_name)
        for target in self.config['targets'].keys():
            if param_name.startswith(target) or squash_name.startswith(target) or name.startswith(target):
                setattr(param, "zero_out_wgrad", True)
                return True

        return False

    def _register_chunk(self, model_chunk, prefix):
        index = 0
        for (param_name, param) in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            if self._is_target_param(param_name, param, prefix):
                name = prefix + squash_param_name(param_name, self.squash_name)
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

    def _register_param_name(self):
        for vpp_stage, model_chunk in enumerate(self.model):
            prefix = f'{vpp_stage}{MonitorConst.NAME_SEP}'
            self._register_chunk(model_chunk, prefix)

    def _is_target_module(self, module_name, targets, vpp_stage):
        if self.all_xy or self.print_struct:
            return vpp_stage + squash_param_name(module_name, self.squash_name)
        for pattern in [
            vpp_stage + squash_param_name(module_name, self.squash_name),
            vpp_stage + module_name,
        ]:
            if pattern in targets:
                return pattern
        return ""

    def _hook_module(self, target_names, module: torch.nn.Module, vpp_stage=''):
        if '_modules' not in module.__dict__:
            # nothing to hook
            return 0

        def fwd_hook_fun(module, module_input, module_output, name):
            if not module.training or is_recomputation():
                # 1 only monitor training stage.
                # 2 when open recompute, skip recomputed forward stage.
                return
            if module not in self.module_fwd_hook_context_by_module:
                self.module_fwd_hook_context_by_module[module] = ModuleHookContext(name)
            context: ModuleHookContext = self.module_fwd_hook_context_by_module[module]
            if not context.struct:
                context.struct = {
                    Const.INPUT: get_param_struct(module_input),
                    Const.OUTPUT: get_param_struct(module_output)
                }
            if self.print_struct:
                self.module_struct[context.module_name].update(context.struct)
                return
            if not context.format_by_arg:
                context.set_format_by_arg(Const.INPUT, self.config['targets'])
                context.set_format_by_arg(Const.OUTPUT, self.config['targets'])
            if not context.format_by_arg:
                return
            if not context.verified:
                context.focused_in_col = validate_config_spec(context.format_by_arg[Const.INPUT],
                                                              module_input, context.module_name,
                                                              Const.INPUT)
                context.focused_out_col = validate_config_spec(context.format_by_arg[Const.OUTPUT],
                                                               module_output, context.module_name,
                                                               Const.OUTPUT)
                context.verified = True
            # expect output be tensor type
            tbtag_tensor_map = {}
            cared_input = module_input if context.focused_in_col is None else module_input[context.focused_in_col]
            tbtag_tensor_map.update(
                self.build_tbtag_tensor_map(
                    f'{context.module_name}.{Const.INPUT}{MonitorConst.NAME_SEP}{context.micro_step}',
                    MonitorConst.ACTV, cared_input))
            cared_output = module_output if context.focused_out_col is None else module_output[context.focused_out_col]
            tbtag_tensor_map.update(
                self.build_tbtag_tensor_map(
                    f'{context.module_name}.{Const.OUTPUT}{MonitorConst.NAME_SEP}{context.micro_step}',
                    MonitorConst.ACTV, cared_output))

            get_metrics(self.ops, tbtag_tensor_map, self.eps, context.actv)
            context.micro_step += 1
            if context.micro_step == self.micro_batch_number:
                context.micro_step = 0
            return

        def bwd_hook_fun(module, input_grad, output_grad):
            context: ModuleHookContext = self.module_bwd_hook_context_by_module[module]
            if not context.struct:
                context.struct = {
                    MonitorConst.INPUT_GRAD: get_param_struct(input_grad),
                    MonitorConst.OUTPUT_GRAD: get_param_struct(output_grad)
                }
            if self.print_struct:
                self.module_struct[context.module_name].update(context.struct)
                return
            if not context.format_by_arg:
                context.set_format_by_arg(MonitorConst.INPUT_GRAD, self.config['targets'])
                context.set_format_by_arg(MonitorConst.OUTPUT_GRAD, self.config['targets'])
            if not context.format_by_arg:
                return
            if not context.verified:
                context.focused_in_col = validate_config_spec(
                    context.format_by_arg[MonitorConst.INPUT_GRAD], 
                    input_grad, context.module_name, MonitorConst.INPUT_GRAD)
                context.focused_out_col = validate_config_spec(
                    context.format_by_arg[MonitorConst.OUTPUT_GRAD],
                    output_grad, context.module_name, MonitorConst.OUTPUT_GRAD)
                context.verified = True

            tbtag_tensor_map = {}
            cared_input_grad = input_grad if context.focused_in_col is None else input_grad[context.focused_in_col]
            tbtag_tensor_map.update(
                self.build_tbtag_tensor_map(
                    f'{context.module_name}.{Const.INPUT}{MonitorConst.NAME_SEP}{context.micro_step}',
                    MonitorConst.ACTV, cared_input_grad))
            cared_output_grad = output_grad if context.focused_out_col is None else output_grad[context.focused_out_col]
            tbtag_tensor_map.update(
                self.build_tbtag_tensor_map(
                    f'{context.module_name}.{Const.OUTPUT}{MonitorConst.NAME_SEP}{context.micro_step}',
                    MonitorConst.ACTV, cared_output_grad))

            if context.micro_step == 0 and context.actvgrad:
                logger.warning(f"actvgrad context of {context.module_name} is not empty when first micro_step, "
                               f"maybe something wrong happened. Now clear it.")
                context.actvgrad.clear()

            get_metrics(self.ops, tbtag_tensor_map, self.eps, self.grad_context.actv)

            context.micro_step += 1
            if context.micro_step == self.micro_batch_number:
                context.micro_step = 0
            return

        if self.backward_only and self.forward_only:
            logger.warning('not enable backward_only and forward_only simultaneously')

        hooked_count = 0
        if self.xy_distribution or self.print_struct:
            for module_name, submodule in module.named_modules():
                name = self._is_target_module(module_name, target_names, vpp_stage)
                if not name:
                    continue
                if not self.backward_only:
                    handle = submodule.register_forward_hook(partial(fwd_hook_fun, name=name))
                    self.handles['xy'].append(handle)
                if not self.forward_only and not self.has_register_backward_hook(name, submodule):
                    handle = submodule.register_full_backward_hook(bwd_hook_fun)
                    self.handles['xy'].append(handle)
                    self.module_bwd_hook_context_by_module[submodule] = ModuleHookContext(name)
                logger.info_on_rank_0(f"> {name} is monitored successfully")
                hooked_count += 1
        return hooked_count

    def _patch_grad_sync(self):
        def patch_sync(sync_grad_func):
            def wrapper(bucket):
                grad_dict = {}
                # Megatron between core_r0.6.0 and core_r0.8.0, this bucket is Bucket.
                # When megatron is core_r0.9.0, this bucket is _ParamAndGradBucketGroup.
                # In megatron version core_r0.9.0, func start_grad_sync from Bucket moved to _ParamAndGradBucketGroup.
                bucket_params_id_list = [id(params) for params in bucket.params]
                for param, name in self.param2name.items():
                    if id(param) not in bucket_params_id_list:
                        continue
                    grad = param.main_grad if self.params_have_main_grad else param.grad
                    if grad is None:
                        logger.warning(f"grad is None: {name}, maybe something wrong happened.")
                        continue
                    tag = self.name2tag.get(name, {}).get(MonitorConst.PRE_GRAD)
                    if tag is None:
                        continue
                    grad_dict[tag] = grad
                    self._register_param_call_id("sync_grad_func", tag)
                get_metrics(self.ops, grad_dict, self.eps, self.grad_context.pre)
                out = sync_grad_func(bucket)
                return out

            return wrapper

        if not self.wg_distribution:
            return

        try:
            from megatron.core.distributed.param_and_grad_buffer import Bucket
            self.origin_start_grad_sync = Bucket.start_grad_sync
            Bucket.start_grad_sync = patch_sync(Bucket.start_grad_sync)
            self.enable_megatron = True
            logger.info("megatron version is >= core_r0.6.0 <= core_r0.8.0")
        except ImportError:
            self.enable_megatron = False

        try:
            from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup
            self.origin_start_grad_sync = _ParamAndGradBucketGroup.start_grad_sync
            _ParamAndGradBucketGroup.start_grad_sync = patch_sync(_ParamAndGradBucketGroup.start_grad_sync)
            self.enable_megatron = True
            logger.info("megatron version is > core_r0.8.0 <= core_r0.9.0")
        except ImportError:
            self.enable_megatron = False

        if not self.enable_megatron:
            self._hook_weights()

    def _hook_weights(self):
        context = self.grad_context

        @torch.no_grad
        def param_hook(*args, context_dict, param, key, name):
            param.micro_step += 1
            self._register_param_call_id("param_hook", key)
            if param.micro_step == self.micro_batch_number:
                param.micro_step = 0
                if self.params_have_main_grad:
                    context_dict[key] = param.main_grad.clone()
                else:
                    context_dict[key] = param.grad.clone()

        logger.info("hooking weights.")
        for param, name in self.param2name.items():
            key = get_summary_writer_tag_name(name, 'acc_grad', self.rank)
            setattr(param, 'micro_step', 0)
            param_tmp = param.expand_as(param)
            grad_acc = param_tmp.grad_fn.next_functions[0][0]
            handle = grad_acc.register_hook(
                partial(param_hook, context_dict=context.acc, param=param, key=key, name=name))
            self.grad_accs.append(grad_acc)
            self.handles['wgrads'].append(handle)

        self.weight_hooked = True

    def _register_param_call_id(self, hook_name: str, key: str):
        """
        :param hook_name:
        :param key: str, '0:relu_0/output_grad'
        :return:
        """
        logger.debug(f"{hook_name} {key}: {self.call_id}")
        self.param_name_call_id[key] = self.call_id
        self.call_id += 1
