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
import time
import json
import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from functools import partial

import pytz
import torch
import torch.distributed as dist
from msprobe.core.common.const import MonitorConst
from msprobe.core.common.file_utils import load_json, save_json
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.monitor.anomaly_analyse import AnomalyDataWriter
from msprobe.pytorch.monitor.anomaly_detect import AnomalyScanner, SummaryWriterWithAD, AnomalyDataFactory, \
    CSVWriterWithAD, BaseWriterWithAD, WriterInput
from msprobe.pytorch.monitor.distributed.wrap_distributed import api_register, create_hooks, op_aggregate, \
    get_process_group
from msprobe.pytorch.monitor.features import get_sign_matches
from msprobe.pytorch.monitor.module_metric import get_metrics, get_summary_writer_tag_name, \
    TensorMetrics, squash_param_name
from msprobe.pytorch.monitor.module_spec_verifier import validate_config_spec
from msprobe.pytorch.monitor.optimizer_collect import OptimizerMonFactory, OptimizerMon
from msprobe.pytorch.monitor.utils import get_param_struct, validate_config, validate_ops, is_recomputation, \
    get_output_base_dir, get_target_output_dir
from msprobe.pytorch.monitor.visualizer import HeatmapVisualizer
from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook
from torch.utils.hooks import BackwardHook

try:
    import torch_npu
except ImportError:
    pass

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

start_step = 0

class OptimizerContext:
    def __init__(self) -> None:
        self.step = start_step
        self.param_effective_rank = defaultdict(float)
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

    def __init__(self, config_file_path, process_group=None, params_have_main_grad=True, opt_ty=None) -> None:
        """
        opt_ty: "Megatron_Float16OptimizerWithFloat16Params" or "Megatron_DistributedOptimizer"
        """
        self.module_fwd_hook_context_by_module = defaultdict(ModuleHookContext)
        self.module_bwd_hook_context_by_module = defaultdict(ModuleHookContext)
        self.optimizer_context = defaultdict(OptimizerContext)
        self.cc_context = defaultdict(CommunicationContext)
        self.grad_context = GradContext()
        self.process_group = get_process_group(process_group)
        self.params_have_main_grad = params_have_main_grad
        self.opt_ty = opt_ty
        self.config = load_json(config_file_path)
        validate_config(self.config)

        self.module_rank_list = self.config.get("module_ranks", [])
        self.format = self.config.get('format', 'tensorboard')
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
            api_register.initialize_hook(*create_hooks(context=self.cc_context, monitor=self))
            api_register.redirect_api()

        self.common_info()

        alert_setting = self.config.get('alert', {"rules": []})
        self.alert_rules = AnomalyScanner.load_rules(alert_setting["rules"])

        # 设置时区，使用 'UTC' 作为示例
        local_tz = pytz.timezone("Asia/Shanghai")  # 根据需要调整为目标时区

        cur_time = datetime.now(local_tz).strftime('%b%d_%H-%M-%S')
        unique_id = str(uuid.uuid4())[:8]
        output_base_dir = get_output_base_dir()

        time_tags = self.config.get("append_output", [])
        if time_tags:
            output_append_dirs = get_target_output_dir(output_base_dir, time_tags[0], time_tags[1])
        if dist.is_initialized():
            rank = dist.get_rank()
            if time_tags and str(rank) in output_append_dirs:
                tensorboard_dir = output_append_dirs[str(rank)]
                logger.info(f"append rank({rank}) result to {tensorboard_dir}")
            else:
                tensorboard_dir = os.path.join(output_base_dir, f"{cur_time}-rank{rank}-{unique_id}")
            pp_stage = dist.get_group_rank(self.process_group, rank)
            group_mates = dist.get_process_group_ranks(self.process_group)
        else:
            rank = 0
            tensorboard_dir = os.path.join(output_base_dir, f"{cur_time}-{unique_id}")
            pp_stage = 0
            group_mates = [0]
        self.rank = rank

        # 初始化AnomalyData工厂
        self.anomaly_data_factory = None
        if alert_setting.get('dump', False):
            self.anomaly_data_factory = AnomalyDataFactory(rank, pp_stage, group_mates)

        if self.format not in FORMAT_MAPPING:
            raise ValueError(f"Unsupported format: {self.format}")
        writer = FORMAT_MAPPING[self.format]
        self.step_count_per_record = self.config.get('step_count_per_record', 1)

        if (rank in self.module_rank_list) or len(self.module_rank_list) == 0:
            self.summary_writer = writer(
                WriterInput(
                    tensorboard_dir,
                    self.alert_rules,
                    unique_id,
                    self.anomaly_data_factory,
                    self.ndigits,
                    self.step_count_per_record
                )
            )
            # 初始化anomaly detected文件目录
            if self.anomaly_data_factory:
                self.anomaly_data_writer = AnomalyDataWriter(os.path.join(output_base_dir, "anomaly_detected"), rank)
                self.anomaly_data_writer.init_detected_json()

        # A HeatmapVisualizer instance is associated with an image
        self.update_heatmap_visualizer = defaultdict(HeatmapVisualizer)
        self.ratio_heatmap_visualizer = defaultdict(HeatmapVisualizer)
        self.micro_batch_number = 1

        self.model = None
        self.weight_hooked = False
        self.optimizer_hooked = False
        self.param_registered = False
        self.vpp = False
        self.dp_group = None
        self.tp_group = None
        self.enable_megatron = False

        self.param2name = defaultdict(str)
        self.name2index = defaultdict()
        self.name2indices = defaultdict()
        self.name2param = {}
        self.param_name_call_id = {}
        self.duplicate_param = {}
        self.name2tag = {}
        self.call_id = 0
        self.grad_accs = []
        self.handles = defaultdict(list)

        self.mix_precision_optimizer_mon = OptimizerMonFactory.create_optimizer_mon(opt_ty)
        self.print_struct = self.config.get("print_struct", False)
        self.struct_printed = False
        self.module_struct = defaultdict(dict)

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
    def set_wrapped_optimizer(_wrapped_optimizer):
        OptimizerMon.set_wrapped_optimizer(_wrapped_optimizer)

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

    def adhoc_check(self, target_tensor: torch.tensor, module_name: str, tensor_name: str, rank_list, ops_list):
        rank = None
        if dist.is_initialized():
            rank = dist.get_rank()
            if (rank not in rank_list) and len(rank_list) != 0:
                return
        self.tensor_metrics.stat_insert(target_tensor, ops_list, module_name, tensor_name, rank)

    def build_tbtag_tensor_map(self, module_name, tag, tensor):
        metrics = {}
        key = get_summary_writer_tag_name(module_name, tag, self.rank)
        if torch.is_tensor(tensor):
            self._register_param_call_id("_hook_module", key)
            metrics[key] = tensor
        return metrics

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
        if not self.opt_ty:
            if self.ur_distribution:
                raise Exception("ur_distribution cannot be enabled with unknown optimizer.")
            if self.mv_distribution:
                raise Exception("mv_distribution cannot be enabled with unknown optimizer.")

    def hook_modules(self, model: torch.nn.Module, grad_acc_steps):
        if self.module_rank_list and (self.rank not in self.module_rank_list):
            return

        if not isinstance(model, list):
            model = [model]
        self.model = model
        self._register_param_name(model)

        self.micro_batch_number = grad_acc_steps

        targets = self.config['targets']
        module_in_all_stage = [key for key in targets.keys() if MonitorConst.VPP_SEP not in key]
        for key in module_in_all_stage:
            struct = targets.pop(key)
            targets.update({f'{vpp_stage}{MonitorConst.VPP_SEP}{key}': struct for vpp_stage in range(len(model))})

        hooked_count = 0
        for vpp_stage, model_chunk in enumerate(model):
            vpp_stage = f'{vpp_stage}{MonitorConst.VPP_SEP}'
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

        if not self.optimizer_hooked:
            self.hook_optimizer()
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
        m_tag_tensor_map = self.generate_param_map('exp_avg', opt_context.param_exp_avg)
        v_tag_tensor_map = self.generate_param_map('efxp_avg_sq', opt_context.param_exp_avg_sq)
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
        return self.grad_context.post, self.grad_context.pre

    def monitor_gnorm_with_ad(self, model, grad_acc_steps=1, optimizer=None, tp_group=None, dp_group=None, start_iteration=0):
        """External interface"""
        global start_step
        start_step = start_iteration
        logger.info(f'grad acc steps {grad_acc_steps}')
        self.hook_optimizer(optimizer)
        self.micro_batch_number = grad_acc_steps

        self.dp_group = dp_group
        self.tp_group = tp_group
        

        self._register_param_name(model)
        self._patch_grad_sync()
        self.hook_modules(model, grad_acc_steps)

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
        self.xy_distribution = xy_distribution

        for handle in self.handles['xy']:
            handle.remove()
        self.handles['xy'].clear()
        self.hook_modules(self.model, self.micro_batch_number)
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

        if self.enable_megatron:
            self.summary_writer.write_metrics(self.ops, self.grad_context.pre, step, 'grad_unreduced')
        else:
            self.summary_writer.write_metrics(self.ops, self.grad_context.acc_metric, step, 'grad_unreduced')
        self.summary_writer.write_metrics(self.ops, self.grad_context.post, step, 'grad_reduced')

    def hook_optimizer(self, optimizer=None):
        # in DDP by default use params_have_main_grad
        def optimizer_pre_step_hook(optimizer, args, kwargs):
            context = self.optimizer_context[optimizer]
            if self.opt_ty in MonitorConst.DEEPSPEED_OPT_TY:
                if not self.name2indices:
                    self.name2indices = self.mix_precision_optimizer_mon.get_param_index(self.param2name,
                                                                                         self.name2index)
                mv_result = self.mix_precision_optimizer_mon.fetch_mv(self, optimizer, self.param2name,
                                                                      self.name2indices)
                self.param2name = mv_result.grad
            else:
                mv_result = self.mix_precision_optimizer_mon.fetch_mv(self, optimizer, self.param2name)
            context.param_exp_avg = mv_result.exp_avg
            context.param_exp_avg_sq = mv_result.exp_avg_sq
            context.param_adam_update = mv_result.update
            context.param_adam_ratio = mv_result.ratio

            if (self.print_struct and not all(value == {} for value in self.module_struct.values())
                    and not self.struct_printed):
                self._save_module_struct()
                if not self.cc_log_only:
                    raise Exception("exit after first step when print model struct")
            if self.cc_log_only and context.step > 0:
                self._smallest_rank_print("> Used communication ops and corresponding stack")
                self._smallest_rank_print(
                    json.dumps({k: [i.split(';') for i in v] for k, v in self.cc_logged_stack.items()}))
                raise Exception("exit after first step when print cc stack")

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

        def optimizer_post_step_hook(optimizer, args, kwargs):
            context = self.optimizer_context[optimizer]
            rank = dist.get_rank() if dist.is_initialized() else None

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
                        get_summary_writer_tag_name(param_name, 'adam_update', rank), context.step, self.summary_writer)
                for param_name, _ in context.param_adam_ratio.items():
                    self.ratio_heatmap_visualizer[param_name].visualize(
                        get_summary_writer_tag_name(param_name, 'adam_ratio', rank), context.step, self.summary_writer)

            if context.metric_dict:
                self.summary_writer.write_metrics(self.ops, context.metric_dict, context.step, 'other')
            context.metric_dict.clear()
            context.step += 1
            if self.anomaly_data_factory:
                self.anomaly_data_writer.write_detected_json(self.summary_writer.get_anomalies())
            self.summary_writer.clear_anomalies()
            self.call_id = 0
            self.param_name_call_id.clear()
            return

        def patch_step(func, optimizer):
            def wrapper(*args, **kwargs):
                optimizer_pre_step_hook(optimizer, args, kwargs)
                out = func(*args, **kwargs)
                optimizer_post_step_hook(optimizer, args, kwargs)
                return out

            return wrapper

        if self.optimizer_hooked:
            return

        if optimizer:
            optimizer.__class__.step = patch_step(optimizer.__class__.step, optimizer)

        else:
            if not self.module_rank_list or (dist.is_initialized() and dist.get_rank() in self.module_rank_list):
                register_optimizer_step_pre_hook(optimizer_pre_step_hook)
                register_optimizer_step_post_hook(optimizer_post_step_hook)
        self.optimizer_hooked = True
        return

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
        squash_name = prefix + squash_param_name(param_name)
        name = prefix + param_name
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

    def _register_param_name(self, model):
        if self.param_registered:
            return

        if not isinstance(model, list):
            model = [model]

        if len(model) > 1:
            self.vpp = True
            self._smallest_rank_print('vpp enabled')

        for vpp_stage, model_chunk in enumerate(model):
            prefix = f'{vpp_stage}{MonitorConst.VPP_SEP}'
            self._register_chunk(model_chunk, prefix)

        self.param_registered = True

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

    def _hook_module(self, target_names, module: torch.nn.Module, vpp_stage=''):
        if '_modules' not in module.__dict__:
            # nothing to hook
            return 0

        def fwd_hook_fun(module, module_input, module_output, name):
            if is_recomputation():
                return
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
            if not context.format_by_arg:
                context.set_format_by_arg(MonitorConst.ACTV_IN, self.config['targets'])
                context.set_format_by_arg(MonitorConst.ACTV_OUT, self.config['targets'])
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
            # expect output be tensor type
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

            get_metrics(self.ops, tbtag_tensor_map, self.eps, context.actv)
            context.micro_step += 1
            if context.micro_step == self.micro_batch_number:
                context.micro_step = 0
                context.step += 1
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
            if not context.format_by_arg:
                context.set_format_by_arg(MonitorConst.ACTVGRAD_IN, self.config['targets'])
                context.set_format_by_arg(MonitorConst.ACTVGRAD_OUT, self.config['targets'])
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

            get_metrics(self.ops, tbtag_tensor_map, self.eps, self.grad_context.actv)

            context.micro_step += 1
            if context.micro_step == self.micro_batch_number:
                context.micro_step = 0
                context.step += 1
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
                if not self.forward_only:
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
                bucket_params_id_list = [id(params) for params in bucket.params_list]
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

        try:
            from megatron.core.distributed.param_and_grad_buffer import Bucket
            self.enable_megatron = True
        except ImportError:
            self.enable_megatron = False

        if not self.wg_distribution:
            return

        if self.enable_megatron:
            Bucket.start_grad_sync = patch_sync(Bucket.start_grad_sync)  # differ in different megatron version
        else:
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
