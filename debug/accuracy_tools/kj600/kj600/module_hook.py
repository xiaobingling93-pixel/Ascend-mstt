import os
import uuid
import json
from collections import defaultdict
from datetime import datetime
import torch
import torch.distributed as dist
from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook
from kj600.module_spec_verifier import validate_config_spec
from kj600.optimizer_collect import MixPrecsionOptimizerMon, print_rank_0, OptimizerMonFactory
from kj600.features import eff_rank, get_sign_matches
from kj600.visualizer import HeatmapVisualizer
from kj600.anomaly_detect import AnomalyScanner, SummaryWriterWithAD
from kj600.anomaly_inform import AnomalyInformFactory
from kj600.module_metric import get_metrics, write_metrics_tensorboard, get_summary_writer_tag_name, TensorMetrics
from kj600.distributed.wrap_distributed import api_register, create_hooks,  op_aggregate
from kj600.utils import print_warn_log, print_info_log, get_param_struct, check_path_length, check_path_pattern_valid, change_mode, FileCheckConst
from kj600.file_check import FileOpen



class ModuleHookContext:
    def __init__(self, module_name) -> None:
        self.step = 0
        self.micro_step = 0
        self.actv = []
        self.actvgrad = []
        self.module_name = module_name
        self.format_by_arg = {}
        self.verified = False
        self.focused_in_col = 0
        self.focused_out_col = 0
        self.ignore_in = False  # no need to care when no key 'input' or 'input_grad' found

    def set_format_by_arg(self, key_name:str, target_config:dict):
        if key_name in target_config[self.module_name]:
            self.format_by_arg[key_name] = target_config[self.module_name][key_name]
        elif key_name in ['input', 'input_grad']:
            self.ignore_in = True
        else:
            raise KeyError(f"Missing key: {key_name} of {self.module_name} in config.json")


class OptimizerContext:
    def __init__(self) -> None:
        self.step = 0
        self.param_effective_rank = defaultdict(float)
        self.param_mg_direction  = defaultdict(float)
        self.param_adam_update = defaultdict()
        self.param_adam_ratio = defaultdict()
        self.param_weight_grad = defaultdict()
        self.param_exp_avg = defaultdict()
        self.param_exp_avg_sq = defaultdict()
        self.metric_list = []


class CommunicationContext:
    def __init__(self) -> None:
        self.data = {}

    @staticmethod
    def _agg(data):
        aggregated_data = {}
        for op, tag2tensorlist in data.items():
            aggregated_data[op] = {}
            for tag, tensorlist in tag2tensorlist.items():
                aggregated_data[op][tag] = op_aggregate(op, tensorlist)
        return aggregated_data

    def reset(self):
        self.data = {}

    def aggregate(self):
        self.data = self._agg(self.data)

class TrainerMon:

    tensor_metrics = TensorMetrics()

    # opt_ty: "Megatron_Float16OptimizerWithFloat16Params" or "Megatron_DistributedOptimizer"
    def __init__(self, config_file_path, params_have_main_grad=True, opt_ty=None) -> None:
        self.module_fwd_hook_context_by_module = defaultdict(ModuleHookContext)
        self.module_bwd_hook_context_by_module = defaultdict(ModuleHookContext)
        self.optimizer_context = defaultdict(OptimizerContext)
        self.cc_context = defaultdict(CommunicationContext)
        self.params_have_main_grad = params_have_main_grad
        with FileOpen(config_file_path, 'r') as f:
            self.config = json.load(config_file_path)
        self.module_rank_list = self.config.get("module_ranks", [])
        self.eps = self.config.get('eps', 1e-8)
        self.ops = self.config.get('ops', [])
        self.xy_distribution = self.config.get('xy_distribution', False)
        if not self.xy_distribution:
            print_rank_0("> module input/output input_grad/output_grad is not monitored. ")
        
        # backward hook cause megatron-lm pipeline parallel schedule assert exception. 
        # TBD: backward hook cause output tensor is view of some base tensor. root cause invesigation pending.
        self.forward_only = self.config.get('forward_only', False) 
        if self.forward_only: 
            print_rank_0("> only module forward is monitored. ")

        self.ur_distribution = self.config.get('ur_distribution', False)
        if not self.ur_distribution:
            print_rank_0("> update vector and ratio vector of adam is not monitored. ")
        self.mv_distribution = self.config.get("mv_distribution", False)
        if not self.mv_distribution:
            print_rank_0("> momentum and variance of adam is not monitored. ")
        self.wg_distribution = self.config.get("wg_distribution", False)
        if not self.wg_distribution:
            print_rank_0("> weight grad of specified module is not monitored. ")
        self.mg_direction = self.config.get('mg_direction', False)
        if not self.mg_direction:
            print_rank_0('> grad and momentum direction will not be compared.')
        self.cc_distribution = self.config.get("cc_distribution", {})
        if not self.cc_distribution.get('enable', False):
            print_rank_0("> cc operator is not monitored.")
            self.cc_log_only = False
        else:
            self.cc_codeline = self.cc_distribution.get('cc_codeline', [])
            self.cc_log_only = self.cc_distribution.get('cc_log_only', False)
            self.cc_logged_stack = defaultdict(set)
            self.cc_pre_hook = self.cc_distribution.get('cc_pre_hook', False)
            api_register.initialize_hook(*create_hooks(context=self.cc_context, monitor=self))
            api_register.redirect_api()

        alert_setting = self.config.get('alert', {"rules":[]})
        self.alert_rules = AnomalyScanner.load_rules(alert_setting["rules"])
        
        anomaly_inform = AnomalyInformFactory.create_informer(**alert_setting["inform"]) if "inform" in alert_setting else None
        
        self.optimizer_hooked = False
        output_base_dir = os.getenv('KJ600_OUTPUT_DIR', './kj600_output')
        cur_time = datetime.now().strftime('%b%d_%H-%M-%S')
        unique_id = str(uuid.uuid4())[:8]
        if dist.is_initialized():
            cur_path = os.path.join(output_base_dir, f"{cur_time}-rank{dist.get_rank()}-{unique_id}")
            if (dist.get_rank() in self.module_rank_list) or len(self.module_rank_list) == 0:
                check_path_length(cur_path)
                check_path_pattern_valid(cur_path)
                self.summary_writer = SummaryWriterWithAD(
                    cur_path, self.alert_rules, unique_id, anomaly_inform)
        else:
            cur_path = os.path.join(output_base_dir, f"{cur_time}-{unique_id}")
            check_path_length(cur_path)
            check_path_pattern_valid(cur_path)
            self.summary_writer = SummaryWriterWithAD(cur_path, self.alert_rules, unique_id, anomaly_inform)

        full_path = os.path.realpath(cur_path)
        change_mode(full_path,FileCheckConst.DATA_DIR_AUTHORITY)

        # A HeatmapVisualizer instance is associated with an image
        self.update_heatmap_visualizer = defaultdict(HeatmapVisualizer)
        self.ratio_heatmap_visualizer = defaultdict(HeatmapVisualizer)
        self.micro_batch_number = 0

        self.param_name_list = []
        self.param2name = defaultdict(str)

        self.mix_precision_optimizer_mon = OptimizerMonFactory.create_optimizer_mon(opt_ty)
        if opt_ty is None:
            if self.ur_distribution:
                raise Exception("ur_distribution cannot be enabled with unknown optimizer.")
            if self.mv_distribution:
                raise Exception("mv_distribution cannot be enabled with unknown optimizer.")
        self.print_struct = self.config.get("print_struct", False)
        self.struct_printed = False
        self.module_struct = {}
        return

    def __del__(self):
        if hasattr(self, "summary_writer"):
            self.summary_writer.close()
    
    @staticmethod
    def set_wrapped_optimizer(_wrapped_optimizer):
        MixPrecsionOptimizerMon.set_wrapped_optimizer(_wrapped_optimizer)

    @staticmethod
    def adhoc_check(target_tensor:torch.tensor, module_name:str, tensor_name:str, rank_list, ops_list):
        rank = None
        if dist.is_initialized():
            rank = dist.get_rank()
            if (rank not in rank_list) and len(rank_list) != 0:
                return
        TrainerMon.tensor_metrics.stat_insert(target_tensor, ops_list, module_name, tensor_name, rank)

    def hook_modules(self, model:torch.nn.Module, grad_acc_steps):
        # fwd=0, bkd=1
        # targets is module name list like ["xx.xxx1", "xxx.xxx2"] which can be obtained when first run. 
        print_rank_0("> module names:")
        for name, _ in model.named_modules():
            print_rank_0(f"\t{name}")
        self.micro_batch_number = grad_acc_steps

        if not self.module_rank_list or (dist.is_initialized() and dist.get_rank() in self.module_rank_list):
            targets = [x for x, _ in model.named_modules()] if self.print_struct else self.config['targets'].keys()
            hooked_count = self._hook_module(targets, model, fwd_or_bkd=0)
            print_rank_0(f"> {hooked_count} out of {len(self.config['targets'])} are monitored.")
        else:
            return

        if not self.optimizer_hooked:
            self.optimizer_hooked = True
            print_rank_0("> parameter names:")
            for name, param in model.named_parameters():
                print_rank_0(f"\t{name}")
                for target_module, _ in self.config['targets'].items():
                    if name.startswith(target_module): # name : language_model.encoder.layers.0.mlp.weight, target_module:language_model.encoder.layers.0
                        self.param_name_list.append(name)
                        self.param2name[param] = name
            self.hook_optimizer()
        return

    def build_tbtag_tensor_map(self, module_name, tag, tensor):
        metrics = {}
        rank = dist.get_rank() if dist.is_initialized() else None
        key = get_summary_writer_tag_name(module_name, tag, rank)
        if tensor is not None:
            metrics[key] = tensor
        return metrics

    def generate_param_metrics(self, tag, param_tensor):
        metrics = {}
        rank = dist.get_rank() if dist.is_initialized() else None
        for param, name in self.param2name.items():
            key = get_summary_writer_tag_name(name, tag, rank)
            if name not in param_tensor or param_tensor[name] is None:
                continue
            metrics[key] = param_tensor[name]
        return metrics
    
    def generate_cc_metrics(self, cc_name, cc_tensor):
        metrics = defaultdict(dict)
        rank = dist.get_rank() if dist.is_initialized() else None
        for op, tag2tensor in cc_tensor.data.items():
            for tag, tensor in tag2tensor.items():
                key = get_summary_writer_tag_name(cc_name, tag, rank)
                metrics[op].update({key: tensor})
        cc_tensor.reset()
        return metrics

    def write_adhoc_check(self, step):
        TrainerMon.tensor_metrics.flush(self.summary_writer)

    def write_xy_tb(self, step):
        if not self.xy_distribution:
            return
        for _, fwd_context in self.module_fwd_hook_context_by_module.items():
            if not len(fwd_context.actv) == self.micro_batch_number:
                print_warn_log(f"fwd_context.actv not equal to micro_batch_number: {len(fwd_context.actv)}, {self.micro_batch_number}")
            for metric_name in self.ops:
                write_metrics_tensorboard(metric_name, self.summary_writer, fwd_context.actv, step)
            fwd_context.actv.clear()

        for _, bwd_context in self.module_bwd_hook_context_by_module.items():
            if not len(bwd_context.actvgrad) == self.micro_batch_number:
                print_warn_log(f"bwd_context.actvgrad not equal to micro_batch_number: {len(bwd_context.actvgrad)}, {self.micro_batch_number}")
            for metric_name in self.ops:
                write_metrics_tensorboard(metric_name, self.summary_writer, bwd_context.actvgrad, step)
            bwd_context.actvgrad.clear()

    def hook_optimizer(self):
        # in DDP by default use params_have_main_grad
        def optimizer_pre_step_hook(optimizer, args, kwargs):
            context = self.optimizer_context[optimizer]
            if self.print_struct and not all(value == {} for value in self.module_struct.values()) and not self.struct_printed:
                self._smallest_rank_print("> module struct:")
                self._smallest_rank_print(json.dumps(self.module_struct, indent=4))
                self.struct_printed = True
                if not self.cc_log_only:
                    raise Exception("exit after first step when print model struct")
            if self.cc_log_only and context.step > 0:
                self._smallest_rank_print("> Used communication ops and corresponding stack")
                self._smallest_rank_print(json.dumps({k:[i.split(';') for i in v] for k,v in self.cc_logged_stack.items()}, indent=4))
                raise Exception("exit after first step when print cc stack")
            

            context.param_exp_avg, context.param_exp_avg_sq, context.param_adam_update, context.param_adam_ratio = self.mix_precision_optimizer_mon.fetch_mv(self,
                optimizer, self.param2name)
            
            for param, name in self.param2name.items():
                if "params_effrank" in self.config and name in self.config["params_effrank"]:
                    context.param_effective_rank[name] = eff_rank(param.detach())
                grad = param.main_grad if self.params_have_main_grad else param.grad
                if grad is None:
                    print_warn_log(f"grad is None: {name}, maybe something wrong happened.")
                    continue
                if self.wg_distribution:
                    context.param_weight_grad[name] = grad
                if self.mg_direction: 
                    if context.step == 0:
                        same_direction_ratio = torch.tensor(1.)
                    else:
                        same_direction_ratio = get_sign_matches(grad, context.param_exp_avg[name])
                    context.param_mg_direction[name] = same_direction_ratio

            tbtag_tensor_map = {}
            if self.wg_distribution:
                tbtag_tensor_map.update(self.generate_param_metrics('weight_grad', context.param_weight_grad))
            if self.mv_distribution:
                tbtag_tensor_map.update(self.generate_param_metrics('exp_avg', context.param_exp_avg))
                tbtag_tensor_map.update(self.generate_param_metrics('exp_avg_sq', context.param_exp_avg_sq))
            if self.mg_direction:
                tbtag_tensor_map.update(self.generate_param_metrics('mg_direction', context.param_mg_direction))
            # if not tbtag_tensor_map:
            #     return
            metric_dict = {}
            for metric_name in self.ops:
                metric_dict[metric_name] = get_metrics(metric_name, tbtag_tensor_map, self.eps)
            for k, c in self.cc_context.items():
                c.aggregate()
                cc_metrics = self.generate_cc_metrics(k, c)
                for op, m in cc_metrics.items():
                    metric_dict[op].update(m)
            if not metric_dict:
                return
            context.metric_list.append(metric_dict)
            return

        def optimizer_post_step_hook(optimizer, args, kwargs):
            context = self.optimizer_context[optimizer]
            rank = dist.get_rank() if dist.is_initialized() else None

            self.write_xy_tb(context.step)
            self.write_adhoc_check(context.step)

            if self.ur_distribution:
                for param_name, _ in context.param_adam_update.items():
                    self.update_heatmap_visualizer[param_name].visualize(get_summary_writer_tag_name(param_name, 'adam_update', rank), context.step, self.summary_writer)
                for param_name, _ in context.param_adam_ratio.items():
                    self.ratio_heatmap_visualizer[param_name].visualize(get_summary_writer_tag_name(param_name, 'adam_ratio', rank), context.step, self.summary_writer)

            for metric_name in self.ops:
                if not context.metric_list:
                    break
                write_metrics_tensorboard(metric_name, self.summary_writer, context.metric_list, context.step)
            context.metric_list.clear()
            context.step += 1

            return
        if not self.module_rank_list or (dist.is_initialized() and dist.get_rank() in self.module_rank_list):
            register_optimizer_step_pre_hook(optimizer_pre_step_hook)
            register_optimizer_step_post_hook(optimizer_post_step_hook)
        return

    def _smallest_rank_print(self, msg):
        if dist.is_initialized():
            if self.module_rank_list:
                if dist.get_rank() == min(self.module_rank_list):
                    print_info_log(msg)
            else:
                if dist.get_rank() == 0:
                    print_info_log(msg)
        else:
            print_info_log(msg)

    def _hook_module(self, target_names, module: torch.nn.Module, fwd_or_bkd):
        if '_modules' not in module.__dict__:
            # nothing to hook
            return 0

        def fwd_hook_fun(module, module_input, module_output):
            context: ModuleHookContext = self.module_fwd_hook_context_by_module[module]
            if self.print_struct:
                self.module_struct[context.module_name].update(
                    {"input": f"{get_param_struct(module_input)}", "output": f"{get_param_struct(module_output)}"})
                return
            if not self.xy_distribution:
                return
            if not context.format_by_arg:
                context.set_format_by_arg('input', self.config['targets'])
                context.set_format_by_arg('output', self.config['targets'])
            if not context.verified:
                if not context.ignore_in:
                    context.focused_in_col = validate_config_spec(context.format_by_arg['input'], module_input, context.module_name, 'input')
                context.focused_out_col = validate_config_spec(context.format_by_arg['output'], module_output, context.module_name, 'output')
                context.verified = True
            # expect output be tensor type
            tbtag_tensor_map = {}
            if not context.ignore_in:
                cared_input = module_input if context.focused_in_col is None else module_input[context.focused_in_col]
                tbtag_tensor_map.update(self.build_tbtag_tensor_map(context.module_name, 'input', cared_input))
            cared_output = module_output if context.focused_out_col is None else module_output[context.focused_out_col]
            tbtag_tensor_map.update(self.build_tbtag_tensor_map(context.module_name, 'output', cared_output))
            metric_dict = {}
            for metric_name in self.ops:
                metric_dict[metric_name] = get_metrics(metric_name, tbtag_tensor_map, self.eps)
            if context.micro_step == 0 and context.actv:
                print_warn_log(
                    f"actv context of {context.module_name} is not empty when first micro_step, maybe something wrong happened. Now clear it.")
                context.actv.clear()
            context.actv.append(metric_dict)

            context.micro_step += 1
            if context.micro_step == self.micro_batch_number:
                context.micro_step = 0
                context.step += 1
            return

        def bwd_hook_fun(module, input_grad, output_grad):
            context: ModuleHookContext = self.module_bwd_hook_context_by_module[module]
            if self.print_struct:
                self.module_struct[context.module_name].update(
                    {"input_grad": f"{get_param_struct(input_grad)}", "output_grad": f"{get_param_struct(output_grad)}"})
                return
            if not self.xy_distribution:
                return
            if not context.format_by_arg:
                context.set_format_by_arg('input_grad', self.config['targets'])
                context.set_format_by_arg('output_grad', self.config['targets'])
            if not context.verified:
                if not context.ignore_in:
                    context.focused_in_col = validate_config_spec(context.format_by_arg['input_grad'], input_grad, context.module_name, 'input_grad')
                context.focused_out_col = validate_config_spec(context.format_by_arg['output_grad'], output_grad, context.module_name, 'output_grad')
                context.verified = True

            tbtag_tensor_map = {}
            if not context.ignore_in:
                cared_input_grad = input_grad if context.focused_in_col is None else input_grad[context.focused_in_col]
                tbtag_tensor_map.update(self.build_tbtag_tensor_map(context.module_name, 'input_grad', cared_input_grad))
            cared_output_grad = output_grad if context.focused_out_col is None else output_grad[context.focused_out_col]
            tbtag_tensor_map.update(self.build_tbtag_tensor_map(context.module_name, 'output_grad', cared_output_grad))
            metric_dict = {}
            for metric_name in self.ops:
                metric_dict[metric_name] = get_metrics(metric_name, tbtag_tensor_map, self.eps)
            if context.micro_step == 0 and context.actvgrad:
                print_warn_log(f"actvgrad context of {context.module_name} is not empty when first micro_step, maybe something wrong happened. Now clear it.")
                context.actvgrad.clear()
            context.actvgrad.append(metric_dict)

            context.micro_step += 1
            if context.micro_step == self.micro_batch_number:
                context.micro_step = 0
                context.step += 1
            return

        hooked_count = 0
        for name, submodule in module.named_modules():
            self.module_struct[name] = {}
            if name in target_names:
                submodule.register_forward_hook(fwd_hook_fun)
                self.module_fwd_hook_context_by_module[submodule] = ModuleHookContext(name)
                if not self.forward_only:
                    submodule.register_full_backward_hook(bwd_hook_fun)
                    self.module_bwd_hook_context_by_module[submodule] = ModuleHookContext(name)
                print_rank_0(f"> {name} is monitored successfully")
                hooked_count += 1
        return hooked_count
