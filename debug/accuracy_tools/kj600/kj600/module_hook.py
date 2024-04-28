import os
import uuid
from collections import defaultdict
from typing import List
from datetime import datetime
import torch
from torch.nn.modules.module import register_module_forward_hook
import torch.distributed as dist
from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook
from torch.utils.tensorboard import SummaryWriter
from kj600.features import square_sum
from kj600.module_spec_verifier import get_config, validate_config_spec
from kj600.optimizer_collect import MixPrecsionOptimizerMon, print_rank_0
from kj600.features import eff_rank
from kj600.visualizer import HeatmapVisualizer


def get_summary_writer_tag_name(module_or_param_name:str, tag:str, rank):
    if rank is None:
        return f"{module_or_param_name}/{tag}"
    else:
        return f"{module_or_param_name}/{rank}/{tag}"


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
        self.param_gnorm = defaultdict(float)
        self.param_exp_avg_norm = defaultdict(float)
        self.param_exp_avg_sq_norm = defaultdict(float)
        self.param_effective_rank = defaultdict(float)
        self.param_adam_update = defaultdict()
        self.param_adam_ratio = defaultdict()


class TrainerMon:
    
    @staticmethod
    def set_wrapped_optimizer(_wrapped_optimizer):
        MixPrecsionOptimizerMon.set_wrapped_optimizer(_wrapped_optimizer)

    def __init__(self, config_file_path) -> None:
        self.module_fwd_hook_context_by_module = defaultdict(ModuleHookContext)
        self.module_bwd_hook_context_by_module = defaultdict(ModuleHookContext)
        self.optimizer_context = defaultdict(OptimizerContext)
        self.params_have_main_grad = True
        self.config = get_config(config_file_path)
        self.module_rank_list = [int(rank) for rank in self.config.get("module_ranks", "").split(',') if rank.strip()]
        self.ur_distribution = self.config.get('ur_distribution', False)

        self.optimizer_hooked = False
        output_base_dir = os.getenv('KJ600_OUTPUT_DIR', './kj600_output')
        cur_time = datetime.now().strftime('%b%d_%H-%M-%S')
        unique_id = str(uuid.uuid4())[:8]
        if dist.is_initialized():
            if (dist.get_rank() in self.module_rank_list) or len(self.module_rank_list) == 0:
                self.summary_writer = SummaryWriter(os.path.join(output_base_dir, f"{cur_time}-rank{dist.get_rank()}-{unique_id}"))
        else:
            self.summary_writer = SummaryWriter(os.path.join(output_base_dir, f"{cur_time}-{unique_id}"))
        # A HeatmapVisualizer instance is associated with an image
        self.update_heatmap_visualizer = defaultdict(HeatmapVisualizer)
        self.ratio_heatmap_visualizer = defaultdict(HeatmapVisualizer)
        self.micro_batch_number = 0

        self.param_name_list = []
        self.param2name = defaultdict(str)

        self.mix_precision_optimizer_mon = MixPrecsionOptimizerMon()
        return
    
    def __del__(self):
        if hasattr(self, "summary_writer"):
            self.summary_writer.close()

    def _hook_module(self, target_name:str, module: torch.nn.Module, fwd_or_bkd):
        paths = target_name.split('.')
        if '_modules' not in module.__dict__:
            # nothing to hook
            return 0
        
        def fwd_hook_fun(module, module_input, module_output):
            context = self.module_fwd_hook_context_by_module[module]
            if not context.format_by_arg:
                context.set_format_by_arg('input', self.config['targets'])
                context.set_format_by_arg('output', self.config['targets'])
            if not context.verified:
                if not context.ignore_in:
                    context.focused_in_col = validate_config_spec(context.format_by_arg['input'], module_input, context.module_name, 'input')
                context.focused_out_col = validate_config_spec(context.format_by_arg['output'], module_output, context.module_name, 'output')
                context.verified = True
            # expect output be tensor type
            if not context.ignore_in:
                cared_input = module_input if context.focused_in_col is None else module_input[context.focused_in_col]
                cared_input_cal_result = square_sum(cared_input)
            else:
                cared_input_cal_result = None
            cared_output = module_output if context.focused_out_col is None else module_output[context.focused_out_col]
            context.actv.append((cared_input_cal_result, square_sum(cared_output)))

            context.micro_step += 1
            if context.micro_step == self.micro_batch_number:
                context.micro_step = 0
                context.step += 1
            return
        
        def bwd_hook_fun(module, input_grad, output_grad):
            context = self.module_bwd_hook_context_by_module[module]
            if not context.format_by_arg:
                context.set_format_by_arg('input_grad', self.config['targets'])
                context.set_format_by_arg('output_grad', self.config['targets'])
            if not context.verified:
                if not context.ignore_in:
                    context.focused_in_col = validate_config_spec(context.format_by_arg['input_grad'], input_grad, context.module_name, 'input_grad')
                context.focused_out_col = validate_config_spec(context.format_by_arg['output_grad'], output_grad, context.module_name, 'output_grad')
                context.verified = True
            if not context.ignore_in:
                cared_input_grad = input_grad if context.focused_in_col is None else input_grad[context.focused_in_col]
                cared_input_grad_cal_result = square_sum(cared_input_grad)
            else:
                cared_input_grad_cal_result = None
            cared_output_grad = output_grad if context.focused_out_col is None else output_grad[context.focused_out_col]
            context.actvgrad.append((cared_input_grad_cal_result, square_sum(cared_output_grad)))
            context.micro_step += 1
            if context.micro_step == self.micro_batch_number:
                context.micro_step = 0
                context.step += 1
            return
        
        for name, submodule in module.named_modules():
            if name == target_name:
                submodule.register_forward_hook(fwd_hook_fun)
                self.module_fwd_hook_context_by_module[submodule] = ModuleHookContext(name)
                submodule.register_full_backward_hook(bwd_hook_fun)
                self.module_bwd_hook_context_by_module[submodule] = ModuleHookContext(name)
                print_rank_0(f"> {name} is monitored successfully")
                return 1
        return 0

    def hook_modules(self, model:torch.nn.Module, global_batch_size, dp, micro_batch_size, fwd_or_bkd, params_have_main_grad=True): 
        # fwd=0, bkd=1
        # targets is module name list like ["xx.xxx1", "xxx.xxx2"] which can be obtained when first run. 
        print_rank_0("> module names:")
        for name, _ in model.named_modules():
            print_rank_0(f"\t{name}")
        self.micro_batch_number = global_batch_size // dp // micro_batch_size
        
        if not self.module_rank_list or (dist.is_initialized() and dist.get_rank() in self.module_rank_list):
            hooked = 0
            for target, _ in self.config['targets'].items():
                hooked += self._hook_module(target, model, fwd_or_bkd=0)
            print_rank_0(f"> {hooked} out of {len(self.config['targets'])} are monitored.")
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
        self.params_have_main_grad = params_have_main_grad
        return
    
    def hook_optimizer(self):
        # in DDP by default use params_have_main_grad
        def optimizer_pre_step_hook(optimizer, args, kwargs):
            context = self.optimizer_context[optimizer]
            for param, name in self.param2name.items():
                grad_for_norm = param.main_grad if self.params_have_main_grad else param.grad
                context.param_gnorm[name] = grad_for_norm.detach().norm()
                if "params_effrank" in self.config and name in self.config["params_effrank"]:
                    context.param_effective_rank[name] = eff_rank(param.detach())

            context.param_exp_avg_norm, context.param_exp_avg_sq_norm, context.param_adam_update, context.param_adam_ratio = self.mix_precision_optimizer_mon.fetch_mv(
                optimizer, self.param2name, self.update_heatmap_visualizer, self.ratio_heatmap_visualizer, self.ur_distribution)
            return
        
        def optimizer_post_step_hook(optimizer, args, kwargs):
            context = self.optimizer_context[optimizer]
            rank = dist.get_rank() if dist.is_initialized() else None
            for _, fwd_context in self.module_fwd_hook_context_by_module.items():
                if not len(fwd_context.actv) == self.micro_batch_number:
                    raise Exception(f"fwd_context.actv not equal to micro_batch_number: {len(fwd_context.actv)}, {self.micro_batch_number}")
                if not fwd_context.ignore_in:
                    x_norm = sum([x.item() for x, _ in fwd_context.actv])
                    self.summary_writer.add_scalar(get_summary_writer_tag_name(fwd_context.module_name, 'input', rank), x_norm, context.step)
                y_norm = sum([y.item() for _, y in fwd_context.actv])
                self.summary_writer.add_scalar(get_summary_writer_tag_name(fwd_context.module_name, 'output', rank), y_norm, context.step)
                fwd_context.actv.clear()
                
            for _, bwd_context in self.module_bwd_hook_context_by_module.items():
                if not len(bwd_context.actvgrad) == self.micro_batch_number:
                    raise Exception(f"fwd_context.actvgrad not equal to micro_batch_number: {len(fwd_context.actvgrad)}, {self.micro_batch_number}")
                if not bwd_context.ignore_in:
                    x_grad_norm = sum([x.item() for x, _ in bwd_context.actvgrad])
                    self.summary_writer.add_scalar(get_summary_writer_tag_name(bwd_context.module_name, 'input_grad', rank), x_grad_norm, context.step)
                y_grad_norm = sum([y.item() for _, y in bwd_context.actvgrad])
                self.summary_writer.add_scalar(get_summary_writer_tag_name(bwd_context.module_name, 'output_grad', rank), y_grad_norm, context.step)
                bwd_context.actvgrad.clear()

            for param_name, grad_norm in context.param_gnorm.items():
                self.summary_writer.add_scalar(get_summary_writer_tag_name(param_name, 'weight_grad', rank), grad_norm.item(), context.step)

            for param_name, exp_avg_norm in context.param_exp_avg_norm.items():
                self.summary_writer.add_scalar(get_summary_writer_tag_name(param_name, 'exp_avg_norm', rank), exp_avg_norm.item(), context.step)
            for param_name, exp_avg_sq_norm in context.param_exp_avg_sq_norm.items():
                self.summary_writer.add_scalar(get_summary_writer_tag_name(param_name, 'exp_avg_sq_norm', rank), exp_avg_sq_norm.item(), context.step)
            if self.ur_distribution:
                for param_name, _ in context.param_adam_update.items():
                    self.update_heatmap_visualizer[param_name].visualize(get_summary_writer_tag_name(param_name, 'adam_update', rank), context.step, self.summary_writer)
                for param_name, _ in context.param_adam_ratio.items():
                    self.ratio_heatmap_visualizer[param_name].visualize(get_summary_writer_tag_name(param_name, 'adam_ratio', rank), context.step, self.summary_writer)
            context.step += 1

            return
        if not self.module_rank_list or (dist.is_initialized() and dist.get_rank() in self.module_rank_list):
            register_optimizer_step_pre_hook(optimizer_pre_step_hook)
            register_optimizer_step_post_hook(optimizer_post_step_hook)
        return
