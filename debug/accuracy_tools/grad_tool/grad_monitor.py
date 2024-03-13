import os
import torch
from grad_tool.level_adapter import Level, LevelAdapter
from grad_tool.grad_stat_csv import GradStatCsv
from grad_tool.utils import get_config, check_numeral_list_ascend, ListCache, data_in_list_target,\
    write_csv, get_rank_id, print_info_log, create_directory, print_warn_log


class GradientMonitor:
    default_bounds = [-10, -1, -0.1, -0.01, -0.001, 0, 0.001, 0.01, 0.1, 1, 10]

    def __init__(self, config_filepath):
        config = get_config(config_filepath)
        self._level_adp: Level = LevelAdapter.level_adapter(config.get("level"))
        self._param_list = config.get('param_list')
        self._target_ranks = config.get("rank")
        print_info_log(f"target rank {self._target_ranks}")
        self._target_step = config.get("step")
        print_info_log(f"target step {self._target_step}")
        self._bounds = config.get("bounds")
        if not self._bounds or len(self._bounds) == 0:
            self._bounds = GradientMonitor.default_bounds
        check_numeral_list_ascend(self._bounds)
        self._output_path = config.get("output_path")
        if not os.path.isdir(self._output_path):
            create_directory(self._output_path)
        else:
            print_warn_log(f"the file in {self._output_path} will be recoverd")
        self._step = -1
        self._list_cache = ListCache()
    
    @staticmethod
    def _hook_fun(param_name, f):
        def backward_hook(grad):
            f(param_name, grad)
        return backward_hook
    
    def _rank_in_targets(self):
        if not hasattr(self, "_rank"):
            raise AttributeError("grad monitor need attribute {_rank}")
        return not torch.distributed.is_initialized() or data_in_list_target(getattr(self, "_rank"), self._target_ranks)
        
    def _model_backward_hook(self, module, gin, gout):
        self._step += 1
        if not hasattr(self, "_rank"):
            setattr(self, "_rank", get_rank_id(gout))
            print_info_log(f"rank_{self._rank} exists")
        if not self._rank_in_targets():
            return
        self._list_cache.flush()
        if not data_in_list_target(self._step, self._target_step):
            return
        print_info_log(f"result generate: rank_{self._rank} step_{self._step}")
        output_path = os.path.join(self._output_path, f"rank_{getattr(self, '_rank')}", f"grad_summary_{self._step}.csv")
        write_csv(output_path, [], GradStatCsv.generate_csv_header(level=self._level_adp, bounds=self._bounds))
        self._list_cache.set_output_file(output_path)
        
    def _save_grad_stat(self, param_name, grad):
        if not self._rank_in_targets():
            return
        if not data_in_list_target(self._step, self._target_step):
            return
        print_info_log(f"param result: rank{self._rank} step{self._step} {param_name}")
        grad_info = GradStatCsv.generate_csv_line(
            level=self._level_adp, 
            param_name=param_name, 
            grad=grad,
            bounds=self._bounds)
        self._list_cache.append(grad_info)
        self._level_adp.save_grad_direction(param_name, grad, f'{self._output_path}/rank_{self._rank}/step_{self._step}')

    def monitor(self, model):
        last_module = None
        for name, module in model.named_modules():
            last_module = module
        last_module.register_backward_hook(self._model_backward_hook)
        for param_name, param in model.named_parameters():
            if not data_in_list_target(param_name, self._param_list):
                continue
            if param is None or param.requires_grad == False:
                continue
            param.register_hook(GradientMonitor._hook_fun(param_name, self._save_grad_stat))

    def save_grad(self, model):
        self._step += 1
        if not hasattr(self, "_rank"):
            setattr(self, "_rank", get_rank_id(next(model.parameters())))
        if not self._rank_in_targets():
            return
        if not data_in_list_target(self._step, self._target_step):
            return
        print_info_log(f"save grad rank_{getattr(self, '_rank')} step_{self._step}")
        output_path = os.path.join(self._output_path, f"rank_{getattr(self, '_rank')}", f"grad_summary_{self._step}.csv")
        write_csv(output_path, [], GradStatCsv.generate_csv_header(level=self._level_adp, bounds=self._bounds))
        self._list_cache.set_output_file(output_path)
        for param_name, param in model.named_parameters():
            if not data_in_list_target(param_name, self._param_list):
                continue
            if param.grad is not None:
                grad = param.grad
            elif param.main_grad is not None:
                grad = param.main_grad
            else:
                continue
            self._save_grad_stat(param_name, grad)
            print_info_log(f"{param_name} is saved")
        self._list_cache.flush()
