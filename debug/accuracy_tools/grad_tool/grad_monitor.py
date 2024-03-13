import os
import torch

from grad_tool.level_adapter import Level, LevelAdapter
from grad_tool.grad_stat_csv import GradStatCsv
from grad_tool.utils import get_config, check_numeral_list_ascend, ListCache, data_in_list_target,\
    write_csv, make_localtime_dir, get_rank_id


class GradientMonitor:
    default_bounds = [-10, -1, -0.1, -0.01, -0.001, 0, 0.001, 0.01, 0.1, 1, 10]

    def __init__(self, config_filepath):
        config = get_config(config_filepath)
        self._level_adp: Level = LevelAdapter.level_adapter(config.get("level"))
        self._param_list = config.get('param_list')
        self._target_ranks = config.get("rank")
        self._target_step = config.get("step")
        self._bounds = config.get("bounds")
        if not self._bounds or len(self._bounds) == 0:
            self._bounds = GradientMonitor.default_bounds
        check_numeral_list_ascend(self._bounds)
        self._output_path = make_localtime_dir(config.get("output_path"))
        self._step = -1
        self._list_cache = ListCache()
    
    @staticmethod
    def hook_fun(param_name, f):
        def backward_hook(grad):
            f(param_name, grad)
        return backward_hook
    
    def model_backward_hook(self, module, gin, gout):
        if not hasattr(self, "_rank"):
            setattr(self, "_rank", get_rank_id(gout))
        if torch.distributed.is_initialized() and not data_in_list_target(getattr(self, "_rank"), self._target_ranks):
            return
        self._list_cache.flush()
        self._step += 1
        if not data_in_list_target(self._step, self._target_step):
            return
        output_path = f'{self._output_path}/rank_{self._rank}/grad_summary_{self._step}.csv'
        write_csv(output_path, [], GradStatCsv.generate_csv_header(level=self._level_adp, bounds=self._bounds))
        self._list_cache.set_output_file(output_path)
        
    def save_grad_stat(self, param_name, grad):
        if not hasattr(self, "_rank"):
            raise AttributeError("grad monitor need attribute {_rank} when save grad stat")
        if torch.distributed.is_initialized() and not data_in_list_target(getattr(self, "_rank"), self._target_ranks):
            return
        if not data_in_list_target(self._step, self._target_step):
            return
        grad_info = GradStatCsv.generate_csv_line(
            level=self._level_adp, 
            param_name=param_name, 
            grad=grad,
            bounds=self._bounds)
        self._list_cache.append(grad_info)
        self._level_adp.save_grad_direction(param_name, grad, f'{self._output_path}/rank_{self._rank}/step_{self._step}')


    def monitor(self, model):
        model.register_full_backward_hook(self.model_backward_hook)
        for param_name, param in model.named_parameters():
            if not data_in_list_target(param_name, self._param_list):
                continue
            param.register_hook(GradientMonitor.hook_fun(param_name, self.save_grad_stat))