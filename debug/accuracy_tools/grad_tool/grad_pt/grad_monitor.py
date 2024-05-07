import os
from collections import defaultdict

import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook
from grad_tool.common.base_monitor import BaseMonitor
from grad_tool.grad_pt.level_adapter import Level, LevelAdapter
from grad_tool.grad_pt.grad_stat_csv import GradStatCsv
from grad_tool.common.utils import check_numeral_list_ascend, data_in_list_target,\
    write_csv, print_info_log, create_directory, print_warn_log
from grad_tool.grad_pt.utils import get_rank_id, print_rank_0


class PtGradientMonitor(BaseMonitor):
    default_bounds = [-10, -1, -0.1, -0.01, -0.001, 0, 0.001, 0.01, 0.1, 1, 10]

    def __init__(self, config_filepath):
        super(PtGradientMonitor, self).__init__(config_filepath)
        self._level_adp: Level = LevelAdapter.level_adapter(self.config.get("level"))
        self._param_list = self.config.get('param_list')
        self._target_ranks = self.config.get("rank")
        print_info_log(f"target rank {self._target_ranks}")
        self._target_step = self.config.get("step")
        print_info_log(f"target step {self._target_step}")
        self._bounds = self.config.get("bounds")
        if not self._bounds or len(self._bounds) == 0:
            self._bounds = PtGradientMonitor.default_bounds
        check_numeral_list_ascend(self._bounds)
        self._output_path = self.config.get("output_path")
        if not os.path.isdir(self._output_path):
            create_directory(self._output_path)
        else:
            print_warn_log(f"the file in {self._output_path} will be recoverd")
        self._step = -1
        self._param2name = defaultdict(str)

    def _rank_in_targets(self):
        if not hasattr(self, "_rank"):
            raise AttributeError("grad monitor need attribute {_rank}")
        return not torch.distributed.is_initialized() or data_in_list_target(getattr(self, "_rank"), self._target_ranks)

    def _hook_optimizer(self):
        def optimizer_pre_step_hook(optimizer, args, kargs):
            self._step += 1
            if not data_in_list_target(self._step, self._target_step):
                return
            output_lines = []
            for param, param_name in self._param2name.items():
                if not data_in_list_target(param_name, self._param_list):
                    continue
                grad = param.main_grad if hasattr(param, "main_grad") else param.grad
                if grad is None:
                    print_info_log(f"grad is None: {param_name}")
                    continue
                grad_info = GradStatCsv.generate_csv_line(
                    level=self._level_adp, 
                    param_name=param_name, 
                    grad=grad,
                    bounds=self._bounds)
                output_lines.append(grad_info)
                self._level_adp.save_grad_direction(param_name, grad, f'{self._output_path}/rank_{self._rank}/step_{self._step}')
            output_path = os.path.join(self._output_path, f"rank_{getattr(self, '_rank')}", f"grad_summary_{self._step}.csv")
            write_csv(output_path, output_lines, GradStatCsv.generate_csv_header(level=self._level_adp, bounds=self._bounds))
        register_optimizer_step_pre_hook(optimizer_pre_step_hook)

    def monitor(self, model):
        print_rank_0("> parameter names:")
        for name, param in model.named_parameters():
            self._param2name[param] = name
            print_rank_0(f"\t{name}")
        setattr(self, "_rank", get_rank_id())
        if torch.distributed.is_initialized() and not data_in_list_target(getattr(self, "_rank"), self._target_ranks):
            return
        self._hook_optimizer()
