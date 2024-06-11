from collections import defaultdict
import torch
import torch.distributed as dist
from kj600.visualizer import HeatmapVisualizer



def print_rank_0(message, debug=False, force=False):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(message)
    else:
        print(message)


class MixPrecsionOptimizerMon:
    wrapped_optimizer = None

    @staticmethod
    def set_wrapped_optimizer(_wrapped_optimizer):
        MixPrecsionOptimizerMon.wrapped_optimizer = _wrapped_optimizer

    def __init__(self) -> None:
        self.fp16_to_fp32_param = {}
        
    # parameter tensors we want to monitor and their names are in params2name_dict
    # base_optimizer is pytorch optimizer, wrapped_optimizer is a normal object with  base_optimizer
    def fetch_mv(self, torch_opt, params2name, update_heatmap_visualizer, ratio_heatmap_visualizer, ur_distribution, mg_direction):
        mix_prec_opt = MixPrecsionOptimizerMon.wrapped_optimizer

        if not self.fp16_to_fp32_param and mix_prec_opt is not None:
            for fp16_group, fp32_group in zip(mix_prec_opt.float16_groups, mix_prec_opt.fp32_from_float16_groups):
                for fp16_param, fp32_param in zip(fp16_group, fp32_group):
                    self.fp16_to_fp32_param[fp16_param] = fp32_param

        exp_avg_norm_dict = defaultdict(float)
        exp_avg_sign_dict = defaultdict(int)
        exp_avg_sq_norm_dict = defaultdict(float)
        update_dict = defaultdict()
        ratio_dict = defaultdict()

        for param, name in params2name.items():
            if param in self.fp16_to_fp32_param:
                param = self.fp16_to_fp32_param[param]
            
            if param in torch_opt.state:
                exp_avg = torch_opt.state[param]["exp_avg"]
                exp_avg_sq = torch_opt.state[param]["exp_avg_sq"]
                exp_avg_norm = exp_avg.detach().norm()
                exp_avg_sq_norm = exp_avg_sq.detach().norm()
                exp_avg_norm_dict[name] = exp_avg_norm
                exp_avg_sq_norm_dict[name] = exp_avg_sq_norm
                if mg_direction:
                    exp_avg_sign_dict[name] = exp_avg.detach().sign()
                if ur_distribution:
                    update_dict[name] = exp_avg / (torch.sqrt(exp_avg_sq) + torch_opt.defaults['eps'])
                    ratio_dict[name] = exp_avg / torch.sqrt(exp_avg_sq)
                    update_heatmap_visualizer[name].pre_cal(update_dict[name])
                    ratio_heatmap_visualizer[name].pre_cal(ratio_dict[name])
                    
        return exp_avg_norm_dict, exp_avg_sign_dict, exp_avg_sq_norm_dict, update_dict, ratio_dict
