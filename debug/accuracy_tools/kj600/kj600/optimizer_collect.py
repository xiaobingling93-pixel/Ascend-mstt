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

    def _fetch_mv_in_adam(self, params2name, torch_opt, monitor):
        exp_avg_dict = defaultdict(float)
        exp_avg_sq_dict = defaultdict(float)
        update_dict = defaultdict()
        ratio_dict = defaultdict()

        for param, name in params2name.items():
            if param in self.fp16_to_fp32_param:
                param = self.fp16_to_fp32_param[param]
            
            if param in torch_opt.state:
                exp_avg = torch_opt.state[param]["exp_avg"]
                exp_avg_sq = torch_opt.state[param]["exp_avg_sq"]
                if monitor.mv_distribution:
                    exp_avg_dict[name] = exp_avg
                    exp_avg_sq_dict[name] = exp_avg_sq
                if monitor.mg_direction:
                    exp_avg_dict[name] = exp_avg
                if monitor.ur_distribution:
                    update_dict[name] = exp_avg / (torch.sqrt(exp_avg_sq) + torch_opt.defaults['eps'])
                    ratio_dict[name] = exp_avg / torch.sqrt(exp_avg_sq)
                    monitor.update_heatmap_visualizer[name].pre_cal(update_dict[name])
                    monitor.ratio_heatmap_visualizer[name].pre_cal(ratio_dict[name])
        return exp_avg_dict, exp_avg_sq_dict, update_dict, ratio_dict

    # parameter tensors we want to monitor and their names are in params2name_dict
    # base_optimizer is pytorch optimizer, wrapped_optimizer is a normal object with  base_optimizer
    def fetch_mv(self, monitor, torch_opt, params2name):
        mix_prec_opt = MixPrecsionOptimizerMon.wrapped_optimizer

        if not self.fp16_to_fp32_param and mix_prec_opt is not None:
            for fp16_group, fp32_group in zip(mix_prec_opt.float16_groups, mix_prec_opt.fp32_from_float16_groups):
                for fp16_param, fp32_param in zip(fp16_group, fp32_group):
                    self.fp16_to_fp32_param[fp16_param] = fp32_param
        return self._fetch_mv_in_adam(params2name, torch_opt, monitor)

class MegatronDistributedOptimizerMon(MixPrecsionOptimizerMon):
    def fetch_mv(self, monitor, torch_opt, params2name):
        mix_prec_opt = MixPrecsionOptimizerMon.wrapped_optimizer
        assert hasattr(mix_prec_opt, "model_float16_groups") and hasattr(mix_prec_opt, "shard_fp32_from_float16_groups"), \
            "megatron distributed optimizer should have model_float16_groups and shard_fp32_from_float16_groups, if not, please check megatron-lm version"
        if not self.fp16_to_fp32_param and mix_prec_opt is not None:
            for fp16_group, shard_fp32_group in zip(mix_prec_opt.model_float16_groups, mix_prec_opt.shard_fp32_from_float16_groups):
                for fp16_param, shard_fp32_param in zip(fp16_group, shard_fp32_group):
                    self.fp16_to_fp32_param[fp16_param] = shard_fp32_param

        return self._fetch_mv_in_adam(params2name, torch_opt, monitor)

class DummyOptimizerMon(MixPrecsionOptimizerMon):
    def fetch_mv(self, monitor, torch_opt, params2name):
        return None, None, None, None

class OptimizerMonFactory:
    @staticmethod
    def create_optimizer_mon(opt_ty:str):
        if opt_ty == "Megatron_Float16OptimizerWithFloat16Params":
            return MixPrecsionOptimizerMon()
        if opt_ty == "Megatron_DistributedOptimizer":
            return MegatronDistributedOptimizerMon()
        if opt_ty == None or opt_ty == "unknown":
            return DummyOptimizerMon()
        assert opt_ty != None, "opt_ty should be Megatron_Float16OptimizerWithFloat16Params or Megatron_DistributedOptimizer or None or unknown"