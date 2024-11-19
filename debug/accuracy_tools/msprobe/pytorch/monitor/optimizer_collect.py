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

from abc import ABC, abstractmethod
from collections import defaultdict

import torch
import torch.distributed as dist

from msprobe.core.common.log import logger
from msprobe.pytorch.monitor.utils import MVResult, MV_Grad_Result


class OptimizerMon(ABC):
    wrapped_optimizer = None

    def __init__(self) -> None:
        self.fp16_to_fp32_param = {}

    @classmethod
    def set_wrapped_optimizer(cls, wrapped_optimizer):
        cls.wrapped_optimizer = wrapped_optimizer

    @abstractmethod
    def fetch_mv(self, monitor, torch_opt, params2name):
        pass

    def _fetch_mv_in_adam(self, monitor, torch_opt, params2name):
        exp_avg_dict = defaultdict(float)
        exp_avg_sq_dict = defaultdict(float)
        update_dict = defaultdict()
        ratio_dict = defaultdict()

        for param, name in params2name.items():
            if param in self.fp16_to_fp32_param:
                param = self.fp16_to_fp32_param[param]

            if param in torch_opt.state:
                state_param = torch_opt.state.get(param, None)
                exp_avg = state_param.get("exp_avg", None)
                exp_avg_sq = state_param.get("exp_avg_sq", None)
                if exp_avg is None or exp_avg_sq is None:
                    logger.warning(f"exp_avg or exp_avg_sq of {name} is None, maybe something wrong happened.")
                    continue
                if monitor.mv_distribution:
                    exp_avg_dict[name] = exp_avg
                    exp_avg_sq_dict[name] = exp_avg_sq
                if monitor.mg_direction:
                    exp_avg_dict[name] = exp_avg
                if monitor.ur_distribution:
                    if 'step' in state_param:
                        step = state_param['step']  # Optimizer from pytorch or FusedAdam from apex(used by megatron)
                    elif 'step' in torch_opt.param_groups[0]:
                        step = torch_opt.param_groups[0]['step']  # AdamW from mindspeed
                    else:
                        logger.warning(f"step of {name} is None, maybe something wrong happened.")
                        continue
                    exp_avg_hat = exp_avg / (1 - torch_opt.defaults['betas'][0] ** step)
                    exp_avg_sq_hat = exp_avg_sq / (1 - torch_opt.defaults['betas'][1] ** step)
                    update_dict[name] = exp_avg_hat / (torch.sqrt(exp_avg_sq_hat) + torch_opt.defaults['eps'])
                    ratio_dict[name] = exp_avg_hat / torch.sqrt(exp_avg_sq_hat)
                    monitor.update_heatmap_visualizer[name].pre_cal(update_dict[name])
                    monitor.ratio_heatmap_visualizer[name].pre_cal(ratio_dict[name])
        return MVResult(exp_avg=exp_avg_dict, exp_avg_sq=exp_avg_sq_dict, update=update_dict, ratio=ratio_dict)

    def _fetch_mv_grad_in_adam(self, monitor, torch_opt, params2name, name2indices, fp32_partitioned_groups_flat):
        exp_avg_dict = defaultdict(float)
        exp_avg_sq_dict = defaultdict(float)
        update_dict = defaultdict()
        ratio_dict = defaultdict()
        param2name = defaultdict()
        mix_prec_opt = OptimizerMon.wrapped_optimizer

        for name in params2name.values():
            start_idx, end_idx = name2indices[name]
            if start_idx > end_idx:
                continue
            fp32_param = fp32_partitioned_groups_flat[start_idx: end_idx]
            fp32_param.grad = fp32_partitioned_groups_flat.grad[start_idx: end_idx]
            param2name[fp32_param] = name
            state_param = list(mix_prec_opt.state.values())[0]
            exp_avg = state_param.get("exp_avg", None)
            exp_avg_sq = state_param.get("exp_avg_sq", None)
            if exp_avg is None or exp_avg_sq is None:
                logger.warning(f"exp_avg or exp_avg_sq of {name} is None, maybe something wrong happened.")
                continue
            exp_avg = exp_avg[start_idx: end_idx]
            exp_avg_sq = exp_avg_sq[start_idx: end_idx]
            if monitor.mv_distribution:
                exp_avg_dict[name] = exp_avg
                exp_avg_sq_dict[name] = exp_avg_sq
            if monitor.mg_direction:
                exp_avg_dict[name] = exp_avg
            if monitor.ur_distribution:
                if 'step' in state_param:
                    step = state_param['step']  # Optimizer from pytorch or FusedAdam from apex(used by megatron)
                elif 'step' in torch_opt.param_groups[0]:
                    step = torch_opt.param_groups[0]['step']  # AdamW from mindspeed
                else:
                    logger.warning(f"step of {name} is None, maybe something wrong happened.")
                    continue
                exp_avg_hat = exp_avg / (1 - torch_opt.defaults['betas'][0] ** step)
                exp_avg_sq_hat = exp_avg_sq / (1 - torch_opt.defaults['betas'][1] ** step)
                update_dict[name] = exp_avg_hat / (torch.sqrt(exp_avg_sq_hat) + torch_opt.defaults['eps'])
                ratio_dict[name] = exp_avg_hat / torch.sqrt(exp_avg_sq_hat)
                monitor.update_heatmap_visualizer[name].pre_cal(update_dict[name])
                monitor.ratio_heatmap_visualizer[name].pre_cal(ratio_dict[name])
        return MV_Grad_Result(exp_avg=exp_avg_dict, exp_avg_sq=exp_avg_sq_dict, update=update_dict, ratio=ratio_dict,
                              grad=param2name)


class MixPrecisionOptimizerMon(OptimizerMon):
    """
    混合精度优化器监控类。在混合精度训练中监控和管理优化器。
    混合精度训练通过适当降低某些计算的精度来加速训练过程并减少内存消耗。
    """

    def fetch_mv(self, monitor, torch_opt, params2name):
        """
        parameter tensors we want to monitor and their names are in params2name_dict
        base_optimizer is pytorch optimizer, wrapped_optimizer is a normal object with  base_optimizer
        """
        mix_prec_opt = self.wrapped_optimizer

        if not self.fp16_to_fp32_param and mix_prec_opt is not None:
            for fp16_group, fp32_group in zip(mix_prec_opt.float16_groups, mix_prec_opt.fp32_from_float16_groups):
                for fp16_param, fp32_param in zip(fp16_group, fp32_group):
                    self.fp16_to_fp32_param[fp16_param] = fp32_param
        return self._fetch_mv_in_adam(monitor, torch_opt, params2name)


class MegatronDistributedOptimizerMon(OptimizerMon):
    """
    分布式优化器监控类。在分布式训练过程中收集和报告优化器的相关信息，以便于调试和性能优化。
    """
    def fetch_mv(self, monitor, torch_opt, params2name):
        mix_prec_opt = self.wrapped_optimizer
        if not (hasattr(mix_prec_opt, "model_float16_groups") and
                hasattr(mix_prec_opt, "shard_fp32_from_float16_groups")):
            raise Exception(
                "megatron distributed optimizer should have model_float16_groups and shard_fp32_from_float16_groups, "
                "if not, please check megatron-lm version")
        if not self.fp16_to_fp32_param and mix_prec_opt is not None:
            for fp16_group, shard_fp32_group in zip(mix_prec_opt.model_float16_groups, mix_prec_opt.shard_fp32_from_float16_groups):
                for fp16_param, shard_fp32_param in zip(fp16_group, shard_fp32_group):
                    self.fp16_to_fp32_param[fp16_param] = shard_fp32_param

        return self._fetch_mv_in_adam(monitor, torch_opt, params2name)


class MegatronFP32OptimizerMon(OptimizerMon):
    def fetch_mv(self, monitor, torch_opt, params2name):
        return self._fetch_mv_in_adam(monitor, torch_opt, params2name)


class DeepSpeedZeroOptimizerStage3Mon(OptimizerMon):
    def get_param_index(self, params2name, name2index):
        mix_prec_opt = OptimizerMon.wrapped_optimizer
        fp16_groups = mix_prec_opt.fp16_partitioned_groups[0]
        name2indices = defaultdict()
        index_length = defaultdict()
        index = 0
        for idx, param in enumerate(fp16_groups):
            index_length[idx] = (index, index + len(param))
            index += len(param)
        for _, name in params2name.items():
            idx = name2index[name]
            start_idx, end_idx = index_length[idx]
            name2indices[name] = (start_idx, end_idx)
        return name2indices

    def fetch_mv(self, monitor, torch_opt, params2name, name2indices):
        mix_prec_opt = OptimizerMon.wrapped_optimizer
        fp32_partitioned_groups_flat = mix_prec_opt.fp32_partitioned_groups_flat[0]
        return self._fetch_mv_grad_in_adam(monitor, torch_opt, params2name, name2indices, fp32_partitioned_groups_flat)


class DeepSpeedZeroOptimizerStage1or2Mon(OptimizerMon):
    def get_param_index(self, params2name, name2index):

        def get_new_index(group_index, start_idx, end_idx, length):
            new_start_idx = start_idx - group_index
            new_end_idx = end_idx - group_index
            return max(new_start_idx, 0), min(new_end_idx, length)

        mix_prec_opt = OptimizerMon.wrapped_optimizer
        bf16_groups = mix_prec_opt.bit16_groups[0]
        flatten_partitioned_fp32_groups = mix_prec_opt.single_partition_of_fp32_groups[0]
        padding = mix_prec_opt.groups_padding[0]
        name2indices = defaultdict()
        index_length = defaultdict()
        index = 0
        group_idx = dist.get_rank(group=mix_prec_opt.real_dp_process_group[0])
        group_index = group_idx * len(flatten_partitioned_fp32_groups)

        if group_idx == dist.get_world_size(group=mix_prec_opt.real_dp_process_group[0]) - 1:
            need_padding = True
        else:
            need_padding = False
        for idx, param in enumerate(bf16_groups):
            param_length = len(param.flatten())
            index_length[idx] = (index, index + param_length)
            index += param_length
        for _, name in params2name.items():
            idx = name2index[name]
            start_idx, end_idx = index_length[idx]
            new_start_idx, new_end_idx = get_new_index(group_index, start_idx, end_idx, len(flatten_partitioned_fp32_groups))
            if need_padding and idx == len(bf16_groups) - 1:
                new_end_idx -= padding
            name2indices[name] = (new_start_idx, new_end_idx)
        return name2indices

    def fetch_mv(self, monitor, torch_opt, params2name, name2indices):
        mix_prec_opt = OptimizerMon.wrapped_optimizer
        fp32_partitioned_groups_flat = mix_prec_opt.single_partition_of_fp32_groups[0]
        return self._fetch_mv_grad_in_adam(monitor, torch_opt, params2name, name2indices, fp32_partitioned_groups_flat)


class DummyOptimizerMon(OptimizerMon):
    def fetch_mv(self, monitor, torch_opt, params2name):
        return MVResult(exp_avg=None, exp_avg_sq=None, update=None, ratio=None)


class OptimizerMonFactory:
    _optimizer_mon_map = {
        "Megatron_Float16OptimizerWithFloat16Params": MixPrecisionOptimizerMon,
        "Megatron_DistributedOptimizer": MegatronDistributedOptimizerMon,
        "Megatron_FP32Optimizer": MegatronFP32OptimizerMon,
        "DeepSpeedZeroOptimizer_Stage1_or_2": DeepSpeedZeroOptimizerStage1or2Mon,
        "DeepSpeedZeroOptimizer_Stage3": DeepSpeedZeroOptimizerStage3Mon,
        "unknown": DummyOptimizerMon
    }

    @staticmethod
    def create_optimizer_mon(opt_ty: str):
        if not opt_ty:
            return DummyOptimizerMon()
        optimizer_mon_class = OptimizerMonFactory._optimizer_mon_map.get(opt_ty)
        if not optimizer_mon_class:
            raise Exception("opt_ty should be one of: " + ", ".join(OptimizerMonFactory._optimizer_mon_map.keys()))
        return optimizer_mon_class()
