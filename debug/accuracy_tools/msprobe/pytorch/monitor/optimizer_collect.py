#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

from collections import defaultdict
import torch


class MixPrecsionOptimizerMon:
    wrapped_optimizer = None

    def __init__(self) -> None:
        self.fp16_to_fp32_param = {}

    @staticmethod
    def set_wrapped_optimizer(_wrapped_optimizer):
        MixPrecsionOptimizerMon.wrapped_optimizer = _wrapped_optimizer

    # parameter tensors we want to monitor and their names are in params2name_dict
    # base_optimizer is pytorch optimizer, wrapped_optimizer is a normal object with  base_optimizer
    def fetch_mv(self, monitor, torch_opt, params2name):
        mix_prec_opt = MixPrecsionOptimizerMon.wrapped_optimizer

        if not self.fp16_to_fp32_param and mix_prec_opt is not None:
            for fp16_group, fp32_group in zip(mix_prec_opt.float16_groups, mix_prec_opt.fp32_from_float16_groups):
                for fp16_param, fp32_param in zip(fp16_group, fp32_group):
                    self.fp16_to_fp32_param[fp16_param] = fp32_param
        return self._fetch_mv_in_adam(params2name, torch_opt, monitor)

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
        res = (exp_avg_dict, exp_avg_sq_dict, update_dict, ratio_dict)
        return res


class MegatronDistributedOptimizerMon(MixPrecsionOptimizerMon):
    def fetch_mv(self, monitor, torch_opt, params2name):
        mix_prec_opt = MixPrecsionOptimizerMon.wrapped_optimizer
        if not (hasattr(mix_prec_opt, "model_float16_groups")
                and hasattr(mix_prec_opt, "shard_fp32_from_float16_groups")):
            raise Exception("megatron distributed optimizer should have model_float16_groups "
                            "and shard_fp32_from_float16_groups, if not, please check megatron-lm version")
        if not self.fp16_to_fp32_param and mix_prec_opt is not None:
            for fp16_group, shard_fp32_group in zip(mix_prec_opt.model_float16_groups,
                                                    mix_prec_opt.shard_fp32_from_float16_groups):
                for fp16_param, shard_fp32_param in zip(fp16_group, shard_fp32_group):
                    self.fp16_to_fp32_param[fp16_param] = shard_fp32_param

        return self._fetch_mv_in_adam(params2name, torch_opt, monitor)


class DummyOptimizerMon(MixPrecsionOptimizerMon):
    def fetch_mv(self, monitor, torch_opt, params2name):
        res = None, None, None, None
        return res


class OptimizerMonFactory:
    @staticmethod
    def create_optimizer_mon(opt_ty:str):
        if opt_ty == "Megatron_Float16OptimizerWithFloat16Params":
            return MixPrecsionOptimizerMon()
        if opt_ty == "Megatron_DistributedOptimizer":
            return MegatronDistributedOptimizerMon()
        if opt_ty is None or opt_ty == "unknown":
            return DummyOptimizerMon()
        raise Exception("opt_ty should be Megatron_Float16OptimizerWithFloat16Params "
                        "or Megatron_DistributedOptimizer or None or unknown")
