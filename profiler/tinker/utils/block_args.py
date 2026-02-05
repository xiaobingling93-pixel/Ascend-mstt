# Copyright (c) 2025, Huawei Technologies Co., Ltd.
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

from dataclasses import dataclass
from typing import List, Dict

from tinker.utils.logger import logger
from tinker.profiler.profile_classes import ProfileArgs


@dataclass
class DetailedInfo:
    """用于整块存储显存开销信息，仿真信息校对时使用"""
    # 时间开销细分项
    fwd: float = 0.0
    block_fwd: List[float] = None
    bwd: float = 0.0
    block_bwd: List[float] = None
    input_comm: float = 0.0
    output_comm: float = 0.0
    # 内存开销细分项
    weight: float = 0.0
    full_precision_weight: float = 0.0
    grad: float = 0.0
    # 当开启`bf16`后 会多存的一份半精度权重尺寸内容
    weight_bf16: float = 0.0
    pipeline_fwd_act: float = 0.0
    optimizer_state: float = 0.0
    inputs: float = 0.0
    activation: float = 0.0
    dist_opt_slice: float = 0.0
    recompute: float = 0.0
    reserved_mem: float = 0.0
    attention_mask_mem: float = 0.0
    dp_dist_opt: int = 1
    num_fwd_act: int = 1
    block_weight: List[float] = None
    block_act: List[float] = None
    first_time_block_act: List[float] = None

    def __post_init__(self):
        self.block_fwd = []
        self.block_bwd = []
        self.block_weight = []
        self.block_act = []
        self.first_time_block_act = []

    def print_info(self):
        self._round_3()
        # 各个成分
        logger.info('Time Cost'.center(60, '-'))
        logger.info(f'block forward time(us): {self.block_fwd}')
        logger.info(f'block backward time with recompute(us): {self.block_bwd}')
        logger.info(f'forward time = {self.fwd / 1000:.3f} ms')
        logger.info(f'backward time = {self.bwd / 1000:.3f} ms')

        logger.info('Memory Cost'.center(60, '-'))
        model_optimizer_mem = self.weight + self.grad + self.weight_bf16 + self.full_precision_weight / self.dp_dist_opt
        logger.info(
            f'model & optimizer({model_optimizer_mem:.3f})'
            f' = {self._v("weight")} + {self._v("grad")} + {self._v("weight_bf16")}'
            f' + {self._v("full_precision_weight")} / {self._v("dp_dist_opt")}'
        )
        logger.info(f'block weights({self.block_weight})')
        logger.info(f'block activations({self.block_act})')
        logger.info(f'first time block activations({self.first_time_block_act})')

    def print_time(self, bubble_time, micro_batch_num, time_cost):
        unit_time = (self.fwd + self.bwd + self.input_comm + self.output_comm) / 1000
        bubble_time = bubble_time / 1000 - unit_time
        logger.info(f'Unit Time({unit_time:.3f} ms)'
                     f' = {self._v("fwd")} + {self._v("bwd")} + {self._v("input_comm")} + {self._v("output_comm")}')
        logger.info(f'Time({time_cost / 1000:.3f})'
                     f' = bubble({bubble_time:.3f}) + mbn({micro_batch_num}) * unit_time({unit_time:.3f})')

    def print_mem_calc(self, mem_cost):
        self._round_3()
        # pipeline_fwd_act计算
        logger.info(
            f'{self._v("pipeline_fwd_act")} = '
            f'{self._v("num_fwd_act")}'
            f' * [{self._v("inputs")} + {self._v("activation")}]'
        )
        logger.info(
            f'Memory({mem_cost:.3f})'
            f' = {self._v("weight")} + {self._v("grad")} + {self._v("weight_bf16")}'
            f' + [{self._v("full_precision_weight")} + {self._v("optimizer_state")}]'
            f' / {self._v("dp_dist_opt")}'
            f' + {self._v("pipeline_fwd_act")}'
            f' + {self._v("attention_mask_mem")}'
            f' + {self._v("recompute")} + {self._v("reserved_mem")}'
        )

    def set_and_print(self, input_comm, output_comm, recompute_mem, reserved_mem_cost, mem_cost):
        self.input_comm = input_comm
        self.output_comm = output_comm
        self.recompute = recompute_mem
        self.reserved_mem = reserved_mem_cost
        self.print_info()
        self.print_mem_calc(mem_cost)

    def _round_3(self):
        for k, v in self.__dict__.items():
            if isinstance(v, float):
                self.__dict__[k] = round(v, 3)

    def _v(self, v):
        return f'{v}({getattr(self, v)})'


class BlockArgs:
    """存block这一层级 所关注的训练优化策略，协同 ProfileArgs 参数，以及 BlockCost 数据下 去支撑 CostModel 中的一些计算"""

    def __init__(self, args, profile_args: ProfileArgs, block_cost: 'BlockCost'):
        # TODO dp dist_opt往这里移
        self.profile_args = profile_args
        self.data = block_cost
        self.num_fwd_act = None
        self.recompute = None
        self.dp = None
        self.dist_opt = None
        self.is_first = False
        self.attention_mask_mem = 0.0
        # 兼容老版本没存数据类型信息
        if not hasattr(args, 'bf16'):
            args.bf16, args.fp16 = True, False
            if 'chatglm' in args.model_name:
                args.bf16, args.fp16 = False, True
        self.bf16 = args.bf16

    @property
    def max_reserved_mem(self):
        return max(self.data.fwd_reserved, self.data.bwd_reserved)

    @property
    def num_npu_block(self):
        """返回这个block涉及的NPU个数，通常一个stage中的block返回值都相等，所以调一个block的值就行"""
        # TODO 后面要把dp挪成自有属性
        return self.profile_args.tp * self.dp

    def update_cost_model_args(self, cost_model_args: Dict[str, int]):
        for k, v in cost_model_args.items():
            setattr(self, k, v)

    # TODO 全都是个入参，考虑把这些函数放到cost model里
    def block_time(self, detail=False, detail_info: DetailedInfo = None) -> float:
        """前向 + 反向 + 重计算 + p2p通信 = fwd + bwd + rec_fwd + in_comm + out_comm"""
        compute_time = self.data.fwd * (1 + self.recompute) + self.data.bwd
        if detail:
            detail_info.fwd += self.data.fwd
            detail_info.block_fwd.append(self.data.fwd)
            detail_info.bwd += self.data.bwd + self.recompute * self.data.fwd
            detail_info.block_bwd.append(self.data.bwd + self.recompute * self.data.fwd)
        return compute_time

    def block_mem(self, detail=False, detail_info: DetailedInfo = None) -> float:
        """
        权重 + 梯度 + 优化器 + 激活值
        = (1 + PM +(1 + PO) / dp_dist_opt) * w + (SB + 1) * (is_first * input + is_recompute * act)
        :return:
        """
        full_precision_weight = self.data.param_master * self.data.w
        weight_mem = self.data.w
        weight_bf16_mem = self.data.w if self.bf16 else 0
        grad_mem = self.data.w
        optimizer_mem = self.data.param_optimizer * self.data.w
        input_mem = self.is_first * self.data.in_size
        activation_mem = self.data.in_size if self.recompute else self.data.act

        dp_dist_opt = self.dp if self.dist_opt else 1

        # memory 不同部分的计算
        # 初始: (2 + bf16 + 2 / dp_dist_opt) * W
        # step0结束: (4 / dp_dist_opt) * W
        # 前反向: pipeline_fwd_act * A + attention_mask
        # 仅 transformer block 生成一次的 attention_mask
        mem = 0
        mem += weight_mem + grad_mem + weight_bf16_mem + full_precision_weight / dp_dist_opt
        mem += optimizer_mem / dp_dist_opt
        mem += self.num_fwd_act * (input_mem + activation_mem)
        mem += self.attention_mask_mem
        if detail:
            detail_info.weight += weight_mem
            detail_info.block_weight.append(weight_mem)
            detail_info.full_precision_weight += full_precision_weight
            detail_info.grad += grad_mem
            detail_info.weight_bf16 += weight_bf16_mem
            detail_info.pipeline_fwd_act += self.num_fwd_act * (input_mem + activation_mem)
            detail_info.optimizer_state += optimizer_mem
            detail_info.inputs += input_mem
            detail_info.activation += activation_mem
            detail_info.block_act.append(activation_mem)
            detail_info.first_time_block_act.append(activation_mem + self.attention_mask_mem)
            detail_info.dist_opt_slice += (grad_mem + optimizer_mem) / dp_dist_opt
            detail_info.attention_mask_mem += self.attention_mask_mem
            detail_info.dp_dist_opt = dp_dist_opt
            detail_info.num_fwd_act = self.num_fwd_act
        return mem


@dataclass
class BlockCost:
    fwd: float
    bwd: float
    in_size: float
    out_size: float
    w: float
    act: float
    fwd_reserved: float
    bwd_reserved: float
    # TODO 这2个变量 必给扔到别的地方
    param_master: int = 2
    param_optimizer: int = 4
