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

import csv
import gc
import json
import logging
import os
import time
import traceback
from collections import namedtuple
from typing import Dict, List, Optional

import numpy as np
import torch
import torch_npu
from torch_npu.npu import amp

# 选择引用的ModelLink版本
from tinker import megatron_patch
from tinker.framework_adapter.modellink_adapter import get_adapter, ModelLinkAdapter
from tinker.megatron_patch.arguments import get_num_layers
from tinker.megatron_patch.microbatches import rebuild_num_microbatches_calculator
from tinker.model.block_infos import get_model_block_infos, BlockInfo
from tinker.profiler.profile_classes import ProfileArgs, InitTensorInfo
from tinker.utils.logger import init_profile_log, profile_logger
from tinker.utils.npu_timer import NPUTimer
from tinker.utils.utils import byte_to_mb

start_profiling_time = time.time()
RETRY_TIMES = 4
EXCEPTIONAL_VALUE = 1000000000


def for_import():
    """
    防止IDE优化删除import过程中包含有效率逻辑但未被使用的import
    """
    torch_npu, amp, transfer_to_npu, megatron_patch


class HeadInputTensor:
    def __init__(self, batch_output):
        tokens, labels, loss_mask, attention_mask, position_ids = batch_output
        self.input_ids = tokens
        self.labels = labels
        self.loss_mask = loss_mask
        self.attention_mask = attention_mask
        self.position_ids = position_ids


class TinkerProfiler:
    DEVICE = torch.device("cuda")

    def __init__(self, adapter):
        self.adapter: ModelLinkAdapter = adapter
        self.args = self.adapter.get_args()
        self.dump_model_info()
        self.args.model_name = self.args.prof_model_name
        self.profiled_results = {}
        self.data_base = None  # from DATA_BASE
        self.profile_args = None  # type: Optional[ProfileArgs]
        self.fwd_timer = NPUTimer()
        self.bwd_timer = NPUTimer()
        self._set_vars()

        # size of inputs and outputs
        self.input_size_dict = {}
        self.output_size_dict = {}
        self.activation_size_dict = {}
        self.weight_size_dict = {}
        self.last_mbs = None
        self.block_tensor_map = None

    @staticmethod
    def calculate_data_size(tensors):
        """
        计算
        :param tensors:
        :return:
        """
        sum_size = 0
        for name, info in tensors.items():
            if info is not None:
                shape = info.shape
                if "mask" not in name and 'rotary' not in name:
                    sum_size += byte_to_mb(np.prod(shape) * info.element_size)
        return sum_size

    @staticmethod
    def profile_fwd_mem(block, input_data):
        mem_allocated = byte_to_mb(torch.cuda.memory_allocated())
        mem_reserved = byte_to_mb(torch.cuda.memory_reserved())

        output_data = block(input_data)

        new_mem_allocated = byte_to_mb(torch.cuda.memory_allocated())
        new_mem_reserved = byte_to_mb(torch.cuda.memory_reserved())

        return output_data, new_mem_reserved - mem_reserved, new_mem_allocated - mem_allocated, new_mem_reserved

    @staticmethod
    def extract_input_tensors(tensors):
        res = {}

        for key, value in tensors.items():
            if value is None:
                res[key] = value
                continue

            init_tensor_info = InitTensorInfo(value.shape, value.requires_grad, value.device, value.dtype,
                                              value.element_size())
            res[key] = init_tensor_info
        return res

    @staticmethod
    def _is_shape_refresh_needed(tensor_name, tensor_value):
        """
        滤掉无需更新的tensor
        :param tensor_name:
        :param tensor_value:
        :return: false 则滤掉
        """
        return 'attention_mask' not in tensor_name and 'rotary_pos_emb' not in tensor_name and tensor_value is not None

    def dump_model_info(self):
        """
        在profile_data中保存seq_length等信息，供后续search等流程使用
        """
        dump_args = ['seq_length', 'fp16', 'bf16']
        dump_dict = {arg: getattr(self.args, arg) for arg in dump_args}
        dump_dict['num_layers'] = get_num_layers()
        dump_file_name = f'model_info_seq{self.args.seq_length}.json'
        dump_file_path = os.path.join(self.args.prof_path, dump_file_name)
        os.makedirs(os.path.dirname(dump_file_path), exist_ok=True)
        with open(dump_file_path, 'w') as f:
            json.dump(dump_dict, f, indent=4)

    def infer_data_size(self, block_infos: List[BlockInfo]):
        """
        仅剩余 weight_size 部分的计算逻辑
        :param block_infos:
        :return:
        """
        for block_info in block_infos:
            cur_input_tensors, cur_output_tensors = self.block_tensor_map[block_info.name]
            block = block_info.get_block()

            self.weight_size_dict[block_info.name] = np.prod(block.weight_size) * self.data_base

            self.input_size_dict[block_info.name] = self.calculate_data_size(cur_input_tensors)
            self.output_size_dict[block_info.name] = self.calculate_data_size(cur_output_tensors)

    def get_inputs(self, block_name):
        """
        获取性能测试输入
        :param block_name:
        :return:
        """
        inputs = {}
        input_tensors = self.block_tensor_map[block_name][0]
        for input_name in input_tensors:
            input_info = input_tensors[input_name]
            # 其他诸如rotary_pos_emb也可能是None，这里需要对每一个张量进行None判断
            if input_info is None:
                inputs[input_name] = None
                continue
            if "mask" in input_name:
                inputs[input_name] = (
                    torch.rand(input_info.shape, requires_grad=input_info.requires_grad,
                               device=input_info.device) < 0.5)
            else:
                inputs[input_name] = (
                    torch.rand(input_info.shape, requires_grad=input_info.requires_grad, device=input_info.device,
                                dtype=input_info.dtype))
        return inputs

    def get_outputs_and_grads(self, output_tensors):
        # keep original output tensors
        origin_outputs = [output_tensor
                          for _, output_tensor in output_tensors.items()
                          if output_tensor is not None and output_tensor.requires_grad]

        output_grads = []
        # add one more dummy op for each output tensor
        for output_tensor in origin_outputs:
            output_tensor_grad = torch.randn(output_tensor.size(), requires_grad=False, device=self.DEVICE,
                                             dtype=self.grad_type)
            origin_grad = torch.autograd.grad(outputs=output_tensor, grad_outputs=output_tensor_grad,
                                              inputs=output_tensor,
                                              allow_unused=False, retain_graph=False)
            output_grads.append(origin_grad[0])

        return origin_outputs, output_grads

    def profile_block(self, block_info: BlockInfo):
        gc.collect()
        block = self.adapter.wrap_block(block_info.get_block())
        self.adapter.pre_profile_block()
        input_data = self.get_inputs(block_info.name)

        # 反向计算使用
        backward_input_tensors = get_backward_input_tensors(block_info, input_data)

        # Profiling forward/backward computation time

        # 让反向中保存计算图以支撑多次调用
        self.adapter.pre_time_profile_backward_step()

        if "post" in block_info.name:
            # 需匹配梯度版本，1次前向 1次反向 交替进行
            for index in range(self.args.prof_repeat_times[0] + self.args.prof_warmup_times):
                # 内存溢出得厉害，提个函数尝试规避下
                self.profile_time(block, input_data, backward_input_tensors, index)

            avg_fwd_time = self.fwd_timer.get_average_time()
            avg_bwd_time = self.bwd_timer.get_average_time()
            self.fwd_timer.reset()
            self.bwd_timer.reset()
        else:
            avg_fwd_time, avg_bwd_time = self.profile_consecutive_time(block, input_data, backward_input_tensors)

        mem_allocated, mem_reserved_bwd, mem_reserved_fwd = self.profile_mem(block, block_info)

        return avg_fwd_time, avg_bwd_time, mem_reserved_fwd, mem_reserved_bwd, mem_allocated

    def profile_consecutive_time(self, block, input_data, backward_input_tensors):
        """
        连续下发fwd和bwd任务并测量性能，减少下发开销对测量结果的影响
        """
        # 前向预热
        for _ in range(self.args.prof_warmup_times):
            output_data = block(input_data)

        remaining_fwd_times = self.args.prof_repeat_times[0]
        remaining_bwd_times = self.args.prof_repeat_times[0]

        # 同步所有rank
        torch.distributed.barrier()
        # 额外跑一次做对齐时间后的预热
        output_data = block(input_data)
        self.fwd_timer.start()

        for _ in range(remaining_fwd_times):
            output_data = block(input_data)
        self.fwd_timer.stop()

        # 模拟loss grad
        origin_outputs, output_grads = self.get_outputs_and_grads(output_data)
        origin_outputs = sum([torch.mean(origin_output) for origin_output in origin_outputs])
        # 反向预热
        self.adapter.pre_time_profile_backward_step()
        # 同步所有rank
        torch.distributed.barrier()
        self.adapter.backward_step(backward_input_tensors, origin_outputs, output_grads)
        torch.distributed.barrier()
        # 额外跑一次对齐时间
        self.adapter.custom_backward(origin_outputs, output_grads[0])
        self.bwd_timer.start()
        for _ in range(remaining_bwd_times):
            self.adapter.custom_backward(origin_outputs, output_grads[0])
        self.bwd_timer.stop()
        avg_fwd_time = self.fwd_timer.get_average_time() / remaining_bwd_times
        avg_bwd_time = self.bwd_timer.get_average_time() / remaining_bwd_times
        self.fwd_timer.reset()
        self.bwd_timer.reset()

        return avg_fwd_time, avg_bwd_time

    def profile_mem(self, block, block_info):
        # Profiling memory
        _mem_reserved_fwd = 0
        _mem_reserved_bwd = 0
        _mem_allocated = 0

        torch.cuda.synchronize()
        input_data = self.get_inputs(block_info.name)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # 用于解决现象：在profile多次因oom退出后、首次运行成功时，测得的内存会为负值；重新profile_fwd_mem后恢复正常
        cnt = RETRY_TIMES
        while cnt > 0 and _mem_allocated <= 0:
            output_data, _mem_reserved_fwd, _mem_allocated, new_mem_reserved = self.profile_fwd_mem(block, input_data)
            cnt -= 1
        _mem_reserved_fwd -= _mem_allocated
        if _mem_reserved_fwd < 0:
            _mem_reserved_fwd = 0

        outputs = []
        output_grads = []
        # 汇总所有输出tensor 并生成对应上游梯度 支撑该层反向计算
        for output_tensor in output_data.values():
            if output_tensor is None:
                continue
            outputs.append(output_tensor)
            output_grads.append(
                torch.randn(output_tensor.size(), requires_grad=False, device=self.DEVICE, dtype=self.grad_type))

        torch.cuda.synchronize()

        outputs = sum([torch.mean(origin_output) for origin_output in outputs])
        output_grads = sum([torch.mean(output_grad) for output_grad in output_grads])

        self.adapter.pre_mem_profile_backward_step()
        torch.cuda.synchronize()

        backward_input_tensors = get_backward_input_tensors(block_info, input_data)
        if backward_input_tensors is None:
            torch.autograd.backward(outputs, grad_tensors=output_grads, retain_graph=True)
        else:
            self.adapter.backward_step(backward_input_tensors, outputs, output_grads)

        mem_reserved_bwd = byte_to_mb(torch.cuda.memory_reserved())

        _mem_reserved_bwd = mem_reserved_bwd - new_mem_reserved
        if _mem_reserved_bwd < 0:
            _mem_reserved_bwd = 0
        torch.cuda.synchronize()
        torch.distributed.barrier()

        return _mem_allocated, _mem_reserved_bwd, _mem_reserved_fwd

    def profile_time(self, block, input_data, backward_input_tensors, index):
        # 1. 运行前向并测量时间
        torch.distributed.barrier()
        if index >= self.args.prof_warmup_times:
            self.fwd_timer.start()

        output_data = block(input_data)

        if index >= self.args.prof_warmup_times:
            self.fwd_timer.stop()

        # 2. 准备反向所需输出
        outputs, output_grads = self.get_outputs_and_grads(output_data)
        # 3. 将输出置为标量张量，以满足框架要求(Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.)
        outputs = sum([torch.mean(origin_output) for origin_output in outputs])
        # 4. 运行反向并测量时间
        if index >= self.args.prof_warmup_times:
            self.bwd_timer.start()
        self.adapter.backward_step(backward_input_tensors, outputs, output_grads)

        if index >= self.args.prof_warmup_times:
            self.bwd_timer.stop()

    def dump_profiled_results(self, block_list):
        if torch.distributed.get_rank() != 0:
            return
        args = self.args
        profile_logger.info(f"====== PROFILING RESULTS ({self.profile_args.model}{self.profile_args.hint}) ======")
        result_title = ["config", "block_name", "forward-compute", "backward_compute", "input_size", "output_size",
                        "weights",
                        "activations", "fwd_reserved", "bwd_reserved"]
        csv_path = os.path.join(args.prof_path, "profiled_data.csv")
        csv_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a') as f_result:
            f_csv = csv.writer(f_result)
            if not csv_exists:
                f_csv.writerow(result_title)
            for block_info in block_list:
                # fwd_time, bwd_time, input_size, output_size, weight_size, activations, reserved_fwd, reserved_bwd
                data = [f'{float(info):.3f}' for info in self.profiled_results[block_info.name]]
                row = [self.profile_args.file_name[:-4]] + [block_info.name] + data
                f_csv.writerow(row)
            f_csv.writerow([])

    def update_micro_batch_size(self, new_mbs):
        # TODO 丑陋的刷新mbs方式
        envs_dict = self.profile_args.__dict__
        envs_dict['mbs'] = new_mbs
        self.profile_args = self.profile_args.update_mbs(new_mbs)
        args = self.args
        args.micro_batch_size = self.profile_args.mbs
        args.global_batch_size = self.profile_args.mbs
        if self.profile_args.ep:
            args.global_batch_size *= self.profile_args.ep
        rebuild_num_microbatches_calculator()

    def get_new_shape(self, shape, mbs):
        shape_list = list(shape)
        for i, element in enumerate(shape_list):
            if element == self.last_mbs:
                shape_list[i] = mbs
        return torch.Size(shape_list)

    def update_tensor_map(self, block_infos: List[BlockInfo], mbs):
        forward_output = None
        # 初始化
        if self.block_tensor_map is None:
            first_input = self._get_first_input()
            block_tensor_map = {}
            for block_info in block_infos:
                genned_block = block_info.get_block()
                wrapped_block = self.adapter.wrap_block(genned_block)
                # 拿到当前 block 的 input_tensors_info 和 input_extra_tensors_info
                input_tensors = block_info.get_input_tensors(first_input, forward_output)
                forward_output = wrapped_block(input_tensors)

                extract_input_tensors_info = self.extract_input_tensors(input_tensors)
                extract_output_tensors_info = self.extract_input_tensors(forward_output)
                block_tensor_map[block_info.name] = extract_input_tensors_info, extract_output_tensors_info
            self.block_tensor_map = block_tensor_map
            self.last_mbs = mbs

        # 不必更新
        if mbs == self.last_mbs:
            return

        # 更新
        for block_info in block_infos:
            tensor_infos = self.block_tensor_map[block_info.name]
            for tensor_info in tensor_infos:
                tensor_info_pairs = [(tensor_name, tensor_value)
                                     for tensor_name, tensor_value in tensor_info.items()
                                     if self._is_shape_refresh_needed(tensor_name, tensor_value)]
                for tensor_name, tensor_value in tensor_info_pairs:
                    new_shape = self.get_new_shape(tensor_value.shape, mbs)
                    new_tensor = InitTensorInfo(new_shape, tensor_value.requires_grad, tensor_value.device,
                                                tensor_value.dtype, tensor_value.element_size)
                    tensor_info[tensor_name] = new_tensor
        self.last_mbs = mbs

    def reset_params(self):
        self.weight_size_dict = {}
        self.input_size_dict = {}
        self.output_size_dict = {}
        self.profiled_results = {}

    def run_profile(self, task_mbs, already_oom=False):
        args = self.args
        block_infos = get_model_block_infos(self.adapter)
        # 重新初始化，避免不同task间的影响；同时相较于原来的unique_name形式，节约了不少显存
        self.reset_params()

        try:
            self.update_micro_batch_size(task_mbs)
        except AssertionError:
            traceback.print_exc()
            profile_logger.error(
                f"Invalid GBS MBS DP pair: [{args.global_batch_size, args.micro_batch_size, args.data_parallel_size}]")
            return False

        # 遍历一轮block，自动生成block中的张量信息，供后续 get_inputs使用
        self.update_tensor_map(block_infos, task_mbs)
        # infer the data size according to block specs

        self.infer_data_size(block_infos)
        # run profiling
        oom = already_oom
        for block_info in block_infos:
            if oom:
                # 因profile_space的日志监控会在block_profiler发生oom时kill进程，因此该处逻辑不会进行；但保留该处以防kill失败
                profile_logger.info(f'[results] already oom, skip {block_info.name}')
                self.profiled_results[block_info.name] = [EXCEPTIONAL_VALUE, EXCEPTIONAL_VALUE,
                                                          self.input_size_dict[block_info.name],
                                                          self.output_size_dict[block_info.name],
                                                          self.weight_size_dict[block_info.name],
                                                          EXCEPTIONAL_VALUE, EXCEPTIONAL_VALUE, EXCEPTIONAL_VALUE]
                continue
            profile_logger.info(f"working on {block_info.name}{self.profile_args.hint} ... ")
            try:
                fwd_time, bwd_time, reserved_fwd, reserved_bwd, allocated_fwd = self.profile_block(block_info)
            except RuntimeError as e:
                if "NPU out of memory" in str(e):
                    # OOM 没必要测试更大的mbs
                    oom = True
                    # break
                profile_logger.error(f'RANK{torch.distributed.get_rank()}: {"-*/" * 20}')
                profile_logger.error(e)
                traceback.print_exc()
                fwd_time = bwd_time = EXCEPTIONAL_VALUE
                reserved_fwd = reserved_bwd = allocated_fwd = EXCEPTIONAL_VALUE
            profile_logger.info(f"[results] {block_info.name}: fwd_compute = {fwd_time:.2f} us, "
                                f"bwd_compute = {bwd_time:.2f} us, fwd_allocated = {allocated_fwd:.1f} MB, "
                                f"fwd_reserved = {reserved_fwd:.1f} MB, bwd_reserved = {reserved_bwd:.1f} MB.")

            self.profiled_results[block_info.name] = [fwd_time, bwd_time,
                                                      self.input_size_dict[block_info.name],
                                                      self.output_size_dict[block_info.name],
                                                      self.weight_size_dict[block_info.name],
                                                      allocated_fwd, reserved_fwd, reserved_bwd]
        self.dump_profiled_results(block_infos)
        return not oom

    def _set_vars(self):
        self.args.iteration = 1
        self.profile_args = ProfileArgs(
            tp=self.args.tensor_model_parallel_size,
            sp=int(self.args.sequence_parallel),
            ep=self.args.expert_model_parallel_size,
            mbs=1,
            algo=0,
            model=f'{self.args.prof_model_name}_{self.args.prof_model_size}'  # from cur_model_name
        )
        self.data_base = byte_to_mb(self.args.params_dtype.itemsize)  # from DATA_BASE
        # fp32存权重梯度
        self.grad_type = torch.float

    def _get_first_input(self):
        """此处使用modellink代码，需密切关注patch、变更情况"""
        return HeadInputTensor(self.adapter.get_head_input())


def get_backward_input_tensors(block_info, input_data):
    """"""
    if "embedding" in block_info.name:
        return None
    input_tensors = []
    for input_name in input_data:
        if input_data[input_name] is not None and input_data[input_name].requires_grad:
            input_tensors.append(input_data[input_name])

    return input_tensors


def main():
    adapter: ModelLinkAdapter = get_adapter()
    adapter.initialize()
    init_profile_log(logging.INFO)
    tinker_profiler = TinkerProfiler(adapter)
    args = adapter.get_args()
    # get profiling tasks
    # "task"s are defined by unique {model, size, mbs} pairs
    all_tasks_mbs = []
    # todo 当前基于传入配置profiling, model_prof_config 可以专注完成搜索范围指定的工作
    model_name = args.prof_model_name
    # TODO 待拓展单次拉起torchrun的profiling维度
    model_prof_config = {"mbs": [1, 2, 4, 8]}
    model_size = args.prof_model_size
    if args.prof_mbs_list is None:
        micro_batch_sizes = model_prof_config["mbs"]
    else:
        micro_batch_sizes = args.prof_mbs_list
    for mbs in micro_batch_sizes:
        # 规避部分 mbs oom
        if mbs >= args.prof_mbs_limit:
            break
        all_tasks_mbs.append(mbs)  # 一个prof_task的tp是固定的

    oom_record = False
    # run profiling tasks
    for task_mbs in all_tasks_mbs:
        if task_mbs >= args.prof_mbs_limit:
            oom_record = True
        run_well = tinker_profiler.run_profile(task_mbs, oom_record)
        if not run_well and not oom_record:
            profile_logger.info(f"[Tinker-Profiler] OOM when mbs={task_mbs}")
            oom_record = True

    end_profiling_time = time.time()

    profile_logger.info(f"[TOTAL PROFILING TIME] {end_profiling_time - start_profiling_time:2f} s")


if __name__ == "__main__":
    main()
