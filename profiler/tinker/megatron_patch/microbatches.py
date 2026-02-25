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

import os
import importlib

from typing import Optional

from megatron.training import get_args


def get_megatron_microbatch_modules():
    # (build_func_path, calculator_path)
    attempts = [
        ("megatron.training.microbatches.build_num_microbatches_calculator",
         "megatron.training.microbatches.NumMicroBatchesCalculator"),
        ("megatron.microbatches.build_num_microbatches_calculator",
         "megatron.microbatches.NumMicroBatchesCalculator"),
        ("megatron.core.num_microbatches_calculator._build_num_microbatches_calculator",
         "megatron.core.num_microbatches_calculator.NumMicroBatchesCalculator"),
    ]

    # 尝试导入
    for build_path, calc_path in attempts:
        try:
            build_module, build_name = build_path.rsplit('.', 1)
            build_module_obj = importlib.import_module(build_module)
            build_func = getattr(build_module_obj, build_name)

            calc_module, calc_name = calc_path.rsplit('.', 1)
            calc_module_obj = importlib.import_module(calc_module)
            calc_class = getattr(calc_module_obj, calc_name)

            # 返回结果
            return build_func, calc_class

        except (ImportError, AttributeError):
            continue

    raise ImportError("Fail to import megatron microbatch modules.")

build_num_microbatches_calculator, NumMicroBatchesCalculator = get_megatron_microbatch_modules()

def get_num_microbatches():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get()


def get_current_global_batch_size():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get_current_global_batch_size()


def update_num_microbatches(consumed_samples, consistency_check=True):
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR.update(consumed_samples, consistency_check)


def _build_num_microbatches_calculator(args):
    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR
    modellink_version = os.getenv('ML_VERSION', "1.1")
    if modellink_version == "2.0.0":
        _GLOBAL_NUM_MICROBATCHES_CALCULATOR = ( 
       build_num_microbatches_calculator(args.rank, args.rampup_batch_size, 
                                         args.global_batch_size,  
                                         args.micro_batch_size,  
                                         args.data_parallel_size))
    elif modellink_version == "2.3.0":
        _GLOBAL_NUM_MICROBATCHES_CALCULATOR = (
            build_num_microbatches_calculator(args.rank, args.rampup_batch_size,
                                              args.global_batch_size,
                                              args.micro_batch_size,
                                              args.data_parallel_size,
                                              args.decrease_batch_size_if_needed))
    else:
        _GLOBAL_NUM_MICROBATCHES_CALCULATOR = build_num_microbatches_calculator(args)


def rebuild_num_microbatches_calculator():
    args = get_args()
    _build_num_microbatches_calculator(args)
