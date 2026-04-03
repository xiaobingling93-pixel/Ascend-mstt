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

from typing import Optional

from megatron.training import get_args

try:
    from megatron.training.microbatches import build_num_microbatches_calculator, NumMicroBatchesCalculator
except ImportError:
    try:
        from megatron.microbatches import build_num_microbatches_calculator, NumMicroBatchesCalculator
    except ImportError:
        from megatron.core.num_microbatches_calculator import build_num_microbatches_calculator, \
                  NumMicroBatchesCalculator

_GLOBAL_NUM_MICROBATCHES_CALCULATOR = None  # type: Optional[NumMicroBatchesCalculator]


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
    else:
        _GLOBAL_NUM_MICROBATCHES_CALCULATOR = build_num_microbatches_calculator(args)


def rebuild_num_microbatches_calculator():
    args = get_args()
    _build_num_microbatches_calculator(args)
