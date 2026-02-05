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

from functools import wraps, partial
from typing import Callable, Optional

STORE_TRUE = 'store_true'
NUM_LAYERS = None  # type: Optional[int]


def get_num_layers():
    return NUM_LAYERS


def extra_args_provider_decorator(extra_args_provider):
    @wraps(extra_args_provider)
    def wrapper(parser):
        if extra_args_provider is not None:
            parser = extra_args_provider(parser)
        parser = process_args(parser)
        return parser

    return wrapper


def parse_args_decorator(parse_args):
    @wraps(parse_args)
    def wrapper(extra_args_provider=None, ignore_unknown_args=False):
        decorated_provider = extra_args_provider_decorator(extra_args_provider)
        args = parse_args(decorated_provider, ignore_unknown_args)
        # 提取layers 然后置1
        global NUM_LAYERS
        NUM_LAYERS = args.num_layers
        args.num_layers = 1
        return args

    return wrapper


def process_args(parser):
    parser.conflict_handler = 'resolve'

    parser = _add_profiler_args(parser)

    return parser


def _add_profiler_args(parser):
    profiler_group = parser.add_argument_group(title='block_profiler')

    profiler_group.add_argument('--prof-path', type=str, default=None, help='')
    profiler_group.add_argument('--prof-model-name', type=str, default='all', help='')
    profiler_group.add_argument('--prof-model-size', type=str, default='all', help='')
    profiler_group.add_argument('--prof-warmup-times', type=int, default=20, help='')
    profiler_group.add_argument('--prof-repeat-times', nargs='+', type=int, default=[50], help='')
    profiler_group.add_argument('--prof-mbs-list', nargs='+', type=int, default=None, help='')
    profiler_group.add_argument('--prof-mbs-limit', type=int, default=None, help='')

    return parser


def profile_args_wrapper(fn: Callable):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        args = fn(*args, **kwargs)
        args = override_profile_args(args)
        return args

    return wrapper


def override_profile_args(args):
    args.data_parallel_size = args.world_size // (args.pipeline_model_parallel_size *
                                                  args.tensor_model_parallel_size * args.context_parallel_size)
    args.global_batch_size = args.data_parallel_size  # 此处仅用于通过validation
    args.micro_batch_size = 1
    args.num_ops_in_each_stage = [1]
    args.virtual_pipeline_model_parallel_size = 1
    args.model_parallel_size_of_each_op = [[args.tensor_model_parallel_size]]
    args.data_parallel_size_of_each_op = [[1]]
    args.model_name = ""
    args.resharding_stages = [True]

    return args
