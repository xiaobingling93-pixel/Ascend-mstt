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
import importlib
from tinker.megatron_patch import microbatches
from tinker.megatron_patch.arguments import parse_args_decorator, profile_args_wrapper
from tinker.megatron_patch.distributed import register_grad_ready_wrapper, start_grad_sync_wrapper

IS_WRAPPER = True
NOT_WRAPPER = False


def patch_func(wrapper_flag, model_name, func_name, patch_func_):
    try:
        model = importlib.import_module(model_name)
        if '.' in func_name:
            func_list = func_name.split('.')
            cls_name = func_list[0]
            func_name = func_list[1]
            model = getattr(model, cls_name)
        setattr(model, func_name, patch_func_ if not wrapper_flag else patch_func_(getattr(model, func_name)))
    except (ModuleNotFoundError, AttributeError):
        pass


_patch_func_model_config = [
    # patch init function for megatron and modellink
    (IS_WRAPPER, ["megatron.arguments", "megatron.training.arguments"], "parse_args", profile_args_wrapper),
    (IS_WRAPPER, ["megatron.initialize", "megatron.training.initialize"], "parse_args", profile_args_wrapper),
    (IS_WRAPPER, ["megatron.arguments", "megatron.training.arguments"], "parse_args", parse_args_decorator),
    (IS_WRAPPER, ["megatron.initialize", "megatron.training.initialize"], "parse_args", parse_args_decorator),

    # patch distributed function for megatron and modellink
    (IS_WRAPPER, ["megatron.core.distributed.grad_buffer",
                  "megatron.core.distributed.param_and_grad_buffer"],
     "Bucket.register_grad_ready", register_grad_ready_wrapper),
    (IS_WRAPPER, ["megatron.core.distributed.grad_buffer", "megatron.core.distributed.param_and_grad_buffer"],
     "Bucket.start_grad_sync", start_grad_sync_wrapper),

    # patch micro_batch function for megatron and modellink
    (NOT_WRAPPER, ["megatron.global_vars", "megatron.training.global_vars"], "_build_num_microbatches_calculator",
     microbatches._build_num_microbatches_calculator),
    (NOT_WRAPPER, ["megatron"], "get_num_microbatches", microbatches.get_num_microbatches),
    (NOT_WRAPPER, ["megatron"], "get_current_global_batch_size", microbatches.get_current_global_batch_size),
    (NOT_WRAPPER, ["megatron"], "update_num_microbatches", microbatches.update_num_microbatches),

]


def patch():
    for _, (wrapper_flap, model_list, func_name, patch_func_) in enumerate(_patch_func_model_config):
        for model_name in model_list:
            patch_func(wrapper_flap, model_name, func_name, patch_func_)
