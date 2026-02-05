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
import types
from dataclasses import dataclass
from typing import List

import torch

from tinker.framework_adapter.modellink_adapter import ModelLinkAdapter
from tinker.model.adapter_utils import MODULE_NAME, get_forward_func_name
from tinker.model.block_adapters import BlockAdapter, mcore_block_adapters, legacy_block_adapters

# 这里直接写死
forward_funcs = importlib.import_module(f'tinker.model.{MODULE_NAME}')


def standardize_forward(forward_func):
    """
    将调用方式从传统参数改为字典，并将输出包装成字典
    """

    def wrapper(self, input_dict):
        # 检查输入是否为字典
        if not isinstance(input_dict, dict):
            raise ValueError("Input must be a dictionary")

        # 调用原始的 new_func，将字典解包为关键字参数
        outputs = forward_func(self, **input_dict)

        # 将输出包装成字典
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        return {k: v for k, v in zip(self.output_name, outputs)}

    return wrapper


def get_weight_size(modules: List[torch.nn.Module]) -> int:
    """根据入参Module 自动计算权重参数尺寸"""
    weight_size = 0
    for module in modules:
        weight_size += sum(p.numel() for p in module.parameters() if p.requires_grad)
    return weight_size


def get_forward_func(block_name):
    """
    通过block名称，匹配gen_block_adapter生成的前向函数
    """
    return getattr(forward_funcs, get_forward_func_name(block_name))


@dataclass
class BlockInfo:
    def __init__(self, block_adapter: BlockAdapter, model: torch.nn.Module):
        # 所有block实例化所需的硬编码内容
        self.adapter: BlockAdapter = block_adapter
        # block名称，仅起到标识作用，从BlockAdapter中获取
        self.name: str = block_adapter.block_name
        # block对应module，延时生成
        self.module: torch.nn.Module = self._get_module(model)

    @staticmethod
    def _get_attr(obj, module_path):
        attribute_paths = module_path.split(".") if module_path else []
        current = obj
        for attr in attribute_paths:
            current = getattr(current, attr)
        return current

    def get_block(self):
        # 1. 替换实例forward
        self.module.forward = types.MethodType(standardize_forward(get_forward_func(self.name)), self.module)
        # 2. 计算权重尺寸，存到可访问的地方，如block实例中
        modules = [self._get_attr(self.module, module_name) for module_name in self.adapter.weight_param_module]
        self.module.weight_size = get_weight_size(modules)
        # 3. 指明block实例的输出列表
        self.module.output_name = self.adapter.return_values
        return self.module

    def get_input_tensors(self, first_input, forward_output):
        input_tensors = {}
        for source in self.adapter.input_source:
            if source.from_forward:
                input_tensor = forward_output[source.source_name]
            else:
                input_tensor = getattr(first_input, source.source_name, None)
            input_tensors[source.name] = input_tensor
        return input_tensors

    def _get_module(self, model):
        return self._get_attr(model, self.adapter.module_path)


def get_block_adapters(args) -> List[BlockAdapter]:
    if args.use_mcore_models:
        # mcore GPTModel
        block_adapters = mcore_block_adapters
    else:
        # legacy GPTModel
        block_adapters = legacy_block_adapters
    return block_adapters


def get_model_block_infos(adapter: ModelLinkAdapter) -> List[BlockInfo]:
    """
    通过block信息，获取需要profile的block列表
    """
    args = adapter.get_args()
    model = adapter.get_model()
    block_adapters = get_block_adapters(args)
    block_infos = []
    for block_adapter in block_adapters:
        block_info = BlockInfo(block_adapter, model)
        block_infos.append(block_info)

    return block_infos