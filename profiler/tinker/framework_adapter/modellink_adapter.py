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
"""
get_args: 获取入参
initialize: 初始化megatron框架、分词器
wrap_block: 模型包装器工具方法
get_head_input: 生成首block输入
backward_step: 跑模型的反向
get_model: 用于生成单层模型实例
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Type, Union

import torch


class ModelLinkAdapter(ABC):
    @abstractmethod
    def get_args(self):
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def wrap_block(self, block):
        pass

    @abstractmethod
    def get_head_input(self):
        pass

    @abstractmethod
    def pre_profile_block(self):
        pass

    @abstractmethod
    def backward_step(self, input_tensors, outputs, output_grads):
        pass

    @abstractmethod
    def custom_backward(self, outputs, output_grads):
        pass

    @abstractmethod
    def get_model(self):
        """
        向args.model_type赋值，然后调用model_provider()返回模型， 子图实例都从这个模型抽取
        """
        pass

    @abstractmethod
    def pre_time_profile_backward_step(self):
        pass

    @abstractmethod
    def pre_mem_profile_backward_step(self):
        pass

    @abstractmethod
    def deallocate_output_tensor(self, outputs, deallocate_outputs):
        pass


class ModelLinkAdapter11(ModelLinkAdapter):

    def __init__(self):
        self.config = None
        self.original_custom_backward = None

    def get_args(self):
        from megatron.training import get_args
        return get_args()

    def initialize(self):
        from modellink.initialize import initialize_megatron
        initialize_megatron()

    def wrap_block(self, block):
        """
        为裸block增加fp16和ddp外壳
        """
        from megatron.core.tensor_parallel import set_defaults_if_not_set_tensor_model_parallel_attributes
        from megatron.legacy.model import Float16Module

        args = self.get_args()

        for param in block.parameters():
            set_defaults_if_not_set_tensor_model_parallel_attributes(param)

        block.cuda(torch.cuda.current_device())
        if args.fp16 or args.bf16:
            block = Float16Module(block, args)

        return block

    def get_head_input(self):
        from pretrain_gpt import get_batch, train_valid_test_datasets_provider
        from megatron.training.training import build_train_valid_test_data_iterators
        train_valid_test_datasets_provider.is_distributed = True
        data_iterator, _, _ = build_train_valid_test_data_iterators(train_valid_test_datasets_provider)
        return get_batch(data_iterator)

    def pre_profile_block(self):
        from megatron.training.arguments import core_transformer_config_from_args
        self.config = core_transformer_config_from_args(self.get_args())

    def pre_time_profile_backward_step(self):
        from megatron.core.pipeline_parallel import schedules
        from tinker.megatron_patch.schedules import custom_backward
        self.original_custom_backward = schedules.custom_backward
        schedules.custom_backward = custom_backward

    def pre_mem_profile_backward_step(self):
        from megatron.core.pipeline_parallel import schedules
        schedules.custom_backward = self.original_custom_backward

    def backward_step(self, input_tensors, outputs, output_grads):
        from megatron.core.enums import ModelType
        from megatron.core.pipeline_parallel import schedules
        schedules.backward_step(input_tensors, outputs, output_grads, ModelType.encoder_or_decoder, self.config)

    def custom_backward(self, outputs, output_grads):
        from tinker.megatron_patch.schedules import custom_backward
        custom_backward(outputs, output_grads)

    def get_model(self):
        from megatron.core.enums import ModelType
        from pretrain_gpt import model_provider
        args = self.get_args()
        args.model_type = ModelType.encoder_or_decoder
        return model_provider()

    def deallocate_output_tensor(self, outputs, deallocate_outputs):
        from megatron.core.pipeline_parallel import schedules
        for _, origin_output in outputs.items():
            schedules.deallocate_output_tensor(origin_output, deallocate_outputs)



class ModelLinkAdapter10(ModelLinkAdapter11):
    def initialize(self):
        from megatron import initialize_megatron
        initialize_megatron()

    def wrap_block(self, block):
        """
        为裸block增加fp16和ddp外壳
        """
        from megatron.core.tensor_parallel import set_defaults_if_not_set_tensor_model_parallel_attributes
        from megatron.model import Float16Module

        args = self.get_args()

        for param in block.parameters():
            set_defaults_if_not_set_tensor_model_parallel_attributes(param)

        block.cuda(torch.cuda.current_device())
        if args.fp16 or args.bf16:
            block = Float16Module(block, args)

        return block

    def get_head_input(self):
        from pretrain_gpt import get_batch, train_valid_test_datasets_provider
        from megatron.training import build_train_valid_test_data_iterators
        train_valid_test_datasets_provider.is_distributed = True
        data_iterator, _, _ = build_train_valid_test_data_iterators(train_valid_test_datasets_provider)
        return get_batch(data_iterator)

    def pre_profile_block(self):
        from megatron.arguments import core_transformer_config_from_args
        self.config = core_transformer_config_from_args(self.get_args())


class ModelLinkAdapter12(ModelLinkAdapter11):
    """
    1.2相关接口相较1.1无变更， 此处新建子类， 用于明确五变更情况
    """
    pass


class ModelLinkAdapter100(ModelLinkAdapter11):
    """
    1.0.0相关接口
    """

    def initialize(self):
        from modellink.training.initialize import initialize_megatron
        initialize_megatron()


class ModelLinkAdapter200(ModelLinkAdapter100):
    """
    2.0.0相关接口相较1.0.0无变更， 此处新建子类， 用于明确五变更情况
    """
    pass


class ModelLinkAdapter230(ModelLinkAdapter200):
    """
    2.3.0相关接口适配
    """
    
    def initialize(self):
        # 按照检视意见：直接使用 megatron 中的 initialize_megatron
        from megatron.training.initialize import initialize_megatron
        initialize_megatron()
    
    def wrap_block(self, block):
        """
        为裸block增加fp16和ddp外壳（2.3.0适配版本）
        """
        from megatron.core.tensor_parallel import set_defaults_if_not_set_tensor_model_parallel_attributes
        import torch
        
        args = self.get_args()
        
        for param in block.parameters():
            set_defaults_if_not_set_tensor_model_parallel_attributes(param)
        
        block.cuda(torch.cuda.current_device())
        if args.fp16 or args.bf16:
            from megatron.core.transformer.module import Float16Module
            from megatron.core.utils import get_model_config
            config = get_model_config(block)
            block = Float16Module(config, block)
        return block


class ModelLinkAdapterTune200(ModelLinkAdapter100):
    """
    2.0.0 tune 相关接口相较1.0.0 输入数据处理变更了
    """

    def get_head_input(self):
        from mindspeed_llm.tasks.posttrain.sft.sft_trainer import SFTTrainer
        from mindspeed_llm.tasks.posttrain.utils import train_valid_test_datasets_provider
        from megatron.training.training import build_train_valid_test_data_iterators
        train_valid_test_datasets_provider.is_distributed = True
        data_iterator, _, _ = build_train_valid_test_data_iterators(train_valid_test_datasets_provider)
        return SFTTrainer.get_batch(data_iterator)


class ModelLinkAdapterTune230(ModelLinkAdapterTune200):
    """
    2.3.0 tune 相关接口相较2.0.0无变更， 此处新建子类， 用于明确五变更情况
    """
    pass


version_map: Dict[str, Type[ModelLinkAdapter]] = {
    '1.0': ModelLinkAdapter10,
    '1.1': ModelLinkAdapter11,
    '1.2': ModelLinkAdapter12,
    '1.0.0': ModelLinkAdapter100,
    '2.0.0': ModelLinkAdapter200,
    '2.3.0': ModelLinkAdapter230
}

tune_version_map: Dict[str, Type[ModelLinkAdapter]] = {
    '2.0.0': ModelLinkAdapterTune200,
    '2.3.0': ModelLinkAdapterTune230
}


def get_adapter() -> ModelLinkAdapter:
    """
    返回指定版本的adapter实例
    """
    toolkit_version = os.getenv('ML_VERSION', '1.1')  # 默认用ModelLink 1.1版本
    tune_flag = int(os.getenv('IS_TUNE', '0'))
    if not tune_flag:
        if toolkit_version not in version_map:
            raise NotImplementedError(f'{toolkit_version}版本的Adapter暂未支持')
        return version_map[toolkit_version]()
    else:
        if toolkit_version not in tune_version_map:
            raise NotImplementedError(f'{toolkit_version}版本的微调Adapter暂未支持')
        return tune_version_map[toolkit_version]()