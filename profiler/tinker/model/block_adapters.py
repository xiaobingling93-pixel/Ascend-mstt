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
from typing import List, Optional


@dataclass
class MethodLocation:
    """
    该类用于定位方法整体位置
    """
    source_method_path: List[str]
    # 定位到整体方法后，实际提取部分的开始关键字
    start_key_word: Optional[str]
    # 定位到整体方法后，实际提取部分的结束关键字
    end_key_word: Optional[str]
    # 是否开启cut_mode模式
    cut_mode: bool = False


@dataclass
class ParamSource:
    name: str
    source_name: Optional[str] = None
    from_forward: bool = True

    def __post_init__(self):
        if self.source_name is None:
            self.source_name = self.name


@dataclass
class BlockAdapter:
    """用于生成ProfileBlock所需的硬编码信息"""
    block_name: str
    # 定位方法整体位置
    method_location: MethodLocation
    # forward 方法返回值列表
    return_values: List[str]
    # forward 方法补充参数
    append_method_signatures: Optional[List[str]]
    # block前向逻辑所在的实例
    module_path: str
    # block中要计算权重参数的子module，硬编码内容
    weight_param_module: List[str]
    # 入参来源
    input_source: List[ParamSource]

    @property
    def source_method_path(self):
        return self.method_location.source_method_path

    @property
    def key_words(self):
        return self.method_location.start_key_word, self.method_location.end_key_word


legacy_block_adapters = [
    BlockAdapter(
        block_name='embedding',
        method_location=MethodLocation(
            source_method_path=['megatron.legacy.model.language_model.TransformerLanguageModel',
                                'megatron.model.language_model.TransformerLanguageModel'],
            start_key_word=None,
            end_key_word='rotary_pos_emb ='
        ),
        return_values=['encoder_input', 'rotary_pos_emb'],
        append_method_signatures=None,
        module_path="language_model",
        weight_param_module=['embedding'],
        input_source=[ParamSource('enc_input_ids', 'input_ids', from_forward=False),
                      ParamSource('enc_position_ids', 'position_ids', from_forward=False)]
    ),
    BlockAdapter(
        block_name='transformer_block',
        method_location=MethodLocation(
            source_method_path=['modellink.model.transformer.parallel_transformer_forward',
                                'megatron.model.transformer.ParallelTransformer',
                                'megatron.legacy.model.transformer.ParallelTransformer'],
            start_key_word=None,
            end_key_word='self.microbatch_count += 1'
        ),
        return_values=['hidden_states'],
        append_method_signatures=None,
        module_path="language_model.encoder",
        weight_param_module=['layers'],
        input_source=[ParamSource('hidden_states', "encoder_input"),
                      ParamSource('rotary_pos_emb'),
                      ParamSource('attention_mask', from_forward=False)]
    ),
    BlockAdapter(
        block_name='final_norm',
        method_location=MethodLocation(
            source_method_path=['modellink.model.transformer.parallel_transformer_forward',
                                'megatron.model.transformer.ParallelTransformer',
                                'megatron.legacy.model.transformer.ParallelTransformer'],
            start_key_word='self.final_norm',
            end_key_word='self.final_norm'
        ),
        return_values=['hidden_states'],
        append_method_signatures=None,
        module_path="language_model.encoder",
        weight_param_module=['final_norm'],
        input_source=[ParamSource('hidden_states')]
    ),
    BlockAdapter(
        block_name='post_process',
        method_location=MethodLocation(
            source_method_path=['modellink.model.gpt_model.GPTModel',
                               'mindspeed_llm.legacy.model.gpt_model.GPTModel'],
            start_key_word='post_language_model_processing',
            end_key_word='fp16_lm_cross_entropy'
        ),
        return_values=["output"],
        append_method_signatures=['lm_output'],
        module_path="",
        weight_param_module=['language_model.output_layer'],
        input_source=[ParamSource('lm_output', 'hidden_states'),
                      ParamSource('labels', from_forward=False)]
    )
]

mcore_block_adapters = [
    BlockAdapter(
        block_name='embedding',
        method_location=MethodLocation(
            source_method_path=['modellink.core.models.gpt.gpt_model.GPTModel', 'mindspeed_llm.core.models.gpt.gpt_model.GPTModel'],
            start_key_word=None,
            end_key_word='rotary_pos_emb ='
        ),
        return_values=['decoder_input', 'rotary_pos_emb'],
        append_method_signatures=None,
        module_path="",
        weight_param_module=['embedding'],
        input_source=[ParamSource('input_ids', from_forward=False),
                      ParamSource('position_ids', from_forward=False)]
    ),
    BlockAdapter(
        block_name='transformer_block',
        method_location=MethodLocation(
            source_method_path=['modellink.core.transformer.transformer_block.transformer_block_forward', 'mindspeed_llm.core.transformer.transformer_block.transformer_block_forward'],
            start_key_word=None,
            end_key_word='group_prefetch_offload_commit_async'
        ),
        return_values=['hidden_states'],
        append_method_signatures=None,
        module_path="decoder",
        weight_param_module=['layers'],
        input_source=[ParamSource('hidden_states', "decoder_input"),
                      ParamSource('rotary_pos_emb'),
                      ParamSource('attention_mask', from_forward=False)]
    ),
    BlockAdapter(
        block_name='final_norm',
        method_location=MethodLocation(
            source_method_path=['modellink.core.transformer.transformer_block.transformer_block_forward', 'mindspeed_llm.core.transformer.transformer_block.transformer_block_forward'],
            start_key_word='final_layernorm',
            end_key_word='final_layernorm'
        ),
        return_values=['hidden_states'],
        append_method_signatures=None,
        module_path="decoder",
        weight_param_module=['final_layernorm'],
        input_source=[ParamSource('hidden_states')]
    ),
    BlockAdapter(
        block_name='post_process',
        method_location=MethodLocation(
            source_method_path=['modellink.core.models.gpt.gpt_model.GPTModel', 'mindspeed_llm.core.models.gpt.gpt_model.GPTModel'],
            start_key_word='decoder_input is not None',
            end_key_word='return hidden_states',
            cut_mode=True
        ),
        return_values=["loss"],
        append_method_signatures=['hidden_states'],
        module_path="",
        weight_param_module=['output_layer'],
        input_source=[ParamSource('hidden_states'),
                      ParamSource('labels', from_forward=False)]
    )
]