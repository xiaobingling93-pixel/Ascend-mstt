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

import re
from dataclasses import dataclass, fields
from typing import Optional

import torch


@dataclass(frozen=True)
class ScriptArgs:
    tp: int = 1
    sp: int = 0
    ep: int = 1

    @property
    def cmd_text_list(self):
        return [str(value) for value in self.__dict__.values()]

    def items(self):
        return self.__dict__.items()

    def is_legal(self, max_npu, args):
        p_times = 1
        for p in [self.tp, self.ep]:
            if p > 1:
                p_times *= p
        illegal = [
            max_npu % p_times,
            self.tp == 1 and self.sp,
            args.hidden_size % self.tp,
            args.ffn_hidden_size % self.tp,
            args.num_attention_heads % self.tp,
            args.group_query_attention and args.num_query_groups % self.tp,
            args.seq_length % self.tp and self.sp,
            self.ep > 1 and not hasattr(args, 'num_experts')
        ]
        if not hasattr(args, 'num_experts'):
            return not any(illegal)
        # MoE
        illegal.extend([
            self.tp > 1 and not self.sp,
            args.num_experts % self.ep,
        ])
        return not any(illegal)


@dataclass(frozen=True)
class ProfileArgs(ScriptArgs):
    """
    测量属性，继承自ScriptArgs

    属性：
        mbs: int - micro barch size
        tp: int - tensor parallel size
        sp: int - sequence parallel size
        ep: int - expert parallel
    """
    mbs: int = 1
    algo: int = 0
    model: Optional[str] = None

    @property
    def hint(self):
        return self._text(", ", " = ")

    @property
    def file_name(self):
        """组织为profile data的csv名称，需与new_from_file_name匹配，互为逆过程"""
        return f"{self.model}{self._text(pre_split='_')}.csv"

    @property
    def npu_used(self):
        p_times = 1
        for p in [self.tp, self.ep]:
            if p and p > 1:
                p_times *= p
        return p_times

    @property
    def is_moe(self):
        return isinstance(self.ep, int) and self.ep > 0

    @classmethod
    def new_from_file_name(cls, file_name: str) -> 'ProfileArgs':
        """从文件名中还原出profile args"""
        parsed_data = {key: int(value) for key, value in re.findall(r"([a-zA-Z]+)(\d+)", file_name)}

        # 获取所有数据类的属性名，并使用默认值
        default_values = {field.name: field.default for field in fields(cls)}

        # 更新默认值为字符串中的值，忽略不存在的属性
        for key, value in parsed_data.items():
            if key in default_values:
                default_values[key] = value

        # 使用解析后的值来创建数据类实例
        return cls(**default_values)

    def update_mbs(self, mbs):
        """支撑一个script内的profile循环 当前功能为 接受要更新的mbs 返回一个新实例"""
        new_dict = self.__dict__
        new_dict.update(dict(mbs=mbs))
        return ProfileArgs(**new_dict)

    def _text(self, pre_split="", post_split=""):
        exclude_info = {"model"}

        text = ""
        for k, v in self.__dict__.items():
            if k in exclude_info:
                continue
            text += f'{pre_split}{k}{post_split}{v}'
        return text


@dataclass(frozen=True)
class InitTensorInfo:
    shape: torch.Size
    requires_grad: bool
    device: torch.device
    dtype: torch.dtype
    element_size: int
