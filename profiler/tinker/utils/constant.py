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

import enum
from dataclasses import dataclass, field
from typing import Dict


class Version(enum.Enum):
    MindSpeed_LLM_1_0_rc1 = "1.0"
    MindSpeed_LLM_1_0_rc2 = "1.1"
    MindSpeed_LLM_1_0_rc3 = "1.2"

    def __str__(self):
        return self.value


VERSION_ALIASES: Dict[str, Version] = {
    # 映射标准化版本
    "1.0": Version.MindSpeed_LLM_1_0_rc1,
    "1.0.RC1": Version.MindSpeed_LLM_1_0_rc1,
    "1.1": Version.MindSpeed_LLM_1_0_rc2,
    "1.0.RC2": Version.MindSpeed_LLM_1_0_rc2,
    "1.2": Version.MindSpeed_LLM_1_0_rc3,
    "1.0.RC3": Version.MindSpeed_LLM_1_0_rc3,
}

PYTHON_STANDARD_INDENT = ' ' * 4

MODULE_NAME = 'genned_block_forward'


def version_parse(version_str: str) -> Version:
    normalized_str = version_str.strip().upper()
    if normalized_str.startswith('V'):
        normalized_str = normalized_str[1:]
    if normalized_str not in VERSION_ALIASES:
        raise ValueError(f"Unrecognized version: {version_str}, supported versions: {VERSION_ALIASES.keys()}")
    return VERSION_ALIASES[normalized_str]


@dataclass
class ProfileParameter:
    model_name: str = "example"
    model_size: str = "7b"
    pretrain_script_path: str = "./pretrain_qwen_7b_ptd.sh"
    version: str = "1.0.0"
    save_path: str = "./profiled_data"
    prof_path: str = None
    prof_sp: str = None
    max_mbs: int = 65536
    task_id: str = "test"
    max_npu: int = 8


@dataclass
class SearchParameter:
    profiled_data_path: str = None
    global_batch_size: int = None
    num_nodes: int = 4
    num_npus_per_node: int = 8
    cpus: int = 20
    memory_limit: int = 57000
    output_dir: str = "./results"
    pretrain_script_path_search: str = None


@dataclass
class SimulateParameter:
    profiled_data_path: str = None
    global_batch_size: int = None
    num_nodes: int = 4
    num_npus_per_node: int = 8
    simu_tp: int = 1
    simu_pp: int = 1
    simu_ep: int = 1
    simu_sp: int = 0
    dist_opt: int = 0
    micro_batch_size: int = 1
    num_layer_list: list = None
    recompute: int = 0


@dataclass
class InitialValues:
    profile: ProfileParameter = field(default_factory=ProfileParameter)
    search: SearchParameter = field(default_factory=SearchParameter)
    simulate: SimulateParameter = field(default_factory=SimulateParameter)