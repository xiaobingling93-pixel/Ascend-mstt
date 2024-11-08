# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
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
from msprobe.core.common.file_utils import load_yaml
from msprobe.core.compare.layer_mapping import generate_api_mapping_by_layer_mapping

DATA_MAPPING = 'data_mapping'
LAYER_MAPPING = 'layer_mapping'


@dataclass
class MappingInfo:
    mapping_type: str = None
    dump_json_n: str = None
    dump_json_b: str = None


class MappingConfig:

    def __init__(self, yaml_file, mapping_info):
        self.mapping_type = mapping_info.mapping_type
        self.config = {}
        if self.mapping_type == DATA_MAPPING:
            yaml_file = load_yaml(yaml_file)
            try:
                self.config = {key: self.validate(key, value) for key, value in yaml_file.items()}
            except Exception as e:
                raise RuntimeError("Line of yaml contains content that is not 'key(str): value(str)'.") from e
        elif self.mapping_type == LAYER_MAPPING:
            try:
                api_mapping = generate_api_mapping_by_layer_mapping(mapping_info.dump_json_n, mapping_info.dump_json_b,
                                                                    yaml_file)
                self.config = api_mapping
            except Exception as e:
                raise RuntimeError("The yaml file format is incorrect.") from e

    @staticmethod
    def validate(key, value):
        if not isinstance(key, str):
            raise ValueError(f"{key} must be a string.")
        if not isinstance(value, str):
            raise ValueError(f"{value} must be a string.")
        return value

    def get_mapping_string(self, origin_string: str):
        return self.config.get(origin_string, origin_string)
