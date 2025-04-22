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

import os
from abc import ABC, abstractmethod

import torch

from msprobe.core.common.const import FileCheckConst


class PackInput:

    def __init__(self, output_zip_path, model, shell_path):
        self.output_zip_path = output_zip_path
        self.shell_path = shell_path
        self.model = model[0] if isinstance(model, list) else model
        self.check_input_params()

    def check_input_params(self):
        if self.model and not isinstance(self.model, torch.nn.Module):
            raise Exception(f"model is not torch.nn.Module or module list.")
        if not isinstance(self.output_zip_path, str) or not self.output_zip_path.endswith(FileCheckConst.ZIP_SUFFIX):
            raise Exception(f"output zip path must be a string and ends with '.zip'")


class BaseChecker(ABC):
    input_needed = None
    target_name_in_zip = None
    multi_rank = False

    @staticmethod
    @abstractmethod
    def pack(pack_input):
        pass

    @staticmethod
    @abstractmethod
    def compare(bench_dir, cmp_dir, output_path):
        pass

    @classmethod
    def compare_ex(cls, bench_dir, cmp_dir, output_path):
        bench_filepath = os.path.join(bench_dir, cls.target_name_in_zip)
        cmp_filepath = os.path.join(cmp_dir, cls.target_name_in_zip)
        if not os.path.exists(bench_filepath) or not os.path.exists(cmp_filepath):
            return None, None, None
        return cls.compare(bench_dir, cmp_dir, output_path)
