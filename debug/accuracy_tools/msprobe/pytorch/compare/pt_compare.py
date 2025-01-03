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

import os.path

import torch

from msprobe.core.common.const import FileCheckConst
from msprobe.core.common.exceptions import FileCheckException
from msprobe.core.common.file_utils import FileChecker, create_directory, load_yaml
from msprobe.core.common.utils import CompareException, check_compare_param, check_configuration_param, get_dump_mode, \
    set_dump_path
from msprobe.core.compare.acc_compare import Comparator, ModeConfig
from msprobe.core.compare.utils import set_stack_json_path
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.common.utils import load_pt


class PTComparator(Comparator):
    def __init__(self, mode_config, data_mapping=None):
        super().__init__(mode_config)

        self.stack_mode = mode_config.stack_mode
        self.auto_analyze = mode_config.auto_analyze
        self.fuzzy_match = mode_config.fuzzy_match
        self.dump_mode = mode_config.dump_mode

        self.frame_name = PTComparator.__name__
        self.data_mapping = data_mapping
        if isinstance(self.data_mapping, str) or self.data_mapping is None:
            self.data_mapping_dict = self.load_mapping_file(self.data_mapping)
        elif isinstance(self.data_mapping, dict):
            self.data_mapping_dict = self.data_mapping
        else:
            raise TypeError(f"The type of parameter `data_mapping` must be dict, str or None, but got "
                            f"{type(self.data_mapping)}")

    @staticmethod
    def load_mapping_file(mapping_file):
        if isinstance(mapping_file, str):
            mapping_dict = load_yaml(mapping_file)
        else:
            mapping_dict = {}
        return mapping_dict

    def read_npy_data(self, dir_path, file_name):
        if not file_name:
            return None
        data_path = os.path.join(dir_path, file_name)
        path_checker = FileChecker(data_path, FileCheckConst.FILE, FileCheckConst.READ_ABLE,
                                   FileCheckConst.PT_SUFFIX, False)
        data_path = path_checker.common_check()
        try:
            # detach because numpy can not process gradient information
            data_value = load_pt(data_path, to_cpu=True).detach()
        except RuntimeError as e:
            # 这里捕获 load_pt 中抛出的异常
            logger.error(f"Failed to load the .pt file at {data_path}.")
            raise CompareException(CompareException.INVALID_FILE_ERROR) from e
        except AttributeError as e:
            # 这里捕获 detach 方法抛出的异常
            logger.error(f"Failed to detach the loaded tensor.")
            raise CompareException(CompareException.DETACH_ERROR) from e
        if data_value.dtype == torch.bfloat16:
            data_value = data_value.to(torch.float32)
        data_value = data_value.numpy()
        return data_value


def compare(input_param, output_path, **kwargs):
    try:
        auto_analyze = kwargs.get('auto_analyze', True)
        fuzzy_match = kwargs.get('fuzzy_match', False)
        data_mapping = kwargs.get('data_mapping', None)
        suffix = kwargs.get('suffix', '')

        set_dump_path(input_param)
        dump_mode = get_dump_mode(input_param)
        if "stack_json_path" in input_param:
            stack_mode = kwargs.get('stack_mode', False)
        else:
            stack_mode = set_stack_json_path(input_param)  # set stack_mode and set "stack_json_path" in input_param
        check_configuration_param(stack_mode, auto_analyze, fuzzy_match, input_param.get('is_print_compare_log', True))
        create_directory(output_path)
        check_compare_param(input_param, output_path, dump_mode, stack_mode)
    except (CompareException, FileCheckException) as error:
        logger.error('Compare failed. Please check the arguments and do it again!')
        raise CompareException(error.code) from error

    mode_config = ModeConfig(stack_mode, auto_analyze, fuzzy_match, dump_mode)
    pt_comparator = PTComparator(mode_config, data_mapping)
    pt_comparator.compare_core(input_param, output_path, suffix=suffix)
