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
from msprobe.pytorch.common.log import logger
from msprobe.core.common.exceptions import FileCheckException
from msprobe.core.compare.acc_compare import Comparator
from msprobe.core.common.utils import check_configuration_param, task_dumppath_get, check_compare_param, \
    CompareException
from msprobe.core.common.file_utils import FileChecker, create_directory, load_yaml
from msprobe.pytorch.common.utils import load_pt


class PTComparator (Comparator):
    def __init__(self, data_mapping=None):
        self.frame_name = PTComparator.__name__
        self.data_mapping = data_mapping
        if isinstance(self.data_mapping, str) or self.data_mapping is None:
            self.data_mapping_dict = self.load_mapping_file(self.data_mapping)
        elif isinstance(self.data_mapping, dict):
            self.data_mapping_dict = self.data_mapping
        else:
            raise TypeError(f"The type of parameter `data_mapping` must be dict, str or None, but got "
                            f"{type(self.data_mapping)}")

    def load_mapping_file(self, mapping_file):
        if isinstance(mapping_file, str):
            mapping_dict = load_yaml(mapping_file)
        else:
            mapping_dict = {}
        return mapping_dict
    
    def read_npy_data(self, dir_path, file_name):
        data_path = os.path.join(dir_path, file_name)
        path_checker = FileChecker(data_path, FileCheckConst.FILE, FileCheckConst.READ_ABLE,
                                FileCheckConst.PT_SUFFIX, False)
        data_path = path_checker.common_check()
        try:
            data_value = load_pt(data_path,
                                 to_cpu=True).detach()  # detach because numpy can not process gradient information
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
    
    
def compare(input_param, output_path, stack_mode=False, auto_analyze=True, fuzzy_match=False, **kwargs):
    try:
        summary_compare, md5_compare = task_dumppath_get(input_param)
        check_configuration_param(stack_mode, auto_analyze, fuzzy_match, input_param.get('is_print_compare_log', True))
        create_directory(output_path)
        check_compare_param(input_param, output_path, summary_compare, md5_compare)
        data_mapping = kwargs.get('data_mapping', None)
    except (CompareException, FileCheckException) as error:
        logger.error('Compare failed. Please check the arguments and do it again!')
        raise CompareException(error.code) from error
    pt_comparator = PTComparator(data_mapping)
    pt_comparator.compare_core(input_param, output_path, stack_mode=stack_mode,
                 auto_analyze=auto_analyze, fuzzy_match=fuzzy_match, summary_compare=summary_compare,
                 md5_compare=md5_compare)
