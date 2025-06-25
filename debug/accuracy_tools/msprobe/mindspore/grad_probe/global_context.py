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
import threading
from typing import Dict, Union, Tuple
import time

from msprobe.core.common.utils import is_int
from msprobe.core.common.file_utils import create_directory, check_path_before_create
from msprobe.core.grad_probe.constant import GradConst
from msprobe.core.grad_probe.utils import check_str, check_bounds_element, check_param_element
from msprobe.mindspore.common.log import logger


class GlobalContext:
    _instance = None
    _instance_lock = threading.Lock()
    _setting = {
        GradConst.LEVEL: None,
        GradConst.PARAM_LIST: None,
        GradConst.STEP: None,
        GradConst.RANK: None,
        GradConst.CURRENT_STEP: 0,
        GradConst.BOUNDS: [-1, 0, 1],
        GradConst.OUTPUT_PATH: None
    }

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance_lock.acquire()
            try:
                cls._instance = object.__new__(cls)
            except Exception as e:
                raise RuntimeError("grad_probe global context init failed") from e
            finally:
                cls._instance_lock.release()
        return cls._instance

    def init_context(self, config_dict: Dict):
        level = config_dict.get(GradConst.LEVEL)
        check_str(level, variable_name="level in yaml")
        if level in GradConst.SUPPORTED_LEVEL:
            self._setting[GradConst.LEVEL] = config_dict.get(GradConst.LEVEL)
        else:
            raise ValueError("Invalid level set in config yaml file, level option: L0, L1, L2")

        self._set_input_list(config_dict, GradConst.PARAM_LIST, (str,), element_check=check_param_element)
        self._set_input_list(config_dict, GradConst.BOUNDS, (float, int), element_check=check_bounds_element)
        self._set_input_list(config_dict, GradConst.STEP, (int,))
        self._set_input_list(config_dict, GradConst.RANK, (int,))

        output_path = config_dict.get(GradConst.OUTPUT_PATH)
        check_str(output_path, variable_name="output_path in yaml")
        try:
            check_path_before_create(output_path)
        except RuntimeError as err:
            raise ValueError(f"Invalid output_path: {output_path}. The error message is {err}.") from err
        self._setting[GradConst.OUTPUT_PATH] = output_path
        if not os.path.isdir(self._setting.get(GradConst.OUTPUT_PATH)):
            create_directory(self._setting.get(GradConst.OUTPUT_PATH))
        else:
            logger.warning("The output_path exists, the data will be covered.")

        self._setting[GradConst.TIME_STAMP] = str(int(time.time()))

    def get_context(self, key: str):
        if key not in self._setting:
            logger.warning(f"Unrecognized {key}.")
        return self._setting.get(key)

    def update_step(self):
        self._setting[GradConst.CURRENT_STEP] += 1

    def step_need_dump(self, step):
        dump_step_list = self.get_context(GradConst.STEP)
        return (not dump_step_list) or (step in dump_step_list)

    def rank_need_dump(self, rank):
        dump_rank_list = self.get_context(GradConst.RANK)
        return (not dump_rank_list) or (rank in dump_rank_list)

    def _get_type_str(self, dtype: Union[int, str, float, Tuple[int, str, float]]):
        if isinstance(dtype, tuple):
            return "/".join([self._get_type_str(element) for element in dtype])
        if dtype == int:
            type_str = "integer"
        elif dtype == float:
            type_str = "float"
        else:
            type_str = "string"
        return type_str

    def _set_input_list(self, config_dict: Dict, name: str,
                        dtype: Union[int, str, float, Tuple[int, str, float]], element_check=None):
        value = config_dict.get(name)
        type_str = self._get_type_str(dtype)
        if value and isinstance(value, list):
            for val in value:
                if not isinstance(val, dtype):
                    logger.warning(f"Invalid {name} which must be None or list of {type_str}, use default value.")
                    return
                elif isinstance(val, int) and not is_int(val):
                    logger.warning(f"Invalid {name} which must be None or list of int, use default value.")
                    return
                if element_check and not element_check(val):
                    logger.warning(f"Given {name} violates some rules, use default value.")
                    return

            self._setting[name] = value
        else:
            logger.warning(f"{name} is None or not a list with valid items, use default value.")


grad_context = GlobalContext()
