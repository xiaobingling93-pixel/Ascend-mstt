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

import csv
import os
import copy
import numpy as np

from msprobe.core.common.const import Const, FileCheckConst
from msprobe.core.common.file_utils import change_mode, FileOpen, save_json, load_json
from msprobe.core.common.log import logger
from msprobe.core.common.exceptions import MsprobeException


class DataWriter:

    def __init__(self) -> None:
        self.dump_file_path = None
        self.stack_file_path = None
        self.construct_file_path = None
        self.free_benchmark_file_path = None
        self.dump_tensor_data_dir = None
        self.debug_file_path = None
        self.flush_size = 1000
        self.cache_data = {}
        self.cache_stack = {}
        self.cache_construct = {}
        self.cache_debug = {}

    @staticmethod
    def write_data_to_csv(result: list, result_header: tuple, file_path: str):
        if not result:
            return
        is_exists = os.path.exists(file_path)
        append = "a+" if is_exists else "w+"
        with FileOpen(file_path, append) as csv_file:
            spawn_writer = csv.writer(csv_file)
            if not is_exists:
                spawn_writer.writerow(result_header)
            spawn_writer.writerows([result,])
        is_new_file = not is_exists
        if is_new_file:
            change_mode(file_path, FileCheckConst.DATA_FILE_AUTHORITY)

    def reset_cache(self):
        self.cache_data = {}
        self.cache_stack = {}
        self.cache_construct = {}

    def initialize_json_file(self, **kwargs):
        if self.debug_file_path and not self.cache_debug:
            # debug level case only create debug.json
            debug_dict = copy.deepcopy(kwargs)
            debug_dict.update({"dump_data_dir": self.dump_tensor_data_dir, Const.DATA: {}})
            self.cache_debug = debug_dict
            save_json(self.debug_file_path, self.cache_debug, indent=1)
            return
        if not self.cache_data:
            kwargs.update({"dump_data_dir": self.dump_tensor_data_dir, Const.DATA: {}})
            self.cache_data = kwargs
            save_json(self.dump_file_path, self.cache_data, indent=1)
        if not self.cache_stack:
            save_json(self.stack_file_path, self.cache_stack, indent=1)
        if not self.cache_construct:
            save_json(self.construct_file_path, self.cache_construct, indent=1)

    def update_dump_paths(self, dump_path_aggregation):
        self.dump_file_path = dump_path_aggregation.dump_file_path
        self.stack_file_path = dump_path_aggregation.stack_file_path
        self.construct_file_path = dump_path_aggregation.construct_file_path
        self.dump_tensor_data_dir = dump_path_aggregation.dump_tensor_data_dir
        self.free_benchmark_file_path = dump_path_aggregation.free_benchmark_file_path
        self.debug_file_path = dump_path_aggregation.debug_file_path

    def flush_data_periodically(self):
        dump_data = self.cache_data.get(Const.DATA)
        if dump_data and isinstance(dump_data, dict) and len(dump_data) % self.flush_size == 0:
            self.write_json()

    def update_data(self, new_data):
        if not isinstance(new_data, dict) or len(new_data.keys()) != 1:
            logger.warning(f"The data info({new_data}) should be a dict with only one outer key.")
            return
        dump_data = self.cache_data.get(Const.DATA)
        if not isinstance(dump_data, dict):
            logger.warning(f"The dump data({dump_data}) should be a dict.")
            return

        key = next(iter(new_data.keys()))
        if key in dump_data:
            dump_data.get(key).update(new_data.get(key))
        else:
            dump_data.update(new_data)

    def update_stack(self, new_data):
        self.cache_stack.update(new_data)

    def update_construct(self, new_data):
        self.cache_construct.update(new_data)

    def update_debug(self, new_data):
        self.cache_debug['data'].update(new_data)

    def write_data_json(self, file_path):
        logger.info(f"dump.json is at {os.path.dirname(os.path.dirname(file_path))}. ")
        save_json(file_path, self.cache_data, indent=1)

    def write_stack_info_json(self, file_path):
        save_json(file_path, self.cache_stack, indent=1)

    def write_construct_info_json(self, file_path):
        save_json(file_path, self.cache_construct, indent=1)

    def write_debug_info_json(self, file_path):
        save_json(file_path, self.cache_debug, indent=1)

    def write_json(self):
        if self.cache_data:
            self.write_data_json(self.dump_file_path)
        if self.cache_stack:
            self.write_stack_info_json(self.stack_file_path)
        if self.cache_construct:
            self.write_construct_info_json(self.construct_file_path)
        if self.cache_debug:
            self.write_debug_info_json(self.debug_file_path)

    def fill_stack_tensor_data(self):
        self.process_stat_data_recursive(self.cache_data)

    def process_stat_data_recursive(self, data, depth=0):
        if depth > Const.MAX_DEPTH:
            logger.error(f"The maximum depth of recursive process stat data, {Const.MAX_DEPTH} is reached.")
            raise MsprobeException(MsprobeException.RECURSION_LIMIT_ERROR)
        if isinstance(data, dict):
            if "tensor_stat" in data.keys():
                tensor_stat = data["tensor_stat"]
                if len(tensor_stat) != Const.TENSOR_STAT_LEN or len(tensor_stat[0]) != len(tensor_stat[1]):
                    logger.warning("Some bad data in async dump")
                else:
                    tensor_stat_index, tensor_stat_data = tensor_stat[0], tensor_stat[1]
                    if hasattr(tensor_stat_data, "device") and tensor_stat_data.device != Const.CPU_LOWERCASE:
                        tensor_stat_data = tensor_stat_data.cpu()
                    for index, stat in zip(tensor_stat_index, tensor_stat_data):
                        data.update({index: stat.item()})
                del data["tensor_stat"]
            else:
                for key in data.keys():
                    self.process_stat_data_recursive(data[key], depth + 1)
        elif isinstance(data, (list, tuple)):
            for i in data:
                self.process_stat_data_recursive(i, depth + 1)