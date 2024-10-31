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

from collections import defaultdict
from msprobe.pytorch.free_benchmark.common.constant import ThresholdConfig


class PreheatCounter:
    def __init__(self) -> None:
        self.api_called_time: dict = defaultdict(int)
        self.api_sample_time: dict = defaultdict(int)
        self.one_step_used_api: dict = defaultdict(int)
        self.api_thd: dict = defaultdict(dict)
        self.preheat_record: dict = defaultdict(dict)
        self.dtype_map: dict = {}
        self.if_preheat: dict = defaultdict(dict)
        self.step = 0

    def clear_step(self):
        self.preheat_record.clear()
        self.api_called_time.clear()
        self.api_sample_time.clear()
    
    def check_step(self, current_step):
        if current_step != self.step:
            self.clear_step()
            self.step = current_step

    def add_api_called_time(self, api_name: str):
        self.api_called_time[api_name] += 1

    def get_api_called_time(self, api_name: str) -> int:
        return self.api_called_time[api_name]
    
    def add_api_sample_time(self, api_name: str):
        self.api_sample_time[api_name] += 1

    def get_api_sample_time(self, api_name: str) -> int:
        return self.api_sample_time[api_name]

    def add_one_step_used_api(self, api_name: str):
        self.one_step_used_api[api_name] += 1

    def get_one_step_used_api(self, api_name: str):
        return self.one_step_used_api[api_name]

    def update_preheat_record(self, api_name, dtype, cmp_result):
        # 记录预热阶段CPU标杆比对的结果
        if str(dtype) not in self.preheat_record[api_name].keys():
            self.preheat_record[api_name][str(dtype)] = list()
        self.preheat_record[api_name][str(dtype)].append(cmp_result)
        self.dtype_map[str(dtype)] = dtype

    def update_api_thd(self, api_name, dtype, threshold, dthreshold):
        self.api_thd[api_name][str(dtype)] = (
            threshold if threshold > dthreshold else dthreshold
        )

    def get_api_thd(self, api_name, dtype):
        if not str(dtype) in self.api_thd[api_name]:
            self.api_thd[api_name][str(dtype)] = ThresholdConfig.PREHEAT_INITIAL_THD
            self.dtype_map[str(dtype)] = dtype
        return self.api_thd[api_name][str(dtype)]

    def set_api_preheat(self, api_name, dtype_str, is_preheat=True):
        # 标记cpu不一致的dtype 不再进行预热
        self.if_preheat[api_name][dtype_str] = is_preheat

    def get_api_preheat(self, api_name, dtype):
        # 标记cpu不一致的dtype 不再进行预热
        if str(dtype) not in self.if_preheat[api_name]:
            return True
        return self.if_preheat[api_name][str(dtype)]

preheat_counter = PreheatCounter()