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

import math
from typing import Any

from msprobe.pytorch.free_benchmark import logger
from msprobe.pytorch.free_benchmark.common.constant import ThresholdConfig
from msprobe.pytorch.free_benchmark.common.counter import preheat_counter
from msprobe.pytorch.free_benchmark.common.enums import DeviceType
from msprobe.pytorch.free_benchmark.common.params import DataParams, HandlerParams
from msprobe.pytorch.free_benchmark.common.utils import Tools
from msprobe.pytorch.free_benchmark.compare.single_benchmark import SingleCompare
from msprobe.pytorch.free_benchmark.result_handlers.base_handler import FuzzHandler


class PreheatHandler(FuzzHandler):

    def __init__(self, params: HandlerParams) -> None:
        super().__init__(params)
        self.pure_name = Tools.get_pure_api_name(self.params.api_name)

    def get_threshold(self, dtype):
        return preheat_counter.get_api_thd(self.pure_name, dtype)

    def compare_npu_and_cpu(self, data_params: DataParams):
        args = Tools.convert_device_and_dtype(
            data_params.args, DeviceType.CPU, change_dtype=True
        )
        kwargs = Tools.convert_device_and_dtype(
            data_params.kwargs, DeviceType.CPU, change_dtype=True
        )
        cpu_result = data_params.origin_func(*args, **kwargs)
        return SingleCompare().compare_seq(data_params.original_result, cpu_result)

    def preheat(self, max_fuzz_ratio, cpu_consistent, first_dtype):
        # 存储当前step所有输出比值和对应npu\cpu比对结果
        preheat_counter.update_preheat_record(
            self.pure_name,
            first_dtype,
            (max_fuzz_ratio, cpu_consistent),
        )
        if self._need_adjust_threshold():
            self._adjust_threshold()

    def handle(self, data_params: DataParams) -> Any:

        if isinstance(data_params.perturbed_result, bool) or not Tools.is_float_tensor(
                data_params.perturbed_result
        ):
            return data_params.original_result

        if self.params.step == 0:
            preheat_counter.add_one_step_used_api(self.pure_name)
            return data_params.original_result

        # 如果当前api,step需要预热
        npu_consistent, max_fuzz_ratio = self.cmp_output_npu(data_params)
        data_params.is_consistent = npu_consistent

        preheat_counter.check_step(self.params.step)

        if self.params.preheat_config.get("preheat_step") <= self.params.step:
            return data_params.original_result

        preheat_counter.add_api_called_time(self.pure_name)

        if not self._is_take_a_sample():
            return data_params.original_result

        cpu_consistent = True
        try:
            cpu_consistent = self.compare_npu_and_cpu(data_params)
        except Exception as e:
            logger.warning_on_rank_0(
                f"[msprobe] Free Benchmark: For {self.params.api_name}, "
                f"when campare to cpu exception raise {e}"
            )
        try:
            first_dtype = Tools.get_first_tensor_dtype(data_params.original_result)
        except RuntimeError:
            logger.warning_on_rank_0(
                f"[msprobe] Free Benchmark: For {self.params.api_name}, "
                f"the output sequence does not contain tensors."
            )
        if preheat_counter.get_api_preheat(self.pure_name, str(first_dtype)):
            self.preheat(max_fuzz_ratio, cpu_consistent, first_dtype)

        return data_params.original_result

    def _is_take_a_sample(self) -> bool:
        need_sample_set = self._get_need_sample_set()
        curr_called_seq = preheat_counter.get_api_called_time(self.pure_name)
        res = curr_called_seq in need_sample_set
        if res:
            total_count = preheat_counter.get_one_step_used_api(self.pure_name)
            logger.info_on_rank_0(
                f"[msprobe] Free benchmark: preheat sample in step{self.params.step}"
                f"api_name {self.params.api_name}, "
                f"curr_called_seq: {curr_called_seq}/{total_count}"
            )
            preheat_counter.add_api_sample_time(self.pure_name)
        return res

    def _get_sample_count_per_step(self) -> set:
        """
        每一个step中应该采集的样本数
        """
        total_count = preheat_counter.get_one_step_used_api(self.pure_name)
        preheat_step = self.params.preheat_config.get("preheat_step")
        max_sample = self.params.preheat_config.get("max_sample")
        return min(math.ceil(total_count / preheat_step), max_sample)

    def _get_need_sample_set(self):
        """
        需要采集的api集合
        """
        # 每一步样本数
        total_count = preheat_counter.get_one_step_used_api(self.pure_name)
        need_sample_set = set()
        if total_count == 0:
            return need_sample_set
        sample_count_per_step = self._get_sample_count_per_step()
        prehead_step = self.params.preheat_config.get("preheat_step")
        for i in range(1, sample_count_per_step + 1):
            count = (prehead_step * (i - 1) + self.params.step) % total_count
            if count == 0:
                count = total_count
            need_sample_set.add(count)
        return need_sample_set

    def _need_adjust_threshold(self) -> bool:
        sample_count_per_step = self._get_sample_count_per_step()
        sampled_time = preheat_counter.get_api_sample_time(self.pure_name)
        res = sampled_time >= sample_count_per_step
        return res

    def _adjust_threshold_for_dtype(self, dtype_str, compare_result):
        con_ratio = [ratio for ratio, is_consistent in compare_result if is_consistent]
        incon_ratio = [ratio for ratio, is_consistent in compare_result if not is_consistent]
        old_thd = preheat_counter.get_api_thd(self.pure_name, dtype_str)
        new_thd = old_thd
        # 正例负例都存在
        if con_ratio and incon_ratio:
            if min(incon_ratio) > max(con_ratio):
                new_thd = min(min(incon_ratio), old_thd)
                preheat_counter.set_api_preheat(self.pure_name, dtype_str, is_preheat=False)
        elif con_ratio:
            # 存在漏报
            if max(con_ratio) > old_thd:
                new_thd = 1 + ((old_thd - 1) * ThresholdConfig.API_THD_STEP)
            else:
                new_thd = 1 + ((old_thd - 1) / ThresholdConfig.API_THD_STEP)
        else:
            new_thd = min(min(incon_ratio), old_thd)
            preheat_counter.set_api_preheat(self.pure_name, dtype_str, is_preheat=False)
        return new_thd

    def _adjust_threshold(self):
        for dtype_str, compare_result in preheat_counter.preheat_record[
            self.pure_name
        ].items():
            new_thd = self._adjust_threshold_for_dtype(dtype_str, compare_result)
            threshold = self._get_default_threshold(
                preheat_counter.dtype_map.get(dtype_str)
            )
            preheat_counter.update_api_thd(
                self.pure_name, dtype_str, new_thd, threshold
            )
