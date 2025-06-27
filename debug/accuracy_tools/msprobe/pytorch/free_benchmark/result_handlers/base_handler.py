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
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np
import torch
from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import FreeBenchmarkException
from msprobe.pytorch.free_benchmark import logger
from msprobe.pytorch.free_benchmark.common.constant import ThresholdConfig
from msprobe.pytorch.free_benchmark.common.enums import (
    FuzzThreshold,
    NormType,
    PerturbationMode,
)
from msprobe.pytorch.free_benchmark.common.params import (
    DataParams,
    HandlerParams,
    make_unequal_row,
)
from msprobe.pytorch.free_benchmark.common.utils import Tools, TorchC


class FuzzHandler(ABC):
    def __init__(self, params: HandlerParams) -> None:
        self.params = params
        self.unequal_rows = []

    @staticmethod
    def pre_process(origin_ouput, perturbed_output):
        if (
            isinstance(origin_ouput, tuple)
            and hasattr(origin_ouput, "values")
            and hasattr(origin_ouput, "indices")
        ):
            origin_ouput = origin_ouput.values
            perturbed_output = perturbed_output.values
        if hasattr(perturbed_output, "dtype"):
            abs_tol = ThresholdConfig.ABS_TOL_VALUE_DICT.get(
                perturbed_output.dtype, FuzzThreshold.F32_THD
            )
        else:
            abs_tol = FuzzThreshold.F32_THD
        return (
            origin_ouput.to(perturbed_output.dtype).to(perturbed_output.device),
            perturbed_output,
            abs_tol,
        )

    @staticmethod
    def tensor_split_for_error_calculate(origin_output, perturbed_output):
        """
        对将投入误差值计算的扰动前后输出张量进行分块
        :param origin_output: 原始输出
        :param perturbed_output: 扰动后输出
        :return origin_output_chunks: 切块后原始输出列表
        :return perturbed_output_chunks: 切块后扰动后输出列表
        """
        single_output_mem = (
            origin_output.element_size() * origin_output.nelement() / Const.ONE_MB
        )
        if single_output_mem == 0 or origin_output.ndim == 0:
            return [origin_output], [perturbed_output]
        # 张量大小和批数之间的关系：chunks_exp=math.log(M,2)-4, chunks=2**chunks_exp (M为对比张量数据大小[Mb])
        chunks_exp = int(math.log(single_output_mem, 2)) - 4
        chunks = 2**chunks_exp
        chunks = max(chunks, 1)
        chunks = min(chunks, ThresholdConfig.TENSOR_SPLIT_MAX_CHUNK)
        origin_output_chunks = TorchC.tensor_split(
            TorchC.reshape(origin_output, (-1,)), chunks
        )
        perturbed_output_chunks = TorchC.tensor_split(
            TorchC.reshape(perturbed_output, (-1,)), chunks
        )
        return origin_output_chunks, perturbed_output_chunks

    @abstractmethod
    def get_threshold(self, dtype):
        pass

    @abstractmethod
    def handle(self, data_params: DataParams) -> Any:
        pass

    def get_ratio_from_specific_norm(
        self, origin_output, perturbed_output, norm_type, abs_tol
    ):
        if norm_type == NormType.ENDLESS_NORM:
            return self.calculate_max_ratio(origin_output, perturbed_output, abs_tol)
        return ThresholdConfig.COMP_CONSISTENT

    def calculate_max_ratio(self, origin_output, perturbed_output, abs_tol):
        origin_output_chunks, perturbed_output_chunks = (
            self.tensor_split_for_error_calculate(origin_output, perturbed_output)
        )
        if len(origin_output_chunks) != len(perturbed_output_chunks):
            err_msg = (
                f"For {self.params.api_name}, the number of compare tensor chunks is different: "
                f"{len(origin_output_chunks)} != {len(perturbed_output_chunks)}. please check!"
            )
            raise FreeBenchmarkException(
                FreeBenchmarkException.OutputIndexError, err_msg
            )

        max_ratio = ThresholdConfig.COMP_CONSISTENT
        for i, chunk_origin in enumerate(origin_output_chunks):
            if chunk_origin.nelement() == 0:
                break
            chunk_perturbed = perturbed_output_chunks[i]
            # 如果乘积最小值 < 极小值乘积的负值，认为存在非极小值符号相反的情况
            if TorchC.lt(
                TorchC.min(TorchC.mul(chunk_origin, chunk_perturbed)), -(abs_tol**2)
            ):
                return ThresholdConfig.SYMBOL_FLIPPING
            # 求A/B B/A的比值前，将值限制在大于极小值范围内
            clamp_origin = TorchC.clamp(TorchC.abs(chunk_origin), min=abs_tol)
            clamp_perturbed = TorchC.clamp(TorchC.abs(chunk_perturbed), min=abs_tol)
            # 对于计算结果为nan的情况，认为两者没有差异
            ratio_tensor = TorchC.nan_to_num(
                TorchC.div(clamp_origin, clamp_perturbed),
                nan=ThresholdConfig.COMP_CONSISTENT,
            )
            # 求A/B 和 B/A比值最大值，其中 B/A的最大值为 A/B的最小值的倒数
            min_ratio, max_ratio = TorchC.stack([*TorchC.aminmax(ratio_tensor)]).tolist()
            min_ratio_reciprocal = np.inf if min_ratio == 0 else 1 / min_ratio
            max_ratio = max(max_ratio, min_ratio_reciprocal)
        return max_ratio

    def ratio_calculate(self, origin_output, perturbed_output, norm_type) -> float:
        try:
            origin_output, perturbed_output, abs_tol = self.pre_process(
                origin_output, perturbed_output
            )
        except Exception as e:
            logger.warning_on_rank_0(
                f"[msprobe] Free Benchmark: For {self.params.api_name}, "
                f"when computing ratio,"
                f" y1 or y2 dtype is not supported {e}"
            )
            return ThresholdConfig.COMP_NAN
        if self.params.fuzz_stage == Const.BACKWARD:
            abs_tol = ThresholdConfig.BACKWARD_OUTPUT_LOWER_BOUND
        else:
            abs_tol = abs_tol**0.5
        return self.get_ratio_from_specific_norm(
            origin_output, perturbed_output, norm_type, abs_tol
        )

    def npu_compare(
        self, origin_output, perturbed_output
    ) -> Tuple[bool, Optional[float]]:

        if isinstance(perturbed_output, int):
            return origin_output == perturbed_output, None
        elif isinstance(perturbed_output, float):
            if perturbed_output == 0:
                origin_output += FuzzThreshold.F32_THD
                perturbed_output += FuzzThreshold.F32_THD
            return (
                math.isclose(origin_output, perturbed_output),
                origin_output / perturbed_output,
            )
        elif not isinstance(perturbed_output, torch.Tensor):
            logger.warning_on_rank_0(
                f"[msprobe] Free Benchmark: For {self.params.api_name} "
                f"The compare for output type {type(perturbed_output)} is not supported"
            )
            return True, 1

        threshold = self.get_threshold(Tools.get_first_tensor_dtype(origin_output))
        ratio = self.ratio_calculate(
            origin_output, perturbed_output, norm_type=NormType.ENDLESS_NORM
        )
        if threshold == 0:
            raise ValueError("Threshold cannot be zero. Check `get_threshold` implementation.")
        if ratio == ThresholdConfig.SYMBOL_FLIPPING:
            is_consistent = False
        else:
            is_consistent = threshold >= ratio >= 1 / threshold
        return is_consistent, ratio

    def cmp_output_npu(self, data_params: DataParams):
        npu_consistent = True
        max_fuzz_ratio = 0
        try:
            if isinstance(data_params.original_result, torch.Tensor):
                is_consistent, ratio = self.npu_compare(
                    data_params.original_result, data_params.perturbed_result
                )
                npu_consistent = is_consistent
                max_fuzz_ratio = (
                    max_fuzz_ratio
                    if not isinstance(ratio, (int, float))
                    else max(max_fuzz_ratio, ratio)
                )
                data_params.is_consistent = is_consistent
                if not is_consistent:
                    self.unequal_rows.append(
                        make_unequal_row(data_params, self.params, ratio=ratio)
                    )

            elif isinstance(data_params.original_result, (list, tuple)):
                for index_, origin_item in enumerate(data_params.original_result):
                    is_consistent, ratio = self.npu_compare(
                        origin_item, data_params.perturbed_result[index_]
                    )
                    npu_consistent = npu_consistent and is_consistent
                    max_fuzz_ratio = (
                        max_fuzz_ratio
                        if not isinstance(ratio, (int, float))
                        else max(max_fuzz_ratio, ratio)
                    )
                    data_params.is_consistent = is_consistent
                    if not is_consistent:
                        self.unequal_rows.append(
                            make_unequal_row(
                                data_params, self.params, ratio=ratio, index=index_
                            )
                        )
        except Exception as e:
            logger.warning_on_rank_0(
                f"[msprobe] Free Benchmark: For {self.params.api_name}, "
                f"when campare the result exception raise {e}"
            )
        return npu_consistent, max_fuzz_ratio

    def get_unequal_rows(self):
        return self.unequal_rows

    def _get_default_threshold(self, dtype):
        if self.params.pert_mode == PerturbationMode.NO_CHANGE:
            threshold = ThresholdConfig.COMP_CONSISTENT
        else:
            threshold = ThresholdConfig.DTYPE_PER_THD.get(
                dtype, ThresholdConfig.DTYPE_PER_THD.get(torch.float32)
            )
        return threshold
