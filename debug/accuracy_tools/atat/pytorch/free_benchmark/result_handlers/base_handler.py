import math
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
from atat.pytorch.free_benchmark import (
    Const,
    print_warn_log_rank_0,
)
from atat.pytorch.free_benchmark.common.utils import TorchC
from atat.pytorch.free_benchmark.common.constant import ThresholdConfig
from atat.pytorch.free_benchmark.common.enums import (
    FuzzThreshold,
    NormType,
    PerturbationMode,
)
from atat.pytorch.free_benchmark.common.params import DataParams, HandlerParams, make_unequal_row


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
            abs_tol = ThresholdConfig.ABS_TOL_VALUE_DICT.get(perturbed_output.dtype)
        else:
            abs_tol = FuzzThreshold.F32_THD.value
        return (
            origin_ouput.to(perturbed_output.dtype).to(perturbed_output.device),
            perturbed_output,
            abs_tol,
        )

    def get_ratio_from_specific_norm(
        self, origin_output, perturbed_output, norm_type, abs_tol
    ):
        if norm_type == NormType.ENDLESS_NORM:
            return self.get_endless_norm(origin_output, perturbed_output, abs_tol)
        return ThresholdConfig.COMP_CONSISTENT

    @staticmethod
    def convert_overflow_ratio_to_consistent(ratio):
        if math.isnan(ratio) or math.isinf(ratio):
            return ThresholdConfig.COMP_CONSISTENT
        return ratio

    def get_endless_norm(self, origin_output, perturbed_output, abs_tol):
        try:
            ratio_tensor1 = TorchC.where(
                TorchC.gt(TorchC.abs(perturbed_output), abs_tol),
                TorchC.div(
                    TorchC.abs(origin_output),
                    TorchC.add(TorchC.abs(perturbed_output), abs_tol),
                ),
                1,
            )
            ratio_tensor2 = TorchC.where(
                TorchC.gt(TorchC.abs(origin_output), abs_tol),
                TorchC.div(
                    TorchC.abs(perturbed_output),
                    TorchC.add(TorchC.abs(origin_output), abs_tol),
                ),
                1,
            )
        except:
            ratio_tensor1 = TorchC.where(
                TorchC.gt(TorchC.abs(perturbed_output.to(torch.float32)), abs_tol),
                TorchC.div(
                    origin_output.to(torch.float32), perturbed_output.to(torch.float32)
                ),
                1,
            )
            ratio_tensor2 = TorchC.where(
                TorchC.gt(TorchC.abs(origin_output.to(torch.float32)), abs_tol),
                TorchC.div(
                    perturbed_output.to(torch.float32), origin_output.to(torch.float32)
                ),
                1,
            )
        norm1 = self.convert_overflow_ratio_to_consistent(
            TorchC.max(ratio_tensor1).item()
        )
        norm2 = self.convert_overflow_ratio_to_consistent(
            TorchC.max(ratio_tensor2).item()
        )
        norm3 = self.convert_overflow_ratio_to_consistent(
            TorchC.min(ratio_tensor1).item()
        )
        if norm3 < 0:
            ratio = ThresholdConfig.SYMBOL_FLIPPING
        else:
            ratio = max(norm1, norm2)
        return ratio

    def ratio_calculate(self, origin_output, perturbed_output, norm_type) -> float:
        try:
            origin_output, perturbed_output, abs_tol = self.pre_process(
                origin_output, perturbed_output
            )
        except Exception as e:
            print_warn_log_rank_0(
                f"[atat] Free Benchmark: For {self.params.api_name}, "
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

    @abstractmethod
    def get_threshold(self, dtype):
        pass

    def _get_default_threshold(self, dtype):
        if self.params.pert_mode == PerturbationMode.NO_CHANGE:
            threshold = ThresholdConfig.COMP_CONSISTENT
        else:
            threshold = ThresholdConfig.DTYPE_PER_THD.get(
                dtype, ThresholdConfig.DTYPE_PER_THD.get(torch.float32)
            )
        return threshold

    def npu_compare(
        self, origin_output, perturbed_output
    ) -> Tuple[bool, Optional[float]]:

        if isinstance(perturbed_output, int):
            return origin_output == perturbed_output, None
        elif isinstance(perturbed_output, float):
            return (
                math.isclose(origin_output, perturbed_output),
                origin_output / perturbed_output,
            )
        elif not isinstance(perturbed_output, torch.Tensor):
            print_warn_log_rank_0(
                f"[atat] Free Benchmark: For {self.params.api_name} "
                f"The compare for output type {type(perturbed_output)} is not supported"
            )

        threshold = self.get_threshold(origin_output.dtype)
        ratio = self.ratio_calculate(
            origin_output, perturbed_output, norm_type=NormType.ENDLESS_NORM
        )
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
                    max_fuzz_ratio if ratio is None else max(max_fuzz_ratio, ratio)
                )
                data_params.is_consistent = is_consistent and data_params.is_consistent
                if not is_consistent and data_params.grad_unequal_flag:
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
                        max_fuzz_ratio if ratio is None else max(max_fuzz_ratio, ratio)
                    )
                    data_params.is_consistent = (
                        is_consistent and data_params.is_consistent
                    )
                    if not is_consistent and data_params.grad_unequal_flag:
                        self.unequal_rows.append(
                            make_unequal_row(
                                data_params, self.params, ratio=ratio, index=index_
                            )
                        )
        except Exception as e:
            print_warn_log_rank_0(
                f"[atat] Free Benchmark: For {self.params.api_name}, "
                f"when campare the result exception raise {e}"
            )
        return npu_consistent, max_fuzz_ratio

    @abstractmethod
    def handle(self, data_params: DataParams) -> Any:
        pass

    def get_unequal_rows(self):
        return self.unequal_rows
