from typing import Any

from msprobe.pytorch.free_benchmark.common.params import DataParams
from msprobe.pytorch.free_benchmark.common.utils import Tools
from msprobe.pytorch.free_benchmark.result_handlers.base_handler import FuzzHandler
from msprobe.pytorch.free_benchmark import logger


class FixHandler(FuzzHandler):

    def get_threshold(self, dtype):
        return self._get_default_threshold(dtype)

    def handle(self, data_params: DataParams) -> Any:
        try:
            return Tools.convert_fuzz_output_to_origin(
                data_params.original_result, data_params.perturbed_result
            )
        except Exception as e:
            logger.warning_on_rank_0(
                f"[msprobe] Free Benchmark: For {self.params.api_name} "
                f"Fix output failed. "
            )
        return data_params.original_result