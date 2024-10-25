from typing import Any

from msprobe.core.common.exceptions import FreeBenchmarkException
from msprobe.pytorch.free_benchmark import logger
from msprobe.pytorch.free_benchmark.common.params import DataParams
from msprobe.pytorch.free_benchmark.common.utils import Tools
from msprobe.pytorch.free_benchmark.result_handlers.base_handler import FuzzHandler


class FixHandler(FuzzHandler):

    def get_threshold(self, dtype):
        return self._get_default_threshold(dtype)

    def handle(self, data_params: DataParams) -> Any:
        try:
            return Tools.convert_fuzz_output_to_origin(
                data_params.original_result, data_params.perturbed_result
            )
        except KeyError as e:
            logger.warning(
                f"[msprobe] Free Benchmark: For {self.params.api_name} "
                f"Fix output failed because of KeyError: {e}"
            )
        except IndexError as e:
            logger.warning(
                f"[msprobe] Free Benchmark: For {self.params.api_name} "
                f"Fix output failed because of IndexError: {e}"
            )
        except FreeBenchmarkException as e:
            logger.warning(
                f"[msprobe] Free Benchmark: For {self.params.api_name} "
                f"Fix output failed because of unsupported type: {e}"
            )
        except Exception as e:
            logger.warning(
                f"[msprobe] Free Benchmark: For {self.params.api_name} "
                f"Fix output failed because of unexcepted: {e}"
            )
        return data_params.original_result