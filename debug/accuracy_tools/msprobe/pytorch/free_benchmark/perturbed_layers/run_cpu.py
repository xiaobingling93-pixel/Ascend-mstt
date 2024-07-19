import torch
from atat.pytorch.free_benchmark import logger
from atat.pytorch.free_benchmark.common.params import DataParams
from atat.pytorch.free_benchmark.common.utils import Tools
from atat.pytorch.free_benchmark.common.enums import DeviceType
from atat.pytorch.free_benchmark.perturbed_layers.base_layer import BaseLayer


class CpuLayer(BaseLayer):

    def handle(self, params: DataParams) -> torch.Any:

        logger.info_on_rank_0(
            f"[atat] Free benchmark: Perturbation is to_cpu of {self.api_name}."
        )
        new_args = Tools.convert_device_and_dtype(params.args, DeviceType.CPU, change_dtype=True)
        new_kwargs = Tools.convert_device_and_dtype(params.kwargs, DeviceType.CPU, change_dtype=True)
        params.perturbed_result = params.origin_func(*new_args, **new_kwargs)
        return params.perturbed_result
