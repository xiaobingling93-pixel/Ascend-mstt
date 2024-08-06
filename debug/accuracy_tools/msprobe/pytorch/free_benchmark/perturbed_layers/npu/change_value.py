import torch
from msprobe.pytorch.free_benchmark import logger
from msprobe.pytorch.free_benchmark.common.enums import PerturbationMode
from msprobe.pytorch.free_benchmark.common.params import DataParams
from msprobe.pytorch.free_benchmark.common.utils import TorchC
from msprobe.pytorch.free_benchmark.perturbed_layers.npu.npu_base_layser import (
    NpuBaseLayer,
)


class ChangeValueLayer(NpuBaseLayer):
    def __init__(self, api_name):
        super().__init__(api_name)
        self.head: int = 0
        self.tail: int = -1

    def change_value(self, tensor_obj):
        """
        交换张量首尾
        """
        if isinstance(tensor_obj, torch.Tensor) and self.pre_check(tensor_obj):
            new_tensor = TorchC.clone(tensor_obj)
            if new_tensor.ndim == 1:
                temp_first = TorchC.clone(new_tensor[self.head])
                temp_last = TorchC.clone(new_tensor[self.tail])
                new_tensor[self.head] = temp_last
                new_tensor[self.tail] = temp_first
            else:
                temp_first = TorchC.clone(new_tensor[self.head][self.head])
                temp_last = TorchC.clone(new_tensor[self.tail][self.tail])
                new_tensor[self.head][self.head] = temp_last
                new_tensor[self.tail][self.tail] = temp_first

            self.is_added = True
            return new_tensor
        if isinstance(tensor_obj, dict):
            return {key: self.change_value(value) for key, value in tensor_obj.items()}
        if isinstance(tensor_obj, (tuple, list)):
            return type(tensor_obj)([self.change_value(value) for value in tensor_obj])
        return tensor_obj

    def handle(self, params: DataParams):
        """
        对输入添加扰动并返回
        """
        logger.info_on_rank_0(
            f"[msprobe] Free benchmark: Perturbation is "
            f"{PerturbationMode.CHANGE_VALUE} of {self.api_name}."
        )
        params.perturbed_value = self.change_value(params.args[params.valid_input_index])
        return self.perturbed_result(params)

    def _check_details(self, tensor_obj):
        """
        判断是否需要添加扰动,  首尾值交换
        """
        if tensor_obj.size(0) < 2:
            logger.info_on_rank_0(
                f"[msprobe] Free Benchmark: For {self.api_name}, "
                f"size 0 must greater than 1. Cancel change value."
            )
            return False
        return True
