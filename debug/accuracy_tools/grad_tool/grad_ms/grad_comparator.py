import os
import torch
import numpy as np

from grad_tool.common.base_comparator import BaseComparator
from grad_tool.common.utils import check_file_or_directory_path


class MsGradComparator(BaseComparator):

    @classmethod
    def _load_grad_files(cls, grad_file1: str, grad_file2: str):
        grad1_suffix = grad_file1.split(".")[-1]
        grad2_suffix = grad_file2.split(".")[-1]
        check_file_or_directory_path(grad_file1)
        grad1 = torch.load(grad_file1).numpy() if grad1_suffix == "pt" else np.load(grad_file1)
        check_file_or_directory_path(grad_file2)
        grad2 = torch.load(grad_file2).numpy() if grad2_suffix == "pt" else np.load(grad_file2)

        if grad1.shape != grad2.shape:
            raise RuntimeError(f"numpy shape is not equal: {grad_file1}, {grad_file2}")
        if grad1.dtype != bool:
            raise TypeError(f"numpy type is not bool: {grad_file1}")
        if grad2.dtype != bool:
            raise TypeError(f"numpy type is not bool: {grad_file2}")
        return grad1, grad2
