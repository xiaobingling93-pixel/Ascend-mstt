import os
import torch
import numpy as np

from grad_tool.common.constant import GradConst


class GradComparator:

    @staticmethod
    def compare(path1: str, path2: str, output_dir: str, framework="PyTorch"):
        if framework not in GradConst.FRAMEWORKS:
            raise RuntimeError(f"{framework} is not supported! Choose from {GradConst.FRAMEWORKS}.")
        if framework == GradConst.PYTORCH:
            from grad_tool.grad_pt.grad_comparator import PtGradComparator as grad_comparator
        else:
            from grad_tool.grad_ms.grad_comparator import MsGradComparator as grad_comparator
        grad_comparator.compare(path1, path2, output_dir)

    @staticmethod
    def compare_distributed(path1: str, path2: str, output_dir: str, framework="PyTorch"):
        if framework not in GradConst.FRAMEWORKS:
            raise RuntimeError(f"{framework} is not supported! Choose from {GradConst.FRAMEWORKS}.")
        if framework == GradConst.PYTORCH:
            from grad_tool.grad_pt.grad_comparator import PtGradComparator as grad_comparator
        else:
            from grad_tool.grad_ms.grad_comparator import MsGradComparator as grad_comparator
        grad_comparator.compare_distributed(path1, path2, output_dir)
