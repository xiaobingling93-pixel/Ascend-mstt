import hashlib
import torch
from grad_tool.grad_pt.level_adapter import Level


class GradExtremeOps:
    @staticmethod
    def tensor_max(tensor):
        return torch._C._VariableFunctionsClass.max(tensor).cpu().detach().float().numpy().tolist()

    @staticmethod
    def tensor_min(tensor):
        return torch._C._VariableFunctionsClass.min(tensor).cpu().detach().float().numpy().tolist()

    @staticmethod
    def tensor_norm(tensor):
        return torch._C._VariableFunctionsClass.norm(tensor).cpu().detach().float().numpy().tolist()


class GradExtremes:
    extremes = {
        "max": GradExtremeOps.tensor_max,
        "min": GradExtremeOps.tensor_min,
        "norm": GradExtremeOps.tensor_norm
    }


class GradStatOps:
    @staticmethod
    def md5_header(**kwargs):
        level: Level = kwargs.get("level")
        return level.MD5_header()

    @staticmethod
    def intervals_header(**kwargs):
        level: Level = kwargs.get("level")
        bounds = kwargs.get("bounds")
        return level.intervals_header(bounds)

    @staticmethod
    def extremes_header(**kwargs):
        return GradExtremes.extremes.keys()

    @staticmethod
    def shape_header(**kwargs):
        return ["shape"]

    @staticmethod
    def md5_content(**kwargs):
        grad = kwargs.get("grad")
        level: Level = kwargs.get("level")
        return level.MD5_content(grad)

    @staticmethod
    def count_distribution(**kwargs):
        level: Level = kwargs.get("level")
        grad = kwargs.get("grad")
        bounds = kwargs.get("bounds")
        return level.count_grad_distribution(grad, bounds)

    @staticmethod
    def extremes_content(**kwargs):
        grad = kwargs.get("grad")
        return [f(grad) for f in GradExtremes.extremes.values()]

    @staticmethod
    def shape_content(**kwargs):
        grad = kwargs.get("grad")
        return [list(grad.shape)]


class GradStatCsv:
    CSV = {
            "MD5": {
                "header": GradStatOps.md5_header,
                "content": GradStatOps.md5_content
            },
            "distribution": {
                "header": GradStatOps.intervals_header,
                "content": GradStatOps.count_distribution
            },
            "extremes": {
                "header": GradStatOps.extremes_header,
                "content": GradStatOps.extremes_content
            },
            "shape": {
                "header": GradStatOps.shape_header,
                "content": GradStatOps.shape_content
            },
        }

    @staticmethod
    def generate_csv_header(**kwargs):
        header = ["param_name"]
        for func in GradStatCsv.CSV.values():
            header.extend(func["header"](**kwargs))
        return header

    @staticmethod
    def generate_csv_line(**kwargs):
        line = [kwargs.get("param_name")]
        for func in GradStatCsv.CSV.values():
            line.extend(func["content"](**kwargs))
        return line
