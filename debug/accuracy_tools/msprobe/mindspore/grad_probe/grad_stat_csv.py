from abc import ABC, abstractmethod
import hashlib

import mindspore
from mindspore import ops, Tensor
from msprobe.core.grad_probe.constant import GradConst


class CsvInput:
    def __init__(self, param_name, grad, bounds):
        self.param_name = param_name
        self.grad = grad
        self.bounds = bounds

class GradStatCsv:
    csv = {}

    @staticmethod
    def get_csv_header(level, csv_input):
        header = ["param_name"]
        for key in level["header"]:
            header.extend(GradStatCsv.csv[key].generate_csv_header(csv_input))
        return header

    @staticmethod
    def get_csv_line(level, csv_input):
        line = [csv_input.param_name]
        for key in level["header"]:
            line.extend(GradStatCsv.csv[key].generate_csv_content(csv_input))
        return line


def register_csv_item(key, cls=None):
    if cls is None:
        # 无参数时，返回装饰器函数
        return lambda cls: register_csv_item(key, cls)
    GradStatCsv.csv[key] = cls
    return cls


class CsvItem(ABC):
    @staticmethod
    @abstractmethod
    def generate_csv_header(csv_input):
        pass

    @staticmethod
    @abstractmethod
    def generate_csv_content(csv_input):
        pass


@register_csv_item(GradConst.MD5)
class CsvMd5(CsvItem):
    def generate_csv_header(csv_input):
        return ["MD5"]

    def generate_csv_content(csv_input):
        grad = csv_input.grad
        tensor_bytes = grad.float().numpy().tobytes()
        md5_hash = hashlib.md5(tensor_bytes)
        return [md5_hash.hexdigest()]


@register_csv_item(GradConst.DISTRIBUTION)
class CsvDistribution(CsvItem):
    def generate_csv_header(csv_input):
        bounds = csv_input.bounds
        intervals = []
        if bounds:
            intervals.append(f"(-inf, {bounds[0]}]")
            for i in range(1, len(bounds)):
                intervals.append(f"({bounds[i-1]}, {bounds[i]}]")
        if intervals:
            intervals.append(f"({bounds[-1]}, inf)")
        intervals.append("=0")
    
        return intervals

    def generate_csv_content(csv_input):
        grad = csv_input.grad
        bounds = csv_input.bounds
        if grad.dtype == mindspore.bfloat16:
            grad = grad.to(mindspore.float32)
        element_num = grad.numel()
        grad_equal_0_num = (grad == 0).sum().item()
        bucketsize_result = ops.bucketize(grad.float(), bounds)
        bucketsize_result = bucketsize_result.astype(mindspore.int8)
        interval_nums = [(bucketsize_result == i).sum().item() for i in range(len(bounds) + 1)]
        interval_nums.append(grad_equal_0_num)
        return_list = [x / element_num if element_num != 0 else 0 for x in interval_nums]
        return return_list


@register_csv_item(GradConst.MAX)
class CsvMax(CsvItem):
    def generate_csv_header(csv_input):
        return ["max"]

    def generate_csv_content(csv_input):
        grad = csv_input.grad
        return [ops.amax(grad).float().numpy().tolist()]


@register_csv_item(GradConst.MIN)
class CsvMin(CsvItem):
    def generate_csv_header(csv_input):
        return ["min"]

    def generate_csv_content(csv_input):
        grad = csv_input.grad
        return [ops.amin(grad).float().numpy().tolist()]


@register_csv_item(GradConst.NORM)
class CsvNorm(CsvItem):
    def generate_csv_header(csv_input):
        return ["norm"]

    def generate_csv_content(csv_input):
        grad = csv_input.grad
        return [ops.norm(grad).float().numpy().tolist()]


@register_csv_item(GradConst.SHAPE)
class CsvShape(CsvItem):
    def generate_csv_header(csv_input):
        return ["shape"]

    def generate_csv_content(csv_input):
        grad = csv_input.grad
        return [list(grad.shape)]