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

from abc import ABC, abstractmethod
from collections import namedtuple
import hashlib
from functools import wraps
import zlib
import torch
from msprobe.core.grad_probe.constant import GradConst

CsvHeaderInput = namedtuple("CsvHeaderInput", ["bounds"])
CsvContentInput = namedtuple("CsvContentInput", ["grad", "bounds"])


class GradStatCsv:
    csv = {}

    @staticmethod
    def generate_csv_header(level, bounds):
        header = ["param_name"]
        for key in level["header"]:
            csv_header_input = CsvHeaderInput(bounds=bounds)
            header.extend(GradStatCsv.csv[key].generate_csv_header(csv_header_input))
        return header

    @staticmethod
    def generate_csv_line(param_name, level, grad, bounds):
        line = [param_name]
        for key in level["header"]:
            csv_content_input = CsvContentInput(grad=grad, bounds=bounds)
            line.extend(GradStatCsv.csv[key].generate_csv_content(csv_content_input))
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
    def generate_csv_header(csv_header_input):
        pass

    @staticmethod
    @abstractmethod
    def generate_csv_content(csv_content_input):
        pass


@register_csv_item(GradConst.MD5)
class CsvMd5(CsvItem):
    @staticmethod
    def generate_csv_header(csv_header_input):
        return ["MD5"]

    @staticmethod
    def generate_csv_content(csv_content_input):
        grad = csv_content_input.grad
        tensor_bytes = grad.cpu().detach().float().numpy().tobytes()
        md5_hash = f"{zlib.crc32(tensor_bytes):08x}"
        return [md5_hash]


@register_csv_item(GradConst.DISTRIBUTION)
class CsvDistribution(CsvItem):
    @staticmethod
    def generate_csv_header(csv_header_input):
        bounds = csv_header_input.bounds
        intervals = []
        if bounds:
            intervals.append(f"(-inf, {bounds[0]}]")
            for i in range(1, len(bounds)):
                intervals.append(f"({bounds[i-1]}, {bounds[i]}]")
        if intervals:
            intervals.append(f"({bounds[-1]}, inf)")
        intervals.append("=0")
    
        return intervals

    @staticmethod
    def generate_csv_content(csv_content_input):
        grad = csv_content_input.grad
        bounds = csv_content_input.bounds
        grad = grad.cpu().detach()
        if grad.dtype == torch.bfloat16:
            grad = grad.to(torch.float32)
        element_num = grad.numel()
        grad_equal_0_num = (grad == 0).sum().item()
        bound = torch.Tensor(bounds)
        bucketsize_result = torch.bucketize(grad, bound)
        interval_nums = [(bucketsize_result == i).sum().item() for i in range(len(bound) + 1)]
        interval_nums.append(grad_equal_0_num)
        return_list = [x / element_num if element_num != 0 else 0 for x in interval_nums]
        return return_list


@register_csv_item(GradConst.MAX)
class CsvMax(CsvItem):
    @staticmethod
    def generate_csv_header(csv_header_input):
        return ["max"]

    @staticmethod
    def generate_csv_content(csv_content_input):
        grad = csv_content_input.grad
        return [torch.max(grad).cpu().detach().float().numpy().tolist()]


@register_csv_item(GradConst.MIN)
class CsvMin(CsvItem):
    @staticmethod
    def generate_csv_header(csv_header_input):
        return ["min"]

    @staticmethod
    def generate_csv_content(csv_content_input):
        grad = csv_content_input.grad
        return [torch.min(grad).cpu().detach().float().numpy().tolist()]


@register_csv_item(GradConst.NORM)
class CsvNorm(CsvItem):
    @staticmethod
    def generate_csv_header(csv_header_input):
        return ["norm"]

    @staticmethod
    def generate_csv_content(csv_content_input):
        grad = csv_content_input.grad
        return [torch.norm(grad).cpu().detach().float().numpy().tolist()]
    

@register_csv_item(GradConst.SHAPE)
class CsvShape(CsvItem):
    @staticmethod
    def generate_csv_header(csv_header_input):
        return ["shape"]

    @staticmethod
    def generate_csv_content(csv_content_input):
        grad = csv_content_input.grad
        return [list(grad.shape)]