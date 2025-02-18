# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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

from mindspore import mint, ops, _no_grad
from mindspore import Tensor
from mindspore import dtype as mstype


@_no_grad()
def square_sum(x: Tensor):
    return (x * x).sum()


@_no_grad()
def get_min(x: Tensor):
    return mint.min(x)


@_no_grad()
def get_mean(x: Tensor):
    return mint.mean(x.astype(mstype.float32))


@_no_grad()
def get_norm(x: Tensor):
    norm_func = mint.norm if hasattr(mint, "norm") else ops.norm
    return norm_func(x.astype(mstype.float32))


@_no_grad()
def get_max(x: Tensor):
    return mint.max(x)


@_no_grad()
def get_zeros(x: Tensor, eps: float):
    return mint.sum(mint.abs(x) < eps) / x.numel()


@_no_grad()
def get_nans(t):
    return ops.isnan(t.astype(mstype.float32)).sum()


FUNC_MAP = {"min"  : get_min,
            "max"  : get_max,
            "mean" : get_mean,
            "norm" : get_norm,
            "nans" : get_nans,
            "zeros": get_zeros
           }