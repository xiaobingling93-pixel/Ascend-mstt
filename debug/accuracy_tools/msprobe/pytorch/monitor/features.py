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

import torch
from torch.autograd.functional import jacobian
from msprobe.pytorch.common.log import logger


@torch.no_grad()
def square_sum(x: torch.tensor):
    return (x * x).sum()


@torch.no_grad()
def get_min(x: torch.tensor):
    return torch.min(x)


@torch.no_grad()
def get_mean(x: torch.tensor):
    return torch.mean(x.to(torch.float64))


@torch.no_grad()
def get_norm(x: torch.tensor):
    return torch.norm(x.to(torch.float64), p=2)


@torch.no_grad()
def get_max(x: torch.tensor):
    return torch.max(x)


@torch.no_grad()
def get_zeros(x: torch.tensor, eps: float):
    return torch.sum(torch.abs(x) < eps) / x.numel()


@torch.no_grad()
def get_sign_matches(x: torch.tensor, y: torch.tensor):
    xs = x.sign()
    ys = y.sign()
    try:
        same_direction_ratio = ((xs * ys).sum() / ys.numel() + 1) / 2
    except RuntimeError as e:
        logger.info(f"RuntimeError: {e}")
        same_direction_ratio = torch.tensor(0.)
    return same_direction_ratio


@torch.no_grad()
def eff_rank(param: torch.tensor, threshold=1e-10):
    U, S, Vh = torch.linalg.svd(param.float())
    rank = torch.sum(S > threshold)
    return rank


# modular neural tangent kernel
@torch.no_grad()
def mNTK(module: torch.nn.Module, x: torch.tensor):
    J_theta_l = jacobian(module, x)
    mntk = torch.matmul(J_theta_l, J_theta_l.t())
    return mntk


@torch.no_grad()
def power_iteration(a, num_iterations):
    b = torch.randn(a.size(1), 1)
    for _ in range(num_iterations):
        b = torch.matmul(a, b)
        b_norm = torch.norm(b)
        b = b / b_norm if b_norm != 0 else 0
    eigval = torch.matmul(torch.matmul(b.t(), a), b)
    return eigval


@torch.no_grad()
def lambda_max_subsample(module: torch.nn.Module, x: torch.tensor, num_iterations=100, subsample_size=None):
    mntk = mNTK(module, x)
    if subsample_size is None:
        subsample_size = min(mntk.size(0), mntk.size(1))
    idx = torch.randperm(mntk.size(0))[:subsample_size]
    subsampled = mntk[idx, :]
    subsampled = subsampled[:, idx]
    eigval = power_iteration(subsampled, num_iterations)
    return eigval


@torch.no_grad()
def cal_histc(tensor_cal, bins_total, min_val, max_val):
    return torch.histc(tensor_cal, bins=bins_total, min=min_val, max=max_val)


@torch.no_grad()
def get_nans(t):
    return torch.isnan(t).sum()
