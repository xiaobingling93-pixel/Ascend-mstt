import torch
from torch.autograd.functional import jacobian


@torch.no_grad()
def square_sum(x: torch.tensor):
    return (x * x).sum()


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
def power_iteration(A, num_iterations):
    b = torch.randn(A.size(1), 1)
    for _ in range(num_iterations):
        b = torch.matmul(A, b)
        b_norm = torch.norm(b)
        b = b / b_norm if b_norm != 0 else 0
    eigval = torch.matmul(torch.matmul(b.t(), A), b)
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


