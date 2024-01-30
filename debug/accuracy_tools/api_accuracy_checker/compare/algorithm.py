# 定义比对算法及比对标准
import torch
import numpy as np
from api_accuracy_checker.compare.compare_utils import CompareConst, check_dtype_comparable
from api_accuracy_checker.common.utils import Const


#cos
def cosine_sim(cpu_output, npu_output):
    msg = ""
    n_value = npu_output.reshape(-1)
    b_value = cpu_output.reshape(-1)
    cos = CompareConst.NA
    np.seterr(divide="ignore", invalid="ignore")
    if n_value.shape != b_value.shape:
        msg = f"Shape of npu and bench outputs don't match. NPU: {n_value.shape}, bench: {b_value.shape}."
        return -1, False, msg
    if len(n_value) == 1:
        msg = "All the data in npu dump data is scalar. Please refer to other compare algorithms."
        return cos, True, msg
    n_value_max = np.max(np.abs(n_value))
    b_value_max = np.max(np.abs(b_value))
    if n_value_max <= np.finfo(float).eps and b_value_max <= np.finfo(float).eps:
        return cos, True, msg
    elif n_value_max <= np.finfo(float).eps:
        msg = "All the data is zero in npu dump data."
        return CompareConst.NA, False, msg
    elif b_value_max <= np.finfo(float).eps:
        msg = "All the data is zero in bench dump data."
        return CompareConst.NA, False, msg
    else:
        n_value = n_value_max.astype(float) / n_value_max
        b_value = b_value_max.astype(float) / b_value_max
        cos = np.dot(n_value, b_value) / (np.linalg.norm(n_value) * np.linalg.norm(b_value))
        if np.isnan(cos):
            msg = "Dump data has NaN when comparing with Cosine Similarity."
        return cos, cos > 0.99, msg


#rmse
def get_rmse(abs_err, inf_nan_mask):
    masked_ae = np.where(inf_nan_mask, 0, abs_err)
    rmse = np.mean(np.square(masked_ae))
    inf_nan_cnt = np.sum(inf_nan_mask)
    rmse = rmse * (abs_err.size / (abs_err.size - inf_nan_cnt + 0.0001) + 0.0001)
    rmse = np.sqrt(rmse)
    return rmse


#误差均衡性
def get_error_balance(bench_data, device_data):
    larger_count = np.sum(np.greater(device_data - bench_data.astype(device_data.dtype), 0))
    smaller_count = np.sum(np.less(device_data - bench_data.astype(device_data.dtype), 0))
    total_count = bench_data.size
    error_balance = abs(larger_count - smaller_count) / total_count
    return error_balance


#小值域错误占比
def get_small_value_err_ratio(small_value_mask, abs_err_greater_mask):
    err_mask = np.logical_and(small_value_mask, abs_err_greater_mask)
    small_value_err_num = np.sum(err_mask)
    small_value_num = np.sum(small_value_mask)
    return 0 if small_value_num == 0 else small_value_err_num / small_value_num


def get_rel_err(abs_err, abs_bench_with_eps, small_value_mask, inf_nan_mask):
    rel_err_tmp = abs_err / abs_bench_with_eps
    rel_err_mask = np.logical_or(small_value_mask, inf_nan_mask)
    rel_err = np.where(rel_err_mask, -1, rel_err_tmp)
    return rel_err


def get_abs_err(bench_data, device_data):
    abs_err = np.abs(device_data - bench_data)
    return abs_err


def get_rel_err_origin(abs_err, b_value):
    rel_err_origin = np.abs(abs_err / b_value)
    return rel_err_origin


def get_max_abs_err(abs_err):
    max_abs_err = abs_err.max()
    bool_result = max_abs_err < 0.001
    return max_abs_err, bool_result


#相对误差最大值
def get_max_rel_err(rel_err):
    return np.max(rel_err)


#相对误差均值
def get_mean_rel_err(rel_err):
    return np.mean(rel_err)


def get_rel_err_ratio(rel_err, thresholding):
    if np.size(rel_err) == 0:
        ratio = 1
    else:
        ratio = np.divide(np.sum(rel_err < thresholding), np.size(rel_err))
    bool_result = ratio > (1 - thresholding)
    return ratio, bool_result
