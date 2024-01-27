# 定义比对算法及比对标准
import torch
import numpy as np
from api_accuracy_checker.compare.compare_utils import CompareConst, check_dtype_comparable
from api_accuracy_checker.common.utils import Const


def get_rel_err1(abs_err, b_value):
    rel_err = np.abs(abs_err / b_value)
    return rel_err


def get_max_abs_err(abs_err):
    max_abs_err = abs_err.max()
    bool_result = max_abs_err < 0.001
    return max_abs_err, bool_result


def get_rel_err_ratio(rel_err, thresholding):
    if np.size(rel_err) == 0:
        ratio = 1
    else:
        ratio = np.divide(np.sum(rel_err < thresholding), np.size(rel_err))
    bool_result = ratio > (1 - thresholding)
    return ratio, bool_result


def max_rel_err_standard(max_rel_errs):
    bool_result = np.array(max_rel_errs) < 0.001
    return np.all(bool_result), bool_result





#误差均衡性
def get_error_balance(bench_data, device_data):
    larger_count = np.sum(np.greater(device_data - bench_data.astype(device_data.dtype), 0))
    smaller_count = np.sum(np.less(device_data - bench_data.astype(device_data.dtype), 0))
    total_count = bench_data.size
    error_balance = abs(larger_count - smaller_count) / total_count
    return error_balance

def get_big_value(bench_data, device_data):
    small_value = 0.001
    bench_big_value = bench_data.copy()
    bench_big_value[bench_data < small_value] = 1
    device_big_value = device_data.copy()
    device_big_value[bench_data < small_value] = 1
    return bench_big_value, device_big_value

def get_small_value(bench_data, device_data):
    small_value = 0.001
    bench_small_value = bench_data.copy()
    bench_small_value[bench_data > small_value] = 1
    device_small_value = device_data.copy()
    device_small_value[bench_data > small_value] = 1
    return bench_small_value, device_small_value

def get_rel_err(abs_err, abs_bench_with_eps, small_value_mask, inf_nan_mask):
    rel_err_tmp = abs_err / abs_bench_with_eps
    rel_err_mask = np.logical_or(small_value_mask, inf_nan_mask)
    rel_err = np.where(rel_err_mask, -1, rel_err_tmp)
    return rel_err

def get_abs_err(bench_data, device_data):
    abs_err = np.abs(device_data - bench_data)
    return abs_err

#相对误差最大值
def get_max_rel_err(rel_err):
    return np.max(rel_err)

#相对误差均值
def get_mean_rel_err(rel_err):
    return np.mean(rel_err)

#绝对阈值法
def get_absolute_threshold(rel_err, abs_err):
    rtol = 0
    etol = 0
    small_value = 0.001
    small_value_atol = 1e-9
    if np.size(rel_err) == 0:
        rel_err_num = 0
    else:
        rel_err_num = np.sum(rel_err > rtol)
    if np.size(abs_err) == 0:
        abs_err_num = 0
    else:
        abs_err_num = np.sum(abs_err > small_value_atol)
    return rel_err_num + abs_err_num

#rmse
def get_rmse(abs_err, inf_nan_mask):
    masked_ae = np.where(inf_nan_mask, 0, abs_err)
    rmse = np.mean(np.square(masked_ae))
    inf_nan_cnt = np.sum(inf_nan_mask)
    rmse = rmse * (abs_err.size / (abs_err.size - inf_nan_cnt + 0.0001) + 0.0001)
    rmse = np.sqrt(rmse)
    return rmse

#小值域
def get_small_value_error(small_value_mask, abs_err_greater_mask):
    err_mask = np.logical_and(small_value_mask, abs_err_greater_mask)
    small_value_err_num = np.sum(err_mask)
    return small_value_err_num

def get_distribution_ratio(rel_err, start, end):

    return np.sum((rel_err>start)&(rel_err<end))

def check_inf_nan_value(bench_data, device_data, inf_nan_mask, abs_bench_with_eps):
    golden_same_dtype = bench_data.astype(device_data.dtype)
    a_min = np.finfo(device_data.dtype).min
    a_max = np.finfo(device_data.dtype).max
    golden_clip = np.clip(golden_same_dtype, a_min, a_max)
    device_clip = np.clip(device_data, a_min, a_max)
    clipped_abs_ae = np.abs(device_clip - golden_clip)
    clipped_re = clipped_abs_ae / abs_bench_with_eps
    pass_mask = np.less_equal(clipped_re, standard.rtol)
    both_nan_mask = np.logical_and(np.isnan(device_data), np.isnan(golden_clip))
    pass_mask = np.logical_or(pass_mask, both_nan_mask)
    not_pass_mask = np.logical_not(pass_mask)
    not_pass_mask = np.logical_and(not_pass_mask, inf_nan_mask)

    inf_nan_err_cnt = np.sum(not_pass_mask)
