import numpy as np
import torch

from compare.compare_utils import CompareConst, precision_configs
from common.utils import Const

FLOAT_EPSILON = np.finfo(float).eps
np.seterr(divide='ignore', invalid='ignore')  # ignore `invalid value encountered in true_divide` warning
print("FLOAT_EPSILON:", FLOAT_EPSILON)
NAN = 'NaN'

#cos
def cosine_sim(bench_output, device_output):
    msg = ""
    n_value = device_output.reshape(-1)
    b_value = bench_output.reshape(-1)
    cos = CompareConst.NA
    np.seterr(divide="ignore", invalid="ignore")
    if n_value.shape != b_value.shape:
        msg = f"Shape of device and bench outputs don't match. device: {n_value.shape}, bench: {b_value.shape}."
        return -1, False, msg
    if len(n_value) == 1:
        msg = "All the data in device dump data is scalar. Please refer to other compare algorithms."
        return cos, True, msg
    n_value_max = np.max(np.abs(n_value))
    b_value_max = np.max(np.abs(b_value))
    if n_value_max <= np.finfo(float).eps and b_value_max <= np.finfo(float).eps:
        return cos, True, msg
    elif n_value_max <= np.finfo(float).eps:
        msg = "All the data is zero in device dump data."
        return CompareConst.NA, False, msg
    elif b_value_max <= np.finfo(float).eps:
        msg = "All the data is zero in bench dump data."
        return CompareConst.NA, False, msg
    else:
        n_value = n_value.astype(float) / n_value_max
        b_value = b_value.astype(float) / b_value_max
        cos = np.dot(n_value, b_value) / (np.linalg.norm(n_value) * np.linalg.norm(b_value))
        if np.isnan(cos):
            msg = "Dump data has NaN when comparing with Cosine Similarity."
        cos = np.clip(cos, -1, 1)
        return cos, cos > 0.99, msg


#rmse
def get_rmse(abs_err, inf_nan_mask):
    masked_ae = np.where(inf_nan_mask, 0, abs_err)
    mse = np.mean(np.square(masked_ae))
    inf_nan_cnt = np.sum(inf_nan_mask)
    mse = mse * (abs_err.size / (abs_err.size - inf_nan_cnt + 0.0001) + 0.0001)
    rmse = np.sqrt(mse)
    return rmse


#误差均衡性
def get_error_balance(bench_data, device_data):
    larger_count = np.sum(np.greater(device_data - bench_data.astype(device_data.dtype), 0))
    smaller_count = np.sum(np.less(device_data - bench_data.astype(device_data.dtype), 0))
    total_count = bench_data.size
    error_balance = abs(larger_count - smaller_count) / total_count if total_count > 0 else 0
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


def get_finite_and_infinite_mask(bench_output, device_output):
    device_finite_mask = np.isfinite(device_output)
    bench_finite_mask = np.isfinite(bench_output.astype(device_output.dtype))
    both_finite_mask = np.logical_and(device_finite_mask, bench_finite_mask)
    inf_nan_mask = np.logical_not(both_finite_mask)
    return both_finite_mask, inf_nan_mask


def get_small_value_mask(abs_bench, both_finite_mask, small_value_threshold):
    small_value_mask = np.less_equal(abs_bench, small_value_threshold)
    small_value_mask = np.logical_and(small_value_mask, both_finite_mask)
    return small_value_mask


def get_msg_and_handle_value(b_value, n_value):
    if n_value.dtype in Const.FLOAT_TYPE:
        zero_mask = (n_value == 0)
        n_value[zero_mask] += np.finfo(n_value.dtype).eps
        b_value[zero_mask] += np.finfo(n_value.dtype).eps
    else:
        b_value, n_value = b_value.astype(float), n_value.astype(float)
        zero_mask = (n_value == 0)
        n_value[zero_mask] += np.finfo(float).eps
        b_value[zero_mask] += np.finfo(float).eps
    return b_value, n_value

 
def compare_bool_tensor(bench_output, npu_out):
    if npu_out.size == 0:
        return CompareConst.NAN, CompareConst.ERROR, "There is not npu calculation result."
    error_nums = (bench_output!= npu_out).sum()
    error_rate = float(error_nums / bench_output.size)
    result = CompareConst.PASS if error_rate == 0 else CompareConst.ERROR
    return error_rate, result, ""


def compare_float_tensor(cpu_output, npu_output, compare_column):
    npu_dtype = npu_output.dtype

    message = ""
    eps = np.finfo(cpu_output.dtype).eps
    abs_bench = np.abs(cpu_output)
    abs_bench_with_eps = abs_bench + eps
    abs_err = get_abs_err(cpu_output, npu_output)
    if npu_dtype in [np.float16, np.float32]:
        dtype_config = precision_configs.get(str(npu_dtype))
        both_finite_mask, inf_nan_mask = get_finite_and_infinite_mask(cpu_output, npu_output)
        small_value_mask = get_small_value_mask(abs_bench, both_finite_mask, dtype_config['small_value'][0])
        abs_err_greater_mask = np.greater(abs_err, dtype_config['small_value_atol'][0])
        compare_column.small_value_err_ratio = get_small_value_err_ratio(small_value_mask, abs_err_greater_mask)
        rel_err = get_rel_err(abs_err, abs_bench_with_eps, small_value_mask, inf_nan_mask)
        compare_column.RMSE = get_rmse(abs_err, np.logical_or(inf_nan_mask, small_value_mask))
        compare_column.EB = get_error_balance(cpu_output, npu_output)
        compare_column.Max_rel_error = get_max_rel_err(rel_err)
        compare_column.Mean_rel_error = get_mean_rel_err(rel_err)
    

    cos_res, cos_status, msg = cosine_sim(cpu_output, npu_output)
    compare_column.cosine_sim = cos_res
    message += msg + "\n"
    if not cos_status:
        return CompareConst.ERROR, compare_column, message

    max_abs_res, max_abs_status = get_max_abs_err(abs_err)
    compare_column.max_abs_err = max_abs_res
    if max_abs_status:
        return CompareConst.PASS, compare_column, message


    rel_err_orign = get_rel_err_origin(abs_err, abs_bench_with_eps)
    if npu_dtype in [np.float16]:
        hundred_res, hundred_status = get_rel_err_ratio(rel_err_orign, 0.01)
        compare_column.rel_err_hundredth = hundred_res
        if not hundred_status:
            return CompareConst.ERROR, compare_column, message
    thousand_res, thousand_status = get_rel_err_ratio(rel_err_orign, 0.001)
    compare_column.rel_err_thousandth = thousand_res
    if npu_dtype in [np.float16]:
        if thousand_status:
            return CompareConst.PASS, compare_column, message
        return CompareConst.WARNING, compare_column, message
    ten_thousand_res, ten_thousand_status = get_rel_err_ratio(rel_err_orign, 0.0001)
    compare_column.rel_err_ten_thousandth = ten_thousand_res
    if npu_dtype in [np.float32, np.float64]:
        if not thousand_status:
            return CompareConst.ERROR, compare_column, message
        if not ten_thousand_status:
            return CompareConst.WARNING, compare_column, message
    return CompareConst.PASS, compare_column, message
    

class CompareColumn:
    def __init__(self):
        self.bench_type = CompareConst.NA
        self.npu_type = CompareConst.NA
        self.shape = CompareConst.NA
        self.cosine_sim = CompareConst.NA
        self.max_abs_err = CompareConst.NA
        self.rel_err_hundredth = CompareConst.NA
        self.rel_err_thousandth = CompareConst.NA
        self.rel_err_ten_thousandth = CompareConst.NA
        self.error_rate = CompareConst.NA
        self.EB = CompareConst.NA
        self.RMSE = CompareConst.NA
        self.small_value_err_ratio = CompareConst.NA
        self.Max_rel_error = CompareConst.NA
        self.Mean_rel_error = CompareConst.NA

    def to_column_value(self, is_pass, message):
        return [self.bench_type, self.npu_type, self.shape, self.cosine_sim, self.max_abs_err, self.rel_err_hundredth,
                self.rel_err_thousandth, self.rel_err_ten_thousandth, self.error_rate, self.EB, self.RMSE, 
                self.small_value_err_ratio, self.Max_rel_error, self.Mean_rel_error, is_pass, message]




