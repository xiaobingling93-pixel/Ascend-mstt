# 定义比对算法及比对标准
import torch
import numpy as np
from api_accuracy_checker.compare.compare_utils import CompareConst, check_dtype_comparable
from api_accuracy_checker.common.utils import Const


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

    def to_column_value(self, is_pass, message):
        return [self.bench_type, self.npu_type, self.shape, self.cosine_sim, self.max_abs_err, self.rel_err_hundredth,
                self.rel_err_thousandth, self.rel_err_ten_thousandth, self.error_rate, is_pass, message]


def compare_torch_tensor(cpu_output, npu_output, compare_column):
    cpu_shape = cpu_output.shape
    npu_shape = npu_output.shape
    npu_dtype = npu_output.dtype
    if npu_dtype == torch.bfloat16:
        cpu_output = cpu_output.to(torch.float32)
        npu_output = npu_output.to(torch.float32)
    cpu_output = cpu_output.numpy()
    npu_output = npu_output.cpu().numpy()
    if cpu_shape != npu_shape:
        return CompareConst.ERROR, compare_column, f"The shape of bench{str(cpu_shape)} " \
                                                   f"and npu{str(npu_shape)} not equal."
    if not check_dtype_comparable(cpu_output, npu_output):
        return CompareConst.ERROR, compare_column, f"Bench out dtype is {cpu_output.dtype} but " \
                                                   f"npu output dtype is {npu_output.dtype}, cannot compare."
    message = ""
    if cpu_output.dtype in [bool, np.uint8, np.int8, np.int16, np.uint16, np.uint32, np.int32, np.int64, np.uint64]:
        message += f"Compare algorithm cosine_sim is not supported for {cpu_output.dtype} data. " \
                   f"Only judged by Error Rate."
        err_rate, status, msg = compare_bool_tensor(cpu_output, npu_output)
        message += msg + "\n"
        compare_column.error_rate = err_rate
        return status, compare_column, message
    # cos
    cos_res, cos_status, msg = cosine_sim(cpu_output, npu_output)
    compare_column.cosine_sim = cos_res
    message += msg + "\n"
    if not cos_status:
        return CompareConst.ERROR, compare_column, message
    # abs err
    b_value, n_value = get_msg_and_handle_value(cpu_output, npu_output)
    abs_err = np.abs(b_value - n_value)
    max_abs_res, max_abs_status = get_max_abs_err(abs_err)
    compare_column.max_abs_err = max_abs_res
    if max_abs_status:
        return CompareConst.PASS, compare_column, message
    # rel err
    rel_err = get_rel_err(abs_err, b_value)
    if npu_dtype in [torch.float16, torch.bfloat16]:
        hundred_res, hundred_status = get_rel_err_ratio(rel_err, 0.01)
        compare_column.rel_err_hundredth = hundred_res
        if not hundred_status:
            return CompareConst.ERROR, compare_column, message
    thousand_res, thousand_status = get_rel_err_ratio(rel_err, 0.001)
    compare_column.rel_err_thousandth = thousand_res
    if npu_dtype in [torch.float16, torch.bfloat16]:
        if thousand_status:
            return CompareConst.PASS, compare_column, message
        return CompareConst.WARNING, compare_column, message
    ten_thousand_res, ten_thousand_status = get_rel_err_ratio(rel_err, 0.0001)
    compare_column.rel_err_ten_thousandth = ten_thousand_res
    if npu_dtype in [torch.float32, torch.float64]:
        if not thousand_status:
            return CompareConst.ERROR, compare_column, message
        if not ten_thousand_status:
            return CompareConst.WARNING, compare_column, message
    return CompareConst.PASS, compare_column, message


def compare_bool_tensor(cpu_output, npu_output):
    error_nums = (cpu_output != npu_output).sum()
    if cpu_output.size == 0:
        return CompareConst.NAN, CompareConst.ERROR, "There is not cpu calculation result."
    error_rate = float(error_nums / cpu_output.size)
    result = CompareConst.PASS if error_rate == 0 else CompareConst.ERROR
    return error_rate, result, ""


def get_msg_and_handle_value(b_value, n_value):
    if n_value.dtype in Const.FLOAT_TYPE:
        zero_mask = (n_value == 0)
        # 给0的地方加上eps防止除0
        n_value[zero_mask] += np.finfo(n_value.dtype).eps
        # 根据n_value为0的位置给n_value也加上eps，否则两者都是0的情况下相对误差会是1
        b_value[zero_mask] += np.finfo(n_value.dtype).eps
    else:
        # int type + float eps 会报错，所以这里要强转
        b_value, n_value = b_value.astype(float), n_value.astype(float)
        zero_mask = (n_value == 0)
        n_value[zero_mask] += np.finfo(float).eps
        b_value[zero_mask] += np.finfo(float).eps
    return b_value, n_value


def get_rel_err(abs_err, b_value):
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


def cosine_standard(compare_result):
    bool_result = np.array(compare_result) > 0.99
    return np.all(bool_result), bool_result


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


def compare_uint8_data(b_value, n_value):
    if (b_value == n_value).all():
        return 1, True
    else:
        return 0, False


def compare_builtin_type(bench_out, npu_out, compare_column):
    if not isinstance(bench_out, (bool, int, float, str)):
        return CompareConst.PASS, compare_column, ""
    if bench_out != npu_out:
        return CompareConst.ERROR, compare_column, ""
    compare_column.error_rate = 0
    return CompareConst.PASS, compare_column, ""


def flatten_compare_result(result):
    flatten_result = []
    for result_i in result:
        if isinstance(result_i, list):
            flatten_result += flatten_compare_result(result_i)
        else:
            flatten_result.append(result_i)
    return flatten_result


def compare_core(bench_out, npu_out):
    compare_column = CompareColumn()
    if not isinstance(bench_out, type(npu_out)):
        return CompareConst.ERROR, compare_column, "bench and npu output type is different."
    if isinstance(bench_out, (list, tuple)):
        status, compare_result, message = [], [], []
        if len(bench_out) != len(npu_out):
            return CompareConst.ERROR, compare_column, "bench and npu output structure is different."
        for b_out_i, n_out_i in zip(bench_out, npu_out):
            status_i, compare_result_i, message_i = compare_core(b_out_i, n_out_i)
            status.append(status_i)
            compare_result.append(compare_result_i)
            message.append(message_i)
    elif isinstance(bench_out, dict):
        b_keys, n_keys = set(bench_out.keys()), set(npu_out.keys())
        if b_keys != n_keys:
            return CompareConst.ERROR, compare_column, "bench and npu output dict keys are different."
        else:
            status, compare_result, message = compare_core(list(bench_out.values()), list(npu_out.values()))
    elif isinstance(bench_out, torch.Tensor):
        copy_bench_out = bench_out.detach().clone()
        copy_npu_out = npu_out.detach().clone()
        compare_column.bench_type = str(copy_bench_out.dtype)
        compare_column.npu_type = str(copy_npu_out.dtype)
        compare_column.shape = tuple(npu_out.shape)
        status, compare_result, message = compare_torch_tensor(copy_bench_out, copy_npu_out,
                                                               compare_column)
    elif isinstance(bench_out, (bool, int, float, str)):
        compare_column.bench_dtype = str(type(bench_out))
        compare_column.npu_dtype = str(type(npu_out))
        compare_column.shape = str(type(npu_out))
        status, compare_result, message = compare_builtin_type(bench_out, npu_out, compare_column)
    elif bench_out is None:
        return CompareConst.PASS, compare_column, "Output is None."
    else:
        return CompareConst.PASS, compare_column, "Unexpected output type in compare_core: {}".format(type(bench_out))

    return status, compare_result, message
