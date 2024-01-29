import numpy as np
import torch
from api_accuracy_checker.common.utils import Const, print_warn_log


detail_test_rows = [[
            "API Name", "Bench Dtype", "NPU Dtype", "Shape",
            "余弦相似度",
            "最大绝对误差",
            "双百指标",
            "双千指标",
            "双万指标",
            "错误率",
            "均方根误差",
            "误差均衡性",
            "相对误差最大值",
            "相对误差平均值",
            "Status",
            "Message"
        ]]


precision_configs = {
    torch.float16 : {
        'small_value' : [
            1e-3
        ],
        'small_value_atol' : [
            1e-5
        ]
    },
    torch.bfloat16: {
        'small_value' : [
            1e-3
        ],
        'small_value_atol' : [
            1e-5
        ]
    },
    torch.float32:{
        'small_value' : [
            1e-6
        ],
        'small_value_atol' : [
            1e-9
        ]
    }
}


class CompareConst:
    NAN = np.nan
    NA = "N/A"
    PASS = 'pass'
    WARNING = 'warning'
    ERROR = 'error'
    SKIP = 'SKIP'
    TRUE = 'TRUE'
    FALSE = 'FALSE'


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
        self.builtin = CompareConst.NA
        self.inf_or_nan = CompareConst.NA
        self.RMSE = CompareConst.NA
        self.EB = CompareConst.NA
        self.Max_rel_error = CompareConst.NA
        self.Mean_rel_error = CompareConst.NA

    def to_column_value(self, is_pass, message):
        return [self.bench_type, self.npu_type, self.shape, self.cosine_sim, self.max_abs_err, self.rel_err_hundredth,
                self.rel_err_thousandth, self.rel_err_ten_thousandth, self.error_rate, self.RMSE, self.EB, 
                self.Max_rel_error, self.Mean_rel_error, is_pass, message]


def check_dtype_comparable(x, y):
    if x.dtype in Const.FLOAT_TYPE:
        if y.dtype in Const.FLOAT_TYPE:
            return True 
        return False 
    if x.dtype in Const.BOOL_TYPE:
        if y.dtype in Const.BOOL_TYPE:
            return True 
        return False 
    if x.dtype in Const.INT_TYPE:
        if y.dtype in Const.INT_TYPE:
            return True 
        return False
    print_warn_log(f"Compare: Unexpected dtype {x.dtype}, {y.dtype}")
    return False
