import numpy as np
import torch
from api_accuracy_checker.common.utils import Const, print_warn_log



detail_test_rows = [[
            "API Name", "Bench Dtype", "NPU Dtype", "Shape",
            "Dropout",
            "Builtin",
            "Inf/Nan",
            "jueduiyuzhi",
            "baiwanfenzhiyi",
            "shiwanfenzhiyi",
            "wanfenzhiyi",
            "wanfenzhiwu",
            "qianfenzhiyi",
            "qianfenzhier",
            "qianfenzhisi",
            "qianfenzhiwu",
            "baifenzhiyi",
            "wuqiong",
            "xiaozhiyu",
            "RMSE",
            "EB",
            "Max_rel_error",
            "Mean_rel_error"
        ]]


precision_configs = {
    torch.float16 : {
        'error_distribution' : [
            ('qianfenzhiyi', 0, 1e-3),
            ('qianfenzhier', 1e-3, 2e-3),
            ('qianfenzhiwu', 2e-3, 5e-3),
            ('baifenzhiyi', 5e-3, 1e-2),
            ('wuqiong', 1e-2, np.inf)
        ],
        'small_value' : [
            1e-3
        ],
        'small_value_atol' : [
            1e-5
        ]
    },
    torch.bfloat16: {
        'error_distribution' : [
            ('qianfenzhisi', 0, 4e-3),
            ('wuqiong', 4e-3, np.inf)
        ],
        'small_value' : [
            1e-3
        ],
        'small_value_atol' : [
            1e-5
        ]
    },
    torch.float32:{
        'error_distribution' : [
            ('baiwanfenzhiyi', 0, 1e-6),
            ('shiwanfenzhiyi', 1e-6, 1e-5),
            ('wanfenzhiyi', 1e-5, 1e-4),
            ('wanfenzhiwu', 1e-4, 5e-4),
            ('wuqiong', 5e-4, np.inf)
        ],
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
        self.dropout = CompareConst.NA
        self.builtin = CompareConst.NA
        self.inf_or_nan = CompareConst.NA
        self.jueduiyuzhi = CompareConst.NA
        self.baiwanfenzhiyi = CompareConst.NA
        self.shiwanfenzhiyi = CompareConst.NA
        self.wanfenzhiyi = CompareConst.NA
        self.wanfenzhiwu = CompareConst.NA
        self.qianfenzhiyi = CompareConst.NA
        self.qianfenzhier = CompareConst.NA
        self.qianfenzhisi = CompareConst.NA
        self.qianfenzhiwu = CompareConst.NA
        self.baifenzhiyi = CompareConst.NA
        self.wuqiong = CompareConst.NA
        self.xiaozhiyu = CompareConst.NA
        self.RMSE = CompareConst.NA
        self.EB = CompareConst.NA
        self.Max_rel_error = CompareConst.NA
        self.Mean_rel_error = CompareConst.NA

    def to_column_value(self):
        return [self.bench_type, self.npu_type, self.shape, self.dropout, self.builtin, self.inf_or_nan,
                self.jueduiyuzhi, self.baiwanfenzhiyi, self.shiwanfenzhiyi, self.wanfenzhiyi, self.wanfenzhiwu, 
                self.qianfenzhiyi, self.qianfenzhier, self.qianfenzhisi, self.qianfenzhiwu, self.baifenzhiyi, 
                self.wuqiong, self.xiaozhiyu, self.RMSE, self.EB, self.Max_rel_error, self.Mean_rel_error]


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
