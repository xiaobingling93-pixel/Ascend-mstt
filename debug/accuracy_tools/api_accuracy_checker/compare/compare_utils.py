import numpy as np
import torch
from api_accuracy_checker.common.utils import Const, print_warn_log


DETAIL_TEST_ROWS = [[
            "API Name", "Bench Dtype", "DEVICE Dtype", "Shape",
            "余弦相似度",
            "最大绝对误差",
            "双百指标",
            "双千指标",
            "双万指标",
            "错误率",
            "误差均衡性",
            "均方根误差",
            "小值域错误占比",
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


Benchmark_Compare_Support_List = ['torch.float16', 'torch.bfloat16', 'torch.float32']


class CompareConst:
    NAN = np.nan
    NA = "N/A"
    PASS = 'pass'
    WARNING = 'warning'
    ERROR = 'error'
    SKIP = 'SKIP'
    TRUE = 'TRUE'
    FALSE = 'FALSE'


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
