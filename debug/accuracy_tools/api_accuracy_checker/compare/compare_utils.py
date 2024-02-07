import time
import numpy as np
import torch
from api_accuracy_checker.common.utils import Const, print_warn_log


current_time = time.strftime("%Y%m%d%H%M%S")
BENCHMARK_COMPARE_RESULT_FILE_NAME = "benchmark_compare_result_" + current_time + ".csv"
BENCHMARK_COMPARE_DETAILS_FILE_NAME = "benchmark_compare_details_" + current_time + ".csv"
Benchmark_Compare_Support_List = ['torch.float16', 'torch.bfloat16', 'torch.float32']
result_mapping = {
    'pass' : True,
    'warning': False,
    'error' :  False
}


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


class CompareConst:
    NAN = np.nan
    NA = "N/A"
    PASS = 'pass'
    WARNING = 'warning'
    ERROR = 'error'
    SKIP = 'SKIP'
    TRUE = 'TRUE'
    FALSE = 'FALSE'
    
    
class BenchmarkCompareColumn:
    API_NAME = 'API Name'
    DEVICE_DTYPE = 'DEVICE Dtype'
    SMALL_VALUE_ERROR_RATE = '小值域错误占比'
    RMSE = '均方根误差'
    MAX_REL_ERR = '相对误差最大值'
    MEAN_REL_ERR = '相对误差平均值'
    EB = '误差均衡性'
    SMALL_VALUE_ERROR_RATIO = '小值域错误比值'
    SMALL_VALUE_ERROR_STATUS = '小值域判定结果'
    RMSE_RATIO = '均方根误差比值'
    RMSE_STATUS = '均方根误差判定结果'
    MAX_REL_ERR_RATIO = '相对误差最大值比值'
    MAX_REL_ERR_STATUS = '相对误差最大值判定结果'
    MEAN_REL_ERR_RATIO = '相对误差平均值比值'
    MEAN_REL_ERR_STATUS = '相对误差平均值判定结果'
    EB_RATIO = '误差均衡性比值'
    EB_STATUS = '误差均衡性判定结果'
    FORWWARD_STATUS = 'Forward Test Success'
    BACKWARD_STATUS = 'Backward Test Success'
    MESSAGE = 'Message'
    
    @staticmethod
    def to_required_columns():
        return [BenchmarkCompareColumn.API_NAME, BenchmarkCompareColumn.DEVICE_DTYPE, 
                BenchmarkCompareColumn.SMALL_VALUE_ERROR_RATE, BenchmarkCompareColumn.RMSE, 
                BenchmarkCompareColumn.MAX_REL_ERR, BenchmarkCompareColumn.MEAN_REL_ERR, BenchmarkCompareColumn.EB]
        
    @staticmethod
    def get_detail_csv_title():
        return [BenchmarkCompareColumn.API_NAME,  
                BenchmarkCompareColumn.SMALL_VALUE_ERROR_RATIO, BenchmarkCompareColumn.SMALL_VALUE_ERROR_STATUS, 
                BenchmarkCompareColumn.RMSE_RATIO, BenchmarkCompareColumn.RMSE_STATUS, 
                BenchmarkCompareColumn.MAX_REL_ERR_RATIO, BenchmarkCompareColumn.MAX_REL_ERR_STATUS, 
                BenchmarkCompareColumn.MEAN_REL_ERR_RATIO, BenchmarkCompareColumn.MEAN_REL_ERR_STATUS, 
                BenchmarkCompareColumn.EB_RATIO, BenchmarkCompareColumn.EB_STATUS]
    
    @staticmethod
    def get_result_csv_title():
        return [BenchmarkCompareColumn.API_NAME, BenchmarkCompareColumn.FORWWARD_STATUS, 
                BenchmarkCompareColumn.BACKWARD_STATUS, BenchmarkCompareColumn.MESSAGE]


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
