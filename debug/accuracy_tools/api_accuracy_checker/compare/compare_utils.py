from api_accuracy_checker.common.utils import Const, print_warn_log
import numpy as np


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
