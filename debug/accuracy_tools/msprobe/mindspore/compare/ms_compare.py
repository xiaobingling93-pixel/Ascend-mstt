import os.path
import numpy as np
from msprobe.core.common.utils import check_compare_param, CompareException, check_configuration_param, task_dumppath_get
from msprobe.core.common.file_check import FileChecker, create_directory
from msprobe.core.common.const import FileCheckConst
from msprobe.core.common.log import logger
from msprobe.core.common.exceptions import FileCheckException
from msprobe.core.compare.acc_compare import Comparator 


class MSComparator (Comparator):
    def __init__(self):
        self.frame_name = MSComparator.__name__
    
    def read_npy_data(self,dir_path, file_name):
        data_path = os.path.join(dir_path, file_name)
        path_checker = FileChecker(data_path, FileCheckConst.FILE, FileCheckConst.READ_ABLE,
                                FileCheckConst.NUMPY_SUFFIX, False)
        data_path = path_checker.common_check()
        data_value = np.load(data_path)      # detach for less memory
        if data_value.dtype == np.float16:
            data_value = data_value.astype(np.float32)
        return data_value    
    
    
def ms_compare(input_param, output_path, stack_mode=False, auto_analyze=True, fuzzy_match=False):
    try:
        summary_compare, md5_compare = task_dumppath_get(input_param)
        check_configuration_param(stack_mode, auto_analyze, fuzzy_match)
        create_directory(output_path)
        check_compare_param(input_param, output_path, summary_compare, md5_compare)
    except (CompareException, FileCheckException) as error:
        logger.error('Compare failed. Please check the arguments and do it again!')
        raise CompareException(error.code) from error
    ms_comparator = MSComparator()
    ms_comparator.compare_core(input_param, output_path, stack_mode=stack_mode,
                 auto_analyze=auto_analyze, fuzzy_match=fuzzy_match, summary_compare=summary_compare,
                 md5_compare=md5_compare)
