import os.path
import numpy as np
from msprobe.core.common.utils import check_compare_param, CompareException, check_configuration_param, \
    task_dumppath_get, load_yaml
from msprobe.core.common.file_check import FileChecker, create_directory
from msprobe.core.common.const import FileCheckConst, Const
from msprobe.core.common.log import logger
from msprobe.core.common.exceptions import FileCheckException
from msprobe.core.compare.acc_compare import Comparator
from msprobe.core.common.utils import CompareException
from msprobe.core.compare.check import check_struct_match, fuzzy_check_op


class MSComparator (Comparator):
    def __init__(self, cell_mapping=None, api_mapping=None):
        self.frame_name = MSComparator.__name__
        self.cell_mapping = cell_mapping
        self.api_mapping = api_mapping
        self.cross_frame = cell_mapping is not None or api_mapping is not None
        self.cell_mapping_dict = self.load_mapping_file(self.cell_mapping)
        self.api_mapping_dict = self.load_mapping_file(self.api_mapping)

    def load_mapping_file(self, mapping_file):
        if isinstance(self.cell_mapping, str):
            mapping_dict = load_yaml(mapping_file)
        else:
            mapping_dict = {}
        return mapping_dict

    def process_cell_mapping(self, a_op_name):
        a_op_name = [op_name.replace("Cell", "Module", 1) for op_name in a_op_name]
        if self.cell_mapping_dict:
            for index, op_name in enumerate(a_op_name):
                cell_name = op_name.split(Const.SEP, 1)[-1].rsplit(Const.SEP, 4)[0]
                if cell_name in self.cell_mapping_dict:
                    a_op_name[index] = op_name.replace(cell_name, self.cell_mapping_dict[cell_name], 1)
        return a_op_name


    def check_op(self, npu_dict, bench_dict, fuzzy_match):
        a_op_name = npu_dict["op_name"]
        b_op_name = bench_dict["op_name"]
   
        if self.api_mapping is not None:
            pass
        if self.cell_mapping is not None:
            a_op_name = self.process_cell_mapping(a_op_name)

        struct_match = check_struct_match(npu_dict, bench_dict, cross_frame=self.cross_frame)
        if not fuzzy_match:
            return a_op_name == b_op_name and struct_match
        is_match = True
        try:
            is_match = fuzzy_check_op(a_op_name, b_op_name)
        except Exception as err:
            logger.warning("%s and %s can not fuzzy match." % (a_op_name, b_op_name))
            is_match = False
        return is_match and struct_match
    
    def read_npy_data(self, dir_path, file_name, load_pt=False):
        data_path = os.path.join(dir_path, file_name)
        if load_pt:
            import torch
            path_checker = FileChecker(data_path, FileCheckConst.FILE, FileCheckConst.READ_ABLE,
                                    FileCheckConst.PT_SUFFIX, False)
            data_path = path_checker.common_check()
            data_value = torch.load(data_path, map_location=torch.device('cpu')).detach()       # detach for less memory
            if data_value.dtype == torch.bfloat16:
                data_value = data_value.to(torch.float32)
            data_value = data_value.numpy()
        else:
            path_checker = FileChecker(data_path, FileCheckConst.FILE, FileCheckConst.READ_ABLE,
                                    FileCheckConst.NUMPY_SUFFIX, False)
            data_path = path_checker.common_check()
            data_value = np.load(data_path)      
        return data_value    

def ms_compare(input_param, output_path, **kwargs):
    try:
        stack_mode = kwargs.get('stack_mode', False)
        auto_analyze = kwargs.get('auto_analyze', True)
        fuzzy_match = kwargs.get('fuzzy_match', False)
        cell_mapping = kwargs.get('cell_mapping', None)
        summary_compare, md5_compare = task_dumppath_get(input_param)
        check_configuration_param(stack_mode, auto_analyze, fuzzy_match)
        create_directory(output_path)
        check_compare_param(input_param, output_path, summary_compare, md5_compare)
    except (CompareException, FileCheckException) as error:
        logger.error('Compare failed. Please check the arguments and do it again!')
        raise CompareException(error.code) from error
    ms_comparator = MSComparator(cell_mapping)
    ms_comparator.compare_core(input_param, output_path, stack_mode=stack_mode,
                 auto_analyze=auto_analyze, fuzzy_match=fuzzy_match, summary_compare=summary_compare,
                 md5_compare=md5_compare)
