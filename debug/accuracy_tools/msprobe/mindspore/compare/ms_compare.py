import os.path
from msprobe.core.common.utils import check_compare_param, CompareException, check_configuration_param, \
    task_dumppath_get, load_yaml, load_npy
from msprobe.core.common.file_check import create_directory
from msprobe.core.common.const import Const
from msprobe.mindspore.common.log import logger
from msprobe.core.common.exceptions import FileCheckException
from msprobe.core.compare.acc_compare import Comparator
from msprobe.core.compare.check import check_struct_match, fuzzy_check_op


class MSComparator(Comparator):
    def __init__(self, cell_mapping=None, api_mapping=None):
        self.frame_name = MSComparator.__name__
        self.cell_mapping = cell_mapping
        self.api_mapping = api_mapping
        self.cross_frame = cell_mapping is not None or api_mapping is not None
        self.cell_mapping_dict = self.load_mapping_file(self.cell_mapping)
        self.api_mapping_dict = {}
        if api_mapping is not None:
            self.ms_to_pt_mapping = self.load_internal_api()
            
    def load_internal_api(self):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        yaml_path = os.path.join(cur_path,"ms_to_pt_api.yaml")
        return load_yaml(yaml_path)

    def load_mapping_file(self, mapping_file):
        if isinstance(mapping_file, str):
            mapping_dict = load_yaml(mapping_file)
        else:
            mapping_dict = {}
        return mapping_dict

    def process_cell_mapping(self, npu_op_name):
        npu_op_name = [op_name.replace("Cell", "Module", 1) for op_name in npu_op_name]
        if self.cell_mapping_dict:
            for index, op_name in enumerate(npu_op_name):
                # get cell name & class name from op_name
                # Cell.fc1.Dense.forward.0.input.0
                cell_name = op_name.split(Const.SEP, 1)[-1].rsplit(Const.SEP, 4)[0]
                if cell_name in self.cell_mapping_dict:
                    npu_op_name[index] = op_name.replace(cell_name, self.cell_mapping_dict[cell_name], 1)
        return npu_op_name

    def check_op(self, npu_dict, bench_dict, fuzzy_match):
        npu_op_name = npu_dict["op_name"].copy()
        bench_op_name = bench_dict["op_name"].copy()
   
        if self.api_mapping is not None:
            npu_op_name = self.process_api_mapping(npu_op_name, bench_op_name)
        if self.cell_mapping is not None:
            npu_op_name = self.process_cell_mapping(npu_op_name)

        struct_match = check_struct_match(npu_dict, bench_dict, cross_frame=self.cross_frame)
        if not fuzzy_match:
            return npu_op_name == bench_op_name and struct_match
        is_match = True
        try:
            is_match = fuzzy_check_op(npu_op_name, bench_op_name)
        except Exception as err:
            logger.warning("%s and %s can not fuzzy match." % (npu_op_name, bench_op_name))
            is_match = False
        return is_match and struct_match
    
    def read_npy_data(self, dir_path, file_name, load_pt_file=False):
        data_path = os.path.join(dir_path, file_name)
        if load_pt_file:
            import torch
            from msprobe.pytorch.common.utils import load_pt
            data_value = load_pt(data_path).detach()
            if data_value.dtype == torch.bfloat16:
                data_value = data_value.to(torch.float32)
            data_value = data_value.numpy()
        else:
            data_value = load_npy(data_path) 
        return data_value    

    def api_replace(self, npu_op_name, target, para):
        for idx, _ in enumerate(npu_op_name):
            npu_op_name[idx] = npu_op_name[idx].replace(target, para)
        return npu_op_name
    
    def process_api_mapping(self, npu_op_name, bench_op_name):
        # get api name & class name from op_name
        # Functional.addcmul.0.forward.input.0
        ms_api_name = npu_op_name[0].rsplit(Const.SEP, 4)[0]
        pt_api_name = bench_op_name[0].rsplit(Const.SEP, 4)[0]
        class_name = ms_api_name.split(Const.SEP)[0]
        if class_name == "Mint":
            return self.api_replace(npu_op_name, "Mint", "Torch")
        elif class_name == "MintFunctional":
            return self.api_replace(npu_op_name, "MintFunctional", "Functional")
        elif self.ms_to_pt_mapping.get(ms_api_name) == pt_api_name:
            return self.api_replace(npu_op_name, ms_api_name, pt_api_name)
        else:
            return npu_op_name
        

def ms_compare(input_param, output_path, **kwargs):
    try:
        stack_mode = kwargs.get('stack_mode', False)
        auto_analyze = kwargs.get('auto_analyze', True)
        fuzzy_match = kwargs.get('fuzzy_match', False)
        cell_mapping = kwargs.get('cell_mapping', None)
        api_mapping = kwargs.get('api_mapping', None)
        summary_compare, md5_compare = task_dumppath_get(input_param)
        check_configuration_param(stack_mode, auto_analyze, fuzzy_match)
        create_directory(output_path)
        check_compare_param(input_param, output_path, summary_compare, md5_compare)
    except (CompareException, FileCheckException) as error:
        logger.error('Compare failed. Please check the arguments and do it again!')
        raise CompareException(error.code) from error
    ms_comparator = MSComparator(cell_mapping, api_mapping)
    ms_comparator.compare_core(input_param, output_path, stack_mode=stack_mode,
                 auto_analyze=auto_analyze, fuzzy_match=fuzzy_match, summary_compare=summary_compare,
                 md5_compare=md5_compare)
