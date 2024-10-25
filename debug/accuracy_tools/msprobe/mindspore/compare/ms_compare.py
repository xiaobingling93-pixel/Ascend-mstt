import os
import re
import copy
import sys
from itertools import zip_longest

from msprobe.core.common.utils import check_compare_param, CompareException, check_configuration_param, \
    task_dumppath_get, struct_json_get, add_time_with_yaml
from msprobe.core.common.file_utils import create_directory, load_yaml, load_npy, load_json, save_yaml, FileOpen
from msprobe.core.common.const import Const, CompareConst
from msprobe.core.common.log import logger
from msprobe.core.common.exceptions import FileCheckException
from msprobe.core.compare.acc_compare import Comparator
from msprobe.core.compare.check import check_struct_match, fuzzy_check_op
from msprobe.mindspore.compare.modify_mapping import modify_mapping_with_stack
from msprobe.mindspore.compare.layer_mapping import get_layer_mapping

class MSComparator(Comparator):
    def __init__(self, cell_mapping=None, api_mapping=None, data_mapping=None, is_cross_framework=False):
        self.frame_name = MSComparator.__name__
        self.cell_mapping = cell_mapping
        self.api_mapping = api_mapping
        self.data_mapping = data_mapping
        if data_mapping:
            self.cross_frame = is_cross_framework
        else:
            self.cross_frame = cell_mapping is not None or api_mapping is not None
        self.cell_mapping_dict = self.load_mapping_file(self.cell_mapping)
        self.api_mapping_dict = self.load_mapping_file(self.api_mapping)
        if api_mapping is not None:
            self.ms_to_pt_mapping = self.load_internal_api()

        if isinstance(self.data_mapping, str) or self.data_mapping is None:
            self.data_mapping_dict = self.load_mapping_file(self.data_mapping)
        elif isinstance(self.data_mapping, dict):
            self.data_mapping_dict = self.data_mapping
        else:
            raise TypeError(f"The type of parameter `data_mapping` must be dict, str or None, but got "
                            f"{type(self.data_mapping)}")

    def load_internal_api(self):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        yaml_path = os.path.join(cur_path, "ms_to_pt_api.yaml")
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
        npu_dict_new, bench_dict_new = copy.deepcopy(npu_dict), copy.deepcopy(bench_dict)  
        npu_op_name, bench_op_name = npu_dict_new.get(CompareConst.OP_NAME), bench_dict_new.get(CompareConst.OP_NAME)
        if self.cell_mapping is not None:
            npu_op_name = self.process_cell_mapping(npu_op_name)
        if self.api_mapping is not None:
            npu_op_name = self.process_internal_api_mapping(npu_op_name, bench_op_name)
            if isinstance(self.api_mapping, str):
                npu_dict_new, bench_dict_new, target_dict = self.transform_user_mapping_api(npu_dict_new, 
                                                                                            bench_dict_new)
                if target_dict:
                    bench_dict = self.reconstitution_bench_dict(npu_dict, copy.deepcopy(bench_dict_new), target_dict)
                    npu_op_name = npu_dict_new.get(CompareConst.OP_NAME) 
                    bench_op_name = bench_dict_new.get(CompareConst.OP_NAME)
        struct_match = check_struct_match(npu_dict_new, bench_dict_new, cross_frame=self.cross_frame)
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
            data_value = load_pt(data_path, True).detach()
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
    
    def process_internal_api_mapping(self, npu_op_name, bench_op_name):
        # get api name & class name from op_name
        # Functional.addcmul.0.forward.input.0
        npu_op_name, bench_op_name = npu_op_name.copy(), bench_op_name.copy()
        ms_api_name = self.get_api_name(npu_op_name[0].split(Const.SEP))
        pt_api_name = self.get_api_name(bench_op_name[0].split(Const.SEP))
        class_name = ms_api_name.split(Const.SEP)[0]
        if class_name == "Mint":
            return self.api_replace(npu_op_name, "Mint", "Torch")
        elif class_name == "MintFunctional":
            return self.api_replace(npu_op_name, "MintFunctional", "Functional")
        elif self.ms_to_pt_mapping.get(ms_api_name) == pt_api_name:
            return self.api_replace(npu_op_name, ms_api_name, pt_api_name)
        else:
            return npu_op_name
    
    def remove_element(self, op_name, struct, summary, idx):
        del op_name[idx]
        del struct[idx]
        del summary[idx]
    
    def get_api_name(self, api_list):
        try:
            api_name = api_list[0] + Const.SEP + api_list[1]
        except IndexError as error:
            logger.error(f'Failed to retrieve API name, please check if the dump data is reasonable')
            raise CompareException(CompareException.INDEX_OUT_OF_BOUNDS_ERROR) from error
        return api_name
    
    def transform_user_mapping_api(self, new_npu_dict, new_bench_dict):
        """
        Transform user mapping API based on new NPU and benchmark dictionaries.
        Parameters:
            new_npu_dict (dict): New NPU operation dictionary.
            new_bench_dict (dict): New benchmark operation dictionary.
        Returns:
            tuple: Updated NPU and benchmark dictionaries, along with the target dictionary.
        """
        npu_op_name, bench_op_name = new_npu_dict.get(CompareConst.OP_NAME), new_bench_dict.get(CompareConst.OP_NAME)
        npu_struct_in = new_npu_dict.get(CompareConst.INPUT_STRUCT)
        bench_struct_in = new_bench_dict.get(CompareConst.INPUT_STRUCT)
        npu_struct_out = new_npu_dict.get(CompareConst.OUTPUT_STRUCT)
        bench_struct_out =  new_bench_dict.get(CompareConst.OUTPUT_STRUCT)
        npu_summary, bench_summary = new_npu_dict.get(CompareConst.SUMMARY), new_bench_dict.get(CompareConst.SUMMARY)
        npu_in_len, bench_in_len = len(npu_struct_in), len(bench_struct_in) 
        npu_out_len, bench_out_len = len(npu_struct_out), len(bench_struct_out)
        ms_api_list, pt_api_list = npu_op_name[0].split(Const.SEP), bench_op_name[0].split(Const.SEP)
        ms_api_name = self.get_api_name(ms_api_list)
        pt_api_name = self.get_api_name(pt_api_list)
        target_dict = {}
        for api_dict in self.api_mapping_dict:
            if api_dict.get("pt_api") == pt_api_name and api_dict.get("ms_api") == ms_api_name:
                ms_user_args_len, pt_user_args_len = len(api_dict.get("ms_args")), len(api_dict.get("pt_args"))
                ms_user_output_len, pt_user_output_len = len(api_dict.get("ms_output")), len(api_dict.get("pt_output"))
                if ms_user_args_len != pt_user_args_len or ms_user_output_len != pt_user_output_len:
                    logger.warning("The user-defined mapping table is incorrect,\
                        make sure that the number of parameters is equal")
                    break
                ms_out_list = api_dict.get("ms_output", [])
                for idx in reversed(range(npu_out_len)):
                    if idx not in ms_out_list:
                        del npu_struct_out[idx]
                        if idx + npu_in_len < len(npu_summary) and idx + npu_in_len < len(npu_op_name): 
                            del npu_summary[idx + npu_in_len]
                            del npu_op_name[idx + npu_in_len]
                pt_out_list = api_dict.get("pt_output", [])
                for idx in reversed(range(bench_out_len)):
                    if idx not in pt_out_list:
                        del bench_struct_out[idx]
                        if idx + bench_in_len < len(bench_summary) and idx + bench_in_len < len(bench_op_name): 
                            del bench_summary[idx + bench_in_len]
                            del bench_op_name[idx + bench_in_len]
                ms_para_list = api_dict.get("ms_args", []) 
                for idx in reversed(range(npu_in_len)):
                    if idx not in ms_para_list:
                        self.remove_element(npu_op_name, npu_struct_in, npu_summary, idx)
                pt_para_list = api_dict.get("pt_args", []) 
                for idx in reversed(range(bench_in_len)):
                    if idx not in pt_para_list:
                        self.remove_element(bench_op_name, bench_struct_in, bench_summary, idx)
                npu_op_name = self.api_replace(npu_op_name, ms_api_name, pt_api_name)
                npu_op_name = self.para_sequence_update(npu_op_name, bench_op_name)
                target_dict = api_dict
                break
        if target_dict:
            new_npu_dict.update({CompareConst.OP_NAME: npu_op_name, CompareConst.INPUT_STRUCT: npu_struct_in, 
                                 CompareConst.OUTPUT_STRUCT: npu_struct_out, CompareConst.SUMMARY: npu_summary})
            new_bench_dict.update({CompareConst.OP_NAME: bench_op_name, CompareConst.INPUT_STRUCT: bench_struct_in,
                                   CompareConst.OUTPUT_STRUCT: bench_struct_out, CompareConst.SUMMARY: bench_summary})
        return new_npu_dict, new_bench_dict, target_dict  
    
    def para_sequence_update(self, npu_op_name, bench_op_name):
        for idx, _ in enumerate(npu_op_name):
            bench_op_name_list = bench_op_name[idx].rsplit(Const.SEP, 1)
            if len(bench_op_name_list) != 0:
                npu_op_name[idx] = npu_op_name[idx][:-1] + bench_op_name_list[-1]
        return npu_op_name

    def reconstitution_bench_dict(self, npu_dict, del_bench_dict, api_dict):
        ms_user_args_list = api_dict.get("ms_args", [])
        ms_user_output_list = api_dict.get("ms_output", [])
        npu_struct_in = npu_dict.get(CompareConst.INPUT_STRUCT)
        npu_struct_out = npu_dict.get(CompareConst.OUTPUT_STRUCT)
        npu_in_len = len(npu_struct_in)
        npu_out_len = len(npu_struct_out)
        if npu_in_len == len(ms_user_args_list) and npu_out_len == len(ms_user_output_list):
            return del_bench_dict
        ms_input_args_list = [i for i in range(npu_in_len)]
        input_sub_list = list(set(ms_input_args_list) - set(ms_user_args_list))
        ms_output_args_list = [i for i in range(npu_out_len)]
        output_sub_list = list(set(ms_output_args_list) - set(ms_user_output_list))
        bench_op_name = del_bench_dict.get(CompareConst.OP_NAME, [])
        bench_struct_in = del_bench_dict.get(CompareConst.INPUT_STRUCT, [])
        bench_struct_out = del_bench_dict.get(CompareConst.OUTPUT_STRUCT, [])
        bench_summary = del_bench_dict.get(CompareConst.SUMMARY, [])
        for idx in input_sub_list:  # Fill in the blank value field in the pt dictionary
            bench_op_name.insert(idx, CompareConst.N_A)
            bench_struct_in.insert(idx, CompareConst.N_A)
            bench_summary.insert(idx, CompareConst.N_A)
        for idx in output_sub_list:  # Fill in the blank value field in the pt dictionary
            bench_op_name.insert(npu_in_len + idx, CompareConst.N_A)
            bench_struct_out.insert(idx, CompareConst.N_A)
            bench_summary.insert(npu_in_len + idx, CompareConst.N_A)
        del_bench_dict.update({CompareConst.OP_NAME: bench_op_name, CompareConst.INPUT_STRUCT: bench_struct_in, 
                               CompareConst.OUTPUT_STRUCT: bench_struct_out, CompareConst.SUMMARY: bench_summary})
        return del_bench_dict
        

def sort_by_execution_sequence(npu_data, bench_data, mapping_list, flag):
    def generate_execution_sequence(data):
        sequence_map = {}
        for index, item in enumerate(data.keys()):
            if flag in item:
                item_split = item.split(Const.SEP)
                item_name = Const.SEP.join(item_split[0:-2])
                item_index = item_split[-1]
                if item_index == 'forward' or item_index == 'backward':
                    item_index = item_split[-2]
                item_key = f"{item_name}.{item_index}"
                sequence_map[item_key] = index
        return sequence_map

    npu_map = generate_execution_sequence(npu_data)
    bench_map = generate_execution_sequence(bench_data)

    def sort_by_map(item):
        first_key = npu_map.get(item[0], sys.maxsize)
        second_key = bench_map.get(item[1], sys.maxsize)
        return first_key, second_key

    return sorted(mapping_list, key=sort_by_map)


def generate_kernel_data(map_value, data, flag):
    if not map_value:
        return [], []
    inputs_name = []
    outputs_name = []
    map_split = map_value.split(Const.SEP)
    map_name = Const.SEP.join(map_split[0:-1])
    map_index = map_split[-1]
    for key, value in data.items():
        if key.find(flag) != -1 and key.find(map_name) != -1:
            if key.split(Const.SEP)[-1] != map_index and key.split(Const.SEP)[-2] != map_index :
                continue
            if flag == 'forward':
                input_args = value.get('input_args', {})
            else:
                input_args = value.get('input', {})
            output_args = value.get('output', {})
            for i in range(len(input_args)):
                inputs_name.append(f"{key}.input.{i}")
            for i in range(len(output_args)):
                outputs_name.append(f"{key}.output.{i}")
    return inputs_name, outputs_name


def generate_file_mapping(npu_json_path, bench_json_path, mapping_list):

    npu_data = load_json(npu_json_path).get("data", {})
    bench_data = load_json(bench_json_path).get("data", {})

    forward_data = []
    mapping_list = sort_by_execution_sequence(npu_data, bench_data, mapping_list, Const.FORWARD)
    for map_value in mapping_list:
        npu_forward_inputs, npu_backward_outputs = generate_kernel_data(map_value[0], npu_data, "forward")
        bench_forward_inputs, bench_backward_outputs = generate_kernel_data(map_value[1], bench_data, "forward")
        inputs_zip = list(zip_longest(npu_forward_inputs, bench_forward_inputs))
        outputs_zip = list(zip_longest(npu_backward_outputs, bench_backward_outputs))
        forward_data.extend(inputs_zip)
        forward_data.extend(outputs_zip)

    backward_data = []
    mapping_list = sort_by_execution_sequence(npu_data, bench_data, mapping_list, Const.BACKWARD)
    for map_value in mapping_list:
        npu_forward_inputs, npu_backward_outputs = generate_kernel_data(map_value[0], npu_data, "backward")
        bench_forward_inputs, bench_backward_outputs = generate_kernel_data(map_value[1], bench_data, "backward")
        inputs_zip = list(zip_longest(npu_forward_inputs, bench_forward_inputs))
        outputs_zip = list(zip_longest(npu_backward_outputs, bench_backward_outputs))
        backward_data.extend(inputs_zip)
        backward_data.extend(outputs_zip)

    kernel_data = forward_data + backward_data
    result = {key: value for key, value in kernel_data if key is not None}

    return result


def check_cross_framework(bench_json_path):
    pattern = r'"data_name":\s*"[^"]+\.pt"'
    with FileOpen(bench_json_path, 'r') as file:
        for line in file:
            if re.search(pattern, line):
                return True
    return False


def ms_compare(input_param, output_path, **kwargs):
    try:
        stack_mode = kwargs.get('stack_mode', False)
        auto_analyze = kwargs.get('auto_analyze', True)
        fuzzy_match = kwargs.get('fuzzy_match', False)
        cell_mapping = kwargs.get('cell_mapping', None)
        api_mapping = kwargs.get('api_mapping', None)
        data_mapping = kwargs.get('data_mapping', None)
        layer_mapping = kwargs.get('layer_mapping', None)

        summary_compare, md5_compare = task_dumppath_get(input_param)
        check_configuration_param(stack_mode, auto_analyze, fuzzy_match, input_param.get('is_print_compare_log', True))
        create_directory(output_path)
        check_compare_param(input_param, output_path, summary_compare, md5_compare)
    except (CompareException, FileCheckException) as error:
        logger.error('Compare failed. Please check the arguments and do it again!')
        raise CompareException(error.code) from error
    if layer_mapping:
        pt_stack, pt_construct = struct_json_get(input_param, Const.PT_FRAMEWORK)
        ms_stack, ms_construct = struct_json_get(input_param, Const.MS_FRAMEWORK)
        mapping = load_yaml(layer_mapping)
        ms_mapping_result = modify_mapping_with_stack(ms_stack, ms_construct)
        pt_mapping_result = modify_mapping_with_stack(pt_stack, pt_construct)
        layer_mapping = get_layer_mapping(ms_mapping_result, pt_mapping_result, mapping)
        data_mapping = generate_file_mapping(input_param.get("npu_json_path"), input_param.get("bench_json_path"), layer_mapping)

        data_mapping_name = add_time_with_yaml(f"data_mapping")
        data_mapping_path = os.path.join(os.path.realpath(output_path), f"{data_mapping_name}")
        save_yaml(data_mapping_path, data_mapping)
    is_cross_framework = check_cross_framework(input_param.get("bench_json_path"))
    ms_comparator = MSComparator(cell_mapping, api_mapping, data_mapping, is_cross_framework)
    ms_comparator.compare_core(input_param, output_path, stack_mode=stack_mode,
                 auto_analyze=auto_analyze, fuzzy_match=fuzzy_match, summary_compare=summary_compare,
                 md5_compare=md5_compare)
