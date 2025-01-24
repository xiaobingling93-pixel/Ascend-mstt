# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from msprobe.core.common.const import CompareConst, Const
from msprobe.core.common.exceptions import FileCheckException
from msprobe.core.common.file_utils import FileOpen, create_directory, load_json, load_npy, load_yaml
from msprobe.core.common.log import logger
from msprobe.core.common.utils import CompareException, check_compare_param, check_configuration_param, \
    check_op_str_pattern_valid, get_dump_mode, set_dump_path
from msprobe.core.compare.acc_compare import Comparator, ModeConfig
from msprobe.core.compare.check import dtype_mapping
from msprobe.core.compare.layer_mapping import generate_data_mapping_by_layer_mapping
from msprobe.core.compare.utils import set_stack_json_path, reorder_op_x_list


class MappingConfig:
    def __init__(self, cell_mapping=None, api_mapping=None, data_mapping=None):
        self.cell_mapping = cell_mapping
        self.api_mapping = api_mapping
        self.data_mapping = data_mapping


class MSComparator(Comparator):
    """
    用于mindspore动态图同框架/跨框架精度比对，支持md5/summary/all模式。
    cell_mapping: mindspore在cell级别(L0)dump数据和pytorch的module之间的映射关系；
    api_mapping: mindspore在api级别(L1)dump数据和pytorch的api之间的映射关系；
    data_mapping: mindspore的cell或api的入参/出参和pytorch之间的映射关系；
    is_cross_framework: 是否跨框架。
    """
    def __init__(self, mode_config, mapping_config=None, is_cross_framework=False):
        super().__init__(mode_config)
        self.frame_name = MSComparator.__name__

        self.stack_mode = mode_config.stack_mode
        self.auto_analyze = mode_config.auto_analyze
        self.fuzzy_match = mode_config.fuzzy_match
        self.dump_mode = mode_config.dump_mode

        if mapping_config:
            self.cell_mapping = mapping_config.cell_mapping
            self.api_mapping = mapping_config.api_mapping
            self.data_mapping = mapping_config.data_mapping

        if self.data_mapping:
            self.cross_frame = is_cross_framework
        else:
            self.cross_frame = self.cell_mapping is not None or self.api_mapping is not None
        self.cell_mapping_dict = self.load_mapping_file(self.cell_mapping)
        self.api_mapping_dict = self.load_mapping_file(self.api_mapping)
        if self.api_mapping is not None:
            self.ms_to_pt_mapping = self.load_internal_api()

        if isinstance(self.data_mapping, str) or self.data_mapping is None:
            self.data_mapping_dict = self.load_mapping_file(self.data_mapping)
        elif isinstance(self.data_mapping, dict):
            self.data_mapping_dict = self.data_mapping
        else:
            raise TypeError(f"The type of parameter `data_mapping` must be dict, str or None, but got "
                            f"{type(self.data_mapping)}")

    def calc_accuracy(self, result_df, header):
        condition_no_bench = result_df[CompareConst.BENCH_NAME] == CompareConst.N_A
        result_df[condition_no_bench] = result_df[condition_no_bench].fillna(CompareConst.N_A)
        result_df.loc[condition_no_bench, CompareConst.ERROR_MESSAGE] = CompareConst.NO_BENCH

        def calc_summary_diff(data_type: str):
            def type_check(val):
                check_series = pd.Series(False, index=val.index)
                val_str = val.astype(str)
                check_series[pd.to_numeric(val_str, errors='coerce').notna() | val_str.str.lower().eq('nan')] = True
                return check_series

            def get_number(val):
                return pd.to_numeric(val.astype(str), errors='coerce')

            ms_val = result_df['NPU ' + data_type]
            pt_val = result_df['Bench ' + data_type]
            diff_name = data_type.capitalize() + ' diff'
            rel_err_name = ('norm' if data_type == 'l2norm' else data_type).capitalize() + 'RelativeErr'
            condition_na = ~type_check(ms_val) | ~type_check(pt_val)
            result_df.loc[condition_na, [diff_name, rel_err_name]] = CompareConst.N_A
            result_df.loc[~(condition_no_bench | condition_na), diff_name] = get_number(ms_val) - get_number(pt_val)
            condition_nan_diff = ~condition_no_bench & ~condition_na & result_df[diff_name].isna()
            condition_not_nan_diff = ~condition_no_bench & ~condition_na & result_df[diff_name].notna()
            result_df.loc[condition_nan_diff, [diff_name, rel_err_name]] = CompareConst.NAN
            condition_pt_zero = pt_val == 0
            result_df.loc[condition_not_nan_diff & condition_pt_zero, rel_err_name] = CompareConst.NAN
            condition_ref_err = condition_not_nan_diff & ~condition_pt_zero
            result_df.loc[condition_ref_err, rel_err_name] = (result_df.loc[condition_ref_err, diff_name] /
                                                              pt_val[condition_ref_err] * 100)
            result_df.loc[condition_ref_err, rel_err_name] = (result_df.loc[condition_ref_err, rel_err_name]
                                                              .abs().astype(str) + '%')
            magnitude = get_number(result_df[diff_name]).abs() / (
                    pd.Series(np.maximum(get_number(ms_val), get_number(pt_val))).abs() + CompareConst.EPSILON)
            return magnitude > CompareConst.MAGNITUDE

        if self.dump_mode == Const.MD5:
            condition_md5_equal = result_df[CompareConst.NPU_MD5] == result_df[CompareConst.BENCH_MD5]
            result_df.loc[condition_md5_equal, CompareConst.RESULT] = CompareConst.PASS
            result_df.loc[~condition_md5_equal & ~condition_no_bench, CompareConst.RESULT] = CompareConst.DIFF
        elif self.dump_mode == Const.SUMMARY:
            warning_list = [calc_summary_diff(data_type) for data_type in ['max', 'min', 'mean', 'l2norm']]
            warning_flag = pd.DataFrame(warning_list).all()
            result_df.loc[~condition_no_bench, [CompareConst.RESULT, CompareConst.ERROR_MESSAGE]] = ''
            result_df.loc[warning_flag, CompareConst.RESULT] = CompareConst.WARNING
            result_df.loc[warning_flag, CompareConst.ERROR_MESSAGE] = 'Need double check api accuracy.'
        else:
            fill_cols = [CompareConst.COSINE, CompareConst.MAX_ABS_ERR, CompareConst.MAX_RELATIVE_ERR,
                         CompareConst.ONE_THOUSANDTH_ERR_RATIO, CompareConst.FIVE_THOUSANDTHS_ERR_RATIO,
                         CompareConst.ERROR_MESSAGE]
            result_df.loc[~condition_no_bench, fill_cols] = ''
            result_df.loc[~condition_no_bench, CompareConst.ACCURACY] = CompareConst.ACCURACY_CHECK_YES
        return result_df[header]

    def make_result_df(self, result):
        header = CompareConst.HEAD_OF_COMPARE_MODE[self.dump_mode][:]

        if self.stack_mode:
            header.append(CompareConst.STACK)
        if self.dump_mode == Const.ALL:
            header.append(CompareConst.DATA_NAME)
        result.rename(columns={'op_name_x': CompareConst.NPU_NAME,
                               'op_name_y': CompareConst.BENCH_NAME,
                               'dtype_x': CompareConst.NPU_DTYPE,
                               'dtype_y': CompareConst.BENCH_DTYPE,
                               'shape_x': CompareConst.NPU_SHAPE,
                               'shape_y': CompareConst.BENCH_SHAPE,
                               'md5_x': CompareConst.NPU_MD5,
                               'md5_y': CompareConst.BENCH_MD5,
                               'data_name_x': CompareConst.DATA_NAME,
                               'stack_info_x': CompareConst.STACK}, inplace=True)

        npu_summary = [CompareConst.NPU_MAX, CompareConst.NPU_MIN, CompareConst.NPU_MEAN, CompareConst.NPU_NORM]
        bench_summary = [CompareConst.BENCH_MAX, CompareConst.BENCH_MIN, CompareConst.BENCH_MEAN,
                         CompareConst.BENCH_NORM]

        def set_summary(summary):
            if summary == CompareConst.N_A:
                return [CompareConst.N_A] * 4
            summary_list = []
            for i in summary:
                if i is None:
                    summary_list.append(CompareConst.N_A)
                elif str(i).lower() == 'nan':
                    summary_list.append(CompareConst.NAN)
                else:
                    summary_list.append(i)
            return summary_list

        result[npu_summary] = result['summary_x'].apply(set_summary).tolist()
        result[bench_summary] = result['summary_y'].apply(set_summary).tolist()
        result_df = pd.DataFrame(columns=header)
        for h in header:
            if h in result.columns:
                result_df[h] = result[h]
        return self.calc_accuracy(result_df, header)

    def load_internal_api(self):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        yaml_path = os.path.abspath(os.path.join(cur_path, CompareConst.INTERNAL_API_MAPPING_FILE))
        return load_yaml(yaml_path)

    def load_mapping_file(self, mapping_file):
        if isinstance(mapping_file, str):
            mapping_dict = load_yaml(mapping_file)
        else:
            mapping_dict = {}
        return mapping_dict

    def process_cell_mapping(self, npu_op_name):
        if not npu_op_name:
            return CompareConst.N_A
        param_grad_flag = Const.PARAMS_GRAD in npu_op_name.split(Const.SEP)
        if not param_grad_flag and not re.search(Const.REGEX_FORWARD_BACKWARD, npu_op_name):
            return CompareConst.N_A
        npu_op_name = npu_op_name.replace("Cell", "Module", 1)
        if self.cell_mapping_dict:
            # get cell name & class name from op_name
            # Cell.fc1.Dense.forward.0.input.0
            cell_name = re.split(r'\.(?:forward|backward|parameters_grad)\.', npu_op_name.split(Const.SEP, 1)[-1])[0]
            if cell_name in self.cell_mapping_dict:
                npu_op_name = npu_op_name.replace(cell_name, self.cell_mapping_dict[cell_name], 1)
        return npu_op_name

    def read_npy_data(self, dir_path, file_name, load_pt_file=False):
        if not file_name:
            return None
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

    def process_internal_api_mapping(self, npu_op_name):
        # get api name & class name from op_name
        # Functional.addcmul.0.forward.input.0
        ms_api_name = self.get_api_name(npu_op_name.split(Const.SEP))
        class_name = ms_api_name.split(Const.SEP)[0]
        if class_name == "Mint":
            return npu_op_name.replace("Mint", "Torch")
        elif class_name == "MintFunctional":
            return npu_op_name.replace("MintFunctional", "Functional")
        elif self.ms_to_pt_mapping.get(ms_api_name):
            return npu_op_name.replace(ms_api_name, self.ms_to_pt_mapping.get(ms_api_name))
        else:
            return npu_op_name

    def get_api_name(self, api_list):
        try:
            api_name = api_list[0] + Const.SEP + api_list[1]
        except IndexError as error:
            logger.error(f'Failed to retrieve API name, please check if the dump data is reasonable')
            raise CompareException(CompareException.INDEX_OUT_OF_BOUNDS_ERROR) from error
        return api_name

    def compare_process(self, file_lists):
        npu_json_path, bench_json_path, stack_json_path = file_lists
        npu_json_data = load_json(npu_json_path)
        bench_json_data = load_json(bench_json_path)
        stack_json_data = load_json(stack_json_path) if self.stack_mode else None

        npu_df = self.gen_data_df(npu_json_data, stack_json_data)
        bench_df = self.gen_data_df(bench_json_data, stack_json_data)
        if self.cell_mapping:
            npu_df[CompareConst.COMPARE_KEY] = npu_df[CompareConst.OP_NAME].apply(self.process_cell_mapping)
        elif self.api_mapping:
            npu_df[CompareConst.COMPARE_KEY] = npu_df[CompareConst.OP_NAME].apply(self.process_internal_api_mapping)
            if isinstance(self.api_mapping, str):
                self.modify_compare_data_with_user_mapping(npu_df, bench_df)
        else:
            npu_df[CompareConst.COMPARE_KEY] = npu_df[CompareConst.OP_NAME]
        npu_df[[Const.DTYPE, Const.SHAPE]] = npu_df[[Const.DTYPE, Const.SHAPE]].astype(str)
        bench_df[[Const.DTYPE, Const.SHAPE]] = bench_df[[Const.DTYPE, Const.SHAPE]].astype(str)
        npu_df[CompareConst.COMPARE_SHAPE] = npu_df[Const.SHAPE]
        bench_df[CompareConst.COMPARE_KEY] = bench_df[CompareConst.OP_NAME]
        bench_df[CompareConst.COMPARE_SHAPE] = bench_df[Const.SHAPE]
        match_result = pd.merge(npu_df, bench_df, on=[CompareConst.COMPARE_KEY, CompareConst.COMPARE_SHAPE],
                                how='outer')
        match_result = match_result[match_result['op_name_x'].notna()].fillna(CompareConst.N_A)

        def gen_dtype_condition():
            npu_dtype = match_result['dtype_x']
            bench_dtype = match_result['dtype_y']
            if self.cross_frame:
                npu_dtype = npu_dtype.map(dtype_mapping).fillna(npu_dtype)
            return ((npu_dtype == bench_dtype) |
                    ((npu_dtype == Const.FLOAT16) & (bench_dtype == Const.FLOAT32)) |
                    ((npu_dtype == Const.FLOAT32) & (bench_dtype == Const.FLOAT16)) |
                    ((npu_dtype == Const.FLOAT16) & (bench_dtype == Const.BFLOAT16)) |
                    ((npu_dtype == Const.BFLOAT16) & (bench_dtype == Const.FLOAT16)) |
                    ((npu_dtype == Const.TORCH_FLOAT16) & (bench_dtype == Const.TORCH_FLOAT32)) |
                    ((npu_dtype == Const.TORCH_FLOAT32) & (bench_dtype == Const.TORCH_FLOAT16)) |
                    ((npu_dtype == Const.TORCH_FLOAT16) & (bench_dtype == Const.TORCH_BFLOAT16)) |
                    ((npu_dtype == Const.TORCH_BFLOAT16) & (bench_dtype == Const.TORCH_FLOAT16)))

        match_result.loc[~gen_dtype_condition(), [i + '_y' for i in bench_df.columns]] = CompareConst.N_A
        return self.make_result_df(match_result)

    def modify_compare_data_with_user_mapping(self, npu_df, bench_df):
        def get_api_indices_dict(op_name_df):
            api_indices_dict = defaultdict(list)
            for op_index, name in enumerate(op_name_df[CompareConst.OP_NAME]):
                api = self.get_api_name(name.split(Const.SEP))
                api_indices_dict[api].append(op_index)
            return api_indices_dict

        ms_api_indices_dict = get_api_indices_dict(npu_df)
        pt_api_indices_dict = get_api_indices_dict(bench_df)

        def gen_input_compare_key(pattern, term):
            flag = True
            for i, prefix in enumerate(mapping_dict.get(f'ms_{term}')):
                if op_name.split(pattern)[1].startswith(str(prefix)):
                    npu_df.loc[index, CompareConst.COMPARE_KEY] = (
                        op_name.replace(pattern + str(prefix),
                                        pattern + str(mapping_dict.get(f'pt_{term}')[i])))
                    flag = False
            return flag

        for mapping_dict in self.api_mapping_dict:
            keys_to_compare = [
                ('ms_args', 'pt_args'),
                ('ms_output', 'pt_output'),
                ('ms_parameters', 'pt_parameters'),
                ('ms_parameters_grad', 'pt_parameters_grad'),
            ]
            if not all(len(mapping_dict.get(k1, [])) == len(mapping_dict.get(k2, [])) for k1, k2 in keys_to_compare):
                logger.warning('The user-defined mapping table is incorrect,\
                                make sure that the number of parameters is equal')
                continue

            ms_api, pt_api = mapping_dict.get('ms_api'), mapping_dict.get('pt_api')
            if ms_api not in ms_api_indices_dict or pt_api not in pt_api_indices_dict:
                continue
            for index in ms_api_indices_dict.get(ms_api):
                op_name = npu_df.loc[index, CompareConst.OP_NAME].replace(ms_api, pt_api, 1)
                if CompareConst.INPUT_PATTERN in op_name:
                    is_abandoned = gen_input_compare_key(CompareConst.INPUT_PATTERN, 'args')
                elif CompareConst.KWARGS_PATTERN in op_name:
                    is_abandoned = gen_input_compare_key(CompareConst.KWARGS_PATTERN, 'args')
                elif CompareConst.OUTPUT_PATTERN in op_name:
                    is_abandoned = gen_input_compare_key(CompareConst.OUTPUT_PATTERN, 'output')
                elif CompareConst.PARAMS_PATTERN in op_name:
                    is_abandoned = gen_input_compare_key(CompareConst.PARAMS_PATTERN, 'parameters')
                elif CompareConst.PARAMS_GRAD_PATTERN in op_name:
                    is_abandoned = gen_input_compare_key(CompareConst.PARAMS_GRAD_PATTERN, 'parameters_grad')
                else:
                    logger.error(f'Excepted op_name: {op_name}')
                    raise CompareException(CompareException.INVALID_DATA_ERROR)
                if is_abandoned:
                    npu_df.loc[index, CompareConst.COMPARE_KEY] = op_name + 'abandoned'

    def gen_data_df(self, data_json, stack_json_data):
        result = {
            CompareConst.OP_NAME: [],
            Const.DTYPE: [],
            Const.SHAPE: [],
            Const.SUMMARY: [],
            'stack_info': []
        }
        if self.dump_mode == Const.ALL:
            result['data_name'] = []
        elif self.dump_mode == Const.MD5:
            result[Const.MD5] = []
        for data_name in data_json['data']:
            check_op_str_pattern_valid(data_name)
            merge_list = self.gen_merge_list(data_json, data_name, stack_json_data)
            if not merge_list:
                continue

            op_name_list = merge_list.get(CompareConst.OP_NAME)
            summary_list = merge_list.get(Const.SUMMARY)
            data_name_list = merge_list.get('data_name')
            op_name_reorder, summary_reorder, data_name_reorder = reorder_op_x_list(op_name_list,
                                                                                    summary_list,
                                                                                    data_name_list)
            for op_name in op_name_reorder:
                result[CompareConst.OP_NAME].append(op_name)
                if (CompareConst.INPUT_PATTERN in op_name) or (CompareConst.KWARGS_PATTERN in op_name):
                    struct = merge_list[CompareConst.INPUT_STRUCT].pop(0)
                elif CompareConst.OUTPUT_PATTERN in op_name:
                    struct = merge_list[CompareConst.OUTPUT_STRUCT].pop(0)
                elif CompareConst.PARAMS_PATTERN in op_name:
                    struct = merge_list[CompareConst.PARAMS_STRUCT].pop(0)
                else:
                    struct = merge_list[CompareConst.PARAMS_GRAD_STRUCT].pop(0)
                result[Const.DTYPE].append(struct[0])
                result[Const.SHAPE].append(struct[1])
                if self.dump_mode == Const.MD5:
                    result[Const.MD5].append(struct[2])
                result[Const.SUMMARY].append(summary_reorder.pop(0))
                result['stack_info'].append(merge_list['stack_info'][0] if self.stack_mode else None)
                if self.dump_mode == Const.ALL:
                    result['data_name'].append(data_name_reorder.pop(0))
        return pd.DataFrame(result)


def check_cross_framework(bench_json_path):
    pattern = r'"data_name":\s*"[^"]+\.pt"'
    with FileOpen(bench_json_path, 'r') as file:
        for line in file:
            if re.search(pattern, line):
                return True
    return False


def ms_compare(input_param, output_path, **kwargs):
    try:
        auto_analyze = kwargs.get('auto_analyze', True)
        fuzzy_match = kwargs.get('fuzzy_match', False)
        cell_mapping = kwargs.get('cell_mapping', None)
        api_mapping = kwargs.get('api_mapping', None)
        data_mapping = kwargs.get('data_mapping', None)
        layer_mapping = kwargs.get('layer_mapping', None)
        suffix = kwargs.get('suffix', '')

        set_dump_path(input_param)
        dump_mode = get_dump_mode(input_param)
        if 'stack_json_path' in input_param:
            stack_mode = kwargs.get('stack_mode', False)
        else:
            stack_mode = set_stack_json_path(input_param)  # set stack_mode and set "stack_json_path" in input_param
        check_configuration_param(stack_mode, auto_analyze, fuzzy_match, input_param.get('is_print_compare_log', True))
        create_directory(output_path)
        check_compare_param(input_param, output_path, dump_mode, stack_mode)
    except (CompareException, FileCheckException) as error:
        logger.error('Compare failed. Please check the arguments and do it again!')
        raise CompareException(error.code) from error
    if layer_mapping:
        data_mapping = generate_data_mapping_by_layer_mapping(input_param, layer_mapping, output_path)

    mode_config = ModeConfig(stack_mode, auto_analyze, fuzzy_match, dump_mode)
    mapping_config = MappingConfig(cell_mapping, api_mapping, data_mapping)
    is_cross_framework = check_cross_framework(input_param.get('bench_json_path'))
    ms_comparator = MSComparator(mode_config, mapping_config, is_cross_framework)
    ms_comparator.compare_core(input_param, output_path, suffix=suffix)
