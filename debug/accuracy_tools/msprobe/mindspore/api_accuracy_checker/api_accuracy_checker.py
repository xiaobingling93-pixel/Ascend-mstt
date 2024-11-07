# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
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
from tqdm import tqdm

from msprobe.core.common.const import Const, CompareConst, MsCompareConst
from msprobe.core.common.file_utils import FileOpen, create_directory, write_csv, load_json
from msprobe.core.common.utils import add_time_as_suffix
from msprobe.mindspore.api_accuracy_checker.api_info import ApiInfo
from msprobe.mindspore.api_accuracy_checker.api_runner import api_runner, ApiInputAggregation
from msprobe.mindspore.api_accuracy_checker.base_compare_algorithm import compare_algorithms
from msprobe.mindspore.api_accuracy_checker.data_manager import DataManager
from msprobe.mindspore.api_accuracy_checker.utils import (check_and_get_from_json_dict, global_context,
                                                          trim_output_compute_element_list)
from msprobe.mindspore.common.log import logger


class BasicInfoAndStatus:
    def __init__(self, api_name, bench_dtype, tested_dtype, shape, status, err_msg) -> None:
        self.api_name = api_name
        self.bench_dtype = bench_dtype
        self.tested_dtype = tested_dtype
        self.shape = shape
        self.status = status
        self.err_msg = err_msg


class ResultCsvEntry:
    def __init__(self) -> None:
        self.forward_pass_status = None
        self.backward_pass_status = None
        self.forward_err_msg = ""
        self.backward_err_msg = ""
        self.overall_err_msg = None


class ApiAccuracyChecker:
    def __init__(self, args):
        self.api_infos = dict()
        self.data_manager = DataManager(args.out_path, args.result_csv_path)  # 在初始化时实例化 DataManager

    @staticmethod
    def run_and_compare_helper(api_info, api_name_str, api_input_aggregation, forward_or_backward):
        '''
        Args:
            api_info: ApiInfo
            api_name_str: str
            api_input_aggregation: ApiInputAggregation
            forward_or_backward: str: Union["forward", "backward"]

        Return:
            output_list: List[tuple(str, str, BasicInfoAndStatus, dict{str: CompareResult})]

        Description:
            get mindspore api output, run torch api and get output.
            compare output.
            record compare result.
        '''
        # get output
        if global_context.get_is_constructed():
            # constructed situation, need use constructed input to run mindspore api getting tested_output
            tested_outputs = api_runner(api_input_aggregation, api_name_str, forward_or_backward, Const.MS_FRAMEWORK)
        else:
            tested_outputs = api_info.get_compute_element_list(forward_or_backward, Const.OUTPUT)
        bench_outputs = api_runner(api_input_aggregation, api_name_str, forward_or_backward, Const.PT_FRAMEWORK)
        tested_outputs = trim_output_compute_element_list(tested_outputs, forward_or_backward)
        bench_outputs = trim_output_compute_element_list(bench_outputs, forward_or_backward)
        if len(tested_outputs) != len(bench_outputs):
            logger.warning(f"ApiAccuracyChecker.run_and_compare_helper: api: {api_name_str}.{forward_or_backward}, "
                           "number of bench outputs and tested outputs is different, comparing result can be wrong. "
                           f"tested outputs: {len(tested_outputs)}, bench outputs: {len(bench_outputs)}")

        # compare output
        output_list = []
        for i, (bench_out, tested_out) in enumerate(zip(bench_outputs, tested_outputs)):
            api_name_with_slot = Const.SEP.join([api_name_str, forward_or_backward, Const.OUTPUT, str(i)])
            bench_dtype = bench_out.get_dtype()
            tested_dtype = tested_out.get_dtype()
            shape = bench_out.get_shape()

            compare_result_dict = dict()
            for compare_algorithm_name, compare_algorithm in compare_algorithms.items():
                compare_result = compare_algorithm(bench_out, tested_out)
                compare_result_dict[compare_algorithm_name] = compare_result

            if compare_result_dict.get(CompareConst.COSINE).pass_status == CompareConst.PASS and \
                    compare_result_dict.get(CompareConst.MAX_ABS_ERR).pass_status == CompareConst.PASS:
                status = CompareConst.PASS
                err_msg = ""
            else:
                status = CompareConst.ERROR
                err_msg = compare_result_dict.get(CompareConst.COSINE).err_msg + \
                          compare_result_dict.get(CompareConst.MAX_ABS_ERR).err_msg
            basic_info_status = \
                BasicInfoAndStatus(api_name_with_slot, bench_dtype, tested_dtype, shape, status, err_msg)
            output_list.append(tuple([api_name_str, forward_or_backward, basic_info_status, compare_result_dict]))
        return output_list

    @staticmethod
    def prepare_api_input_aggregation(api_info, forward_or_backward=Const.FORWARD):
        '''
        Args:
            api_info: ApiInfo
            forward_or_backward: str
        Returns:
            ApiInputAggregation
        '''
        forward_inputs = api_info.get_compute_element_list(Const.FORWARD, Const.INPUT)
        kwargs = api_info.get_kwargs()
        if forward_or_backward == Const.FORWARD:
            gradient_inputs = None
        else:
            gradient_inputs = api_info.get_compute_element_list(Const.BACKWARD, Const.INPUT)
        return ApiInputAggregation(forward_inputs, kwargs, gradient_inputs)

    def parse(self, api_info_path):
        api_info_dict = load_json(api_info_path)

        # init global context
        task = check_and_get_from_json_dict(api_info_dict, MsCompareConst.TASK_FIELD,
                                            "task field in api_info.json", accepted_type=str,
                                            accepted_value=(MsCompareConst.STATISTICS_TASK,
                                                            MsCompareConst.TENSOR_TASK))
        is_constructed = task == MsCompareConst.STATISTICS_TASK
        if not is_constructed:
            dump_data_dir = check_and_get_from_json_dict(api_info_dict, MsCompareConst.DUMP_DATA_DIR_FIELD,
                                                         "dump_data_dir field in api_info.json", accepted_type=str)
        else:
            dump_data_dir = ""
        global_context.init(is_constructed, dump_data_dir)

        api_info_data = check_and_get_from_json_dict(api_info_dict, MsCompareConst.DATA_FIELD,
                                                     "data field in api_info.json", accepted_type=dict)
        for api_name, api_info in api_info_data.items():
            is_mint = api_name.split(Const.SEP)[0] in \
                      (MsCompareConst.MINT, MsCompareConst.MINT_FUNCTIONAL)
            if not is_mint:
                continue
            forbackward_str = api_name.split(Const.SEP)[-1]
            if forbackward_str not in (Const.FORWARD, Const.BACKWARD):
                logger.warning(f"api: {api_name} is not recognized as forward api or backward api, skip this.")
            api_name = Const.SEP.join(api_name.split(Const.SEP)[:-1])  # www.xxx.yyy.zzz --> www.xxx.yyy
            if api_name not in self.api_infos:
                self.api_infos[api_name] = ApiInfo(api_name)

            if forbackward_str == Const.FORWARD:
                self.api_infos[api_name].load_forward_info(api_info)
            else:
                self.api_infos[api_name].load_backward_info(api_info)

    def run_and_compare(self):
        for api_name_str, api_info in tqdm(self.api_infos.items()):
            if not self.data_manager.is_unique_api(api_name_str):
                continue

            if not api_info.check_forward_info():
                logger.debug(f"api: {api_name_str} is lack of forward infomation, skip forward and backward check.")
                continue
            try:
                forward_inputs_aggregation = self.prepare_api_input_aggregation(api_info, Const.FORWARD)
            except Exception as e:
                logger.warning(f"exception occurs when getting inputs for {api_name_str} forward api. "
                               f"skip forward and backward check. detailed exception information: {e}.")
                continue
            forward_output_list = None
            try:
                forward_output_list = \
                    self.run_and_compare_helper(api_info, api_name_str, forward_inputs_aggregation, Const.FORWARD)
            except Exception as e:
                logger.warning(f"exception occurs when running and comparing {api_name_str} forward api. "
                               f"detailed exception information: {e}.")
            self.data_manager.record(forward_output_list)

            if not api_info.check_backward_info():
                self.data_manager.save_results(api_name_str)  # 不存在反向，则直接保存前向结果

                logger.debug(f"api: {api_name_str} is lack of backward infomation, skip backward check.")
                continue
            try:
                backward_inputs_aggregation = self.prepare_api_input_aggregation(api_info, Const.BACKWARD)
            except Exception as e:
                logger.warning(f"exception occurs when getting inputs for {api_name_str} backward api. "
                               f"skip backward check. detailed exception information: {e}.")
                continue
            backward_output_list = None
            try:
                backward_output_list = \
                    self.run_and_compare_helper(api_info, api_name_str, backward_inputs_aggregation, Const.BACKWARD)
            except Exception as e:
                logger.warning(f"exception occurs when running and comparing {api_name_str} backward api. "
                               f"detailed exception information: {e}.")
            self.data_manager.record(backward_output_list)

            self.data_manager.save_results(api_name_str)


