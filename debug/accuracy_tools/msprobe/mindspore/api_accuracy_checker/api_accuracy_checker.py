import json
import os

from msprobe.core.common.file_utils import FileOpen, create_directory, write_csv
from msprobe.core.common.utils import add_time_as_suffix
from msprobe.core.common.const import Const, CompareConst, MsCompareConst
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.api_accuracy_checker.api_info import ApiInfo
from msprobe.mindspore.api_accuracy_checker.api_runner import api_runner, ApiInputAggregation
from msprobe.mindspore.api_accuracy_checker.base_compare_algorithm import compare_algorithms
from msprobe.mindspore.api_accuracy_checker.utils import (check_and_get_from_json_dict, global_context,
                                                          trim_output_compute_element_list)


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
    def __init__(self):
        self.api_infos = dict()
        self.results = dict()

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
        with FileOpen(api_info_path, "r") as f:
            api_info_dict = json.load(f)

        # init global context
        task = check_and_get_from_json_dict(api_info_dict, MsCompareConst.TASK_FIELD,
                                            "task field in api_info.json",accepted_type=str,
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
            api_name = Const.SEP.join(api_name.split(Const.SEP)[:-1]) # www.xxx.yyy.zzz --> www.xxx.yyy
            if api_name not in self.api_infos:
                self.api_infos[api_name] = ApiInfo(api_name)

            if forbackward_str == Const.FORWARD:
                self.api_infos[api_name].load_forward_info(api_info)
            else:
                self.api_infos[api_name].load_backward_info(api_info)

    def run_and_compare(self):
        for api_name_str, api_info in self.api_infos.items():
            if not api_info.check_forward_info():
                logger.warning(f"api: {api_name_str} is lack of forward infomation, skip forward and backward check.")
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
            self.record(forward_output_list)

            if not api_info.check_backward_info():
                logger.warning(f"api: {api_name_str} is lack of backward infomation, skip backward check.")
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
            self.record(backward_output_list)

    def record(self, output_list):
        if output_list is None:
            return
        for output in output_list:
            api_real_name, forward_or_backward, basic_info, compare_result_dict = output
            key = tuple([api_real_name, forward_or_backward])
            if key not in self.results:
                self.results[key] = []
            self.results[key].append(tuple([basic_info, compare_result_dict]))


    def to_detail_csv(self, csv_dir):
        # detail_csv
        detail_csv = []
        detail_csv_header_basic_info = [
            MsCompareConst.DETAIL_CSV_API_NAME,
            MsCompareConst.DETAIL_CSV_BENCH_DTYPE,
            MsCompareConst.DETAIL_CSV_TESTED_DTYPE,
            MsCompareConst.DETAIL_CSV_SHAPE,
        ]
        detail_csv_header_compare_result = list(compare_algorithms.keys())
        detail_csv_header_status = [
            MsCompareConst.DETAIL_CSV_PASS_STATUS,
            MsCompareConst.DETAIL_CSV_MESSAGE,
        ]

        detail_csv_header = detail_csv_header_basic_info + detail_csv_header_compare_result + detail_csv_header_status
        detail_csv.append(detail_csv_header)

        for _, results in self.results.items():
            # detail csv
            for res in results:
                basic_info, compare_result_dict = res
                csv_row_basic_info = \
                    [basic_info.api_name, basic_info.bench_dtype, basic_info.tested_dtype, basic_info.shape]
                csv_row_compare_result = list(compare_result_dict.get(algorithm_name).compare_value \
                                            for algorithm_name in detail_csv_header_compare_result)
                csv_row_status = [basic_info.status, basic_info.err_msg]
                csv_row = csv_row_basic_info  + csv_row_compare_result + csv_row_status
                detail_csv.append(csv_row)

        file_name = os.path.join(csv_dir, add_time_as_suffix(MsCompareConst.DETAIL_CSV_FILE_NAME))
        create_directory(csv_dir)
        write_csv(detail_csv, file_name, mode="w")


    def to_result_csv(self, csv_dir):
        result_csv_dict = dict()
        for key, results in self.results.items():
            api_real_name, forward_or_backward = key
            forward_or_backward_pass_status = CompareConst.PASS
            forward_or_backward_overall_err_msg = ""
            # detail csv
            for res in results:
                basic_info, _ = res
                if basic_info.status != CompareConst.PASS:
                    forward_or_backward_pass_status = CompareConst.ERROR
                forward_or_backward_overall_err_msg += basic_info.err_msg
            forward_or_backward_overall_err_msg = \
                "" if forward_or_backward_pass_status == CompareConst.PASS else forward_or_backward_overall_err_msg

            #result_csv_dict
            if api_real_name not in result_csv_dict:
                result_csv_dict[api_real_name] = ResultCsvEntry()
            if forward_or_backward == Const.FORWARD:
                result_csv_dict[api_real_name].forward_pass_status = forward_or_backward_pass_status
                result_csv_dict[api_real_name].forward_err_msg = forward_or_backward_overall_err_msg
            else:
                result_csv_dict[api_real_name].backward_pass_status = forward_or_backward_pass_status
                result_csv_dict[api_real_name].backward_err_msg = forward_or_backward_overall_err_msg

        #result_csv
        result_csv = []
        result_csv_header = [
            MsCompareConst.DETAIL_CSV_API_NAME,
            MsCompareConst.RESULT_CSV_FORWARD_TEST_SUCCESS,
            MsCompareConst.RESULT_CSV_BACKWARD_TEST_SUCCESS,
            MsCompareConst.DETAIL_CSV_MESSAGE,
        ]
        result_csv.append(result_csv_header)

        for api_name, result_csv_entry in result_csv_dict.items():
            if result_csv_entry.forward_pass_status == CompareConst.PASS and \
                result_csv_entry.backward_pass_status == CompareConst.PASS:
                overall_err_msg = ""
            else:
                overall_err_msg = result_csv_entry.forward_err_msg + result_csv_entry.backward_err_msg
            row = [api_name, result_csv_entry.forward_pass_status,
                   result_csv_entry.backward_pass_status, overall_err_msg]
            result_csv.append(row)

        file_name = os.path.join(csv_dir, add_time_as_suffix(MsCompareConst.RESULT_CSV_FILE_NAME))
        create_directory(csv_dir)
        write_csv(result_csv, file_name, mode="w")