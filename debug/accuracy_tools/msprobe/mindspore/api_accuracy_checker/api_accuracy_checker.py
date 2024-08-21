from msprobe.core.common.file_check import FileOpen
from msprobe.core.common.const import Const
from msprobe.core.common.log import logger
from msprobe.mindspore.api_accuracy_checker.api_info import ApiInfo
from msprobe.mindspore.api_accuracy_checker.api_runner import api_runner, ApiInputAggregation
from msprobe.mindspore.api_accuracy_checker.const import MsApiAccuracyCheckerConst
from msprobe.mindspore.api_accuracy_checker.utils import check_and_get_from_json_dict, global_context
import json


class ApiAccuracyChecker:
    def __init__(self):
        self.api_infos = dict()
        self.results = None

    def parse(self, api_info_path):
        with FileOpen(api_info_path, "r") as f:
            api_info_dict = json.load(f)

        # init global context
        task = check_and_get_from_json_dict(api_info_dict, MsApiAccuracyCheckerConst.TASK_FIELD,
                                            "task field in api_info.json",accepted_type=str,
                                            accepted_value=(MsApiAccuracyCheckerConst.STATISTICS_TASK,
                                                            MsApiAccuracyCheckerConst.TENSOR_TASK))
        is_constructed = task == MsApiAccuracyCheckerConst.STATISTICS_TASK
        if not is_constructed:
            dump_data_dir = check_and_get_from_json_dict(api_info_dict, MsApiAccuracyCheckerConst.DUMP_DATA_DIR_FIELD,
                                                         "dump_data_dir field in api_info.json", accepted_type=str)
        else:
            dump_data_dir = ""
        global_context.init(is_constructed, dump_data_dir)

        api_info_data = check_and_get_from_json_dict(api_info_dict, MsApiAccuracyCheckerConst.DATA_FIELD,
                                                     "data field in api_info.json", accepted_type=dict)
        for api_name, api_info in api_info_data.items():
            forbackward_str = api_name.split(".")[-1]
            if forbackward_str not in (Const.FORWARD, Const.BACKWARD):
                logger.warning(f"api: {api_name} is not recognized as forward api or backward api, skip this.")
            api_name = Const.SEP.join(api_name.split(".")[:-1]) # www.xxx.yyy.zzz --> www.xxx.yyy
            if api_name not in self.api_infos:
                self.api_infos[api_name] = ApiInfo(api_name)

            if forbackward_str == Const.FORWARD:
                self.api_infos[api_name].load_forward_info(api_info)
            else:
                self.api_infos[api_name].load_backward_info(api_info)


    def run_and_compare(self):
        for api_name_str, api_info in self.api_infos:
            if not api_info.check_forward_info():
                logger.warning(f"api: {api_name_str} is lack of forward infomation skip checking")
                continue
            # first run forapi_infoward and compare
            inputs = api_info.get_compute_element_list(Const.FORWARD, Const.INPUT)
            kwargs =api_info.get_kwargs()

            if global_context.get_is_constructed():
                tested

    def record(self, compare_result_collection):
        pass


    def to_csv(self):
        pass