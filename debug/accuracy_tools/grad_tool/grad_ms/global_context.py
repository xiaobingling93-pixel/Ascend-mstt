import os
import threading
from typing import Dict, List, Union

from grad_tool.common.utils import print_warn_log
from grad_tool.common.constant import GradConst
from grad_tool.common.utils import path_valid_check, create_directory


class GlobalContext:

    _instance = None
    _instance_lock = threading.Lock()
    _setting = {
        GradConst.LEVEL: GradConst.LEVEL0,
        GradConst.PARAM_LIST: None,
        GradConst.CURRENT_STEP: 0,
        GradConst.BOUNDS: [-1., 0., 1.],
        GradConst.OUTPUT_PATH: "./grad_stat"
    }

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance_lock.acquire()
            cls._instance = object.__new__(cls)
            cls._instance_lock.release()
        return cls._instance

    def init_context(self, config_dict: Dict):
        if config_dict.get(GradConst.LEVEL, None) in GradConst.SUPPORTED_LEVEL:
            self._setting[GradConst.LEVEL] = config_dict.get(GradConst.LEVEL)
        else:
            print_warn_log("Invalid level set in config yaml file, use L0 instead.")
        self._set_input_list(config_dict, GradConst.PARAM_LIST, str)
        self._set_input_list(config_dict, GradConst.BOUNDS, float)
        output_path = config_dict.get(GradConst.OUTPUT_PATH)
        if output_path:
            try:
                path_valid_check(output_path)
            except RuntimeError as err:
                print_warn_log(f"Invalid output_path, use default output_path. The error message is {err}.")
                output_path = None
        if output_path:
            self._setting[GradConst.OUTPUT_PATH] = output_path
        if not os.path.isdir(self._setting.get(GradConst.OUTPUT_PATH)):
            create_directory(self._setting.get(GradConst.OUTPUT_PATH))
        else:
            print_warn_log("The output_path exists, the data will be covered.")

    def get_context(self, key: str):
        if key not in self._setting:
            print_warn_log(f"Unrecognized {key}.")
        return self._setting.get(key)

    def update_step(self):
        self._setting[GradConst.CURRENT_STEP] += 1

    def _set_input_list(self, config_dict: Dict, name: str, dtype: Union[int, str, float]):
        value = config_dict.get(name)
        if dtype == int:
            type_str = "integer"
        elif dtype == float:
            type_str = "float"
        else:
            type_str = "string"
        if value and isinstance(value, list):
            for val in value:
                if not isinstance(val, dtype):
                    print_warn_log(f"Invalid {name} which must be None or list of {type_str}")
                    return
            self._setting[name] = value
        else:
            print_warn_log(f"{name} is None or not a list with valid items, use default value.")

    def step_need_dump(self, step):
        dump_step_list = self.get_context(GradConst.STEP)
        return (not dump_step_list) or (step in dump_step_list)

    def rank_need_dump(self, rank):
        dump_rank_list = self.get_context(GradConst.RANK)
        return (not dump_rank_list) or (rank in dump_rank_list)


grad_context = GlobalContext()
