import os
import csv

from msprobe.core.common.file_utils import change_mode, FileOpen
from msprobe.core.common.log import logger
from msprobe.core.common.const import Const, FileCheckConst
from msprobe.core.common.file_utils import remove_path, load_json, save_json


class DataWriter:

    def __init__(self, init_json=None) -> None:
        self.dump_count = 0
        self.init_json = init_json
        self.dump_file_path = None  # os.path.join(dump_dir, DataWriter.dump_json_name)
        self.stack_file_path = None  # os.path.join(dump_dir, DataWriter.stack_json_name)
        self.construct_file_path = None  # os.path.join(dump_dir, DataWriter.construct_json_name)
        self.free_benchmark_file_path = None
        self.dump_tensor_data_dir = None
        self.buffer_size = 1000
        self.cache_data = {Const.DATA: {}}
        self.cache_stack = {}
        self.cache_construct = {}

    @staticmethod
    def write_data_to_csv(result: list, result_header: tuple, file_path: str):
        if not result:
            return
        is_exists = os.path.exists(file_path)
        append = "a+" if is_exists else "w+"
        with FileOpen(file_path, append) as csv_file:
            spawn_writer = csv.writer(csv_file)
            if not is_exists:
                spawn_writer.writerow(result_header)
            spawn_writer.writerows([result,])
        is_new_file = not is_exists
        if is_new_file:
            change_mode(file_path, FileCheckConst.DATA_FILE_AUTHORITY)

    def initialize_json_file(self, **kwargs):
        kwargs.update({"dump_data_dir": self.dump_tensor_data_dir, Const.DATA: {}})
        save_json(self.dump_file_path, kwargs)

        empty_dict = {}
        remove_path(self.stack_file_path)
        save_json(self.stack_file_path, empty_dict)

        remove_path(self.construct_file_path)
        save_json(self.construct_file_path, empty_dict)

    def update_dump_paths(self, dump_file_path, stack_file_path, construct_file_path, dump_data_dir, 
                          free_benchmark_file_path):
        self.dump_file_path = dump_file_path
        self.stack_file_path = stack_file_path
        self.construct_file_path = construct_file_path
        self.dump_tensor_data_dir = dump_data_dir
        self.free_benchmark_file_path = free_benchmark_file_path

    def update_data(self, new_data):
        key = next(iter(new_data.keys()))  # assert len(new_data.keys()) == 1
        if key in self.cache_data[Const.DATA]:
            self.cache_data[Const.DATA][key].update(new_data[key])
        else:
            self.cache_data[Const.DATA].update(new_data)

    def flush_data_when_buffer_is_full(self):
        if len(self.cache_data[Const.DATA]) >= self.buffer_size:
            self.write_data_json(self.dump_file_path)

    def update_stack(self, new_data):
        self.cache_stack.update(new_data)

    def update_construct(self, new_data):
        self.cache_construct.update(new_data)

    def write_data_json(self, file_path):
        logger.info(f"dump.json is at {os.path.dirname(os.path.dirname(file_path))}. ")
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            data_to_write = load_json(file_path)
        else:
            self.init_json['data_path'] = self.dump_tensor_data_dir
            data_to_write = self.init_json
        data_to_write[Const.DATA].update(self.cache_data[Const.DATA])
        save_json(file_path, data_to_write, indent=1)
        self.cache_data[Const.DATA].clear()

    def write_stack_info_json(self, file_path):
        save_json(file_path, self.cache_stack, indent=1)

    def write_construct_info_json(self, file_path):
        save_json(file_path, self.cache_construct, indent=1)

    def write_json(self):
        self.write_data_json(self.dump_file_path)
        self.write_stack_info_json(self.stack_file_path)
        self.write_construct_info_json(self.construct_file_path)
