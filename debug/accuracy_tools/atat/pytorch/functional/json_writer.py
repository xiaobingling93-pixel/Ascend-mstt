import os
from pathlib import Path
import json
from ..common.log import print_info_log_rank_0


class DataWriter:  # TODO: UT
    # dump_json_name = "dump.json"
    # stack_json_name = "stack.json"
    # construct_json_name = "construct.json"

    def __init__(self, init_json=None) -> None:
        self.dump_count = 0
        self.init_json = init_json
        self.dump_file_path = None  # os.path.join(dump_dir, DataWriter.dump_json_name)
        self.stack_file_path = None  # os.path.join(dump_dir, DataWriter.stack_json_name)
        self.construct_file_path = None  # os.path.join(dump_dir, DataWriter.construct_json_name)
        self.dump_tensor_data_dir = None
        self.buffer_size = 1000
        self.cache_data = {"data": {}}
        self.cache_stack = {}
        self.cache_construct = {}

    def initialize_json_file(self, **kwargs):
        kwargs.update({"dump_data_dir": self.dump_tensor_data_dir, "data": {}})
        with open(self.dump_file_path, 'w') as f:
            json.dump(kwargs, f)

        if os.path.exists(self.stack_file_path):
            os.remove(self.stack_file_path)
        Path(self.stack_file_path).touch()

        if os.path.exists(self.construct_file_path):
            os.remove(self.construct_file_path)
        Path(self.construct_file_path).touch()

    def update_dump_paths(self, dump_file_path, stack_file_path, construct_file_path, dump_data_dir):
        self.dump_file_path = dump_file_path
        self.stack_file_path = stack_file_path
        self.construct_file_path = construct_file_path
        self.dump_tensor_data_dir = dump_data_dir

    def update_data(self, new_data):
        key = next(iter(new_data.keys()))  # assert len(new_data.keys()) == 1
        if key in self.cache_data["data"]:
            self.cache_data["data"][key].update(new_data[key])
        else:
            self.cache_data["data"].update(new_data)

    def flush_data_when_buffer_is_full(self):
        if len(self.cache_data["data"]) >= self.buffer_size:
            self.write_data_json(self.dump_file_path)

    def update_stack(self, new_data):
        self.cache_stack.update(new_data)

    def update_construct(self, new_data):
        self.cache_construct.update(new_data)

    def write_data_json(self, file_path):
        import fcntl
        print_info_log_rank_0(f"dump.json is at {os.path.dirname(os.path.dirname(file_path))}. ")
        if Path(file_path).exists() and os.path.getsize(file_path) > 0:
            with open(file_path, "r+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                data_to_write = json.load(f)
                fcntl.flock(f, fcntl.LOCK_UN)
        else:
            self.init_json['data_path'] = self.dump_tensor_data_dir
            data_to_write = self.init_json
        data_to_write['data'].update(self.cache_data['data'])
        with open(file_path, 'w+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(data_to_write, f, indent=1)
            fcntl.flock(f, fcntl.LOCK_UN)

        self.cache_data["data"].clear()

    def write_stack_info_json(self, file_path):
        import fcntl
        with open(file_path, 'w+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(self.cache_stack, f, indent=1)
            fcntl.flock(f, fcntl.LOCK_UN)

    def write_construct_info_json(self, file_path):
        import fcntl
        with open(file_path, 'w+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(self.cache_construct, f, indent=1)
            fcntl.flock(f, fcntl.LOCK_UN)

    def write_json(self):
        self.write_data_json(self.dump_file_path)
        self.write_stack_info_json(self.stack_file_path)
        self.write_construct_info_json(self.construct_file_path)