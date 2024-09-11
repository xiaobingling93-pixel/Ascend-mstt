import json

from msprobe.core.common.exceptions import ParseJsonException
from msprobe.core.common.file_utils import FileOpen


def parse_json_info_forward_backward(json_path):
    def parse_data_name_with_pattern(data_name, pattern):
        name_struct = data_name.split('.')
        if not name_struct[-1] == pattern:
            raise ParseJsonException(ParseJsonException.UnexpectedNameStruct,
                f"{data_name} in file {json_path}")
        api_name = '.'.join(name_struct[:-1])
        return api_name

    with FileOpen(json_path, 'r') as f:
        dump_json = json.load(f)

    real_data_path = dump_json.get("dump_data_dir")
    dump_data = dump_json.get("data")
    if not dump_data:
        raise ParseJsonException(ParseJsonException.InvalidDumpJson, "dump数据中没有data字段")

    forward_data = {}
    backward_data = {}
    for data_name, data_item in dump_data.items():
        if "Module" in data_name:
            continue
        if "forward" in data_name:
            api_name = parse_data_name_with_pattern(data_name, "forward")
            forward_data.update({api_name: data_item})
        elif "backward" in data_name:
            api_name = parse_data_name_with_pattern(data_name, "backward")
            backward_data.update({api_name: data_item})
        else:
            raise ParseJsonException(ParseJsonException.UnexpectedNameStruct,
                f"{data_name} in file {json_path}.")

    return forward_data, backward_data, real_data_path
