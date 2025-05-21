# Copyright (c) 2025-2025, Huawei Technologies Co., Ltd.
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
import time
import re
import atexit
from multiprocessing import Pool

import numpy as np
import mindspore as ms
from mindspore import nn, ops

from msprobe.mindspore.common.log import logger
from msprobe.core.common.const import Const as CoreConst
from msprobe.core.common.file_utils import load_npy, save_json, remove_path, load_yaml
from msprobe.core.common.const import FileCheckConst


CONSTRUCT_FILE_NAME = "construct.json"
DEFAULT_RANK_DIR = "rank0"
KEY_LAYERS = "layers"
construct = {}
cell_list = []
KEY_SIDE_EFFECT = "side_effect_io"
KEY_TOPLAYER = "TopLayer"
KEY_FORWARD = CoreConst.FORWARD
KEY_BACKWARD = CoreConst.BACKWARD
KEY_INPUT = CoreConst.INPUT
KEY_OUTPUT = CoreConst.OUTPUT
td = ops.TensorDump()
if (ms.__version__ >= "2.5.0"):
    td_in = ops.TensorDump("in")
else:
    td_in = ops.TensorDump()
gd = ops.DumpGradient()
td.add_prim_attr(KEY_SIDE_EFFECT, False)
td_in.add_prim_attr(KEY_SIDE_EFFECT, False)
np_ms_dtype_dict = {
    "bool": ms.bool_,
    "int8": ms.int8,
    "byte": ms.byte,
    "int16": ms.int16,
    "short": ms.short,
    "int32": ms.int32,
    "intc": ms.intc,
    "int64": ms.int64,
    "intp": ms.intp,
    "uint8": ms.uint8,
    "ubyte": ms.ubyte,
    "uint16": ms.uint16,
    "ushort": ms.ushort,
    "uint32": ms.uint32,
    "uintc": ms.uintc,
    "uint64": ms.uint64,
    "uintp": ms.uintp,
    "float16": ms.float16,
    "half": ms.half,
    "float32": ms.float32,
    "single": ms.single,
    "float64": ms.float64,
    "double": ms.double,
    "bfloat16": ms.bfloat16,
    "complex64": ms.complex64,
    "complex128": ms.complex128
}


def gen_file_path(dump_path, cell_prefix, suffix, io_type, index):
    step_path = os.path.join(dump_path, "{step}")
    rank_path = os.path.join(step_path, "{rank}")
    data_path = os.path.join(rank_path, CoreConst.DUMP_TENSOR_DATA)
    file_name = cell_prefix + CoreConst.SEP + suffix + CoreConst.SEP + io_type + CoreConst.SEP + str(index)
    return os.path.join(data_path, file_name)


def need_tensordump_in(cell_obj, attr, index):
    if not hasattr(cell_obj, attr):
        return False
    attr_values = getattr(cell_obj, attr)
    if index >= len(attr_values):
        return False
    return attr_values[index] == "in"


def cell_construct_wrapper(func, self):
    def new_construct(self, *args, **kwargs):
        new_args = []
        out_list = []

        index = 0
        item = None
        backward_or_all = self.data_mode in ["backward", "all"]
        forward_or_all = self.data_mode in ["forward", "all"]
        # The inputs of the cell.
        for index, item in enumerate(args):
            if backward_or_all and ops.is_tensor(item):
                if need_tensordump_in(self, 'input_dump_mode', index):
                    item = gd(gen_file_path(self.dump_path, self.cell_prefix, KEY_BACKWARD, KEY_OUTPUT, index),
                              item, "in")
                else:
                    item = gd(gen_file_path(self.dump_path, self.cell_prefix, KEY_BACKWARD, KEY_OUTPUT, index),
                              item, "out")
            if forward_or_all and ops.is_tensor(item):
                if need_tensordump_in(self, 'input_dump_mode', index):
                    temp = td_in(
                        gen_file_path(self.dump_path, self.cell_prefix, KEY_FORWARD, KEY_INPUT, index),
                        item
                    )
                else:
                    temp = td(
                        gen_file_path(self.dump_path, self.cell_prefix, KEY_FORWARD, KEY_INPUT, index),
                        item
                    )
                item = ops.depend(item, temp)
            new_args.append(item)

        out = func(*new_args, **kwargs)

        # The outputs of the cell.
        if isinstance(out, tuple):
            for index, item in enumerate(out):
                if backward_or_all and ops.is_tensor(item):
                    if need_tensordump_in(self, 'output_dump_mode', index):
                        item = gd(gen_file_path(self.dump_path, self.cell_prefix, KEY_BACKWARD, KEY_INPUT, index),
                                  item, "in")
                    else:
                        item = gd(gen_file_path(self.dump_path, self.cell_prefix, KEY_BACKWARD, KEY_INPUT, index),
                                  item, "out")
                if forward_or_all and ops.is_tensor(item):
                    if need_tensordump_in(self, 'output_dump_mode', index):
                        temp = td_in(
                            gen_file_path(self.dump_path, self.cell_prefix, KEY_FORWARD, KEY_OUTPUT, index),
                            item
                        )
                    else:
                        temp = td(
                            gen_file_path(self.dump_path, self.cell_prefix, KEY_FORWARD, KEY_OUTPUT, index),
                            item
                        )
                    item = ops.depend(item, temp)
                    out_list.append(item)
                elif forward_or_all and not ops.is_tensor(item):
                    out_list.append(item)
            out_list = tuple(out_list)
            return out_list
        else:
            if backward_or_all:
                if need_tensordump_in(self, 'output_dump_mode', index):
                    out = gd(gen_file_path(self.dump_path, self.cell_prefix, KEY_BACKWARD, KEY_INPUT, 0),
                             out, "in")
                else:
                    out = gd(gen_file_path(self.dump_path, self.cell_prefix, KEY_BACKWARD, KEY_INPUT, 0),
                             out, "out")
            if forward_or_all and ops.is_tensor(out):
                if need_tensordump_in(self, 'output_dump_mode', index):
                    temp = td_in(
                        gen_file_path(self.dump_path, self.cell_prefix, KEY_FORWARD, KEY_OUTPUT, 0),
                        out
                    )
                else:
                    temp = td(
                        gen_file_path(self.dump_path, self.cell_prefix, KEY_FORWARD, KEY_OUTPUT, 0),
                        out
                    )
                out = ops.depend(out, temp)
            return out

    return new_construct.__get__(self, type(self))


# 获取目录下所有文件名并根据TensorDump落盘自增id从小到大排序
def sort_filenames(path):
    filenames = os.listdir(path)
    id_pattern = re.compile(rf'{CoreConst.REPLACEMENT_CHARACTER}(\d+){CoreConst.NUMPY_SUFFIX}$')
    filenames.sort(key=lambda x: int(id_pattern.findall(x)[0]))
    return filenames


# 删除重复dump的文件：自定义文件名相同，并且数据相同
def del_same_file(path, filenames):
    result_list = []
    seen_prefixes = {}
    for current_filename in filenames:
        parts = current_filename.rsplit(CoreConst.REPLACEMENT_CHARACTER, 1)
        prefix = parts[0]
        if prefix not in seen_prefixes:
            result_list.append(current_filename)
            seen_prefixes[prefix] = current_filename
        else:
            current_file_path = os.path.join(path, current_filename)
            current_file = load_npy(current_file_path)
            prev_filename = seen_prefixes[prefix]
            prev_file_path = os.path.join(path, prev_filename)
            prev_file = load_npy(prev_file_path)
            if np.array_equal(current_file, prev_file):
                remove_path(current_file_path)
                logger.warning(f"{current_file_path} is deleted!")
            else:
                result_list.append(current_filename)
    return result_list


def rename_filename(path):
    filenames = sort_filenames(path)
    filenames = del_same_file(path, filenames)

    filename_dict = {}
    for filename in filenames:
        name_field = filename.rsplit(CoreConst.REPLACEMENT_CHARACTER, 1)[0]

        if name_field in filename_dict:
            filename_dict[name_field] += 1
        else:
            filename_dict[name_field] = 0

        cell_index = filename_dict[name_field]

        # 修改文件名，增加重复调用Cell的序号
        if CoreConst.FORWARD_PATTERN in filename:
            #Format: Cell.{cell_name}.{class_name}.{forward/backward}.{number}.{input/output}.{index}_{dtype}_{id}.npy
            newFileName = filename.replace(CoreConst.FORWARD_PATTERN, CoreConst.FORWARD_PATTERN + str(cell_index) + CoreConst.SEP)
        if CoreConst.BACKWARD_PATTERN in filename:
            newFileName = filename.replace(CoreConst.BACKWARD_PATTERN, CoreConst.BACKWARD_PATTERN + str(cell_index) + CoreConst.SEP)
        os.rename(os.path.join(path, filename), os.path.join(path, newFileName))
    logger.info(f"==========The rename_filename phase is Finished!==========")


# Extract the field between the first "." and the third to last ".", i.e. {cell_name}
def get_cell_name(str):
    parts = str.split(CoreConst.SEP)
    if len(parts) < 4:
        return None
    start_index = 1
    end_index = len(parts) - 3
    return CoreConst.SEP.join(parts[start_index:end_index])


# Extract the field between the last "." and the second to last ".", i.e. {data_made}
def get_data_mode(str):
    last_dot_index = str.rfind(CoreConst.SEP)
    second_last_dot_index = str.rfind(CoreConst.SEP, 0, last_dot_index)
    data_mode = str[second_last_dot_index + 1:last_dot_index]
    return data_mode


# 判断二者之间是否存在父子关系
def check_relation(cell_name, parent_cell_name):
    layers_pattern = rf"{CoreConst.SEP}{KEY_LAYERS}{CoreConst.SEP}\d+$"
    last_dot_index = cell_name.rfind(CoreConst.SEP)
    if last_dot_index == -1:
        return False
    # 如果cell_name最后一个'.'之前的字段等于parent_cell_name，则判定存在父子关系
    sub_cell_name = cell_name[:last_dot_index]
    if sub_cell_name == parent_cell_name:
        return True
    elif re.search(layers_pattern, cell_name):
        # 如果cell_name以".layer.{layer_id}"结尾，且去掉该字段后等于parent_cell_name，则判定存在父子关系
        sub_cell_name = re.sub(layers_pattern, '', cell_name)
        if sub_cell_name == parent_cell_name:
            return True
    return False


def get_construct(cell_list_input):
    for cell in cell_list_input:
        cell_name = get_cell_name(cell)
        cell_data_mode = get_data_mode(cell)
        found_flag = False
        for parent_cell in cell_list_input:
            parent_cell_name = get_cell_name(parent_cell)
            parent_data_mode = get_data_mode(parent_cell)
            has_relation = check_relation(cell_name, parent_cell_name)
            if has_relation and parent_data_mode == cell_data_mode:
                construct.update({cell: parent_cell})
                found_flag = True
                break
        if not found_flag:
            construct.update({cell: None})


def generate_construct(path):
    global construct
    filenames = sort_filenames(path)

    # 提取文件名中Cell.{cell_name}.{class_name}.{data_mode}.{重复调用此cell的序号}字段，并存入cell_list
    for filename in filenames:
        point_position = 3
        mid_field = filename.rsplit(CoreConst.SEP, point_position)[0]
        if KEY_INPUT in filename:
            if mid_field in cell_list:
                cell_list.remove(mid_field)
            cell_list.append(mid_field)
        else:
            if mid_field not in cell_list:
                index = filenames.index(filename)
                output_field = mid_field + KEY_OUTPUT
                find_flag = False
                for filename_other in cell_list[index + 1:]:
                    if output_field in filename_other:
                        find_flag = True
                if find_flag is False:
                    cell_list.append(mid_field)

    get_construct(cell_list)

    # 生成JSON文件
    rank_dir = os.path.dirname(path)
    json_path = os.path.join(rank_dir, CONSTRUCT_FILE_NAME)
    save_json(json_path, construct, indent=1)

    # 清空'construct'继续处理下一个路径下的数据
    construct = {}
    logger.info(f"Construct data saved to {json_path}")


def process_file(file_path):
    try:
        # 读取.npy文件内容
        npy_content = load_npy(file_path)
        logger.debug(f"Loaded {file_path}: shape is {npy_content.shape}, dtype is {npy_content.dtype}")

        # 文件名举例:Cell.network._backbone.loss.CrossEntropyLoss.forward.0.input.0_float32_165.npy
        parts = os.path.basename(file_path).split(CoreConst.SEP)
        data_dtype = ""
        # 获取0_float32_165或者0_in_float32_165中的float32
        data_dtype_list = parts[-2].split('_')
        if len(data_dtype_list) > 1:
            data_dtype = data_dtype_list[-2]
        # op_name是Cell.network._backbone.loss.CrossEntropyLoss.forward.0
        op_name = CoreConst.SEP.join(parts[:-3])
        ms_dtype = np_ms_dtype_dict.get(data_dtype)
        if ms_dtype is None:
            logger.warning(f"Get dtype None from file {file_path}")

        #修改落盘文件名字，去掉TensorDump自带的数据类型和自增id字段
        data_file_name = os.path.basename(file_path)
        data_file_dir = os.path.dirname(file_path)
        parts = data_file_name.split(CoreConst.SEP)
        if len(parts) >= 2:
            param_index = parts[-2].split(CoreConst.REPLACEMENT_CHARACTER)[0]
            pre_parts = CoreConst.SEP.join(parts[:-2])
            new_file_name = pre_parts + CoreConst.SEP + param_index + CoreConst.NUMPY_SUFFIX
            os.rename(os.path.join(data_file_dir, data_file_name), os.path.join(data_file_dir, new_file_name))
            logger.debug(f"{data_file_name} is renamed to {new_file_name}")
        else:
            logger.warning(f"Failed to rename {data_file_name}.")
            new_file_name = data_file_name

        tensor_json = {
            CoreConst.TYPE: 'mindspore.Tensor',
            CoreConst.DTYPE: str(ms_dtype),
            CoreConst.SHAPE: list(npy_content.shape),
            CoreConst.MAX: npy_content.max().item(),
            CoreConst.MIN: npy_content.min().item(),
            CoreConst.MEAN: npy_content.mean().item(),
            CoreConst.NORM: np.linalg.norm(npy_content).item(),
            CoreConst.DATA_NAME: new_file_name
        }

        # 根据文件名的最后一个部分（输入或输出）确定是添加到input_args还是output
        if parts[-3] == KEY_INPUT:
            return op_name, CoreConst.INPUT_ARGS, tensor_json
        elif parts[-3] == KEY_OUTPUT:
            return op_name, KEY_OUTPUT, tensor_json
        else:
            return None, None, None

    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None, None, None


def custom_sort(item, key_to_index):
    key = item[0]
    return key_to_index.get(key, float('inf'))


def generate_dump_info(path):
    if not os.path.exists(path):
        logger.error("The provided path does not exist.")
        return

    dump_data = {"task": "tensor", "level": "L0", "dump_data_dir": path, "data": {}}

    with Pool(processes=10) as pool:
        file_paths = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(FileCheckConst.NUMPY_SUFFIX):
                    file_paths.append((os.path.join(root, file),))
        file_paths.sort()
        results = pool.starmap(process_file, file_paths)

    # 收集结果
    for op_name, key, tensor_json in results:
        if op_name:
            if op_name not in dump_data.get(CoreConst.DATA, {}):
                dump_data.get(CoreConst.DATA, {})[op_name] = {CoreConst.INPUT_ARGS: [],
                                                              CoreConst.INPUT_KWARGS: {},
                                                              KEY_OUTPUT: []}
            if key not in dump_data.get(CoreConst.DATA, {}).get(op_name, {}):
                dump_data.get(CoreConst.DATA, {}).get(op_name, {})[key] = []
            dump_data.get(CoreConst.DATA, {}).get(op_name, {}).get(key, []).append(tensor_json)

    # 根据cell_list排序
    data_dict = dump_data.get(CoreConst.DATA, {})
    key_to_index = {key: index for index, key in enumerate(cell_list)}
    sorted_data_dict = dict(sorted(data_dict.items(), key=lambda item: custom_sort(item, key_to_index)))
    dump_data[CoreConst.DATA] = sorted_data_dict

    # 将数据写入dump.json
    json_path = os.path.join(os.path.dirname(path), 'dump.json')
    save_json(json_path, dump_data, indent=1)

    logger.info(f"Dump data saved to {json_path}")


def generate_stack_info(path):
    if not os.path.exists(path):
        logger.error("The provided path does not exist.")
        return

    stack_data = {}
    file_paths = []
    # 传入的path为工具生成的./dump_tensor_data，内容为npy文件
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(FileCheckConst.NUMPY_SUFFIX):
                file_paths.append(os.path.join(root, file))
    file_paths.sort()
    for file_path in file_paths:
        # 文件名举例:Cell.network._backbone.loss.CrossEntropyLoss.forward.0.input.0_float32_165.npy
        parts = os.path.basename(file_path).split(CoreConst.SEP)
        # op_name是Cell.network._backbone.loss.CrossEntropyLoss.forward.0
        op_name = CoreConst.SEP.join(parts[:-3])
        stack_data.update({op_name: []})

    # 将数据写入stack.json
    json_path = os.path.join(os.path.dirname(path), 'stack.json')
    save_json(json_path, stack_data, indent=1)

    logger.info(f"Stack data saved to {json_path}")


def is_download_finished(directory, interval=3):
    """
    判断指定目录在一段时间后是否有数据被下载完成
    :param directory: 指定目录的路径
    :param interval: 检查的时间间隔（秒），默认为 3 秒
    :return: 如有数据被下载完成返回 True，否则返回 False
    """
    # 检查目录是否存在
    if not os.path.exists(directory):
        logger.warning(f"The specified directory {directory} does not exist.")
        return False
    initial_modification_time = os.path.getmtime(directory)
    time.sleep(interval)
    current_modification_time = os.path.getmtime(directory)
    # 比较初始和当前修改时间
    if current_modification_time > initial_modification_time:
        return False
    else:
        return True


def process(dump_path):
    rank_id = os.environ.get('RANK_ID')
    rank_dir = DEFAULT_RANK_DIR
    if rank_id is not None:
        rank_dir = CoreConst.RANK + str(rank_id)

    step_dir_list = os.listdir(dump_path)
    for step_dir in step_dir_list:
        step_path = os.path.join(dump_path, step_dir)
        rank_path = os.path.join(step_path, rank_dir)
        npy_path = os.path.join(rank_path, CoreConst.DUMP_TENSOR_DATA)
        while True:
            is_finished = is_download_finished(npy_path)
            if not is_finished:
                logger.info(f"There is data being downloaded in the specified directory, continue checking...")
            else:
                logger.info(f"There is no data being downloaded in the specified directory, Stop checking.")
                break
        logger.info(f"==========Start processing data that has already been stored on the disk!==========")
        rename_filename(npy_path)
        generate_construct(npy_path)
        generate_dump_info(npy_path)
        generate_stack_info(npy_path)
        if rank_id is None:
            new_rank_path = os.path.join(step_path, CoreConst.RANK)
            try:
                os.rename(rank_path, new_rank_path)
                logger.info(f"Directory was successfully renamed to: {new_rank_path}")
            except Exception as e:
                logger.error(f"Error renamed to {new_rank_path}: {e}")
        logger.info(f"==========JSON file generation completed!==========")


def get_yaml_keys(yaml_data):
    keys = []
    for key, _ in yaml_data.items():
        keys.append(key)
    return keys


def get_tensordump_mode(input_str):
    left_index = input_str.find('(')
    right_index = input_str.find(')')

    # 提取括号内的字符串
    if left_index != -1 and right_index != -1:
        inner_str = input_str[left_index + 1:right_index]
        # 分割字符串得到元素列表
        elements = inner_str.split(',')
        if len(elements) >= 2:
            # 去除元素前后的空格
            first_element = elements[0].strip()
            second_element = elements[1].strip()
            return first_element, second_element
    return None, None


def str_to_list(input_str):
    # 去除首尾的方括号
    input_str = input_str.strip('[]')
    # 按逗号分割并去除元素两端的空格
    return [item.strip() for item in input_str.split(',')]


def set_tensordump_mode(cell, input_str):
    first_str, second_str = get_tensordump_mode(input_str)
    inputs_mode = []
    outputs_mode = []
    if first_str and second_str:
        inputs_mode = str_to_list(first_str)
        outputs_mode = str_to_list(second_str)
    if inputs_mode and outputs_mode:
        cell.input_dump_mode = inputs_mode
        cell.output_dump_mode = outputs_mode


def start(net=None, dump_path="./", data_mode=CoreConst.ALL):
    if net is None:
        return

    td_config_path = ""
    try:
        import mindformers
        mindformers_file = mindformers.__file__
        mindformers_dir = os.path.dirname(mindformers_file)
        td_config_path = os.path.join(mindformers_dir, "configuration", "layer_mapping.yaml")
    except ImportError:
        logger.warning("The configuration file in mindformers was not loaded, the default mode will be used.")

    if td_config_path == "":
        yaml_data = {}
    else:
        yaml_data = load_yaml(td_config_path)
    first_layer_key = get_yaml_keys(yaml_data)

    black_list = ["grad_reducer", ""]
    for name, cell in net.cells_and_names():
        class_name = cell.__class__.__name__
        # 跳过黑名单cell
        if name in black_list:
            logger.info(f"Cell {name}.{class_name} is skipped!")
            continue
        # 跳过框架内部的cell
        if class_name.startswith(CoreConst.REPLACEMENT_CHARACTER):
            logger.info(f"Cell {name}.{class_name} is skipped!")
            continue
        else:
            #Format: Cell.{cell_name}.{class_name}
            cell.cell_prefix = CoreConst.SEP.join([CoreConst.CELL, name, cell.__class__.__name__])

        # 根据yaml配置文件设置cell的TensorDump模式
        if class_name in first_layer_key:
            layer_data = yaml_data.get(class_name)
            if layer_data:
                for child_name, child_cell in cell.cells_and_names():
                    if child_name in layer_data:
                        set_tensordump_mode(child_cell, layer_data[child_name])
        top_layer_data = yaml_data.get(KEY_TOPLAYER)
        if top_layer_data and name in top_layer_data:
            set_tensordump_mode(cell, top_layer_data[name])

        # 替换construct函数
        cell.construct = cell_construct_wrapper(cell.construct, cell)
        logger.info(f"Cell {name}: construct function is wrapped!")
        cell.dump_path = dump_path
        cell.data_mode = data_mode

    logger.info(f"==========The cell_dump_process_start phase is Finished!==========")
    atexit.register(process, dump_path=dump_path)