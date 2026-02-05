# Copyright (c) 2025, Huawei Technologies Co., Ltd.
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

import glob
import json
import os
import re
from pathlib import Path
from typing import List

from tinker.utils.constant import Version, version_parse
from tinker.utils.logger import logger
from tinker.version import optimizer_version


def project_root():
    """
    获取tinker父目录
    Returns:
    """
    # 一般不会动，写死
    root_path = os.path.abspath(__file__)
    for _ in range(3):
        root_path = os.path.dirname(root_path)
    return root_path


def read_file(file_path: str):
    """
    读取文件
    :param file_path 文件路径
    Returns: 字符串
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def extract_line(reserve_str, content):
    """
    提取content中以reserve_str开头的指定行
    :param reserve_str 起始字符串
    :param content 待识别内容
    Returns:
    """
    res = list()
    lines = content.splitlines()
    for line in lines:
        if line.startswith(reserve_str):
            res.append(line)
    return res


def extract_line_ignore_blank(reserve_str, content):
    """
    提取content中以reserve_str开头的指定行（忽略空格）
    :param reserve_str 起始字符串
    :param content 待识别内容
    Returns:
    """
    res = list()
    lines = content.splitlines()
    for line in lines:
        if line.lstrip().startswith(reserve_str):
            res.append(line)
    return res


def extract_between(start_word, end_word, content):
    """
    提取 start_word 和 end_word 之间的字符串（包含start_word和end_word）
    :param start_word 起始字符串
    :param end_word 结束字符串
    :param content 待识别内容
    Returns:
    """

    # 正则表达式匹配 start_word 和 end_word 之间的内容，包括这两个标记
    pattern = re.escape(start_word) + r'(.*?)' + re.escape(end_word)

    # 使用 re.search 查找第一个匹配项
    match = re.search(pattern, content, re.DOTALL)

    if match:
        # 返回匹配到的内容
        return match.group(0)
    else:
        # 如果没有找到匹配项，返回 None
        return None


def del_line(del_params: list, content: str):
    """
    删除content中以del_params开头的内容
    :param del_params 待删除关键字
    :param content 给定内容
    Returns:
    """
    # 按行分割字符串
    lines = content.splitlines()

    # 过滤掉以 start_strs 中任意一个字符串开头的行
    filtered_lines = [line for line in lines if not any(start_str in line for start_str in del_params)]

    # 将过滤后的行重新组合成字符串
    return '\n'.join(filtered_lines)


def del_content(start_word, end_word, content: str):
    """
    删除content 中 start_word 开始到 end_word结束的内容，不能是同一行（最相邻原则）
    :param start_word:
    :param end_word:
    :param content:
    :return:
    """
    lines = content.splitlines()
    start_idx = -1
    end_idx = -1
    del_idx_pair = None
    for i, line in enumerate(lines):
        if line.startswith(start_word):
            start_idx = i
            continue
        if line.startswith(end_word):
            end_idx = i
            if start_idx != -1:
                del_idx_pair = (start_idx, end_idx)
                break
            continue

    if del_idx_pair is None:
        raise RuntimeError('cannot find del word pair in content')

    del lines[del_idx_pair[0]: del_idx_pair[1] + 1]

    return '\n'.join(lines)


def write_lines(final_res: list, dest_file: str):
    """
    写入指定内容至文件
    :param final_res 待写入内容
    :param dest_file 写入路径
    Returns:
    """
    try:
        with open(dest_file, 'w', encoding='utf-8') as file:
            for line in final_res:
                file.write(line + '\n')
    except Exception as e:
        raise RuntimeError(f'write to file: {dest_file} failed.') from e


def load_infos(args):
    """读取前序流程保存的模型结构等信息，并保存到全局变量args中"""
    model_info = find_files(args.profiled_data_path, 'model_info*.json')
    if not model_info:
        logger.info('model_info未找到，seq_length取4096')
        args.seq_length = 4096
    else:
        with open(model_info, 'r') as file:
            data = json.load(file)
        for k, v in data.items():
            if k == 'num_layers' and v == 1:  # 留None 避免存下减层后脚本跑出来的num_layers: 1
                continue
            setattr(args, k, v)

    # 记录使用测量数据基于的感知器版本
    task_info = find_files(args.profiled_data_path, 'VERSION*.json')
    if task_info:
        with open(task_info, 'r') as file:
            data = json.load(file)
        if 'version_profiler' not in data:
            args.version_profiler = data['version']
            args.version_framework = Version.MindSpeed_LLM_1_0_rc3
        else:
            args.version_profiler = data['version_profiler']
        args.model_name = data.get('model_name')
        args.model_size = data.get('model_size')
        if args.pretrain_script_path_search is None:
            args.pretrain_script_path = data.get('pretrain_script_path')
    args.version_optimizer = optimizer_version()


def find_files(dir_path, pattern):
    load_path = os.path.join(dir_path, pattern)
    files = glob.glob(load_path)
    if files:
        return files[0]
    return None


def extract_and_format_model_size(model_size: str):
    """
    提取模型尺寸中的数字部分，可能是小数（除去b\B）
    :param model_size 用户输入的模型尺寸，待统一化
    Returns:
    """
    model_size_search = re.search(r'\d+(?:\.\d+)?[bB]?', model_size)
    if model_size_search is None:
        raise RuntimeError(f'The model size {model_size} is not valid, accept pattern like xxb, xxB or xx.')
    # 这里除去b\B
    model_size = model_size_search.group(0)[:-1]
    return f'{model_size}b'


def byte_to_mb(x):
    """
    将以字节为单位的内存尺寸转换为MB为单位的内存尺寸
    :param x: 内存开销(Bytes)
    :return: 内存开销(MB)
    """
    return x / 1024.0 / 1024.0


def find_keywords_line_idx(source_code: str, key_word: str):
    """
    提取 source_code 中 key_word 所在行号的列表
    :param source_code 用户输入的模型尺寸，待统一化
    :param key_word 用户输入的模型尺寸，待统一化
    Returns: line 索引列表
    """
    lines = source_code.splitlines()
    res = []
    # 遍历每一行，查找关键字
    for line_idx, line in enumerate(lines):
        if key_word in line:
            res.append(line_idx)
    if not res:
        raise RuntimeError(f'Cannot find key word: {key_word} in source code')
    return res


def get_lines(module_code: str, start_idx: int, end_idx: int):
    """
    获取 module_code 中指定起止位置代码
    :param module_code: 给定代码段
    :param start_idx: 给起始点
    :param end_idx: 给截止点
    :return: 区间代码段
    """
    # 提取module_code中第start_idx+1行 到 end_idx+1行的内容（左闭右开）
    lines = module_code.splitlines()

    # 截取第 i 行到第 j 行的内容，注意切片的结束索引是 j + 1
    selected_lines = lines[start_idx:end_idx]  # 假设传入的行号是从 1 开始的

    # 将截取的行重新连接成一个字符串，使用换行符分隔
    return '\n'.join(selected_lines)


def path_to_package(file_system_path):
    """
    将路径形式转为包形式
    :param file_system_path: 给定路径
    :return: 包形式字符串
    """
    # 创建 Path 对象
    path = Path(file_system_path)
    # 使用 parts 属性获取路径的各个部分
    parts = path.parts
    # 使用 str.join 方法将部分拼接成包路径
    package_path = '.'.join(parts)
    return package_path


def extract_arg_value_from_json(json_path: str):
    """提取配置文件parameter_config.json中的参数值"""
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data


def check_path_exist(path: str):
    """校验路径存在"""
    if not os.path.exists(path):
        logger.error(f'The file path {path} does not exist.')
        raise Exception


def check_path_type(path: str, path_type: str):
    """校验路径类型,文件or目录"""
    if type == 'file':
        if not os.path.isfile(path):
            logger.error(f'The {path} should be a file!')
            raise Exception
    if type == 'dir':
        if not os.path.isdir(path):
            logger.error(f'The {path} should be a directory!')
            raise Exception


def check_file_suffix(path: str, suffix: str):
    """校验文件类型"""
    if suffix:
        if not path.endswith(suffix):
            logger.error('The {path} should be a {suffix} file!')
            raise Exception


def check_path_before_create(path: str):
    """创建目录/文件前的路径校验"""
    parent_dir = os.path.dirname(path)
    check_path_exist(parent_dir)
    check_path_type(parent_dir, 'dir')


def check_files_in_dir(path: str):
    """校验目录下存在文件"""
    if os.path.isdir(path) and len(os.listdir(path)) == 0:
        logger.error(f'No files in {path}')
        raise Exception


def convert_to_pp_stage_block_idx(num_layer_list: List[int], num_all_blocks_len: int):
    """
    格式转换
    :param num_layer_list: 一种可能的划分方式, num_layer_list中的元素为每个stage的长度
    :param num_all_blocks_len: 加上头尾blocks的长度
    :return:
    """
    interval_layer_list = list()
    start_num = 1
    for stage_length in num_layer_list:
        interval_layer_list.append((start_num, start_num + stage_length - 1))
        start_num += stage_length
    # 处理首尾
    first_tuple = interval_layer_list[0]
    interval_layer_list[0] = (0, first_tuple[1])
    last_tuple = interval_layer_list[-1]
    interval_layer_list[-1] = (last_tuple[0], num_all_blocks_len - 1)
    return interval_layer_list


def convert_to_num_layers(interval_layer_list):
    num_layer_list = [interval[1] - interval[0] + 1 for interval in interval_layer_list]
    num_layer_list[0] -= 1
    num_layer_list[-1] -= 2
    num_layers = ','.join(map(str, num_layer_list))
    return num_layers
