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
import time
import glob
from typing import Dict, List
from pathlib import Path

import pandas as pd

from msprobe.core.common.const import Const
from msprobe.core.common.file_utils import (
    check_file_or_directory_path,
    FileOpen,
    create_directory,
    write_csv,
    check_path_before_create,
    read_csv,
    write_df_to_csv
)
from msprobe.mindspore.code_mapping.graph import GraphNode
from msprobe.mindspore.common.log import logger


# 定义Trie节点
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_key = False
        self.value = None


# 定义Trie树
class Trie:
    def __init__(self):
        self.root = TrieNode()

    # 向Trie中插入一个键
    def insert(self, key, value):
        node = self.root
        for key_char in key:
            if key_char not in node.children:
                node.children[key_char] = TrieNode()
            node = node.children[key_char]
        # 标记结束位置
        node.is_end_of_key = True
        node.value = value

    # 在name字符串中查找所有匹配的键
    def search_in_string(self, string):
        matched_values = []
        for i in range(len(string)):
            node = self.root
            j = i
            # 从字符串的每个字符开始，逐字符查找匹配
            while j < len(string) and string[j] in node.children:
                node = node.children[string[j]]
                if node.is_end_of_key:
                    matched_values.append(node.value)
                j += 1
        return matched_values


# 定义匹配函数
def match_codes(trie, name):
    matched_nodes = trie.search_in_string(name)
    matched_codes = [Const.NEW_LINE.join(node.code_info) for node in matched_nodes]
    return Const.NEW_LINE.join(matched_codes)


def match_names(trie, name):
    matched_nodes = trie.search_in_string(name)
    matched_names = [node.scope for node in matched_nodes]
    return Const.NEW_LINE.join(matched_names)


def map_op_names_to_codes_and_scopes(df, match_dict):
    # 构建Trie树并插入所有键
    trie = Trie()
    for key, value in match_dict.items():
        trie.insert(key, value)

    df[Const.CODE_STACK] = df[Const.OP_NAME].apply(lambda name: match_codes(trie, name))
    df[Const.SCOPE_NAME] = df[Const.OP_NAME].apply(lambda name: match_names(trie, name))
    return df


def find_npy_files(npy_path):
    """
    查找指定路径下所有的.npy文件。

    Parameters:
        npy_path (str): 搜索的路径，可以是文件或目录。

    Returns:
        List[Path]: 找到的.npy文件路径列表。
    """
    npy_files = []
    npy_path_obj = Path(npy_path)

    # 检查当前路径是否是一个以 .npy 结尾的文件
    if npy_path_obj.suffix == Const.NUMPY_SUFFIX and npy_path_obj.is_file():
        check_file_or_directory_path(npy_path_obj)
        npy_files.append(npy_path_obj.resolve())
        return npy_files

    # 如果是目录，使用Path.rglob查找所有.npy文件
    if npy_path_obj.is_dir():
        base_depth = len(npy_path_obj.resolve().parts)
        for root, dirs, files in os.walk(npy_path_obj):
            current_depth = len(Path(root).resolve().parts) - base_depth
            if current_depth >= 10:
                dirs[:] = []

            for filename in files:
                if filename.endswith(Const.NUMPY_SUFFIX):
                    file_path = Path(root) / filename
                    check_file_or_directory_path(file_path)
                    npy_files.append(file_path.resolve())
    else:
        logger.info(f"The specified path is neither an .npy file nor a directory: {npy_path}")

    return npy_files


def write_to_csv(param: Dict, output_dir: str):
    """
    将参数写入CSV文件。

    Parameters:
        param (Dict): 要写入的数据，格式为{文件名: (代码堆栈, 作用域名称)}。
        output_dir (str): 输出目录路径。
    """
    create_directory(output_dir)

    # 使用时间戳生成文件名
    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    file_path = Path(output_dir) / f"code_mapping_{timestamp}.csv"
    check_path_before_create(file_path)
    data = [(name, res1, res2) for name, (res1, res2) in param.items()]
    df = pd.DataFrame(data, columns=[Const.FILE_PATH, Const.CODE_STACK, Const.SCOPE_NAME])
    write_df_to_csv(df, file_path)


def find_statistic_files(path):
    if not os.path.isdir(path):
        if os.path.basename(path) == 'statistic.csv':
            return [path]
        else:
            return []
    pattern = os.path.join(path, '**', "statistic.csv")

    statistic_files = list(glob.glob(pattern, recursive=True))
    return statistic_files


def check_and_fix_header(file_path: str):
    """
    检查 CSV 文件的表头是否以逗号结尾，如果没有则添加一个逗号。

    Parameters:
        file_path (str): CSV 文件的路径。

    Returns:
        bool: 如果表头被修改，返回 True；否则，返回 False。
    """

    with FileOpen(file_path, "r") as f:
        lines = f.readlines()

    if not lines:
        logger.warning(f"The file {file_path} is empty.")
        return False

    # 获取表头并去除末尾的换行符
    header = lines[0].rstrip(Const.NEW_LINE).rstrip('\r')

    if not header.endswith(','):
        logger.info(f"The header does not end with a comma. Adding a comma to the file: {file_path}.")
        # 添加逗号并恢复换行符
        lines[0] = header + Const.CSV_NEWLINE_SEPARATOR

        # 写回修复后的内容到文件
        with FileOpen(file_path, "w") as f:
            f.writelines(lines)
        logger.info(f"Added a trailing comma to the file: {file_path}.")
        return True
    else:
        logger.info(f"The header already ends with a comma. No modification needed for the file: {file_path}.")
        return False


def bind_for_statistic(statistic_files: List[str], match_dict: Dict):
    """
    处理统计文件，绑定代码信息。

    Parameters:
        statistic_files (List[str]): 统计文件路径列表。
        match_dict (Dict): 匹配字典，用于复杂映射。
    """
    for statistic_file in statistic_files:
        # 使用FileOpen安全打开文件
        header_modified = check_and_fix_header(statistic_file)
        if header_modified:
            logger.info(f"The header of the file {statistic_file} has been fixed.")

        df = read_csv(statistic_file, as_pd=True)

        # 进行复杂映射
        df = map_op_names_to_codes_and_scopes(df, match_dict)

        # 使用write_csv安全写入文件
        write_df_to_csv(df, statistic_file)


def bind_code_info_for_data(input_dir: str, nodes: Dict[str, GraphNode]) -> Dict[str, str]:
    # 待重构后优化性能
    match_dict = {}
    for node in nodes.values():
        # 屏蔽子图节点
        if node.is_subgraph:
            continue
        # 获取规范化后的scope name
        scope_name = node.scope.replace(Const.SCOPE_SEPARATOR, Const.REPLACEMENT_CHARACTER)
        match_dict[scope_name] = node
    npy_files = find_npy_files(input_dir)

    bind_result = {}
    if not npy_files:
        statistic_files = find_statistic_files(input_dir)
        if statistic_files:
            bind_for_statistic(statistic_files, match_dict)
        return bind_result

    for npy_file in npy_files:
        directory, file_name = os.path.split(npy_file)  # 拆分路径
        name_without_ext = os.path.splitext(file_name)[0]  # 提取文件名（去掉扩展名）
        if name_without_ext.isdigit():
            # 3. 读取find.csv文件
            csv_file_path = os.path.join(directory, 'mapping.csv')
            check_file_or_directory_path(csv_file_path)
            df = read_csv(csv_file_path, header=None)

            # 4. 查找是否有与xxx.npy匹配的条目
            matching_row = df[df[0] == file_name]  # 假设A列存储文件名
            if not matching_row.empty:
                corresponding_name = matching_row[1].values[0]
            else:
                corresponding_name = None
            name_without_ext = os.path.splitext(corresponding_name)[0]
        npy_path = os.path.realpath(npy_file)

        parts = name_without_ext.split(".")
        if len(parts) < 2:
            logger.error(
                f'File name "{file_name}" in "{directory}" '
                f'does not conform to expected format (missing scope separator ".")!'
            )
            raise Exception(
                f'File name "{file_name}" has incorrect format, cannot extract node scope!'
            )
        node_scope = parts[1]

        trie = Trie()
        for key, value in match_dict.items():
            trie.insert(key, value)
        bind_code = match_codes(trie, node_scope)
        bind_name = match_names(trie, node_scope)
        bind_result[npy_path] = (bind_code, bind_name)
    return bind_result
