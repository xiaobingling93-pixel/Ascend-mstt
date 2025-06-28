# Copyright (c) 2025, Huawei Technologies.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import math
import os
import json
import re
import stat
import sys
from functools import cmp_to_key
from pathlib import Path
from tensorboard.util import tb_logging
from .global_state import GraphState, FILE_NAME_REGEX, MAX_FILE_SIZE

logger = tb_logging.get_logger()
FILE_PATH_MAX_LENGTH = 4096


class GraphUtils:

    @staticmethod
    def get_graph_data(meta_data):
        if not meta_data:
            return None, 'Error: no query parameters provided'
        try:
            run_name = meta_data.get('run')
            runs = GraphState.get_global_value('runs', {})
            run = runs.get(run_name) or run_name
            tag = meta_data.get('tag')
            current_tag = GraphState.get_global_value('current_tag')
            current_run = GraphState.get_global_value('current_run')
            if current_tag == tag and current_run == run:
                return GraphState.get_global_value('current_file_data'), None  # 直接返回获取结果
            else:
                json_data, error_message = GraphUtils.safe_load_data(run, f"{tag}.vis", False)
                if error_message:
                    return None, error_message
                GraphState.set_global_value('current_file_data', json_data)
                GraphState.set_global_value('current_tag', tag)
                GraphState.set_global_value('current_run', run)
                return json_data, error_message
        except Exception as e:
            logger.error(f'Error: fail to get graph data by {meta_data}, error: {e}')
            return None, 'Error: fail to get graph data'

    @staticmethod
    def get_parent_node_list(graph_data, node_name):
        """获取父节点列表"""
        # 如果 graph_data 为空或 node_name 不存在，直接返回空列表
        if not graph_data or not node_name:
            return []

        node_list = []  # 存储结果的列表
        current_node = node_name  # 当前节点名称
        nodes = graph_data['node']  # 提取节点数据，避免重复访问

        while current_node:
            node_list.append(current_node)  # 将当前节点加入列表
            # 获取当前节点的 upnode
            current_node = nodes.get(current_node, {}).get('upnode')

            # 检测循环引用（防止死循环）
            if current_node in node_list:
                raise ValueError(f"检测到循环引用：节点 {current_node} 已存在于路径中")

        return list(reversed(node_list))  # 返回结果列表

    @staticmethod
    def split_graph_data_by_microstep(graph_data, micro_step):
        graph_nodes = graph_data.get('node', {})
        if str(micro_step) == str(-1):
            return graph_nodes
        splited_graph_data = {}

        def traverse_npu(graph_nodes, subnodes):
            for node_name in subnodes:
                node_data = graph_nodes.get(node_name, {})
                micro_step_id = node_data.get('micro_step_id')
                if str(micro_step_id) == str(micro_step) or micro_step_id is None:
                    splited_graph_data[node_name] = (node_data)
                    traverse_npu(graph_nodes, node_data.get('subnodes', []))

        root = graph_data.get('root')
        root_subnodes = graph_nodes.get(root, {}).get('subnodes', [])
        traverse_npu(graph_nodes, root_subnodes)
        return splited_graph_data

    @staticmethod
    def walk_with_max_depth(logdir, max_depth):
        for root, dirs, files in os.walk(logdir):
            # 计算当前 root 相对于 top 的深度
            depth = root[len(logdir):].count(os.sep) + 1
            if depth >= max_depth:
                del dirs[:]
            yield root, dirs, files

    @staticmethod   
    def safe_json_loads(json_str, default_value=None):
        """
        安全地解析 JSON 字符串，带长度限制和异常处理。
        :param json_str: 要解析的 JSON 字符串
        :param default_value: 如果解析失败返回的默认值
        :return: 解析后的 Python 对象 或 default_value
        """
        # 类型检查
        if not isinstance(json_str, str):
            logger.error("Input is not a string.")
            return default_value

        # 长度限制
        if len(json_str) > MAX_FILE_SIZE:
            logger.error(f"Input length exceeds {MAX_FILE_SIZE} characters.")
            return default_value

        try:
            result = json.loads(json_str)
            GraphUtils.remove_prototype_pollution(result)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return default_value
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return default_value

    @staticmethod   
    def remove_prototype_pollution(obj, current_depth=1, max_depth=200):
        """
        递归删除对象中的原型污染字段，如 '__proto__', 'constructor', 'prototype'。
        
        :param obj: 要清理的对象
        :param current_depth: 当前递归深度，默认从 1 开始
        :param max_depth: 最大允许递归深度
        """
        if current_depth > max_depth:
            logger.warning(f"Reached maximum recursion depth of {max_depth}. Stopping further recursion.")
            return
        
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                if key in ('__proto__', 'constructor', 'prototype'):
                    del obj[key]
                else:
                    GraphUtils.remove_prototype_pollution(obj[key], current_depth + 1, max_depth)
        elif isinstance(obj, list):
            for item in obj:
                GraphUtils.remove_prototype_pollution(item, current_depth + 1, max_depth)

    @staticmethod
    def remove_prefix(node_data, prefix):
        if node_data is None:
            return {}
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in node_data.items()}

    @staticmethod
    def convert_to_float(value):
        try:
            return float(value)
        except ValueError:
            return float('nan')

    @staticmethod
    def format_relative_err(value):
        """格式化相对误差为百分比，保留四位小数"""
        if value is None or math.isnan(value):
            return "NaN"
        else:
            return "{:.4%}".format(value)

    @staticmethod
    def nan_to_str(value):
        """将 NaN 转换为 'NaN' 字符串"""
        return "NaN" if math.isnan(value) else value

    @staticmethod
    def is_relative_to(path, base):
        abs_path = os.path.abspath(path)
        abs_base = os.path.abspath(base)
        return os.path.commonpath([abs_path, abs_base]) == str(abs_base)

    @staticmethod
    def bytes_to_human_readable(size_bytes, decimal_places=2):
        """
        将字节大小转换为更易读的格式（如 KB、MB、GB 等）。
        
        :param size_bytes: int 或 float，表示字节大小
        :param decimal_places: 保留的小数位数，默认为 2
        :return: str，人类可读的大小表示
        """
        if size_bytes == 0:
            return "0 B"

        units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        unit_index = 0

        while size_bytes >= 1024 and unit_index < len(units) - 1:
            size_bytes /= 1024.0
            unit_index += 1

        return f"{size_bytes:.{decimal_places}f} {units[unit_index]}"

    @staticmethod
    def has_group_permission(path, permission):
        """
        检查指定路径是否对其所属组具有指定权限。
        
        :param path: 要检查的文件或目录的路径。
        :param permission: 要检查的权限，使用 stat 中定义的常量，如 stat.S_IWGRP 表示写权限。
        :return: 如果所属组有指定权限，则返回 True；否则返回 False。
        """
        try:
            # 获取文件的状态信息
            st = os.stat(path)
        except FileNotFoundError:
            return False
        
        # 使用位运算检查权限
        group_permissions = st.st_mode & (stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP)
        
        # 判断指定的权限是否存在
        if group_permissions & permission:
            return True
        else:
            return False

    @staticmethod
    def safe_save_data(data, run_name, tag):
        runs = GraphState.get_global_value('runs', {})
        run = runs.get(run_name) or run_name
        safe_base_dir = GraphState.get_global_value('logdir')
        if run is None or tag is None:
            error_message = 'The query parameters "run" and "tag" are required'
            return None, error_message
        try:
            # 构建文件路径并标准化
            file_path = os.path.join(run, tag)
            file_path = os.path.normpath(file_path)
            # 检查 run 目录是否存在，如果不存在则创建
            if not os.path.exists(run):
                os.makedirs(run, exist_ok=True)
                os.chmod(run, 0o750)
            # 权限校验：检查目录所属组是否有写权限
            if not os.stat(file_path).st_mode and not GraphUtils.has_group_permission(file_path, stat.stat.S_IWGRP):
                raise PermissionError(f"No write permission for directory: {run}\n")
            # 检查 tag 是否为合法文件名
            if not re.match(FILE_NAME_REGEX, tag):
                raise ValueError(f"Invalid tag: {tag}.")
            # 检查文件路径是否合法，防止路径遍历攻击
            if not file_path.startswith(os.path.abspath(run)):
                raise ValueError(f"Invalid file path: {file_path}. Potential path traversal attack.\n")
            # 基础路径校验
            if not GraphUtils.is_relative_to(file_path, safe_base_dir):
                raise ValueError(f"Path out of bounds: {file_path}")
            if os.path.islink(file_path):
                raise RuntimeError("The target file is a symbolic link")
            if os.path.islink(run):
                raise RuntimeError(f"Parent directory contains a symbolic link")
            # # 尝试写入文件
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
            os.chmod(file_path, 0o640)
            # 最终校验（防御TOCTOU攻击）
            if os.path.islink(file_path):
                raise RuntimeError("The file has been replaced with a symbolic link")
            return True, None
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid data: {e}")
            return None, 'Invalid data'
        except OSError as e:
            logger.error(f"Failed to create directory: {run}. Error: {e}\n")
            return None, 'failed to create directory {run}'
        except Exception as e:
            logger.error(f'Error: File "{file_path}" is not accessible. Error: {e}')
            return None, 'failed to save file'

    @staticmethod
    def safe_load_data(run_name, tag, only_check=False):
        runs = GraphState.get_global_value('runs', {})
        run_dir = runs.get(str(run_name)) or run_name
        """Load a single .vis file from a given directory based on the tag."""
        if run_dir is None or tag is None:
            error_message = 'The query parameters "run" and "tag" are required'
            return None, error_message
        try:
            file_path = os.path.join(run_dir, tag)
            # 目录安全校验
            success, error = GraphUtils.safe_check_load_file_path(run_dir, True)
            if not success:
                raise RuntimeError(error)
            # 文件安全校验
            success, error = GraphUtils.safe_check_load_file_path(file_path)
            if not success:
                raise RuntimeError(error)
            # 读取文件比较耗时，支持onlyCheck参数，仅进行安全校验
            if only_check:
                return True, None
            # 尝试解析 JSON 文件,校验文件内容是否合理
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f), None
        except json.JSONDecodeError:
            logger.error(f'Error: File "{file_path}" is not a valid JSON file!')
            return None, "File is not a valid JSON file!"
        except Exception as e:
            logger.error(f'Error: File "{file_path}" is not accessible. Error: {e}')
            return None, e

    @staticmethod
    def safe_check_load_file_path(file_path, isDir=False):
        # 权限常量定义
        PERM_GROUP_WRITE = 0o020
        PERM_OTHER_WRITE = 0o002
        PERM_OTHER_READ = 0o004
        file_path = os.path.normpath(file_path)  # 标准化路径
        real_path = os.path.realpath(file_path)
        safe_base_dir = GraphState.get_global_value('logdir')
        st = os.stat(real_path)
        try:
            # 安全验证：路径长度检查
            if len(real_path) > FILE_PATH_MAX_LENGTH:
                raise RuntimeError(f"Path length exceeds limit")
            # 安全验证：路径归属检查（防止越界访问）
            if not GraphUtils.is_relative_to(file_path, safe_base_dir):
                raise RuntimeError(f"Path out of bounds")
            # 安全检查：文件存在性验证
            if not os.path.exists(real_path):
                raise FileNotFoundError(f"File does not exist")
            # 安全验证：禁止符号链接文件
            if os.path.islink(file_path):
                raise RuntimeError(f"Detected symbolic link file")
            # 安全验证：文件类型检查（防御TOCTOU攻击）
            # 文件类型
            if not isDir and not os.path.isfile(real_path):
                raise RuntimeError(f"Path is not a regular file")
            # 目录类型
            if isDir and not Path(real_path).is_dir():
                raise RuntimeError(f"Directory does not exist")
            # 可读性检查
            if not st.st_mode & PERM_OTHER_READ:
                raise RuntimeError(f"Directory lacks read permission for others")
            # 文件大小校验
            if not isDir and os.path.getsize(file_path) > MAX_FILE_SIZE:
                file_size = GraphUtils.bytes_to_human_readable(os.path.getsize(file_path))
                max_size = GraphUtils.bytes_to_human_readable(MAX_FILE_SIZE)
                raise RuntimeError(
                    f"File size exceeds limit ({file_size} > {max_size})")
            # 非windows系统下，属主检查
            if os.name != 'nt':
                current_uid = os.getuid() 
                # 如果是root用户，跳过后续权限检查
                if current_uid == 0:
                    return True
                # 属主检查
                if st.st_uid != current_uid:
                    raise RuntimeError(f"Directory is not owned by the current user")
                # group和其他用户不可写检查
                if st.st_mode & PERM_GROUP_WRITE or st.st_mode & PERM_OTHER_WRITE:
                    raise RuntimeError(f"Directory has group or other write permission")
            return True, None
        except Exception as e:
            logger.error(e)
            return False, e
       
    @staticmethod
    def find_config_files(run_name):
        """
        在指定目录下查找所有以 .vis.config 结尾的文件（不包括子目录）
        :param directory: 要搜索的目标目录路径
        :return: 包含所有匹配文件路径的列表
        """
        runs = GraphState.get_global_value('runs', {})
        run = runs.get(run_name)
        dir_path = Path(run)
        try:
            if GraphUtils.safe_check_load_file_path(run, True):
                return [
                    file.name for file in dir_path.iterdir()
                    if file.is_file() and file.name.endswith('.vis.config')
                ]
        except Exception as e:
            logger.error(e)
            return []

    @staticmethod
    def compare_tag_names(a: str, b: str) -> int:
        """自然排序比较函数，支持路径格式和数字顺序"""

        # 辅助函数：标准化路径并分割组件
        def split_components(s: str) -> list:
            s = s.replace('\\', '/')  # 统一路径分隔符
            return [c for c in re.split(r'[/_]+', s) if c]  # 按 /_ 分割非空组件

        # 辅助函数：将组件拆分为数字/字符串混合的 tokens
        def tokenize(component: str) -> list:
            tokens = []
            buffer = []
            is_num = False

            for char in component:
                if char.isdigit() == is_num and buffer:
                    buffer.append(char)
                else:
                    if buffer:
                        tokens.append(int(''.join(buffer)) if is_num else ''.join(buffer))
                    buffer = [char]
                    is_num = char.isdigit()

            if buffer:
                tokens.append(int(''.join(buffer)) if is_num else ''.join(buffer))
            return tokens

        # 逐级比较组件
        a_comps = split_components(a)
        b_comps = split_components(b)

        for a_part, b_part in zip(a_comps, b_comps):
            a_tokens = tokenize(a_part)
            b_tokens = tokenize(b_part)

            # 比较 token 序列
            for a_tok, b_tok in zip(a_tokens, b_tokens):
                if isinstance(a_tok, int) and isinstance(b_tok, int):
                    if a_tok != b_tok:
                        return a_tok - b_tok
                elif isinstance(a_tok, int):
                    return -1  # 数字优先于字母
                elif isinstance(b_tok, int):
                    return 1
                elif a_tok != b_tok:
                    return -1 if a_tok < b_tok else 1

            # 处理子序列长度差异
            if len(a_tokens) != len(b_tokens):
                return len(a_tokens) - len(b_tokens)

        # 处理组件数量差异
        return len(a_comps) - len(b_comps)

    @staticmethod
    def sort_data(data: dict) -> dict:
        """自然排序比较函数，支持路径格式和数字顺序"""
        sorted_data = {}
        sorted_keys = sorted(data.keys(), key=cmp_to_key(GraphUtils.compare_tag_names))
        for k in sorted_keys:
            # 对每个键对应的值列表进行排序
            sorted_values = sorted(data[k], key=cmp_to_key(GraphUtils.compare_tag_names))
            sorted_data[k] = sorted_values

        return sorted_data

    @staticmethod
    def process_vis_file(dir_path, file, run_tag_pairs):
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path) and file.endswith('.vis'):
            run = dir_path
            run_name = os.path.basename(run)
            GraphState.set_global_value('runs', run, run_name)
            tag = file[:-4]  # Use the filename without extension as tag
            _, error = GraphUtils.safe_load_data(run_name, tag, True)
            if error:
                logger.error(f'Error: File run:"{run}, tag:{tag}" is not accessible. Error: {error}')
                return
            run_tag_pairs.setdefault(run_name, []).append(tag)
