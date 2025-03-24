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

from tensorboard.util import tb_logging
from .global_state import get_global_value, set_global_value, FILE_NAME_REGEX

logger = tb_logging.get_logger()
MAX_FILE_SIZE = 1024 * 1024 * 1024  # 最大文件大小限制为1GB


class GraphUtils:

    @staticmethod
    def get_graph_data(meta_data):
        tag = meta_data.get('tag')
        graph_data = GraphUtils.check_jsondata(tag)
        if graph_data is None:
            run = meta_data.get('run')
            return GraphUtils.get_jsondata(run, tag)  # 直接返回获取结果
        return graph_data, None

    # 检查是否使用缓存
    @staticmethod
    def check_jsondata(tag):
        _current_tag = get_global_value('current_tag')
        if _current_tag == tag:
            return get_global_value('current_file_data')
        else:
            return None

    # 读取json文件
    @staticmethod
    def get_jsondata(run, tag):
        json_data = None
        error_message = None
        logdir = get_global_value('logdir')
        if run is None or tag is None:
            error_message = 'The query parameters "run" and "tag" are required'
            return json_data, error_message
        run_dir = os.path.join(logdir, run)
        run_dir = os.path.normpath(run_dir)  # 标准化路径
        file_path = GraphUtils._load_json_file(run_dir, tag)
        if not file_path:
            error_message = f'vis file for tag "{tag}" not found in run "{run}"'
            return json_data, error_message
        json_data = GraphUtils._read_json_file(file_path)
        set_global_value('current_file_data', json_data)
        set_global_value('current_tag', tag)
        if json_data is None:
            error_message = f'Error reading vis file for tag "{tag}" in run "{run}"'
        return json_data, error_message

    @staticmethod
    def save_data(data, run, tag):

        # 检查 tag 是否为合法文件名
        if not re.match(FILE_NAME_REGEX, tag):
            raise ValueError(f"Invalid tag: {tag}.")

        # 检查 run 目录是否存在，如果不存在则创建
        if not os.path.exists(run):
            try:
                os.makedirs(run, exist_ok=True)
                os.chmod(run, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)
            except OSError as e:
                raise PermissionError(f"Failed to create directory: {run}. Error: {e}\n") from e

        # 检查 data 是否为有效的 JSON 可序列化对象
        try:
            json.dumps(data, ensure_ascii=False, indent=4)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid data: {e}") from e

        # 构建文件路径并标准化
        file_path = os.path.join(run, f"{tag}.vis")
        file_path = os.path.normpath(file_path)

        # 检查文件路径是否合法，防止路径遍历攻击
        if not file_path.startswith(os.path.abspath(run)):
            raise ValueError(f"Invalid file path: {file_path}. Potential path traversal attack.\n")

        # 权限校验：检查目录是否有写权限
        if not os.access(run, os.W_OK):
            raise PermissionError(f"No write permission for directory: {run}\n")

        # 尝试写入文件
        try:
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
            # 设置文件权限为仅所有者可读写 (0o600)
            os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR)
        except IOError as e:
            raise IOError(f"Failed to write to file {file_path}: {e}\n") from e

    @staticmethod
    def remove_prefix(node_data, prefix):
        if node_data is None:
            return {}
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in node_data.items()}

    # 字符串转float
    @staticmethod
    def convert_to_float(value):
        try:
            # 尝试将值转换为float
            return float(value)
        except ValueError:
            # 如果转换失败，返回默认值0.0
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
    def _load_json_file(run_dir, tag):
        """Load a single .vis file from a given directory based on the tag."""
        file_path = os.path.join(run_dir, f"{tag}.vis")
        file_path = os.path.normpath(file_path)  # 标准化路径
        if os.path.exists(file_path):
            # 校验文件的读权限
            if not os.access(file_path, os.R_OK):
                logger.error(f'Error: No read permission for file "{file_path}"')
                return None
            # 校验文件大小
            if os.path.getsize(file_path) > MAX_FILE_SIZE:
                logger.error(f'Error: File "{file_path}" exceeds the size limit of {MAX_FILE_SIZE // (1024 * 1024)}MB')
                return None
            set_global_value('current_file_path', file_path)
            return file_path
        else:
            logger.error(f'Error: File "{file_path}" does not exist.')
        return None

    @staticmethod
    def _read_json_file(file_path):
        """Read and parse a JSON file from disk."""
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # 尝试解析 JSON 文件
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f'Error: File "{file_path}" is not a valid JSON file!')
            except Exception as e:
                logger.error(f'Unexpected error while reading file "{file_path}": {e}')
        else:
            logger.error(f'Error: File "{file_path}" is not accessible.')
        return None
