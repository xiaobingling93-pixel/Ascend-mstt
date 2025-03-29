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
        run = meta_data.get('run')
        tag = meta_data.get('tag')
        current_tag = get_global_value('current_tag')
        if current_tag == tag:
            return get_global_value('current_file_data'), None  # 直接返回获取结果
        else:
            json_data, error_message = GraphUtils.safe_load_data(run, tag)
            if error_message:
                return None, error_message
            set_global_value('current_file_data', json_data)
            set_global_value('current_tag', tag)
            return json_data, error_message

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
    def safe_save_data(data, run, tag):
        safe_base_dir = get_global_value('logdir')
        if run is None or tag is None:
            error_message = 'The query parameters "run" and "tag" are required'
            return None, error_message
        try:
            # 构建文件路径并标准化
            file_path = os.path.join(run, f"{tag}.vis")
            file_path = os.path.normpath(file_path)
            # 权限校验：检查目录是否有写权限
            if not os.access(run, os.W_OK):
                raise PermissionError(f"No write permission for directory: {run}\n")
            # 检查 run 目录是否存在，如果不存在则创建
            if not os.path.exists(run):
                os.makedirs(run, exist_ok=True)
                os.chmod(run, 0o750)
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
            if not os.path.isfile(file_path):
                raise RuntimeError("The target path is not a regular file")
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
    def safe_load_data(run_dir, tag, only_check=False):
        """Load a single .vis file from a given directory based on the tag."""
        if run_dir is None or tag is None:
            error_message = 'The query parameters "run" and "tag" are required'
            return None, error_message
        try:
            file_path = os.path.join(run_dir, f"{tag}.vis")
            file_path = os.path.normpath(file_path)  # 标准化路径
            # 解析真实路径（包含符号链接跟踪）
            real_path = os.path.realpath(file_path)
            safe_base_dir = get_global_value('logdir')
            # 安全验证1：路径归属检查（防止越界访问）
            if not GraphUtils.is_relative_to(file_path, safe_base_dir):
                raise RuntimeError(f"Path out of bounds:")
            # 安全验证2：禁止符号链接文件
            if os.path.islink(file_path):
                raise RuntimeError(f"Detected symbolic link file")
            if os.path.islink(run_dir):
                raise RuntimeError(f"Parent directory contains a symbolic link")
            # 安全验证3：二次文件类型检查（防御TOCTOU攻击）
            if not os.path.isfile(real_path):
                raise RuntimeError(f"Path is not a regular file")
            # 安全检查4：文件存在性验证
            if not os.path.exists(real_path):
                raise FileNotFoundError(f"File does not exist")
            # 权限验证
            if not os.stat(real_path).st_mode & stat.S_IRUSR:
                raise PermissionError(f"File has no read permissions")
            # 文件大小验证
            if os.path.getsize(real_path) > MAX_FILE_SIZE:
                raise RuntimeError(f"File size exceeds limit ({os.path.getsize(real_path)} > {MAX_FILE_SIZE/1024}MB)")
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
            return None, 'failed to load file'

