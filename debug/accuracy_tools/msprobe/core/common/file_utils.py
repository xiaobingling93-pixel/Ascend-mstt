#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2022-2023. Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
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
"""
import csv
import fcntl
import os
import json
import re
import shutil
import yaml
import numpy as np
import pandas as pd

from msprobe.core.common.log import logger
from msprobe.core.common.exceptions import FileCheckException
from msprobe.core.common.const import FileCheckConst


class FileChecker:
    """
    The class for check file.

    Attributes:
        file_path: The file or dictionary path to be verified.
        path_type: file or dictionary
        ability(str): FileCheckConst.WRITE_ABLE or FileCheckConst.READ_ABLE to set file has writability or readability
        file_type(str): The correct file type for file
    """

    def __init__(self, file_path, path_type, ability=None, file_type=None, is_script=True):
        self.file_path = file_path
        self.path_type = self._check_path_type(path_type)
        self.ability = ability
        self.file_type = file_type
        self.is_script = is_script

    @staticmethod
    def _check_path_type(path_type):
        if path_type not in [FileCheckConst.DIR, FileCheckConst.FILE]:
            logger.error(f'The path_type must be {FileCheckConst.DIR} or {FileCheckConst.FILE}.')
            raise FileCheckException(FileCheckException.ILLEGAL_PARAM_ERROR)
        return path_type

    def common_check(self):
        """
        功能：用户校验基本文件权限：软连接、文件长度、是否存在、读写权限、文件属组、文件特殊字符
        注意：文件后缀的合法性，非通用操作，可使用其他独立接口实现
        """
        check_path_exists(self.file_path)
        check_link(self.file_path)
        self.file_path = os.path.realpath(self.file_path)
        check_path_length(self.file_path)
        check_path_type(self.file_path, self.path_type)
        self.check_path_ability()
        if self.is_script:
            check_path_owner_consistent(self.file_path)
        check_path_pattern_vaild(self.file_path)
        check_common_file_size(self.file_path)
        check_file_suffix(self.file_path, self.file_type)
        return self.file_path

    def check_path_ability(self):
        if self.ability == FileCheckConst.WRITE_ABLE:
            check_path_writability(self.file_path)
        if self.ability == FileCheckConst.READ_ABLE:
            check_path_readability(self.file_path)
        if self.ability == FileCheckConst.READ_WRITE_ABLE:
            check_path_readability(self.file_path)
            check_path_writability(self.file_path)


class FileOpen:
    """
    The class for open file by a safe way.

    Attributes:
        file_path: The file or dictionary path to be opened.
        mode(str): The file open mode
    """
    SUPPORT_READ_MODE = ["r", "rb"]
    SUPPORT_WRITE_MODE = ["w", "wb", "a", "ab"]
    SUPPORT_READ_WRITE_MODE = ["r+", "rb+", "w+", "wb+", "a+", "ab+"]

    def __init__(self, file_path, mode, encoding='utf-8'):
        self.file_path = file_path
        self.mode = mode
        self.encoding = encoding
        self._handle = None

    def __enter__(self):
        self.check_file_path()
        binary_mode = "b"
        if binary_mode not in self.mode:
            self._handle = open(self.file_path, self.mode, encoding=self.encoding)
        else:
            self._handle = open(self.file_path, self.mode)
        return self._handle

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle:
            self._handle.close()

    def check_file_path(self):
        support_mode = self.SUPPORT_READ_MODE + self.SUPPORT_WRITE_MODE + self.SUPPORT_READ_WRITE_MODE
        if self.mode not in support_mode:
            logger.error("File open not support %s mode" % self.mode)
            raise FileCheckException(FileCheckException.ILLEGAL_PARAM_ERROR)
        check_link(self.file_path)
        self.file_path = os.path.realpath(self.file_path)
        check_path_length(self.file_path)
        self.check_ability_and_owner()
        check_path_pattern_vaild(self.file_path)
        if os.path.exists(self.file_path):
            check_common_file_size(self.file_path)

    def check_ability_and_owner(self):
        if self.mode in self.SUPPORT_READ_MODE:
            check_path_exists(self.file_path)
            check_path_readability(self.file_path)
            check_path_owner_consistent(self.file_path)
        if self.mode in self.SUPPORT_WRITE_MODE and os.path.exists(self.file_path):
            check_path_writability(self.file_path)
            check_path_owner_consistent(self.file_path)
        if self.mode in self.SUPPORT_READ_WRITE_MODE and os.path.exists(self.file_path):
            check_path_readability(self.file_path)
            check_path_writability(self.file_path)
            check_path_owner_consistent(self.file_path)


def check_link(path):
    abs_path = os.path.abspath(path)
    if os.path.islink(abs_path):
        logger.error('The file path {} is a soft link.'.format(path))
        raise FileCheckException(FileCheckException.SOFT_LINK_ERROR)


def check_path_length(path, name_length=None):
    file_max_name_length = name_length if name_length else FileCheckConst.FILE_NAME_LENGTH
    if len(path) > FileCheckConst.DIRECTORY_LENGTH or \
            len(os.path.basename(path)) > file_max_name_length:
        logger.error('The file path length exceeds limit.')
        raise FileCheckException(FileCheckException.ILLEGAL_PATH_ERROR)


def check_path_exists(path):
    if not os.path.exists(path):
        logger.error('The file path %s does not exist.' % path)
        raise FileCheckException(FileCheckException.ILLEGAL_PATH_ERROR)


def check_path_readability(path):
    if not os.access(path, os.R_OK):
        logger.error('The file path %s is not readable.' % path)
        raise FileCheckException(FileCheckException.FILE_PERMISSION_ERROR)


def check_path_writability(path):
    if not os.access(path, os.W_OK):
        logger.error('The file path %s is not writable.' % path)
        raise FileCheckException(FileCheckException.FILE_PERMISSION_ERROR)


def check_path_executable(path):
    if not os.access(path, os.X_OK):
        logger.error('The file path %s is not executable.' % path)
        raise FileCheckException(FileCheckException.FILE_PERMISSION_ERROR)


def check_other_user_writable(path):
    st = os.stat(path)
    if st.st_mode & 0o002:
        logger.error('The file path %s may be insecure because other users have write permissions. ' % path)
        raise FileCheckException(FileCheckException.FILE_PERMISSION_ERROR)


def check_path_owner_consistent(path):
    file_owner = os.stat(path).st_uid
    if file_owner != os.getuid() and os.getuid() != 0:
        logger.error('The file path %s may be insecure because is does not belong to you.' % path)
        raise FileCheckException(FileCheckException.FILE_PERMISSION_ERROR)


def check_path_pattern_vaild(path):
    if not re.match(FileCheckConst.FILE_VALID_PATTERN, path):
        logger.error('The file path %s contains special characters.' % (path))
        raise FileCheckException(FileCheckException.ILLEGAL_PATH_ERROR)


def check_file_size(file_path, max_size):
    try:
        file_size = os.path.getsize(file_path)
    except OSError as os_error:
        logger.error(f'Failed to open "{file_path}". {str(os_error)}')
        raise FileCheckException(FileCheckException.INVALID_FILE_ERROR) from os_error
    if file_size >= max_size:
        logger.error(f'The size ({file_size}) of {file_path} exceeds ({max_size}) bytes, tools not support.')
        raise FileCheckException(FileCheckException.FILE_TOO_LARGE_ERROR)


def check_common_file_size(file_path):
    if os.path.isfile(file_path):
        for suffix, max_size in FileCheckConst.FILE_SIZE_DICT.items():
            if file_path.endswith(suffix):
                check_file_size(file_path, max_size)
                return
        check_file_size(file_path, FileCheckConst.COMMOM_FILE_SIZE)
        


def check_file_suffix(file_path, file_suffix):
    if file_suffix:
        if not file_path.endswith(file_suffix):
            logger.error(f"The {file_path} should be a {file_suffix} file!")
            raise FileCheckException(FileCheckException.INVALID_FILE_ERROR)


def check_path_type(file_path, file_type):
    if file_type == FileCheckConst.FILE:
        if not os.path.isfile(file_path):
            logger.error(f"The {file_path} should be a file!")
            raise FileCheckException(FileCheckException.INVALID_FILE_ERROR)
    if file_type == FileCheckConst.DIR:
        if not os.path.isdir(file_path):
            logger.error(f"The {file_path} should be a dictionary!")
            raise FileCheckException(FileCheckException.INVALID_FILE_ERROR)


def make_dir(dir_path):
    dir_path = os.path.realpath(dir_path)
    check_path_before_create(dir_path)
    if os.path.isdir(dir_path):
        return
    try:
        os.makedirs(dir_path, mode=FileCheckConst.DATA_DIR_AUTHORITY, exist_ok=True)
    except OSError as ex:
        raise FileCheckException(FileCheckException.ILLEGAL_PATH_ERROR,
                                 f"Failed to create {dir_path}. "
                                 f"Please check the path permission or disk space. {str(ex)}") from ex
    file_check = FileChecker(dir_path, FileCheckConst.DIR)
    file_check.common_check()


def create_directory(dir_path):
    """
    Function Description:
        creating a safe directory with specified permissions
    Parameter:
        dir_path: directory path
    Exception Description:
        when invalid data throw exception
    """
    dir_path = os.path.realpath(dir_path)
    check_path_before_create(dir_path)
    parent_dir = os.path.dirname(dir_path)
    if not os.path.isdir(parent_dir):
        create_directory(parent_dir)
    make_dir(dir_path)


def check_path_before_create(path):
    if path_len_exceeds_limit(path):
        raise FileCheckException(FileCheckException.ILLEGAL_PATH_ERROR, 'The file path length exceeds limit.')

    if not re.match(FileCheckConst.FILE_PATTERN, os.path.realpath(path)):
        raise FileCheckException(FileCheckException.ILLEGAL_PATH_ERROR,
                                 'The file path {} contains special characters.'.format(path))


def check_file_or_directory_path(path, isdir=False):
    """
    Function Description:
        check whether the path is valid
    Parameter:
        path: the path to check
        isdir: the path is dir or file
    Exception Description:
        when invalid data throw exception
    """
    if isdir:
        path_checker = FileChecker(path, FileCheckConst.DIR, FileCheckConst.WRITE_ABLE)
    else:
        path_checker = FileChecker(path, FileCheckConst.FILE, FileCheckConst.READ_ABLE)
    path_checker.common_check()


def change_mode(path, mode):
    if not os.path.exists(path) or os.path.islink(path):
        return
    try:
        os.chmod(path, mode)
    except PermissionError as ex:
        raise FileCheckException(FileCheckException.FILE_PERMISSION_ERROR,
                                 'Failed to change {} authority. {}'.format(path, str(ex))) from ex


def path_len_exceeds_limit(file_path):
    return len(os.path.realpath(file_path)) > FileCheckConst.DIRECTORY_LENGTH or \
        len(os.path.basename(file_path)) > FileCheckConst.FILE_NAME_LENGTH


def check_file_type(path):
    """
    Function Description:
        determine if it is a file or a directory
    Parameter:
        path: path
    Exception Description:
        when neither a file nor a directory throw exception
    """
    if os.path.isdir(path):
        return FileCheckConst.DIR
    elif os.path.isfile(path):
        return FileCheckConst.FILE
    else:
        logger.error(f'{path} does not exist, please check!')
        raise FileCheckException(FileCheckException.INVALID_FILE_ERROR)


def load_yaml(yaml_path):
    path_checker = FileChecker(yaml_path, FileCheckConst.FILE, FileCheckConst.READ_ABLE, FileCheckConst.YAML_SUFFIX)
    checked_path = path_checker.common_check()
    try:
        with FileOpen(checked_path, "r") as f:
            yaml_data = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"The yaml file failed to load. Please check the path: {checked_path}.")
        raise RuntimeError(f"Load yaml file {checked_path} failed.") from e
    return yaml_data


def load_npy(filepath):
    check_file_or_directory_path(filepath)
    try:
        npy = np.load(filepath)
    except Exception as e:
        logger.error(f"The numpy file failed to load. Please check the path: {filepath}.")
        raise RuntimeError(f"Load numpy file {filepath} failed.") from e
    return npy


def load_json(json_path):
    try:
        with FileOpen(json_path, "r") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            data = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        logger.error(f'load json file "{os.path.basename(json_path)}" failed.')
        raise RuntimeError(f"Load json file {json_path} failed.") from e
    return data


def save_json(json_path, data, indent=None):
    json_path = os.path.realpath(json_path)
    check_path_before_create(json_path)
    try:
        with FileOpen(json_path, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(data, f, indent=indent)
            fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        logger.error(f'Save json file "{os.path.basename(json_path)}" failed.')
        raise RuntimeError(f"Save json file {json_path} failed.") from e
    change_mode(json_path, FileCheckConst.DATA_FILE_AUTHORITY)


def save_yaml(yaml_path, data):
    yaml_path = os.path.realpath(yaml_path)
    check_path_before_create(yaml_path)
    try:
        with FileOpen(yaml_path, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            yaml.dump(data, f, sort_keys=False)
            fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        logger.error(f'Save yaml file "{os.path.basename(yaml_path)}" failed.')
        raise RuntimeError(f"Save yaml file {yaml_path} failed.") from e
    change_mode(yaml_path, FileCheckConst.DATA_FILE_AUTHORITY)


def move_file(src_path, dst_path):
    check_file_or_directory_path(src_path)
    check_path_before_create(dst_path)
    try:
        shutil.move(src_path, dst_path)
    except Exception as e:
        logger.error(f"move file {src_path} to {dst_path} failed")
        raise RuntimeError(f"move file {src_path} to {dst_path} failed") from e
    change_mode(dst_path, FileCheckConst.DATA_FILE_AUTHORITY)


def save_npy(data, filepath):
    filepath = os.path.realpath(filepath)
    check_path_before_create(filepath)
    try:
        np.save(filepath, data)
    except Exception as e:
        logger.error(f"The numpy file failed to save. Please check the path: {filepath}.")
        raise RuntimeError(f"Save numpy file {filepath} failed.") from e
    change_mode(filepath, FileCheckConst.DATA_FILE_AUTHORITY)


def save_npy_to_txt(data, dst_file='', align=0):
    if os.path.exists(dst_file):
        logger.info("Dst file %s exists, will not save new one." % dst_file)
        return
    shape = data.shape
    data = data.flatten()
    if align == 0:
        align = 1 if len(shape) == 0 else shape[-1]
    elif data.size % align != 0:
        pad_array = np.zeros((align - data.size % align,))
        data = np.append(data, pad_array)
    check_path_before_create(dst_file)
    try:
        np.savetxt(dst_file, data.reshape((-1, align)), delimiter=' ', fmt='%g')
    except Exception as e:
        logger.error("An unexpected error occurred: %s when savetxt to %s" % (str(e), dst_file))
    change_mode(dst_file, FileCheckConst.DATA_FILE_AUTHORITY)


def save_workbook(workbook, file_path):
    """
    保存工作簿到指定的文件路径
    workbook: 要保存的工作簿对象
    file_path: 文件保存路径
    """
    file_path = os.path.realpath(file_path)
    check_path_before_create(file_path)
    try:
        workbook.save(file_path)
    except Exception as e:
        logger.error(f'Save result file "{os.path.basename(file_path)}" failed')
        raise RuntimeError(f"Save result file {file_path} failed.") from e
    change_mode(file_path, FileCheckConst.DATA_FILE_AUTHORITY)


def write_csv(data, filepath, mode="a+", malicious_check=False):
    def csv_value_is_valid(value: str) -> bool:
        if not isinstance(value, str):
            return True  
        try:
            # -1.00 or +1.00 should be consdiered as digit numbers
            float(value)
        except ValueError:
            # otherwise, they will be considered as formular injections
            return not bool(re.compile(FileCheckConst.CSV_BLACK_LIST).search(value))
        return True
    
    if malicious_check:
        for row in data:
            for cell in row:
                if not csv_value_is_valid(cell):
                    raise RuntimeError(f"Malicious value [{cell}] is not allowed " \
                                       f"to be written into the csv: {filepath}.")

    file_path = os.path.realpath(filepath)
    check_path_before_create(filepath)
    try:
        with FileOpen(filepath, mode, encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerows(data)
    except Exception as e:
        logger.error(f'Save csv file "{os.path.basename(file_path)}" failed')
        raise RuntimeError(f"Save csv file {file_path} failed.") from e
    change_mode(filepath, FileCheckConst.DATA_FILE_AUTHORITY)


def read_csv(filepath):
    check_file_or_directory_path(filepath)
    try:
        csv_data = pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"The csv file failed to load. Please check the path: {filepath}.")
        raise RuntimeError(f"Read csv file {filepath} failed.") from e
    return csv_data


def remove_path(path):
    if not os.path.exists(path):
        return
    try:
        if os.path.islink(path) or os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)
    except PermissionError as err:
        logger.error("Failed to delete {}. Please check the permission.".format(path))
        raise FileCheckException(FileCheckException.ILLEGAL_PATH_ERROR) from err
    except Exception as e:
        logger.error("Failed to delete {}. Please check.".format(path))
        raise RuntimeError(f"Delete {path} failed.") from e


def get_json_contents(file_path):
    ops = get_file_content_bytes(file_path)
    try:
        json_obj = json.loads(ops)
    except ValueError as error:
        logger.error('Failed to load json.')
        raise FileCheckException(FileCheckException.INVALID_FILE_ERROR) from error
    if not isinstance(json_obj, dict):
        logger.error('Json file content is not a dictionary!')
        raise FileCheckException(FileCheckException.INVALID_FILE_ERROR)
    return json_obj


def get_file_content_bytes(file):
    with FileOpen(file, 'rb') as file_handle:
        return file_handle.read()
