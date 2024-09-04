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
import os
import re

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
    if file_owner != os.getuid():
        logger.error('The file path %s may be insecure because is does not belong to you.' % path)
        raise FileCheckException(FileCheckException.FILE_PERMISSION_ERROR)


def check_path_pattern_vaild(path):
    if not re.match(FileCheckConst.FILE_VALID_PATTERN, path):
        logger.error('The file path %s contains special characters.' %(path))
        raise FileCheckException(FileCheckException.ILLEGAL_PATH_ERROR)


def check_file_size(file_path, max_size):
    file_size = os.path.getsize(file_path)
    if file_size >= max_size:
        logger.error(f'The size of file path {file_path} exceeds {max_size} bytes.')
        raise FileCheckException(FileCheckException.FILE_TOO_LARGE_ERROR)


def check_common_file_size(file_path):
    if os.path.isfile(file_path):
        for suffix, max_size in FileCheckConst.FILE_SIZE_DICT.items():
            if file_path.endswith(suffix):
                check_file_size(file_path, max_size)
                break


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
    try:
        os.makedirs(dir_path, mode=FileCheckConst.DATA_DIR_AUTHORITY, exist_ok=True)
    except OSError as ex:
        raise FileCheckException(FileCheckException.ILLEGAL_PATH_ERROR,
            'Failed to create {}. Please check the path permission or disk space .{}'.format(dir_path, str(ex))) from ex
    file_check = FileChecker(dir_path, FileCheckConst.DIR)
    file_check.common_check()


def check_path_before_create(path):
    if path_len_exceeds_limit(path):
        raise FileCheckException(FileCheckException.ILLEGAL_PATH_ERROR, 'The file path length exceeds limit.')

    if not re.match(FileCheckConst.FILE_PATTERN, os.path.realpath(path)):
        raise FileCheckException(FileCheckException.ILLEGAL_PATH_ERROR,
                                 'The file path {} contains special characters.'.format(path))


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
        logger.error('Neither a file nor a directory.')
        raise FileCheckException(FileCheckException.INVALID_FILE_ERROR)
