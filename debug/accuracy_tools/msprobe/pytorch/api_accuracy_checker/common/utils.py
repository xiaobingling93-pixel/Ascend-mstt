#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2023-2023. Huawei Technologies Co., Ltd. All rights reserved.
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
import json
import os
import re
import csv

import torch

try:
    import torch_npu
except ImportError:
    IS_GPU = True
else:
    IS_GPU = False

from msprobe.pytorch.common.log import logger
from msprobe.core.common.file_check import FileChecker, FileOpen, change_mode, create_directory
from msprobe.core.common.const import Const, FileCheckConst
from msprobe.core.common.utils import CompareException


class DumpException(CompareException):
    pass


def write_csv(data, filepath):
    with FileOpen(filepath, 'a', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerows(data)


def check_object_type(check_object, allow_type):
    """
    Function Description:
        Check if the object belongs to a certain data type
    Parameter:
        check_object: the object to be checked
        allow_type: legal data type
    Exception Description:
        when invalid data throw exception
    """
    if not isinstance(check_object, allow_type):
        logger.error(f"{check_object} not of {allow_type} type")
        raise CompareException(CompareException.INVALID_DATA_ERROR)


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
        if not os.path.exists(path):
            logger.error('The path {} is not exist.'.format(path))
            raise CompareException(CompareException.INVALID_PATH_ERROR)

        if not os.path.isdir(path):
            logger.error('The path {} is not a directory.'.format(path))
            raise CompareException(CompareException.INVALID_PATH_ERROR)

        if not os.access(path, os.W_OK):
            logger.error(
                'The path {} does not have permission to write. Please check the path permission'.format(path))
            raise CompareException(CompareException.INVALID_PATH_ERROR)
    else:
        if not os.path.isfile(path):
            logger.error('{} is an invalid file or non-exist.'.format(path))
            raise CompareException(CompareException.INVALID_PATH_ERROR)

    if not os.access(path, os.R_OK):
        logger.error(
            'The path {} does not have permission to read. Please check the path permission'.format(path))
        raise CompareException(CompareException.INVALID_PATH_ERROR)


def get_json_contents(file_path):
    ops = get_file_content_bytes(file_path)
    try:
        json_obj = json.loads(ops)
    except ValueError as error:
        logger.error('Failed to load "%s". %s' % (file_path, str(error)))
        raise CompareException(CompareException.INVALID_FILE_ERROR) from error
    if not isinstance(json_obj, dict):
        logger.error('Json file %s, content is not a dictionary!' % file_path)
        raise CompareException(CompareException.INVALID_FILE_ERROR)
    return json_obj


def get_file_content_bytes(file):
    with FileOpen(file, 'rb') as file_handle:
        return file_handle.read()


class SoftlinkCheckException(Exception):
    pass


def check_need_convert(api_name):
    convert_type = None
    for key, value in Const.CONVERT_API.items():
        if api_name not in value:
            continue
        else:
            convert_type = key
    return convert_type


def api_info_preprocess(api_name, api_info_dict):
    """
    Function Description:
        Preprocesses the API information.
    Parameter:
        api_name: Name of the API.
        api_info_dict: argument of the API.
    Return api_info_dict:
        convert_type: Type of conversion.
        api_info_dict: Processed argument of the API.
    """
    convert_type = check_need_convert(api_name)
    if api_name == 'cross_entropy':
        api_info_dict = cross_entropy_process(api_info_dict)
    return convert_type, api_info_dict


def cross_entropy_process(api_info_dict):
    """
    Function Description:
        Preprocesses the cross_entropy API information.
    Parameter:
        api_info_dict: argument of the API.
    Return api_info_dict:
        api_info_dict: Processed argument of the API.
    """
    if 'args' in api_info_dict and len(api_info_dict['args']) > 1 and 'Min' in api_info_dict['args'][1]:
        if api_info_dict['args'][1]['Min'] <= 0:
            # The second argument in cross_entropy should be -100 or not less than 0
            api_info_dict['args'][1]['Min'] = 0
    return api_info_dict


def initialize_save_path(save_path, dir_name):
    data_path = os.path.join(save_path, dir_name)
    if os.path.exists(data_path):
        logger.warning(f"{data_path} already exists, it will be overwritten")
    else:
        os.mkdir(data_path, mode=FileCheckConst.DATA_DIR_AUTHORITY)
    data_path_checker = FileChecker(data_path, FileCheckConst.DIR)
    data_path_checker.common_check()
    return data_path


def write_pt(file_path, tensor):
    if os.path.exists(file_path):
        raise ValueError(f"File {file_path} already exists")
    torch.save(tensor, file_path)
    full_path = os.path.realpath(file_path)
    change_mode(full_path, FileCheckConst.DATA_FILE_AUTHORITY)
    return full_path


def get_real_data_path(file_path):
    targets = ['forward_real_data', 'backward_real_data', 'ut_error_data\d+']
    pattern = re.compile(r'({})'.format('|'.join(targets)))
    match = pattern.search(file_path)
    if match:
        target_index = match.start()
        target_path = file_path[target_index:]
        return target_path
    else:
        raise DumpException(DumpException.INVALID_PATH_ERROR)


def get_full_data_path(data_path, real_data_path):
    if not data_path:
        return data_path
    full_data_path = os.path.join(real_data_path, data_path)
    return os.path.realpath(full_data_path)


class UtDataProcessor:
    def __init__(self, save_path):
        self.save_path = save_path
        self.index = 0

    def save_tensors_in_element(self, api_name, element):
        self.index = 0
        self._save_recursive(api_name, element)

    def _save_recursive(self, api_name, element):
        if isinstance(element, torch.Tensor):
            api_args = api_name + Const.SEP + str(self.index)
            create_directory(self.save_path)
            file_path = os.path.join(self.save_path, f'{api_args}.pt')
            write_pt(file_path, element.contiguous().cpu().detach())
            self.index += 1
        elif element is None or isinstance(element, (bool, int, float, str, slice)):
            self.index += 1
        elif isinstance(element, (list, tuple)):
            for item in element:
                self._save_recursive(api_name, item)
        elif isinstance(element, dict):
            for value in element.values():
                self._save_recursive(api_name, value)
        else:
            self.index += 1
