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
import re
import hashlib

import torch

from msprobe.pytorch.common.log import logger


def merge_keys(dir_0, dir_1):
    output_list = list(dir_0.keys())
    output_list.extend(list(dir_1.keys()))
    return set(output_list)


def compare_dict(bench_dict, cmp_dict):
    result = []
    for key in set(bench_dict.keys()) | set(cmp_dict.keys()):
        if key in bench_dict and key in cmp_dict:
            if bench_dict[key] != cmp_dict[key]:
                result.append(f"{key}: {bench_dict[key]} -> {cmp_dict[key]}")
        elif key in bench_dict:
            result.append(f"{key}: [deleted] -> {bench_dict[key]}")
        else:
            result.append(f"{key}: [added] -> {cmp_dict[key]}")
    return result


def config_checking_print(msg):
    logger.info(f"[config checking log] {msg}")


def tensor_to_hash(tensor):
    """Compute the hash value of a tensor"""
    tensor_bytes = tensor.clone().detach().cpu().numpy().tobytes()
    return bytes_hash(tensor_bytes)


def get_tensor_features(tensor):
    features = {
        "max": lambda x: torch.max(x).item(),
        "min": lambda x: torch.min(x).item(),
        "mean": lambda x: torch.mean(x).item(),
        "norm": lambda x: torch.norm(x).item(),
    }

    if not tensor.is_floating_point() or tensor.dtype == torch.float64:
        tensor = tensor.float()
    return {key: features.get(key)(tensor) for key in features}


def compare_dicts(dict1, dict2, path=''):
    deleted = []
    added = []
    changed = []
    result = {}

    for key in dict1:
        if key not in dict2:
            deleted.append(f"[Deleted]: {path + key}")
            result[key] = "[deleted]"
        else:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                sub_deleted, sub_added, sub_changed, sub_result = compare_dicts(
                    dict1[key], dict2[key], path + key + '/')
                deleted.extend(sub_deleted)
                added.extend(sub_added)
                changed.extend(sub_changed)
                if sub_result:
                    result[key] = sub_result
            elif dict1[key] != dict2[key]:
                changed.append(f"[Changed]: {path + key} : {dict1[key]} -> {dict2[key]}")
                result[key] = f"[changed]: {dict1[key]} -> {dict2[key]}"
    for key in dict2:
        if key not in dict1:
            added.append(f"[Added]: {path + key}")
            result[key] = "[added]"
    return deleted, added, changed, result


def bytes_hash(obj: bytes):
    hex_dig = hashlib.sha256(obj).hexdigest()
    short_hash = int(hex_dig, 16) % (2 ** 16)
    return short_hash
