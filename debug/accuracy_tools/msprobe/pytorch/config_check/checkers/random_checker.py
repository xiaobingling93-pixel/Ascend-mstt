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


import random
from functools import wraps
from typing import Callable
import inspect
import os
import json
from collections import defaultdict

import numpy as np
import torch
import pandas as pd
from msprobe.pytorch.config_check.config_checker import register_checker_item, register_pre_forward_fun_list
from msprobe.pytorch.common.utils import get_rank_id
from msprobe.core.common.file_utils import create_file_in_zip, load_json, save_excel
from msprobe.pytorch.config_check.checkers.base_checker import BaseChecker
from msprobe.pytorch.config_check.utils.utils import config_checking_print


random_log_dict = defaultdict(dict)


def load_json_files(directory):
    json_data = {}
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if file.startswith('rank') and file.endswith('.json'):
            json_data.update(load_json(file_path))
    return json_data


def get_file_and_line(position):
    parts = position.rsplit(':', 1)
    if len(parts) == 2:
        file_name = os.path.basename(parts[0])
        line_num = parts[1]
        return f"{file_name}:{line_num}"
    return position


def compare_json_files(bench_data, cmp_data):
    results = []
    for op in set(bench_data) | set(cmp_data):
        bench_records = bench_data.get(op, {})
        cmp_records = cmp_data.get(op, {})
        all_positions = set()
        for position in set(bench_records) | set(cmp_records):
            all_positions.add(get_file_and_line(position))

        for position in all_positions:
            bench_count = 0
            cmp_count = 0
            for original_position, count in bench_records.items():
                if get_file_and_line(original_position) == position:
                    bench_count += count
            for original_position, count in cmp_records.items():
                if get_file_and_line(original_position) == position:
                    cmp_count += count
            results.append([op, position, bench_count == cmp_count, bench_count, cmp_count])
    return results


def compare_random(bench_dir='bench', cmp_dir='cmp'):
    bench_data = load_json_files(bench_dir)
    cmp_data = load_json_files(cmp_dir)
    results = compare_json_files(bench_data, cmp_data)
    df = pd.DataFrame(results, columns=RandomChecker.result_header)
    return df


def track_random_call(func: Callable, name: str):
    @wraps(func)
    def wrapper(*args, **kwargs):
        frame = inspect.currentframe()
        caller_frame = frame.f_back
        caller_info = inspect.getframeinfo(caller_frame)
        location = f"{os.path.abspath(caller_info.filename)}:{caller_info.lineno}"
        
        global random_log_dict
        random_log_dict.setdefault(name, {})
        random_log_dict[name][location] = random_log_dict[name].get(location, 0) + 1
        
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            raise e
        finally:
            del frame, caller_frame
            
    return wrapper


def apply_patches():
    random_patches = {
        'random': random.random,
        'randint': random.randint,
        'uniform': random.uniform,
        'choice': random.choice
    }
    for name, func in random_patches.items():
        setattr(random, name, track_random_call(func, f"random.{name}"))
    
    np_random_patches = {
        'rand': np.random.rand,
        'randint': np.random.randint,
        'choice': np.random.choice,
        'normal': np.random.normal
    }
    for name, func in np_random_patches.items():
        setattr(np.random, name, track_random_call(func, f"np.random.{name}"))
    
    torch_patches = {
        'rand': torch.rand,
        'randint': torch.randint,
        'randn': torch.randn,
        'rand_like': torch.rand_like,
        'randint_like': torch.randint_like,
        'randn_like': torch.randn_like,
        'manual_seed': torch.manual_seed
    }
    for name, func in torch_patches.items():
        setattr(torch, name, track_random_call(func, f"torch.{name}"))
    
    tensor_patches = {
        'exponential_': torch.Tensor.exponential_,
        'geometric_': torch.Tensor.geometric_,
        'log_normal_': torch.Tensor.log_normal_,
        'cauchy_': torch.Tensor.cauchy_
    }
    for name, func in tensor_patches.items():
        setattr(torch.Tensor, name, track_random_call(func, f"torch.Tensor.{name}"))
    


@register_checker_item("random")
class RandomChecker(BaseChecker):
    input_needed = None

    target_name_in_zip = "random"
    result_header = ['op', 'position', 'equal', 'bench_count', 'cmp_count']
    write_once = False

    @staticmethod
    def pack(pack_input):
        output_zip_path = pack_input.output_zip_path

        def collect_input(model, args, kwargs, step):
            if RandomChecker.write_once:
                return

            random_log_filepath = os.path.join(RandomChecker.target_name_in_zip, f"rank{get_rank_id()}.json")
            create_file_in_zip(output_zip_path, random_log_filepath, json.dumps(random_log_dict, indent=4))
            config_checking_print(f"add first random_log input features to zip")
            RandomChecker.write_once = True
            
        register_pre_forward_fun_list(collect_input)

    @staticmethod
    def compare(bench_dir, cmp_dir, output_path):
        bench_random_log_pack_path = os.path.join(bench_dir, RandomChecker.target_name_in_zip)
        cmp_random_log_pack_path = os.path.join(cmp_dir, RandomChecker.target_name_in_zip)

        df = compare_random(bench_random_log_pack_path, cmp_random_log_pack_path)
        pass_check = False not in df['equal'].values
        return RandomChecker.target_name_in_zip, pass_check, df
        
