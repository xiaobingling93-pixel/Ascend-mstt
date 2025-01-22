# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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

import argparse
import os
import re
from glob import glob

import pandas as pd

from msprobe.pytorch.common.log import logger


def parse_logfile(logfile):
    grad_norm = []
    step = []
    with open(logfile) as f:
        for line in f.readlines():
            if 'consumed samples' in line:
                grad_norm.append(float(re.findall('(?<=grad norm\: )[\d\.]*', line)[0]))
    return grad_norm


def parse_monitor_output(output_dir):
    reduced = {}
    unreduced = {}
    for directory in glob(output_dir + '*'):
        rank = int(re.findall('(?<=rank)[\d]*', directory)[0])
        unreduced[rank] = []
        reduced[rank] = []
        for file in os.listdir(directory):
            df = pd.read_csv(os.path.join(directory, file))
            if '_unreduced_' in file:
                unreduced[rank].append(df)
                pass
            elif '_reduced_' in file:
                reduced[rank].append(df)
            else:
                logger.info(f'unexpected file {file} in {directory}')
    return reduced, unreduced


def valid_reduce(reduced, unreduced, tp_size, dp_size, sequence_parallel):
    steps = len(reduced[0])
    world_size = len(reduced)
    errors = []
    for _, row in unreduced[0][0].iterrows():
        param = row['param_name']
        is_tp_duplicate = False
        for step in range(2):
            # sum reduced
            reduced_mean = 0.
            for rank in range(world_size):
                if len(reduced[rank]) == 0:
                    continue
                df = reduced[rank][step]
                value = list(df[df['param_name'] == param]['mean'])
                if not value:
                    if step == 0:
                        is_tp_duplicate = True
                    continue
                reduced_mean += value[0]

            # sum unreduced
            unreduced_mean = 0.
            for rank in range(world_size):
                df = unreduced[rank][step]
                value = list(df[df['param_name'] == param]['mean'])
                if not value:
                    continue
                unreduced_mean += list(df[df['param_name'] == param]['mean'])[0]

            unreduced_mean /= dp_size
            if is_tp_duplicate and (not sequence_parallel or 'embedding' in param):
                unreduced_mean /= tp_size
            try:
                assert_equal(unreduced_mean, reduced_mean)
            except AssertionError as e:
                errors.append([param, step, e, is_tp_duplicate])
    if errors:
        logger.info(errors)
    else:
        logger.info(f'grad mean is in consist between unreduced grad and reduced grad monitord.')


def assert_equal(a, b):
    if b == 0 or a == 0:
        return
    if b == 0:
        rel_diff = a
    elif a == 0:
        rel_diff = b
    else:
        rel_diff = abs(a / b - 1)
    assert rel_diff < 0.01, f'{a}, {b}, {rel_diff}'


def valid_total_norm(total_norm, reduced, duplicate_embedding):
    steps = len(total_norm)
    world_size = len(reduced)
    errors = []
    for step in range(steps):
        calculated_norm = 0.
        for rank in range(world_size):
            if len(reduced[rank]) == 0:
                if step == 0:
                    logger.info(f'rank {rank} is duplicated in dp group')
                continue
            for _, row in reduced[rank][step].iterrows():
                if duplicate_embedding and 'word_embedding' in row['param_name']:
                    continue
                calculated_norm += row['norm'] ** 2
        try:
            assert_equal(calculated_norm ** 0.5, total_norm[step])
        except AssertionError as e:
            errors.append([step, e])
    if errors:
        logger.info('total norm errors: ', errors)
    else:
        logger.info('grad norm in consist between training log and reduced gradients monitored')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--monitor_output', '-m', type=str, required=True,
                        help='path prefix to the output of monitor e.g. monitor_output/Aug12_07-16')
    parser.add_argument('--logfile', '-l', type=str, required=True, help='path to the training log file')
    parser.add_argument('--tp_size', '-t', type=int, required=True, help='tp parallel size')
    parser.add_argument('--dp_size', '-d', type=int, required=True, help='dp parallel size')
    parser.add_argument('--pp_size', '-p', type=int, required=True, help='pp parallel size')
    parser.add_argument('--untie_embeddings_and_output_weights', '-u', action="store_true", default=False,
                        help='whether untie_embeddings_and_output_weights in pp parallel')
    parser.add_argument('--sequence_parallel', '-s', action="store_true", default=False,
                        help='whether sequence parallel is enabled. Add -s to store true')

    args = parser.parse_args()

    assert args.tp_size > 0, 'if tp not enabled, set tp_size = 1'
    assert args.dp_size > 0, 'if tp not enabled, set dp_size = 1'
    assert args.pp_size > 0, 'if tp not enabled, set pp_size = 1'

    total_norm = parse_logfile(args.logfile)
    reduced, unreduced = parse_monitor_output(args.monitor_output)

    duplicate_embedding = not args.untie_embeddings_and_output_weights and args.pp_size > 1

    valid_total_norm(total_norm, reduced, duplicate_embedding)
    valid_reduce(reduced, unreduced, args.tp_size, args.dp_size, args.sequence_parallel)
