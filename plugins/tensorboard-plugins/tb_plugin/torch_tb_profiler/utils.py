# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
#
# Copyright(c) 2023 Huawei Technologies.
# All rights reserved
#
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
#
# Modifications: Add visualization of PyTorch Ascend profiling.
# --------------------------------------------------------------------------
import logging
import math
import os
import time
from contextlib import contextmanager
from math import pow

from . import consts


def get_logging_level():
    log_level = os.environ.get('TORCH_PROFILER_LOG_LEVEL', 'INFO').upper()
    if log_level not in logging._levelToName.values():
        log_level = logging.getLevelName(logging.INFO)
    return log_level


logger = None


def get_logger():
    global logger
    if logger is None:
        logger = logging.getLogger(consts.PLUGIN_NAME)
        logger.setLevel(get_logging_level())
    return logger


def is_gpu_chrome_trace_file(path):
    return consts.WORKER_PATTERN.match(path)


def is_worker_span_dir(path):
    return consts.WORKER_SPAN_PATTERN.match(path)


def is_npu_trace_path(path):
    return consts.TRACE_PATTERN.match(path)


def href(text, url):
    """"return html formatted hyperlink string

    Note:
        target="_blank" causes this link to be opened in new tab if clicked.
    """
    return f'<a href="{url}" target="_blank">{text}</a>'


class Canonicalizer:
    def __init__(
            self,
            time_metric='us',
            memory_metric='B',
            *,
            input_time_metric='us',
            input_memory_metric='B'):
        # raw timestamp is in microsecond
        time_metric_to_factor = {
            'us': 1,
            'ms': 1e3,
            's':  1e6,
        }
        # raw memory is in bytes
        memory_metric_to_factor = {
            'B':  pow(1024, 0),
            'KB': pow(1024, 1),
            'MB': pow(1024, 2),
            'GB': pow(1024, 3),
        }

        # canonicalize the memory metric to a string
        self.canonical_time_metrics = {
            'micro': 'us', 'microsecond': 'us', 'us': 'us',
            'milli': 'ms', 'millisecond': 'ms', 'ms': 'ms',
            '': 's', 'second': 's', 's': 's',
        }
        # canonicalize the memory metric to a string
        self.canonical_memory_metrics = {
            '': 'B', 'B': 'B',
            'K': 'KB', 'KB': 'KB',
            'M': 'MB', 'MB': 'MB',
            'G': 'GB', 'GB': 'GB',
        }

        self.time_metric = self.canonical_time_metrics.get(time_metric)
        self.memory_metric = self.canonical_memory_metrics.get(memory_metric)

        # scale factor scale input to output
        self.time_factor = time_metric_to_factor.get(self.canonical_time_metrics.get(input_time_metric)) /\
            time_metric_to_factor.get(self.time_metric)
        self.memory_factor = memory_metric_to_factor.get(self.canonical_memory_metrics.get(input_memory_metric)) /\
            memory_metric_to_factor.get(self.memory_metric)

    def convert_time(self, t):
        return self.time_factor * t

    def convert_memory(self, m):
        return self.memory_factor * m


class DisplayRounder:
    """Round a value for display purpose."""

    def __init__(self, ndigits):
        self.ndigits = ndigits
        self.precision = pow(10, -ndigits)

    def __call__(self, v: float):
        _v = abs(v)
        if _v >= self.precision or v == 0:
            return round(v, 3)
        else:
            ndigit = abs(math.floor(math.log10(_v)))
            return round(v, ndigit)


@contextmanager
def timing(description: str, force: bool = False) -> None:
    if force or os.environ.get('TORCH_PROFILER_BENCHMARK', '0') == '1':
        start = time.time()
        yield
        elapsed_time = time.time() - start
        logger.info(f'{description}: {elapsed_time}')
    else:
        yield
