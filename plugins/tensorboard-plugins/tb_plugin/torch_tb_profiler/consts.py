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
import enum
import re
from collections import namedtuple

PLUGIN_NAME = 'pytorch_profiler'

WORKER_PATTERN = re.compile(r"""^(.*?) # worker name
        (\.\d+)? # optional timestamp like 1619499959628 used as span name
        \.pt\.trace\.json$""", re.X)

TRACE_PATTERN = re.compile(r"""^trace_view\.json$""")
WORKER_SPAN_PATTERN = re.compile(r"""^(.*?)_(\d+(\.\d+)?)_ascend_pt$""")

NODE_PROCESS_PATTERN = re.compile(r"""^(.*)_(\d+)""")
MONITOR_RUN_REFRESH_INTERNAL_IN_SECONDS = 10
MAX_GPU_PER_NODE = 64
MAX_FILE_SIZE = 500 * 1024 * 1024
MAX_LINUX_PATH_LENGTH = 4096
MAX_WINDOWS_PATH_LENGTH = 260

View = namedtuple('View', 'id, name, display_name')
OVERALL_VIEW = View(1, 'overall', 'Overview')
OP_VIEW = View(2, 'operator', 'Operator')
KERNEL_VIEW = View(3, 'kernel', 'Kernel')
TRACE_VIEW = View(4, 'trace', 'Trace')
DISTRIBUTED_VIEW = View(5, 'distributed', 'Distributed')
MEMORY_VIEW = View(6, 'memory', 'Memory')
MODULE_VIEW = View(7, 'module', 'Module')
LIGHTNING_VIEW = View(8, 'lightning', 'Lightning')

TOOLTIP_GPU_UTIL = \
    'GPU Utilization:\n' \
    'GPU busy time / All steps time. The higher, the better. ' \
    'GPU busy time is the time during which there is at least one GPU kernel running on it. ' \
    'All steps time is the total time of all profiler steps(or called as iterations).\n'
TOOLTIP_SM_EFFICIENCY = \
    'Est. SM Efficiency:\n' \
    'Estimated Stream Multiprocessor Efficiency. The higher, the better. ' \
    'This metric of a kernel, SM_Eff_K = min(blocks of this kernel / SM number of this GPU, 100%). ' \
    "This overall number is the sum of all kernels' SM_Eff_K weighted by kernel's execution duration, " \
    'divided by all steps time.\n'
TOOLTIP_OCCUPANCY_COMMON = \
    'Est. Achieved Occupancy:\n' \
    'For most cases such as memory bandwidth bounded kernels, the higher the better. ' \
    'Occupancy is the ratio of active warps on an SM ' \
    'to the maximum number of active warps supported by the SM. ' \
    'The theoretical occupancy of a kernel is upper limit occupancy of this kernel, ' \
    'limited by multiple factors such as kernel shape, kernel used resource, ' \
    'and the GPU compute capability.\n' \
    'Est. Achieved Occupancy of a kernel, OCC_K = ' \
    'min(threads of the kernel / SM number / max threads per SM, theoretical occupancy of the kernel). '
TOOLTIP_OCCUPANCY_OVERVIEW = \
    "This overall number is the weighted average of all kernels' OCC_K " \
    "using kernel's execution duration as weight. " \
    'It shows fine-grained low-level GPU utilization.\n'
TOOLTIP_TENSOR_CORES = \
    'Kernel using Tensor Cores:\n' \
    'Total GPU Time for Tensor Core kernels / Total GPU Time for all kernels.\n'
TOOLTIP_OCCUPANCY_TABLE = \
    "This \"Mean\" number is the weighted average of all calls' OCC_K of the kernel, " \
    "using each call's execution duration as weight. " \
    'It shows fine-grained low-level GPU utilization.'
TOOLTIP_BLOCKS_PER_SM = \
    'Blocks Per SM = blocks of this kernel / SM number of this GPU.\n' \
    'If this number is less than 1, it indicates the GPU multiprocessors are not fully utilized.\n' \
    '\"Mean Blocks per SM\" is the weighted average of all calls of this kernel, ' \
    "using each call's execution duration as weight."
TOOLTIP_OP_TC_ELIGIBLE = \
    'Whether this operator is eligible to use Tensor Cores.'
TOOLTIP_OP_TC_SELF = \
    'Time of self-kernels with Tensor Cores / Time of self-kernels.'
TOOLTIP_OP_TC_TOTAL = \
    'Time of kernels with Tensor Cores / Time of kernels.'
TOOLTIP_KERNEL_USES_TC = \
    'Whether this kernel uses Tensor Cores.'
TOOLTIP_KERNEL_OP_TC_ELIGIBLE = \
    'Whether the operator launched this kernel is eligible to use Tensor Cores.'
TOOLTIP_OP_TC_ELIGIBLE_AICORE = \
    'Whether this operator is eligible to use AICore.'
TOOLTIP_OP_TC_SELF_AICORE = \
    'Time of Device Self Duration With AICore / Device Self Duration.'
TOOLTIP_OP_TC_TOTAL_AICORE = \
    'Time of Device Total Duration With AICore / Device Total Duration.'


class InputFilesType(enum.Enum):
    KERNEL_DETAILS_CSV = 'kernel_details.csv'
    MEMORY_RECORD_CSV = 'memory_record.csv'
    MEMORY_OPERATOR_CSV = 'operator_memory.csv'
    MEMORY_COMPONENT_CSV = 'npu_module_mem.csv'
    OPERATOR_DETAILS_CSV = 'operator_details.csv'
    DISTRIBUTED_STEP_CSV = 'step_trace_time.csv'
    DISTRIBUTED_COMMUNICATION_JSON = 'communication.json'


INPUT_FILE_LIST = [e.value for e in InputFilesType]
