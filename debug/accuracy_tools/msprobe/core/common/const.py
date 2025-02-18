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

import os
import stat

import numpy as np


class Const:
    """
    Class for const
    """
    TOOL_NAME = "msprobe"

    ipv4_pattern = "([1-9]?\d|1\d{2}|2[0-4]\d|25[0-5])(\.([1-9]?\d|1\d{2}|2[0-4]\d|25[0-5])){3}$"
    SEP = "."
    REGEX_PREFIX_MAX_LENGTH = 20
    REGEX_PREFIX_PATTERN = r"^[a-zA-Z0-9_-]+$"
    REGEX_FORWARD_BACKWARD = r'\.(forward|backward)\.'
    FILE_PATTERN = r'^[a-zA-Z0-9_./-]+$'
    STRING_BLACKLIST = r"^[＋－＝％＠\+\-=%@]|;[＋－＝％＠\+\-=%@]"
    COMMA = ","
    FLOAT_EPSILON = np.finfo(float).eps
    OFF = 'OFF'
    BACKWARD = 'backward'
    FORWARD = 'forward'
    PROGRESS_TIMEOUT = 3000
    EXCEPTION_NONE = None
    JIT = 'Jit'
    PRIMITIVE_PREFIX = 'Primitive'
    DEFAULT_LIST = []
    DEFAULT_PATH = './'
    WHITE_LIST = 'white_list'
    BLACK_LIST = 'black_list'
    DUMP_TENSOR_DATA = 'dump_tensor_data'
    NONE = None
    THREE_SEGMENT = 3
    FOUR_SEGMENT = 4
    SIX_SEGMENT = 6
    SEVEN_SEGMENT = 7
    MAX_DEPTH = 10

    # dump mode
    ALL = "all"
    LIST = "list"
    RANGE = "range"
    STACK = "stack"
    ACL = "acl"
    API_LIST = "api_list"
    API_STACK = "api_stack"
    DUMP_MODE = [ALL, LIST, RANGE, STACK, ACL, API_LIST, API_STACK]
    AUTO = "auto"
    ONLINE_DUMP_MODE = [ALL, LIST, AUTO, OFF]
    SUMMARY = "summary"
    MD5 = "md5"
    VALUE = "value"
    SUMMARY_MODE = [ALL, SUMMARY, MD5]

    WRITE_FLAGS = os.O_WRONLY | os.O_CREAT
    WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR
    OVERWRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC

    PKL_SUFFIX = ".pkl"
    NUMPY_SUFFIX = ".npy"
    NUMPY_PATTERN = "*.npy"
    PT_SUFFIX = ".pt"
    ONE_GB = 1073741824  # 1 * 1024 * 1024 * 1024
    TEN_GB = 10737418240  # 10 * 1024 * 1024 * 1024
    ONE_MB = 1048576  # 1 * 1024 * 1024
    FILE_PATTERN = r'^[a-zA-Z0-9_./-]+$'
    DISTRIBUTED_PREFIX_LENGTH = 60
    # env dump path
    KWARGS = 'kwargs'
    INPUT = 'input'
    OUTPUT = 'output'
    INPUT_ARGS = 'input_args'
    INPUT_KWARGS = 'input_kwargs'
    GRAD_INPUT = 'grad_input'
    GRAD_OUTPUT = 'grad_output'
    PARAMS = 'parameters'
    PARAMS_GRAD = 'parameters_grad'
    START = "start"
    STOP = "stop"
    ENV_ENABLE = "1"
    ENV_DISABLE = "0"
    MAX_SEED_VALUE = 4294967295  # 2**32 - 1
    STATISTICS = "statistics"
    TENSOR = "tensor"
    OVERFLOW_CHECK = "overflow_check"
    FREE_BENCHMARK = "free_benchmark"
    RUN_UT = "run_ut"
    GRAD_PROBE = "grad_probe"
    STRUCTURE = "structure"
    TASK_LIST = [TENSOR, STATISTICS, OVERFLOW_CHECK, FREE_BENCHMARK, RUN_UT, GRAD_PROBE, STRUCTURE]
    DUMP_DATA_COLLECTION_LIST = [STATISTICS, TENSOR, STRUCTURE]
    DUMP_DATA_MODE_LIST = [ALL, INPUT, OUTPUT, FORWARD, BACKWARD]
    LEVEL_L0 = "L0"
    LEVEL_L1 = "L1"
    LEVEL_L2 = "L2"
    LEVEL_MIX = "mix"
    LEVEL_DEBUG = "debug"
    LEVEL_LIST = [LEVEL_L0, LEVEL_L1, LEVEL_L2, LEVEL_MIX, LEVEL_DEBUG]
    ATTR_NAME_PREFIX = "wrap_"
    ATTR_NAME_PREFIX_LEN = len(ATTR_NAME_PREFIX)
    KERNEL_DUMP = "kernel_dump"
    DATA = "data"
    PT_FRAMEWORK = "pytorch"
    MS_FRAMEWORK = "mindspore"
    MT_FRAMEWORK = "mindtorch"
    UNKNOWN_FRAMEWORK = "unknown"
    DIRECTORY_LENGTH = 4096
    FILE_NAME_LENGTH = 255
    FLOAT_TYPE = [np.half, np.single, float, np.double, np.float64, np.longdouble, np.float32, np.float16]
    BOOL_TYPE = [bool, np.uint8]
    INT_TYPE = [np.int32, np.int64]
    NPU = 'NPU'
    NPU_LOWERCASE = 'npu'
    CPU_LOWERCASE = 'cpu'
    CUDA_LOWERCASE = 'cuda'
    DEVICE = 'device'
    DISTRIBUTED = 'Distributed'
    DUMP_PREFIX = ["Distributed", "Functional", "Torch", "Tensor", "Mint", "MintFunctional", "Primitive",
                   "Aten", "VF", "NPU", "Jit"]
    MODULE_PREFIX = ["Module", "Cell"]
    FORWARD_NAME_SUFFIX = ".forward"

    # struct json param
    ORIGIN_DATA = "origin_data"
    SCOPE = "scope"
    STACK = "stack"

    ATEN = "Aten"
    MODULE_WHITE_LIST = ["torch", "numpy"]

    FUNC_SKIP_LIST = ["construct", "__call__"]
    FILE_SKIP_LIST = ["msprobe", "MindSpeed"]
    DATA_TYPE_SKIP_LIST = ["Primitive", "Jit"]

    STACK_FILE_INDEX = 0
    STACK_FUNC_INDEX = 2
    STACK_FUNC_ELE_INDEX = 1

    SCOPE_ID_INDEX = -1
    SCOPE_DIRECTION_INDEX = -2
    TYPE_NAME_INDEX = -3
    PARAMS_GRAD_TYPE_NAME_INDEX = -2
    LAYER_NAME_INDEX = -4
    PARAMS_GRAD_NAME_INDEX = -3
    API_TYPE_INDEX = 0
    LEFT_MOVE_INDEX = -1
    RIGHT_MOVE_INDEX = 1
    LAST_INDEX = -1

    TOP_LAYER = "TopLayer"
    CELL = "Cell"
    MODULE = "Module"
    FRAME_FILE_LIST = ["site-packages/torch", "package/torch", "site-packages/mindspore", "package/mindspore"]
    INPLACE_LIST = [
        "broadcast", "all_reduce", "reduce", "all_gather", "gather", "scatter", "reduce_scatter",
        "_reduce_scatter_base", "_all_gather_base", "send", "recv", "irecv", "isend", "all_to_all_single", "all_to_all",
        "all_gather_into_tensor", "reduce_scatter_tensor"
    ]

    CONVERT = {
        "int32_to_int64": ["torch.int32", "torch.int64"],
        "int64_to_fp32": ["torch.int64", "torch.float32"]
    }

    CONVERT_API = {
        "int32_to_int64": ["cross_entropy"],
        "int64_to_fp32": ["histc"]
    }

    FA_SPECIAL_SPARSE_MODE = [2, 3, 4]

    FILL_CHAR_NUMS = 50
    TOOL_ENDS_SUCCESSFULLY = f"{TOOL_NAME} ends successfully."
    WITHOUT_CALL_STACK = "The call stack retrieval failed."

    STEP = "step"
    RANK = "rank"
    HYPHEN = "-"
    STEP_RANK_MINIMUM_VALUE = 0
    STEP_RANK_MAXIMUM_VALUE = int(1e6)

    # data type const
    TORCH_INT_DTYPE = ["torch.int8", "torch.int32", "torch.int64"]
    TORCH_FLOAT_DTYPE = ["torch.bfloat16", "torch.float16", "torch.float32", "torch.float64"]
    FLOAT16 = "Float16"
    FLOAT32 = "Float32"
    BFLOAT16 = "BFloat16"
    TORCH_FLOAT16 = "torch.float16"
    TORCH_FLOAT32 = "torch.float32"
    TORCH_BFLOAT16 = "torch.bfloat16"

    DTYPE = 'dtype'
    SHAPE = 'shape'
    MAX = 'Max'
    MIN = 'Min'
    MEAN = 'Mean'
    NORM = 'Norm'

    CODE_STACK = 'Code Stack'
    OP_NAME = 'Op Name'
    SCOPE_NAME = 'Scope Name'
    CODE_STACKS = 'Code Stacks'
    FILE_PATH = 'File Path'
    NEW_LINE = '\n'
    CSV_NEWLINE_SEPARATOR = ',\n'
    # 分隔符常量
    SCOPE_SEPARATOR = "/"
    REPLACEMENT_CHARACTER = "_"

    OPTIMIZER = "optimizer"
    CLIP_GRAD = "clip_grad"
    END_PREFIX = "end_"

    TENSOR_STAT_LEN = 2


class CompareConst:
    """
    Class for compare module const
    """
    SPACE = " "
    # compare result column name
    NPU_NAME = "NPU Name"
    BENCH_NAME = "Bench Name"
    NPU_DTYPE = "NPU Dtype"
    BENCH_DTYPE = "Bench Dtype"
    NPU_SHAPE = "NPU Tensor Shape"
    BENCH_SHAPE = "Bench Tensor Shape"
    NPU_MAX = "NPU max"
    NPU_MIN = "NPU min"
    NPU_MEAN = "NPU mean"
    NPU_NORM = "NPU l2norm"
    BENCH_MAX = "Bench max"
    BENCH_MIN = "Bench min"
    BENCH_MEAN = "Bench mean"
    BENCH_NORM = "Bench l2norm"
    MAX_DIFF = "Max diff"
    MIN_DIFF = "Min diff"
    MEAN_DIFF = "Mean diff"
    NORM_DIFF = "L2norm diff"
    COSINE = "Cosine"
    MAX_ABS_ERR = "MaxAbsErr"
    MAX_RELATIVE_ERR = "MaxRelativeErr"
    MIN_RELATIVE_ERR = "MinRelativeErr"
    MEAN_RELATIVE_ERR = "MeanRelativeErr"
    NORM_RELATIVE_ERR = "NormRelativeErr"
    ACCURACY = "Accuracy Reached or Not"
    STACK = "NPU_Stack_Info"
    DATA_NAME = "Data_name"
    ERROR_MESSAGE = "Err_message"
    ONE_THOUSANDTH_ERR_RATIO = "One Thousandth Err Ratio"
    FIVE_THOUSANDTHS_ERR_RATIO = "Five Thousandths Err Ratio"
    NPU_MD5 = "NPU MD5"
    BENCH_MD5 = "BENCH MD5"
    RESULT = "Result"
    MAGNITUDE = 0.5
    OP_NAME = "op_name"
    STRUCT = "struct"
    INPUT_STRUCT = "input_struct"
    KWARGS_STRUCT = "kwargs_struct"
    OUTPUT_STRUCT = "output_struct"
    PARAMS_STRUCT = "params_struct"
    PARAMS_GRAD_STRUCT = "params_grad_struct"
    SUMMARY = "summary"
    COMPARE_RESULT = "compare_result"
    COMPARE_MESSAGE = "compare_message"
    MAX_EXCEL_LENGTH = 1048576
    YES = "Yes"
    NO = "No"
    STATISTICS_INDICATOR_NUM = 4
    EPSILON = 1e-10
    COMPARE_ENDS_SUCCESSFULLY = "msprobe compare ends successfully."
    DEFAULT_RATIO_VALUE = 10000
    THOUSANDTH_PASS_VALUE = 0.999
    ZERO_SHAPE = '(0,)'

    BENCHMARK_COMPARE_ALGORITHM_NAME = "标杆比对法"
    ULP_COMPARE_ALGORITHM_NAME = "ULP误差比对法"
    BINARY_CONSISTENCY_ALGORITHM_NAME = "二进制一致法"
    ABSOLUTE_THRESHOLD_ALGORITHM_NAME = "绝对阈值法"
    THOUSANDTH_STANDARD_ALGORITHM_NAME = "双千指标法"
    ACCUMULATIVE_ERROR_COMPARE_ALGORITHM_NAME = "累积误差比对法"

    ABSOLUTE_THRESHOLD = 'absolute_threshold'
    BINARY_CONSISTENCY = 'binary_consistency'
    ULP_COMPARE = 'ulp_compare'
    THOUSANDTH_STANDARD = 'thousandth_threshold'
    BENCHMARK = 'benchmark'
    ACCUMULATIVE_ERROR_COMPARE = 'accumulative_error_compare'

    SMALL_VALUE_ERR_RATIO = "small_value_err_ratio"
    RMSE_RATIO = "rmse_ratio"
    MAX_REL_ERR_RATIO = "max_rel_err_ratio"
    MEAN_REL_ERR_RATIO = "mean_rel_err_ratio"
    EB_RATIO = "eb_ratio"

    SMALL_VALUE = "small_value"
    RMSE = "rmse"
    MAX_REL_ERR = "max_rel_err"
    MEAN_REL_ERR = "mean_rel_err"
    EB = "eb"

    SMALL_VALUE_ERR_STATUS = "small_value_err_status"
    RMSE_STATUS = "rmse_status"
    MAX_REL_ERR_STATUS = "max_rel_err_status"
    MEAN_REL_ERR_STATUS = "mean_rel_err_status"
    EB_STATUS = "eb_status"

    MEAN_ULP_ERR = "mean_ulp_err"
    ULP_ERR_PROPORTION = "ulp_err_proportion"
    ULP_ERR_PROPORTION_RATIO = "ulp_err_proportion_ratio"

    ULP_ERR_STATUS = "ulp_err_status"

    COMPARE_RESULT_HEADER = [
        NPU_NAME, BENCH_NAME, NPU_DTYPE, BENCH_DTYPE, NPU_SHAPE, BENCH_SHAPE, COSINE, MAX_ABS_ERR, MAX_RELATIVE_ERR,
        ONE_THOUSANDTH_ERR_RATIO, FIVE_THOUSANDTHS_ERR_RATIO,
        NPU_MAX, NPU_MIN, NPU_MEAN, NPU_NORM, BENCH_MAX, BENCH_MIN, BENCH_MEAN, BENCH_NORM, ACCURACY, ERROR_MESSAGE
    ]

    SUMMARY_COMPARE_RESULT_HEADER = [
        NPU_NAME, BENCH_NAME, NPU_DTYPE, BENCH_DTYPE, NPU_SHAPE, BENCH_SHAPE, MAX_DIFF, MIN_DIFF, MEAN_DIFF, NORM_DIFF,
        MAX_RELATIVE_ERR, MIN_RELATIVE_ERR, MEAN_RELATIVE_ERR, NORM_RELATIVE_ERR,
        NPU_MAX, NPU_MIN, NPU_MEAN, NPU_NORM, BENCH_MAX, BENCH_MIN, BENCH_MEAN, BENCH_NORM, RESULT, ERROR_MESSAGE
    ]

    MD5_COMPARE_RESULT_HEADER = [
        NPU_NAME, BENCH_NAME, NPU_DTYPE, BENCH_DTYPE, NPU_SHAPE, BENCH_SHAPE, NPU_MD5, BENCH_MD5, RESULT
    ]

    COMPARE_RESULT_HEADER_STACK = COMPARE_RESULT_HEADER + [STACK]

    SUMMARY_COMPARE_RESULT_HEADER_STACK = SUMMARY_COMPARE_RESULT_HEADER + [STACK]

    MD5_COMPARE_RESULT_HEADER_STACK = MD5_COMPARE_RESULT_HEADER + [STACK]

    HEAD_OF_COMPARE_MODE = {
        Const.ALL: COMPARE_RESULT_HEADER,
        Const.SUMMARY: SUMMARY_COMPARE_RESULT_HEADER,
        Const.MD5: MD5_COMPARE_RESULT_HEADER
    }

    ALL_COMPARE_INDEX = [COSINE, MAX_ABS_ERR, MAX_RELATIVE_ERR, ONE_THOUSANDTH_ERR_RATIO, FIVE_THOUSANDTHS_ERR_RATIO]
    SUMMARY_COMPARE_INDEX = [MAX_DIFF, MIN_DIFF, MEAN_DIFF, NORM_DIFF,
                             MAX_RELATIVE_ERR, MIN_RELATIVE_ERR, MEAN_RELATIVE_ERR, NORM_RELATIVE_ERR]

    # dtype match
    MS_TYPE = [
        [Const.FLOAT16, Const.FLOAT32], [Const.FLOAT32, Const.FLOAT16],
        [Const.FLOAT16, Const.BFLOAT16], [Const.BFLOAT16, Const.FLOAT16]
    ]
    TORCH_TYPE = [
        [Const.TORCH_FLOAT16, Const.TORCH_FLOAT32], [Const.TORCH_FLOAT32, Const.TORCH_FLOAT16],
        [Const.TORCH_FLOAT16, Const.TORCH_BFLOAT16], [Const.TORCH_BFLOAT16, Const.TORCH_FLOAT16]
    ]

    # read_op
    IO_NAME_MAPPING = {
        Const.INPUT_ARGS: '.input',
        Const.INPUT_KWARGS: '.input',
        Const.INPUT: '.input',
        Const.OUTPUT: '.output',
        Const.PARAMS: '.parameters'
    }

    # state to struct mapping
    STATE_TO_STRUCT_MAPPING = {
        Const.INPUT: INPUT_STRUCT,
        Const.KWARGS: INPUT_STRUCT,
        Const.OUTPUT: OUTPUT_STRUCT,
        Const.PARAMS: PARAMS_STRUCT,
        Const.PARAMS_GRAD: PARAMS_GRAD_STRUCT
    }

    STRUCT_COMPARE_KEY = [
        INPUT_STRUCT,
        OUTPUT_STRUCT,
        PARAMS_STRUCT,
        PARAMS_GRAD_STRUCT
    ]

    # compare standard
    HUNDRED_RATIO_THRESHOLD = 0.01
    THOUSAND_RATIO_THRESHOLD = 0.001
    FIVE_THOUSAND_RATIO_THRESHOLD = 0.005
    TEN_THOUSAND_RATIO_THRESHOLD = 0.0001
    COSINE_THRESHOLD = 0.9999
    ULP_FLOAT32_THRESHOLD = 32
    ULP_FLOAT16_THRESHOLD = 1

    # compare result data
    READ_NONE = 'No data'
    NONE = 'None'
    SHAPE_UNMATCH = 'shape unmatched'
    DIFF = 'Different'
    UNSUPPORTED = 'unsupported'
    NAN = 'Nan'
    PASS = 'pass'
    WARNING = 'Warning'
    ERROR = 'error'
    TRUE = 'TRUE'
    FALSE = 'FALSE'
    SKIP = 'SKIP'
    N_A = 'N/A'
    INF = 'inf'
    NEG_INF = '-inf'
    BFLOAT16_MIN = -3.3895313892515355e+38
    BFLOAT16_MAX = 3.3895313892515355e+38
    BFLOAT16_EPS = 3.90625e-3  # 2 ** -8

    # accuracy standards
    COS_THRESHOLD = 0.99
    MAX_ABS_ERR_THRESHOLD = 0.001
    MAX_RELATIVE_ERR_THRESHOLD = 0.001
    COS_MAX_THRESHOLD = 0.9
    MAX_ABS_ERR_MAX_THRESHOLD = 1
    ACCURACY_CHECK_YES = "Yes"
    ACCURACY_CHECK_NO = "No"
    ACCURACY_CHECK_UNMATCH = "Unmatched"

    # error message
    NO_BENCH = "No bench data matched."

    # compare const
    FLOAT_TYPE = [np.half, np.single, float, np.double, np.float64, np.longdouble]

    # highlight xlsx color const
    RED = "FFFF0000"
    YELLOW = "FFFF00"
    BLUE = "0000FF"

    # run_ut const
    MAX_TOKENS = 65536
    SPECIAL_SPARSE_MOED = 4

    # highlight rules const
    OVERFLOW_LIST = ['nan\t', 'inf\t', '-inf\t', 'nan', 'inf', '-inf']
    MAX_DIFF_RED = 1e+10
    ORDER_MAGNITUDE_DIFF_YELLOW = 1
    ONE_THOUSAND_ERROR_IN_RED = 0.9
    ONE_THOUSAND_ERROR_OUT_RED = 0.6
    ONE_THOUSAND_ERROR_DIFF_YELLOW = 0.1
    COSINE_DIFF_YELLOW = 0.1
    MAX_RELATIVE_OUT_RED = 0.5
    MAX_RELATIVE_OUT_YELLOW = 0.1
    MAX_RELATIVE_IN_YELLOW = 0.01
    MS_GRAPH_BASE = {
        NPU_NAME: None, BENCH_NAME: None, NPU_DTYPE: None, BENCH_DTYPE: None, NPU_SHAPE: None, BENCH_SHAPE: None,
        NPU_MAX: None, NPU_MIN: None, NPU_MEAN: None, NPU_NORM: None, BENCH_MAX: None, BENCH_MIN: None,
        BENCH_MEAN: None, BENCH_NORM: None, ACCURACY: '', ERROR_MESSAGE: ''
    }
    MS_GRAPH_NPY = {
        COSINE: None, MAX_ABS_ERR: None, MAX_RELATIVE_ERR: None, ONE_THOUSANDTH_ERR_RATIO: None,
        FIVE_THOUSANDTHS_ERR_RATIO: None
    }
    MS_GRAPH_STATISTIC = {
        MAX_DIFF: None, MIN_DIFF: None, MEAN_DIFF: None, NORM_DIFF: None, MAX_RELATIVE_ERR: None,
        MIN_RELATIVE_ERR: None, MEAN_RELATIVE_ERR: None, NORM_RELATIVE_ERR: None
    }
    INPUT_PATTERN = Const.SEP + Const.INPUT + Const.SEP
    KWARGS_PATTERN = Const.SEP + Const.KWARGS + Const.SEP
    OUTPUT_PATTERN = Const.SEP + Const.OUTPUT + Const.SEP
    PARAMS_PATTERN = Const.SEP + Const.PARAMS + Const.SEP
    PARAMS_GRAD_PATTERN = Const.SEP + Const.PARAMS_GRAD + Const.SEP
    COMPARE_KEY = 'compare_key'
    COMPARE_SHAPE = 'compare_shape'
    INTERNAL_API_MAPPING_FILE = 'ms_to_pt_api.yaml'
    UNREADABLE = 'unreadable data'


class FileCheckConst:
    """
    Class for file check const
    """
    READ_ABLE = "read"
    WRITE_ABLE = "write"
    READ_WRITE_ABLE = "read and write"
    DIRECTORY_LENGTH = 4096
    FILE_NAME_LENGTH = 255
    FILE_VALID_PATTERN = r"^[a-zA-Z0-9_.:/-]+$"
    FILE_PATTERN = r'^[a-zA-Z0-9_./-]+$'
    PKL_SUFFIX = ".pkl"
    NUMPY_SUFFIX = ".npy"
    JSON_SUFFIX = ".json"
    PT_SUFFIX = ".pt"
    CSV_SUFFIX = ".csv"
    XLSX_SUFFIX = ".xlsx"
    YAML_SUFFIX = ".yaml"
    IR_SUFFIX = ".ir"
    MAX_PKL_SIZE = 1073741824  # 1 * 1024 * 1024 * 1024
    MAX_NUMPY_SIZE = 10737418240  # 10 * 1024 * 1024 * 1024
    MAX_JSON_SIZE = 1073741824  # 1 * 1024 * 1024 * 1024
    MAX_PT_SIZE = 10737418240  # 10 * 1024 * 1024 * 1024
    MAX_CSV_SIZE = 1073741824  # 1 * 1024 * 1024 * 1024
    MAX_XLSX_SIZE = 1073741824  # 1 * 1024 * 1024 * 1024
    MAX_YAML_SIZE = 1073741824  # 1 * 1024 * 1024 * 1024
    MAX_IR_SIZE = 1073741824  # 1 * 1024 * 1024 * 1024
    COMMOM_FILE_SIZE = 1048576  # 1 * 1024 * 1024
    DIR = "dir"
    FILE = "file"
    DATA_DIR_AUTHORITY = 0o750
    DATA_FILE_AUTHORITY = 0o640
    FILE_SIZE_DICT = {
        PKL_SUFFIX: MAX_PKL_SIZE,
        NUMPY_SUFFIX: MAX_NUMPY_SIZE,
        JSON_SUFFIX: MAX_JSON_SIZE,
        PT_SUFFIX: MAX_PT_SIZE,
        CSV_SUFFIX: MAX_CSV_SIZE,
        XLSX_SUFFIX: MAX_XLSX_SIZE,
        YAML_SUFFIX: MAX_YAML_SIZE,
        IR_SUFFIX: MAX_IR_SIZE
    }
    CSV_BLACK_LIST = r'^[＋－＝％＠\+\-=%@]|;[＋－＝％＠\+\-=%@]'


class OverflowConst:
    """
    Class for Overflow
    """
    OVERFLOW_ORIGINAL_MODE = 0
    OVERFLOW_DEBUG_MODE = 1


class MsCompareConst:
    # api_info field
    MINT = "Mint"
    MINT_FUNCTIONAL = "MintFunctional"
    TENSOR_API = "Tensor"

    API_NAME_STR_LENGTH = 4
    MAX_RECURSION_DEPTH = 20

    # Mindtorch api_info field
    MINDTORCH_TENSOR = "Tensor"
    MINDTORCH = "Torch"
    MINDTORCH_FUNC = "Functional"
    MINDTORCH_NPU = "NPU"
    MINDTORCH_DIST = "Distributed"



    MT_VALID_API_TYPES = [
        MINDTORCH, MINDTORCH_FUNC, MINDTORCH_TENSOR
    ]

    TASK_FIELD = "task"
    STATISTICS_TASK = "statistics"
    FRAMEWORK = "framework"
    TENSOR_TASK = "tensor"
    DUMP_DATA_DIR_FIELD = "dump_data_dir"
    DATA_FIELD = "data"

    # supported api yaml
    SUPPORTED_API_LIST_FILE = "checker_support_api.yaml"
    SUPPORTED_TENSOR_LIST_KEY = "tensor"

    # detail_csv
    DETAIL_CSV_API_NAME = "API Name"
    DETAIL_CSV_BENCH_DTYPE = "Bench Dtype"
    DETAIL_CSV_TESTED_DTYPE = "Tested Dtype"
    DETAIL_CSV_SHAPE = "Shape"
    DETAIL_CSV_PASS_STATUS = "Status"
    DETAIL_CSV_MESSAGE = "Message"
    DETAIL_CSV_FILE_NAME = "accuracy_checking_details"

    # result_csv
    RESULT_CSV_FORWARD_TEST_SUCCESS = "Forward Test Success"
    RESULT_CSV_BACKWARD_TEST_SUCCESS = "Backward Test Success"
    RESULT_CSV_FILE_NAME = "accuracy_checking_result"

    EPSILON = 1e-8

    class ProcessStatus:
        SUCCESS = "success"
        API_NOT_FOUND = "api_not_found"
        EXCEPTION_SKIP = "exception_skip"


class MsgConst:
    """
    Class for log messages const
    """
    MSPROBE_LOG_LEVEL = "MSPROBE_LOG_LEVEL"
    LOG_LEVEL_ENUM = ["0", "1", "2", "3", "4"]
    LOG_LEVEL = ["DEBUG", "INFO", "WARNING", "ERROR"]

    class LogLevel:
        class DEBUG:
            value = 0

        class INFO:
            value = 1

        class WARNING:
            value = 2

        class ERROR:
            value = 3

    SPECIAL_CHAR = ["\n", "\r", "\u007F", "\b", "\f", "\t", "\u000B", "%08", "%0a", "%0b", "%0c", "%0d", "%7f"]

    NOT_CREATED_INSTANCE = "PrecisionDebugger instance is not created."


class GraphMode:
    NPY_MODE = "NPY_MODE"
    STATISTIC_MODE = "STATISTIC_MODE"
    ERROR_MODE = "ERROR_MODE"


class MonitorConst:
    """
    Class for monitor const
    """
    OP_LIST = ["norm", "min", "max", "zeros", "nans", "id", "mean"]
    MONITOR_OUTPUT_DIR = "MONITOR_OUTPUT_DIR"
    DEFAULT_MONITOR_OUTPUT_DIR = "./monitor_output"
    DATABASE = "database"
    EMAIL = "email"
    OPT_TY = ['Megatron_DistributedOptimizer', 'Megatron_Float16OptimizerWithFloat16Params']
    DEEPSPEED_OPT_TY = (
        "DeepSpeedZeroOptimizer_Stage0",
        "DeepSpeedZeroOptimizer_Stage1_or_2",
        "DeepSpeedZeroOptimizer_Stage3"
    )
    DEEPSPEED_ZERO_OPT_FILTER = "DeepSpeedZeroOptimizer"
    RULE_NAME = ['AnomalyTurbulence']

    SLICE_SIZE = 20480
    # used for name
    DOT = "."
    NAME_SEP = ":"
    INPUT_GRAD = "input_grad"
    OUTPUT_GRAD = "output_grad"
    ACTV_IN = "input"
    ACTV_OUT = "output"
    ACTVGRAD_IN = "input_grad"
    ACTVGRAD_OUT = "output_grad"
    # used for tasks
    ACTV = "actv"
    ACTVGRAD = "actv_grad"
    POST_GRAD = "post_grad"
    PRE_GRAD = "pre_grad"
    ACC_GRAD = "acc_grad"
    PREFIX_POST = "post"
    PREFIX_PRE = "pre"
    EXP_AVG = "exp_avg"
    EXP_AVG_SQ = "exp_avg_sq"
    PARAM = "param"

    CSV_HEADER = ["vpp_stage", "name", "step"]
    CSV_HEADER_XY = ["vpp_stage", "name", "step", "micro_step"]
    OUTPUT_DIR_PATTERN = r"([\w-]{0,20})-rank(\d{1,5})-"
    ANOMALY_JSON = "anomaly.json"
    ANALYSE_JSON = "anomaly_analyse.json"
    TENSORBOARD = "tensorboard"
    CSV = "csv"
    API = "api"
    HEADER_NAME = 'name'
