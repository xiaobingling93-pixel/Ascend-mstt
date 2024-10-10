import numpy as np
import mindspore as ms

from msprobe.core.common.const import Const as CoreConst


class Const:
    CELL = "cell"
    API = "api"
    KERNEL = "kernel"
    TOOL_LEVEL_DICT = {
        CoreConst.LEVEL_L0: CELL,
        CoreConst.LEVEL_L1: API,
        CoreConst.LEVEL_L2: KERNEL
    }
    PYNATIVE_MODE = "pynative"
    GRAPH_GE_MODE = "graph_ge"
    GRAPH_KBYK_MODE = "graph_kbyk"
    JIT_LEVEL = "jit_level"
    JIT_LEVEL_O0 = "O0"
    JIT_LEVEL_O1 = "O1"
    JIT_LEVEL_O2 = "O2"
    ASCEND_910A = "ascend910"

    OPS_PREFIX = "mindspore.ops."
    Tensor_PREFIX = "mindspore.Tensor."
    MINT_PREFIX = "mindspore.mint."
    MINT_NN_FUNC_PREFIX = "mindspore.mint.nn.functional."
    COMM_PREFIX = "mindspore.communication.comm_func."
    COMMUNICATION_API_LIST = [
        "mindspore.communication.comm_func.all_gather_into_tensor",
        "mindspore.communication.comm_func.gather_into_tensor",
        "mindspore.communication.comm_func.all_reduce",
        "mindspore.communication.comm_func.reduce",
        "mindspore.communication.comm_func.reduce_scatter_tensor"
        ]
    TENSOR_DATA_PREFIX = "Tensor."
    STUB_TENSOR_DATA_PREFIX = "Tensor."
    OPS_DATA_PREFIX = "Functional."
    MINT_DATA_PREFIX = "Mint."
    MINT_NN_FUNC_DATA_PREFIX = "MintFunctional."
    DISTRIBUTED_DATA_PREFIX = "Distributed."

    SUPPORTED_API_LIST_FILE = "support_wrap_ops.yaml"
    SUPPORTED_TENSOR_LIST_KEY = "tensor"
    SUPPORTED_OPS_LIST_KEY = "ops"
    SUPPORTED_MINT_LIST_KEY = "mint.ops"
    SUPPORTED__MINT_NN_FUNC_LIST_KEY = "mint.nn.functional"
    SUPPORTED_COMM_LIST_KEY = "communication.comm_func"

    DROPOUT_API_NAME_PREFIX = "dropout"


class FreeBenchmarkConst:
    ADD_NOISE = "add_noise"
    BIT_NOISE = "bit_noise"
    NO_CHANGE = "no_change"
    EXCHANGE_VALUE = "change_value"
    IMPROVE_PRECISION = "improve_precision"
    CHECK = "check"
    FIX = "fix"
    DEFAULT_DEVICE = "npu"
    DEFAULT_STAGE = CoreConst.FORWARD
    DEFAULT_DUMP_LEVEL = "L1"
    DEFAULT_PERT_TYPE = IMPROVE_PRECISION
    DEFAULT_HANDLER_TYPE = CHECK
    DEVICE_LIST = [DEFAULT_DEVICE]
    STAGE_LIST = [CoreConst.FORWARD]
    DUMP_LEVEL_LIST = [DEFAULT_DUMP_LEVEL]
    PERT_TYPE_LIST = [IMPROVE_PRECISION, ADD_NOISE, BIT_NOISE, NO_CHANGE, EXCHANGE_VALUE]
    HANDLER_TYPE_LIST = [CHECK, FIX]
    NO_CHANGE_ERROR_THRESHOLD = 1.0
    SYMBOL_FLIPPING_RATIO = 8.0

    API_PREFIX_DICT = {
        "ops": Const.OPS_PREFIX,
        "Tensor": Const.Tensor_PREFIX,
        "mint": Const.MINT_PREFIX,
        "mint.nn.functional": Const.MINT_NN_FUNC_PREFIX,
        "communication": Const.COMM_PREFIX
    }

    PERT_VALUE_DICT = {
        ms.bfloat16: 1e-4,
        ms.float16: 1e-6,
        ms.float32: 1e-8,
        ms.float64: 1e-16
    }

    ERROR_THRESHOLD = {
        ms.float16: 1.002,
        ms.float32: 1.0002
    }

    PERT_BIT_DICT = {
        ms.float16: np.int16,
        ms.float32: np.int32,
        ms.float64: np.int64
    }

    MS_NUMPY_DTYPE_DICT = {
        ms.int16: np.int16,
        ms.int32: np.int32,
        ms.int64: np.int64,
        ms.float16: np.float16,
        ms.float32: np.float32,
        ms.float64: np.float64
    }
