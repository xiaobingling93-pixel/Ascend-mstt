import numpy as np
import mindspore as ms

from msprobe.core.common.const import Const as CoreCost


class Const:
    CELL = "cell"
    API = "api"
    KERNEL = "kernel"
    TOOL_LEVEL_DICT = {
        CoreCost.LEVEL_L0: CELL,
        CoreCost.LEVEL_L1: API,
        CoreCost.LEVEL_L2: KERNEL
    }
    PYNATIVE_MODE = "pynative"
    GRAPH_GE_MODE = "graph_ge"
    GRAPH_KBYK_MODE = "graph_kbyk"


class FreeBenchmarkConst:
    DEFAULT_DEVICE = "npu"
    DEFAULT_STAGE = "forward"
    DEFAULT_DUMP_LEVEL = CoreCost.LEVEL_L1
    DEFAULT_PERT_TYPE = "improve_precision"
    DEFAULT_HANDLER_TYPE = "check"
    FIX_HANDLER_MODE = "fix"
    ADD_NOISE = "add_noise"
    BIT_NOISE = "bit_noise"
    NO_CHANGE = "no_change"
    IMPROVE_PRECISION = "improve_precision"
    CHECK = "check"
    FIX = "fix"
    DEVICE_LIST = ["npu"]
    STAGE_LIST = ["forward"]
    DUMP_LEVEL_LIST = [CoreCost.LEVEL_L1]
    PERT_TYPE_LIST = [IMPROVE_PRECISION, ADD_NOISE, BIT_NOISE, NO_CHANGE]
    HANDLER_TYPE_LIST = [CHECK, FIX]
    COMMUNICATION_API_LIST = [
        "mindspore.communication.comm_func.all_gather_into_tensor",
        "mindspore.communication.comm_func.gather_into_tensor",
        "mindspore.communication.comm_func.all_reduce",
        "mindspore.communication.comm_func.reduce",
        "mindspore.communication.comm_func.reduce_scatter_tensor"
        ]
    NO_CHANGE_ERROR_THRESHOLD = 1.0
    SYMBOL_FLIPPING_RATIO = 8.0
    OPS_PREFIX = "mindspore.ops."
    Tensor_PREFIX = "mindspore.Tensor."
    MINT_PREFIX = "mindspore.mint."
    MINT_NN_FUNC_PREFIX = "mindspore.mint.nn.functional."
    COMM_PREFIX = "mindspore.communication.comm_func."

    API_PREFIX_DICT = {
        "ops": OPS_PREFIX,
        "Tensor": Tensor_PREFIX,
        "mint": MINT_PREFIX,
        "mint.nn.functional": MINT_NN_FUNC_PREFIX,
        "communication": COMM_PREFIX
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
