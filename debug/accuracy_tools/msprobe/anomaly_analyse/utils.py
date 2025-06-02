from dataclasses import dataclass
# from msprobe.core.common.file_utils import check_file_or_directory_path
# from msprobe.core.common.const import CompareConst
from debug.accuracy_tools.msprobe.core.common.const import CompareConst
from debug.accuracy_tools.msprobe.core.common.file_utils import check_file_or_directory_path


@dataclass
class RankPath:
    rank: int
    dump_path: str
    construct_path: str
    stack_path: str

    def __init__(self, rank, dump_path, construct_path, stack_path):
        self.rank = rank
        check_file_or_directory_path(dump_path)
        self.dump_path = dump_path
        check_file_or_directory_path(construct_path)
        self.construct_path = construct_path
        check_file_or_directory_path(stack_path)
        self.stack_path = stack_path


class FileCache:
    """
    lazy load file
    """
    instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.instance:
            cls.instance = super().__new__(cls, *args, **kwargs)
        return cls.instance

    def __init__(self):
        self._buffer = {}
        self._frequency = {}
        self._call_cnt = {}

    def load_json(self, json_path):
        if json_path in self._buffer:
            self._call_cnt[json_path] = max(self._call_cnt[json_path] - 1, 0)
            self._frequency[json_path] += 1
            return self._buffer.get(json_path)
        self._memory_control()
        content = load_json(json_path)
        self._buffer[json_path] = content
        self._frequency[json_path] = 1
        self._call_cnt[json_path] = 0
        return content

    def _memory_control(self):
        pass


def is_communication_op(op_name):
    # 定义通信算子的关键字，覆盖各种通信操作，如all_reduce, send, broadcast等
    # 从wrap文件中读取，先硬编码在文件中
    communication_keywords = [
        'send',  # send 算子
        'recv',  # recv 算子
        'broadcast',  # broadcast 算子
        'all_reduce',  # all_reduce 算子
        'reduce',  # reduce 算子
        'all_gather',  # all_gather 算子
        'gather',  # gather 算子
        'isend',  # isend 算子
        'irecv',  # irecv 算子
        'scatter',  # scatter 算子
        'reduce_scatter',  # reduce_scatter 算子
        '_reduce_scatter_base',  # _reduce_scatter_base 算子
        '_all_gather_base',  # _all_gather_base 算子
        'all_to_all_single',  # all_to_all_single 算子
        'all_to_all',  # all_to_all 算子
        'all_gather_into_tensor',  # all_gather_into_tensor 算子
        'reduce_scatter_tensor'  # reduce_scatter_tensor 算子
    ]
    return op_name.startswith('Distributed.') and any(keyword in op_name for keyword in communication_keywords)


def is_ignore_op(op_name):
    ignore_keywords = [
        'Torch.empty'
    ]
    return any(keyword in op_name for keyword in ignore_keywords)


def check_item_anomaly(param):
    def has_nan_inf(dict_obj, key):
        return str(dict_obj.get(key)).lower() in CompareConst.OVERFLOW_LIST

    items = []
    if isinstance(param, list):
        items = param
    elif isinstance(param, dict):
        items = param.values()
    for item in items:
        if not isinstance(item, dict):
            continue
        if has_nan_inf(item, 'Max') or has_nan_inf(item, 'Min'):
            return True
    return False


class AnomalyAnalyseConst:
    P2P_API_MAPPING = {'send': 'recv', 'recv': 'send', 'isend': 'irecv', 'irecv': 'isend'}
    SRC = 'src'
    DST = 'dst'
    LINK = 'link'