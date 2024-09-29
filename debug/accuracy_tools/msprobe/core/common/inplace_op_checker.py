import os
from msprobe.core.common.file_utils import load_yaml


class InplaceOpChecker:
    OP_FUNCTIONAL = 'functional'
    OP_TENSOR = 'tensor'
    OP_TORCH = 'torch'
    OP_DISTRIBUTED = 'distributed'

    INPLACE_OPS_DICT = None

    @classmethod
    def load_ops(cls):
        if cls.INPLACE_OPS_DICT is None:
            cls.INPLACE_OPS_DICT = dict()
        cur_path = os.path.dirname(os.path.realpath(__file__))
        yaml_path = os.path.join(cur_path, "inplace_ops.yaml")
        all_ops = load_yaml(yaml_path)
        cls.INPLACE_OPS_DICT[cls.OP_FUNCTIONAL] = all_ops.get('inplace_functional_op')
        cls.INPLACE_OPS_DICT[cls.OP_TENSOR] = all_ops.get('inplace_tensor_op')
        cls.INPLACE_OPS_DICT[cls.OP_TORCH] = all_ops.get('inplace_torch_op')
        cls.INPLACE_OPS_DICT[cls.OP_DISTRIBUTED] = all_ops.get('inplace_distributed_op')

    @classmethod
    def check(cls, api, category='distributed'):
        """
            给定api和分类，检查其是否为inplace操作
        """
        if not cls.INPLACE_OPS_DICT:
            cls.load_ops()

        if category not in cls.INPLACE_OPS_DICT.keys():
            return False
        return api in cls.INPLACE_OPS_DICT[category]


InplaceOpChecker.load_ops()
