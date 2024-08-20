# Copyright 2024 Huawei Technologies Co., Ltd
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
# ============================================================================

import os
import mindspore as ms
from msprobe.mindspore.dump.hook_cell.hook_cell import HOOKCell
from msprobe.core.common.utils import Const, load_yaml


cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")


def load_ops_functions():
    ops_func = {f: getattr(ms.ops, f) for f in dir(ms.ops)}
    mint_ops_func = {f: getattr(ms.mint, f) for f in dir(ms.mint)}
    mint_func_ops_func = {f: getattr(ms.mint.nn.functional, f) for f in dir(ms.mint.nn.functional)}
    return ops_func, mint_ops_func, mint_func_ops_func


def get_functional_ops():
    ops_func, mint_ops_func, mint_func_ops_func = load_ops_functions()
    config = load_yaml(yaml_path)
    wrap_functional = config.get("ops")
    wrap_mint = config.get("mint.ops")
    wrap_mint_functional = config.get("mint.nn.functional")
    return (
        set(wrap_functional) & set(ops_func.keys()),
        set(wrap_mint) & set(mint_ops_func.keys()),
        set(wrap_mint_functional) & set(mint_func_ops_func.keys())
    )


class HOOKFunctionalOP(object):
    pass


class HOOKMintOP(object):
    pass


class HOOKMintNNFunctionalOP(object):
    pass


class FunctionalOPTemplate(HOOKCell):
    def __init__(self, op_name, op_dict, prefix, hook):
        self.op_name = op_name
        self.op_func = op_dict[op_name]
        self.prefix_op_name_ = prefix + str(op_name.split(Const.SEP)[-1]) + Const.SEP
        super().__init__(hook)

    def construct(self, *args, **kwargs):
        if self.op_name.startswith('dropout'):
            return args[0] if args else kwargs.get('input')
        return self.op_func(*args, **kwargs)


def wrap_functional_op(op_name, op_dict, prefix, hook):
    def op_template(*args, **kwargs):
        return FunctionalOPTemplate(op_name, op_dict, prefix, hook)(*args, **kwargs)
    return op_template


def wrap_functional_ops_and_bind(ops, op_dict, prefix, hook, hook_class):
    for op_name in ops:
        if callable(op_dict[op_name]):
            setattr(hook_class, Const.ATTR_NAME_PREFIX + op_name, wrap_functional_op(op_name, op_dict, prefix, hook))


def setup_hooks(hook):
    functional_ops, mint_ops, mint_func_ops = get_functional_ops()
    wrap_functional_ops_and_bind(
        functional_ops, {f: getattr(ms.ops, f) for f in dir(ms.ops)}, "Functional.", hook, HOOKFunctionalOP)
    wrap_functional_ops_and_bind(
        mint_ops, {f: getattr(ms.mint, f) for f in dir(ms.mint)}, "Mint.", hook, HOOKMintOP)
    wrap_functional_ops_and_bind(
        mint_func_ops, {f: getattr(ms.mint.nn.functional, f) for f in dir(ms.mint.nn.functional)}, "MintFunctional.", hook, HOOKMintNNFunctionalOP)

