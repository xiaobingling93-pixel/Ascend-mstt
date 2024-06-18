import os
from functools import wraps
from collections import defaultdict
import yaml
import re
import inspect
import functools
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.hooks as full_hooks

from ..module_metric import get_metrics

try:
    import torch_npu
except ImportError:
    is_gpu = True
else:
    is_gpu = False


cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "distributed_ops.yaml")
with open(yaml_path) as f:
    WrapDistributedOps = yaml.safe_load(f).get('distributed')

npu_distributed_api = ['isend', 'irecv']

distributed_func = {}
for f in dir(dist):
    distributed_func[f] = getattr(dist, f)

ORIGIN_WAIT = getattr(dist.Work, 'wait')
PENDING_ASYNC_CC_BY_HANDLE = {}


def get_distributed_ops():
    global WrapDistributedOps
    _all_distributed_ops = dir(dist)
    return set(WrapDistributedOps) & set(_all_distributed_ops)


class DistributedOPTemplate(nn.Module):
    def __init__(self, op_name, hook):
        super(DistributedOPTemplate, self).__init__()
        self.op_name_ = op_name
        self.prefix_op_name_ = str(op_name)
        self.register_forward_hook(hook(), with_kwargs=True)

    def forward(self, *args, **kwargs):
        return distributed_func.get(self.op_name_)(*args, **kwargs)


class ApiRegistry:
    def __init__(self):
        self.distributed_attr_origin = {}
        self.distributed_attr_hooked = {}

    @staticmethod
    def store_ori_attr(ori_api_group, api_list, api_ori_attr):
        for api in api_list:
            if '.' in api:
                sub_module_name, sub_op = api.rsplit('.', 1)
                sub_module = getattr(ori_api_group, sub_module_name)
                api_ori_attr[api] = getattr(sub_module, sub_op)
            else:
                api_ori_attr[api] = getattr(ori_api_group, api)

    @staticmethod
    def set_api_attr(api_group, attr_dict):
        for cc_api_name, cc_api_entry_func in attr_dict.items():
            if '.' in cc_api_name:
                sub_module_name, sub_op = cc_api_name.rsplit('.', 1)
                sub_module = getattr(api_group, sub_module_name, None)
                if sub_module is not None:
                    setattr(sub_module, sub_op, cc_api_entry_func)
            else:
                setattr(api_group, cc_api_name, cc_api_entry_func)

    def redirect_api(self):
        self.set_api_attr(dist, self.distributed_attr_hooked)
        self.set_api_attr(dist.distributed_c10d, self.distributed_attr_hooked)
        self.redirect_wait()

    def restore_api(self):
        self.set_api_attr(dist, self.distributed_attr_origin)
        self.set_api_attr(dist.distributed_c10d, self.distributed_attr_origin)
        setattr(dist.Work, 'wait', ORIGIN_WAIT)
   
    def initialize_hook(self, hook):
        self.store_ori_attr(dist, get_distributed_ops(), self.distributed_attr_origin)
        for op_name in get_distributed_ops():
            self.distributed_attr_hooked[op_name] = DistributedOPTemplate(op_name, hook)

    def redirect_wait(self):
        global ORIGIN_WAIT
        global PENDING_ASYNC_CC_BY_HANDLE

        def wrapped_wait(work):
            def wrapped_wait(*args, **kwargs):
                ORIGIN_WAIT(*args, **kwargs)
                if args[0] in PENDING_ASYNC_CC_BY_HANDLE:
                    store_func = PENDING_ASYNC_CC_BY_HANDLE.pop(args[0])
                    store_func()
            return wrapped_wait

        dist.Work.wait = wrapped_wait(dist.Work)


def get_callstack():
    callstack = []
    for (_, path, line, func, code, _) in inspect.stack():
        stack_line = f'{path}[{line}]'
        callstack.append(stack_line)
    return callstack

def op_aggregate(op, t1, t2):
    if op == 'min':
        return min(t1, t2)
    if op == 'max':
        return max(t1, t2)
    if op == 'norm':
        return (t1**2+t2**2)**0.5
    if op == 'zeros': # TODO wrong
        return (t1+t2)/2

def update_data(old, new):
    updated = {op:{} for op in new.keys()}
    if old:
        for op, tag2tensor in old.items():
            for tag, t_old in tag2tensor.items():
                t_new = new[op][tag]
                updated[op][tag] = op_aggregate(op, t_old, t_new)     
    else:
        updated = new
    return updated

def is_target_line(codeline):
    stack = get_callstack()
    whole_stack = ';'.join(stack)
    if codeline == []:
        return True
    for pattern in codeline:
        if re.search(pattern, whole_stack):
            return True
    return False

def catch_data(cc_context, ops, module, args, out=None):
    tensor_args = {}
    for arg in args:
        if isinstance(arg, torch.Tensor):
            tensor_args[f'input_{len(tensor_args)}'] = arg
        elif isinstance(arg, list):
            arg = torch.stack(arg)
            tensor_args[f'input_{len(tensor_args)}'] = arg
    new_data = {op: get_metrics(op, tensor_args, 1e-8) for op in ops}
    cc_context.indata=update_data(cc_context.indata, new_data)
    if out and isinstance(out, dist.Work):
        tensor_res = {}
        for res in out.result():
            if isinstance(res, torch.Tensor):
                tensor_res[f'output_{len(tensor_res)}'] = res
        new_data = {op: get_metrics(op, tensor_res, 1e-8) for op in ops}
        cc_context.outdata=update_data(cc_context.outdata, new_data)

def create_store_func(context, ops, module, args, out):
    def store_data():
        catch_data(context, ops, module, args, out)
    return store_data

def create_hook(context, monitor):
    def cc_hook(module, args, kwargs, out=None):
        if monitor.cc_log_only:
            stack = ';'.join(get_callstack()[4:7])
            monitor.cc_logged_stack[module.prefix_op_name_].add(stack)
            return out
        args = args + tuple(kwargs.values())
        if (dist.is_initialized() and dist.get_rank() not in monitor.module_rank_list and monitor.module_rank_list != []):
            return out
        if not is_target_line(monitor.cc_codeline):
            return out
        if out: # async
            PENDING_ASYNC_CC_BY_HANDLE[out] = create_store_func(context[module.prefix_op_name_], monitor.ops, module, args, out)
            return out
        catch_data(context[module.prefix_op_name_], monitor.ops, module, args, out)
        return out
    return cc_hook

api_register = ApiRegistry()

