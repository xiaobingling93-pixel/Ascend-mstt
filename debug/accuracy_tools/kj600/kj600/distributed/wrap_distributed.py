import os
import yaml
import re
import inspect
import torch
import torch.nn as nn
import torch.distributed as dist

from ..module_metric import get_metrics

try:
    import torch_npu
except ImportError:
    pass

PREFIX_POST = "post"

OpsPath = os.path.join(os.path.dirname(__file__), "distributed_ops.yaml")
with open(OpsPath) as f:
    WrapDistributedOps = yaml.safe_load(f).get('distributed')

StackBlackListPath = os.path.join(os.path.dirname(__file__), "stack_blacklist.yaml")
with open(StackBlackListPath) as f:
    StackBlackList = yaml.safe_load(f).get('stack')

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
    def __init__(self, op_name, pre_hooks, post_hooks):
        super(DistributedOPTemplate, self).__init__()
        self.op_name_ = str(op_name)
        self.__name__ = self.op_name_
        for pre_hook in pre_hooks:
            self.register_forward_pre_hook(pre_hook, with_kwargs=True)
        for hook in post_hooks:
            self.register_forward_hook(hook, with_kwargs=True)

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
   
    def initialize_hook(self, pre_hooks, post_hooks):
        self.store_ori_attr(dist, get_distributed_ops(), self.distributed_attr_origin)
        for op_name in get_distributed_ops():
            self.distributed_attr_hooked[op_name] = DistributedOPTemplate(op_name, pre_hooks, post_hooks)

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


def stack_filter(stack):
    for pattern in StackBlackList:
        if re.search(pattern, stack):
            return False
    return True

def get_callstack():
    callstack = []
    for (_, path, line, func, code, _) in inspect.stack():
        stack_line = f'{path}[{line}]'
        if stack_filter(stack_line):
            callstack.append(stack_line+'   '+func)
    return callstack

@torch.no_grad()
def op_aggregate(op, tensorlist):
    if isinstance(tensorlist, torch.Tensor):
        return tensorlist
    if not tensorlist:
        return torch.nan
    if op == 'min':
        return min(tensorlist)
    if op == 'max':
        return max(tensorlist)
    if op == 'norm':
        return sum(tensorlist)
    if op == 'zeros': # TODO wrong
        return sum(tensorlist) / len(tensorlist) if len(tensorlist) != 0 else 0
    return torch.nan

def update_data(old, new):
    for op, tag2tensorlist in new.items():
        if op not in old:
            old[op] = {}
        for tag, tensor in tag2tensorlist.items():
            if tag not in old[op]:
                old[op][tag] = [tensor]
            else:
                old[op][tag].append(tensor)
    return old
    

def is_target_line(codeline):
    stack = get_callstack()
    whole_stack = ';'.join(stack)
    if codeline == []:
        return True
    for pattern in codeline:
        if re.search(pattern, whole_stack):
            return True
    return False

@torch.no_grad()
def catch_data(cc_context, ops, args, prefix):
    tensor_args = {}
    for arg in args:
        if isinstance(arg, torch.Tensor):
            tensor_args[f'{prefix}_{len(tensor_args)}'] = arg
        elif isinstance(arg, list):
            if isinstance(arg[0], torch.Tensor):
                stacked_arg = torch.stack(arg)
            elif isinstance(arg[0], dist.P2POp):
                stacked_arg = torch.stack([op.tensor for op in arg])
            tensor_args[f'{prefix}_{len(tensor_args)}'] = stacked_arg
            
    new_data = {op: get_metrics(op, tensor_args, 1e-8) for op in ops}
    cc_context.data=update_data(cc_context.data, new_data)

def create_async_callback_func(context, ops, args, prefix):
    def store_data():
        catch_data(context, ops, args, prefix)
    return store_data

def get_tensor_dtype(args):
    dtypes = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            dtypes.append(arg.dtype)
        else:
            dtypes.append(None)
    return dtypes

def get_group_members(args):
    group = None
    for arg in args:
        if isinstance(arg, dist.ProcessGroup):
            group = arg
    if group is None:
        group = dist.GroupMember.WORLD
    return dist.get_process_group_ranks(group)


def create_hooks(context, monitor):

    def cc_log_hook(module, args, kwargs):
        all_args = args + tuple(kwargs.values())
        dtypes = '|'.join([str(i) if i else '' for i in get_tensor_dtype(all_args)])
        stack = ';'.join(get_callstack())
        group_members = '|'.join([str(i) for i in get_group_members(all_args)])
        monitor.cc_logged_stack[module.op_name_].add(';'.join([dtypes, group_members, stack]))
        return
        
    def cc_pre_hook(module, args, kwargs): 
        if not is_target_line(monitor.cc_codeline):
            return
        args = args + tuple(kwargs.values())
        catch_data(context[module.op_name_], monitor.ops, args, 'pre')
        return

    def cc_hook(module, args, kwargs, out=None):
        if not is_target_line(monitor.cc_codeline):
            return out
        args = args + tuple(kwargs.values())
        if out: # async
            if isinstance(out, dist.Work):
                PENDING_ASYNC_CC_BY_HANDLE[out] = create_async_callback_func(context[module.op_name_], monitor.ops, args, PREFIX_POST)
            elif isinstance(out, list): # batch_isend_irecv
                for o in out:
                    PENDING_ASYNC_CC_BY_HANDLE[o] = create_async_callback_func(context[module.op_name_], monitor.ops, args, PREFIX_POST)
            return out
        catch_data(context[module.op_name_], monitor.ops, args, PREFIX_POST)
        return out
    
    pre_hooks = []
    hooks = []
    if (dist.is_initialized() and dist.get_rank() not in monitor.module_rank_list and monitor.module_rank_list != []):
        return [pre_hooks, hooks]
    
    pre_hooks.append(cc_log_hook)
    if monitor.cc_log_only:
        return [pre_hooks, hooks]
    
    if monitor.cc_pre_hook:
        pre_hooks.append(cc_pre_hook)
    
    hooks.append(cc_hook)
    
    return [pre_hooks, hooks]

api_register = ApiRegistry()

