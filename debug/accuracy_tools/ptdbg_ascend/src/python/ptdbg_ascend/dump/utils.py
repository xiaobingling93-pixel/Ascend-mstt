import os
import re
import shutil
import sys
from pathlib import Path
import torch
import torch.distributed as dist

from ..dump import dump
from ..common.utils import print_error_log, CompareException, DumpException, Const, get_time, print_info_log, \
    check_mode_valid, get_api_name_from_matcher, check_switch_valid, check_dump_mode_valid, check_summary_only_valid, generate_compare_script, \
    check_is_npu, check_file_valid, make_dump_path_if_not_exists, check_path_before_create
from ..common.file_check_util import FileChecker, FileCheckConst, check_path_length, check_path_pattern_vaild

from ..common.version import __version__

dump_count = 0
range_begin_flag, range_end_flag = False, False


def check_list_or_acl_mode(name_prefix):
    global dump_count
    for item in DumpUtil.dump_switch_scope:
        if name_prefix.startswith(item):
            dump_count = dump_count + 1
            return True
    return False


def check_range_mode(name_prefix):
    global range_begin_flag
    global range_end_flag
    if name_prefix.startswith(DumpUtil.dump_switch_scope[0]):
        range_begin_flag = True
        return True
    if name_prefix.startswith(DumpUtil.dump_switch_scope[1]):
        range_end_flag = True
        return True
    if range_begin_flag and not range_end_flag:
        return True
    return False


def check_stack_mode(name_prefix):
    if len(DumpUtil.dump_switch_scope) == 0:
        return True
    elif len(DumpUtil.dump_switch_scope) == 1:
        return name_prefix.startswith(DumpUtil.dump_switch_scope[0])
    elif len(DumpUtil.dump_switch_scope) == 2:
        return check_range_mode(name_prefix)
    else:
        print_error_log("dump scope is invalid, Please set the scope mode in"
                        " set_dump_switch with 'all', 'list', 'range', 'stack', 'acl', 'api_list'!")
    return False


class DumpUtil(object):
    dump_root = None
    dump_data_dir = None
    dump_path = None
    dump_switch = None
    dump_switch_mode = Const.ALL # all, api_stack, list, stack...
    dump_switch_scope = []
    dump_init_enable = False
    dump_api_list = []
    dump_filter_switch = None
    dump_mode = ['forward', 'backward', 'input', 'output']
    backward_input = {}
    dump_dir_tag = 'ptdbg_dump'
    dump_config = None
    dataloader_iter = 0
    target_iter = None
    iter_num = 0
    target_rank = None
    summary_only = False
    need_replicate = False

    @staticmethod
    def set_dump_path(save_path):
        DumpUtil.dump_path = save_path
        DumpUtil.dump_init_enable = True

    @staticmethod
    def set_acl_config(acl_config):
        if not acl_config:
            raise ValueError("acl_config must be configured when mode is 'acl'")
        acl_config_checker = FileChecker(acl_config, FileCheckConst.FILE, FileCheckConst.READ_ABLE,
                                         FileCheckConst.JSON_SUFFIX)
        acl_config = acl_config_checker.common_check()
        DumpUtil.dump_config = acl_config

    @staticmethod
    def set_dump_switch(switch, mode=None, scope=None, api_list=None, filter_switch=None, dump_mode=None, summary_only=False):
        DumpUtil.dump_switch = switch
        if mode is not None:
            DumpUtil.dump_switch_mode = mode
        DumpUtil.dump_init_enable = True
        if scope is not None:
            DumpUtil.dump_switch_scope = scope
        if api_list is not None:
            DumpUtil.dump_api_list = [api.lower() for api in api_list]
        if filter_switch is not None:
            DumpUtil.dump_filter_switch = filter_switch
        if dump_mode is not None:
            DumpUtil.dump_mode = dump_mode if isinstance(dump_mode, list) else [dump_mode]

        if mode == Const.ACL:
            DumpUtil.dump_switch_scope = [api_name.replace("backward", "forward") for api_name in scope]
        DumpUtil.summary_only = summary_only

    check_mapper = {
        Const.LIST: check_list_or_acl_mode,
        Const.ACL: check_list_or_acl_mode,
        Const.RANGE: check_range_mode,
        Const.STACK: check_stack_mode
    }

    @staticmethod
    def check_switch_scope(name_prefix):
        if DumpUtil.dump_switch_mode in DumpUtil.check_mapper:
            check_func = DumpUtil.check_mapper[DumpUtil.dump_switch_mode]
            return check_func(name_prefix)
        return False

    @staticmethod
    def get_dump_path():
        if DumpUtil.dump_path:
            return DumpUtil.dump_path

        if DumpUtil.dump_switch_mode == Const.ALL:
            raise RuntimeError("get_dump_path: the file path is empty,"
                               " you must use set_dump_path to set a valid dump path!!!")
        else:
            dir_path = os.path.realpath("./")
            dump_file_name = "scope_dump_{}_{}_{}.pkl".format(
                DumpUtil.dump_switch_mode, DumpUtil.dump_switch_scope[0], get_time())
            DumpUtil.dump_path = os.path.join(dir_path, dump_file_name)
            return DumpUtil.dump_path

    @staticmethod
    def get_dump_switch():
        return DumpUtil.dump_switch == "ON"


def set_dump_path(fpath=None, dump_tag='ptdbg_dump'):
    fpath = load_env_dump_path(fpath)
    check_file_valid(fpath)
    if not re.match(Const.FILE_PATTERN, dump_tag):
        print_error_log('The file path {} contains special characters.'.format(dump_tag))
        raise CompareException(CompareException.INVALID_PATH_ERROR)
    real_path = os.path.realpath(fpath)
    make_dump_path_if_not_exists(real_path)
    fpath_checker = FileChecker(real_path, FileCheckConst.DIR, FileCheckConst.WRITE_ABLE)
    fpath_checker.common_check()
    DumpUtil.set_dump_path(real_path)
    DumpUtil.dump_dir_tag = dump_tag


def get_tensor_rank(in_feat, out_feat):
    if dist.is_initialized():
        return dist.get_rank()
        
    def get_tensor_rank_single(x):
        if isinstance(x, (list, tuple)):
            if len(x) > 0:
                return get_tensor_rank_single(x[0])
            return None
        elif isinstance(x, torch.Tensor):
            device = x.device
            if device.type == 'cpu':
                return None
            else:
                return device.index
        return None
    in_rank = get_tensor_rank_single(in_feat)
    if in_rank is None:
        out_rank = get_tensor_rank_single(out_feat)
        if out_rank is None:
            return None
        return out_rank
    return in_rank


def create_dirs_if_not_exist(rank, dump_file):
    dump_path, file_name = os.path.split(dump_file)
    rank_dir = os.path.join(dump_path, f"rank{rank}")
    dump_file = os.path.join(rank_dir, file_name)
    if not os.path.isdir(rank_dir):
        Path(rank_dir).mkdir(mode=FileCheckConst.DATA_DIR_AUTHORITY, exist_ok=True)
    return dump_file


def generate_dump_path_str():
    if DumpUtil.dump_switch_mode == 'acl':
        if DumpUtil.dump_config == '':
            print_error_log("Please provide dump config for register hook before turning on dump switch!")
            raise DumpException(DumpException.NONE_ERROR)
        dump_path = f"according to dump config {DumpUtil.dump_config}"
    else:
        dump_dir, dump_file = os.path.split(DumpUtil.dump_path)
        if not dump_file.endswith(".pkl"):
            dump_dir = DumpUtil.dump_path
        dump_path = f"to {dump_dir}"
    return dump_path


def set_dump_switch(switch, mode=Const.ALL, scope=None, api_list=None, filter_switch=Const.OFF, dump_mode=None,
                    summary_only=False):
    if scope is None:
        scope = []
    if api_list is None:
        api_list = []
    if dump_mode is None:
        dump_mode = [Const.ALL]
    check_switch_valid(switch)
    if not DumpUtil.dump_path:
        set_dump_path()
    DumpUtil.set_dump_switch(switch, summary_only=summary_only)
    dump_path_str = generate_dump_path_str()
    if switch == "OFF":
        dump.write_to_disk()
        if check_is_npu() and DumpUtil.dump_switch_mode in [Const.ALL, Const.API_STACK, Const.LIST, Const.RANGE, Const.API_LIST]:
            generate_compare_script(DumpUtil.dump_data_dir, dump.get_pkl_file_path(), DumpUtil.dump_switch_mode)
    set_dump_switch_print_info(switch, mode, dump_path_str)
    set_dump_switch_config(mode=mode, scope=scope, api_list=api_list, filter_switch=filter_switch, dump_mode=dump_mode,
                           summary_only=summary_only)


def set_dump_switch_config(mode=Const.ALL, scope=None, api_list=None, filter_switch=Const.OFF, dump_mode=None,
                           summary_only=False):
    if scope is None:
        scope = []
    if api_list is None:
        api_list = []
    if dump_mode is None:
        dump_mode = [Const.ALL]
    try:
        check_mode_valid(mode, scope, api_list)
        check_switch_valid(filter_switch)
        dump_mode = check_dump_mode_valid(dump_mode)
        summary_only = check_summary_only_valid(summary_only)
    except (CompareException, AssertionError) as err:
        print_error_log(str(err))
        raise CompareException(CompareException.INVALID_PARAM_ERROR) from err
    switch = DumpUtil.dump_switch
    DumpUtil.set_dump_switch("OFF", mode=mode, scope=scope, api_list=api_list, filter_switch=filter_switch,
                             dump_mode=dump_mode, summary_only=summary_only)
    DumpUtil.dump_switch = switch


def set_dump_switch_print_info(switch, mode, dump_path_str):
    global dump_count
    if switch == "ON":
        print_info_log(f"Dump switch is turned on. Dump data will be saved {dump_path_str}. ")
        if mode == Const.LIST:
            dump_count = 0
    else:
        print_info_log(f"Dump switch is turned off. ")
        if mode == Const.LIST:
            print_info_log("The number of matched dump is {}".format(dump_count))


def check_if_in_api_list(name):
    if not DumpUtil.dump_api_list:
        return False
    for api in DumpUtil.dump_api_list:
        if api.lower() in name.lower():
            return True
    return False


def set_backward_input(backward_input):
    for index, api_name in enumerate(DumpUtil.dump_switch_scope):
        DumpUtil.backward_input[api_name] = backward_input[index]


def make_dump_data_dir(dump_file_name):
    dump_path, file_name = os.path.split(os.path.realpath(dump_file_name))
    name_body, name_extension = os.path.splitext(file_name)
    output_dir = os.path.join(dump_path, f"{name_body}")
    check_path_before_create(output_dir)
    if not os.path.exists(output_dir):
        Path(output_dir).mkdir(mode=0o750, exist_ok=True)
    else:
        shutil.rmtree(output_dir, ignore_errors=True)
        Path(output_dir).mkdir(mode=0o750, exist_ok=True)
    return output_dir


def make_dump_dirs():
    dump_file_name, dump_file_name_body = "dump.pkl", "dump"
    dump_root_dir = load_env_dump_path(DumpUtil.dump_path)
    tag_dir = os.path.join(dump_root_dir, DumpUtil.dump_dir_tag + f'_v{__version__}')
    check_path_length(tag_dir)
    check_path_pattern_vaild(tag_dir)
    Path(tag_dir).mkdir(mode=0o750, parents=True, exist_ok=True)
    DumpUtil.dump_dir = tag_dir
    dump_file_path = os.path.join(tag_dir, dump_file_name)
    DumpUtil.set_dump_path(dump_file_path)


def check_writable(dump_file):
    if not os.access(dump_file, os.W_OK):
        print_error_log(
            'The path {} does not have permission to write. Please check the path permission'.format(
                dump_file))
        raise DumpException(DumpException.INVALID_PATH_ERROR)


def load_env_dump_path(dump_path):
    if not dump_path:
        dump_path = os.getenv(Const.ASCEND_WORK_PATH)
        if dump_path:
            try:
                dump_path = os.path.join(str(dump_path), Const.DUMP_DIR)
            except TypeError as err:
                print_error_log("Generating dump path from environment variables ASCEND_WORK_PATH failed.")
                raise DumpException(DumpException.INVALID_PATH_ERROR) from err
        else:
            print_error_log("Dump path is None, you can configure it in the following ways:\n"
                            "1. Configure set_dump_path function.\n"
                            "2. Configure the dump_path parameter of PrecisionDebugger.\n"
                            "3. Set environment variables ASCEND_WORK_PATH.")
            raise DumpException(DumpException.INVALID_PATH_ERROR)
    return dump_path
