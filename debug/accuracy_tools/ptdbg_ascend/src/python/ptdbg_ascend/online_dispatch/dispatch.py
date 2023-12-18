import os
import time
import json
from pathlib import Path
from multiprocessing import Manager, Pool

import yaml
import torch

from torch.utils._python_dispatch import TorchDispatchMode

try:
    import torch_npu
except ImportError:
    is_npu = False
else:
    is_npu = True

from ..common.utils import Const, CompareConst, add_time_as_suffix, check_file_or_directory_path, \
    check_path_before_create
from ..common.version import __version__
from .dump_compare import dispatch_workflow, dispatch_multiprocess, error_call, TimeStatistics, \
    DispatchRunParam, save_csv
from .utils import get_callstack, data_to_cpu, logger_debug, logger_error, logger_warn, logger_logo, get_sys_info
from ..common.file_check_util import FileOpen


class PtdbgDispatch(TorchDispatchMode):
    def __init__(self, dump_mode=Const.OFF, api_list=None, debug=False, dump_path=None, tag=None, process_num=0):
        super(PtdbgDispatch, self).__init__()
        logger_logo()
        if not is_npu:
            logger_error("Please confirm you run environment installed torch_npu!")
            return

        if dump_path is None:
            logger_error("Please set dump_path when dump_mode is config!")
        check_file_or_directory_path(dump_path, True)

        self.device_id = torch_npu._C._npu_getDevice()
        self.dump_mode = dump_mode
        self.dump_api_list = self.get_dump_api(api_list)
        self.debug_flag = debug
        self.api_index = 0
        self.single_api_index_dict = {}
        self.device_dump_path_cpu = None
        self.device_dump_path_npu = None
        self.all_summery = []
        self.call_stack_list = []
        # guarantee file uniqueness
        time.sleep(1)
        time_now = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
        if tag is None:
            dir_name = f'ptdbg_v{__version__}_rank{self.device_id}_{time_now}'
        else:
            dir_name = f'ptdbg_v{__version__}_{tag}_rank{self.device_id}_{time_now}'
        self.root_path = os.path.join(os.path.realpath(dump_path), dir_name)
        self.root_cpu_path = os.path.join(self.root_path, f'cpu')
        self.root_npu_path = os.path.join(self.root_path, f'npu')
        file_name = add_time_as_suffix(f'compare_result_rank{self.device_id}')
        self.csv_path = os.path.join(self.root_path, file_name)
        check_path_before_create(self.root_cpu_path)
        check_path_before_create(self.root_npu_path)
        Path(self.root_cpu_path).mkdir(mode=0o750, parents=True, exist_ok=True)
        Path(self.root_npu_path).mkdir(mode=0o750, parents=True, exist_ok=True)

        self.aten_ops_blacklist = []
        self.npu_adjust_autogard = []
        yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "torch_ops_config.yaml")
        with FileOpen(yaml_path, 'r') as f:
            yaml_file = yaml.safe_load(f)
            self.aten_ops_blacklist = yaml_file.get('aten_ops_blacklist')
            self.npu_adjust_autogard = yaml_file.get('npu_adjust_autogard')

        self.process_num = process_num
        self.lock = None
        if process_num > 0:
            self.pool = Pool(process_num)
            self.lock = Manager().Lock()

        if debug:
            logger_debug(f'Main pid:{os.getpid()} device:{self.device_id} dump_list:{self.dump_api_list} '
                         f'dump_mode:{self.dump_mode} cpu_path[{self.root_cpu_path}], npu_path[{self.root_npu_path}], '
                         f'process[{process_num}]')

    @staticmethod
    def get_dump_api(api_list):
        aten_api_list = dir(torch.ops.aten)
        dump_api_list = []
        if api_list is not None:
            for aten_api in api_list:
                if aten_api in aten_api_list:
                    dump_api_list.append(aten_api)
                else:
                    logger_warn(f'{aten_api} is not aten api will not dump, please refer to torch.ops.aten')
        return dump_api_list

    def get_dump_flag(self, aten_api):
        dump_flag = False
        auto_dump_flag = False
        if self.dump_mode == Const.ALL:
            dump_flag = True
        if self.dump_mode == Const.LIST and aten_api in self.dump_api_list:
            dump_flag = True
        if self.dump_mode == Const.AUTO:
            auto_dump_flag = True
        return dump_flag, auto_dump_flag

    @staticmethod
    def check_fun(func, run_param):
        if hasattr(torch.ops.aten, run_param.aten_api):
            aten_func = getattr(torch.ops.aten, run_param.aten_api)
            if hasattr(aten_func, run_param.aten_api_overload_name):
                aten_overload_func = getattr(aten_func, run_param.aten_api_overload_name)
                if id(aten_overload_func) == id(func):
                    run_param.func_namespace = "aten"
                    return True
        return False

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)

        if not is_npu:
            return
        logger_debug(f'start write compare csv: Rank[{self.device_id}], Pid[{os.getpid()}')

        if self.process_num > 0:
            self.pool.close()
            self.pool.join()
            summery_path = os.path.join(self.root_cpu_path, f'summery.json')
            if not os.path.exists(summery_path):
                logger_error("Please check train log, An exception may have occurred!")
                return
            check_file_or_directory_path(summery_path, False)
            fp_handle = open(summery_path, "r")
            while True:
                json_line_data = fp_handle.readline()
                if json_line_data == '\n':
                    continue
                if len(json_line_data) == 0:
                    break
                msg = json.loads(json_line_data)
                self.all_summery[msg[0]] = msg[1]
            fp_handle.close()

        if self.debug_flag:
            input_num = 0
            output_num = 0
            total_num = 0

            for list_data in self.all_summery:
                for data in list_data:
                    logger_debug(f'summery: Device[{self.device_id}], Pid[{os.getpid()}], Data[{data}]')
                    if "_input" in data[CompareConst.NPU_NAME]:
                        input_num = input_num + 1
                    if "_output" in data[CompareConst.NPU_NAME]:
                        output_num = output_num + 1
                    total_num = total_num + 1
            logger_debug(f'Dispatch exit: Device[{self.device_id}], Pid[{os.getpid()} Input[{input_num}] '
                         f'Output[{output_num}] Total[{total_num}] API_Total[{self.api_index}]]')

        save_csv(self.all_summery, self.call_stack_list, self.csv_path)

    def enable_autogard(self, aten_api):
        if aten_api in self.npu_adjust_autogard:
            torch._C._dispatch_tls_set_dispatch_key_excluded(torch._C.DispatchKey.AutogradFunctionality, False)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if not is_npu:
            logger_error("Please confirm you run environment installed torch_npu!")
            return func(*args, **kwargs)

        func_name_split_list = func.__name__.split(".")
        aten_api = func_name_split_list[0]
        try:
            aten_api_overload_name = func_name_split_list[1]
        except IndexError:
            logger_error(f"Please check the func name {func.__name__}!")
            return func(*args, **kwargs)

        self.enable_autogard(aten_api)
        if aten_api in self.aten_ops_blacklist:
            npu_out = func(*args, **kwargs)
            return npu_out

        call_stack = get_callstack()
        self.call_stack_list.append(call_stack)
        self.api_index += 1
        if aten_api not in self.single_api_index_dict:
            self.single_api_index_dict[aten_api] = 1
        else:
            self.single_api_index_dict[aten_api] += 1

        run_param = DispatchRunParam(self.debug_flag, self.device_id, self.root_npu_path, self.root_cpu_path,
                                     self.process_num)
        run_param.dump_flag, run_param.auto_dump_flag = self.get_dump_flag(aten_api)
        run_param.func_name = func.__name__
        run_param.aten_api = aten_api
        run_param.aten_api_overload_name = aten_api_overload_name
        run_param.single_api_index = self.single_api_index_dict[aten_api]
        run_param.api_index = self.api_index

        if self.debug_flag:
            logger_debug(f'Dispatch Info: Rank[{self.device_id}], Pid[{os.getpid()}], Func[{func.__name__}], '
                         f'Name[{run_param.aten_api}_{run_param.single_api_index}], '
                         f'Count[{self.api_index}], Sys[{get_sys_info()}]')

        cpu_args = []
        cpu_kwargs = []
        data_to_cpu(args, 0, cpu_args)
        data_to_cpu(kwargs, 0, cpu_kwargs)
        cpu_args = cpu_args[0]
        cpu_kwargs = cpu_kwargs[0]

        with TimeStatistics("NPU RUN", run_param):
            npu_out = func(*args, **kwargs)
        npu_out_cpu = []
        data_to_cpu(npu_out, 0, npu_out_cpu)
        npu_out_cpu = npu_out_cpu[0]

        with TimeStatistics("CPU RUN", run_param):
            cpu_out = func(*cpu_args, **cpu_kwargs)

        if isinstance(cpu_out, torch.Tensor) and cpu_out.dtype in [torch.bfloat16, torch.float16, torch.half]:
            cpu_out = cpu_out.float()

        if self.process_num == 0:
            self.all_summery.append([])
            dispatch_workflow(run_param, cpu_args, cpu_kwargs, self.all_summery, func, npu_out_cpu, cpu_out, self.lock)
        else:
            self.lock.acquire()
            self.all_summery.append([])
            self.lock.release()
            run_param.process_flag = True
            if self.check_fun(func, run_param):
                self.pool.apply_async(func=dispatch_multiprocess,
                                      args=(run_param, cpu_args, cpu_kwargs, self.all_summery, npu_out_cpu, cpu_out,
                                            self.lock),
                                      error_callback=error_call)
            else:
                logger_error("can not get correct function please set process_num=0")
        return npu_out
