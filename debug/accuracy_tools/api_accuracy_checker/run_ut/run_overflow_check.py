import argparse
import os
import sys
import torch_npu
import torch
from tqdm import tqdm
from api_accuracy_checker.run_ut.run_ut import exec_api, generate_device_params, get_api_info
from api_accuracy_checker.common.utils import print_info_log, print_warn_log, get_json_contents, print_error_log
from ptdbg_ascend.src.python.ptdbg_ascend.common.file_check_util import check_link


def check_tensor_overflow(x):
    if isinstance(x, torch.Tensor) and x.numel() != 0 and x.dtype != torch.bool:
        if len(x.shape) == 0:
            tensor_max = x.cpu().detach().float().numpy().tolist()
            tensor_min = tensor_max
        else:
            tensor_max = torch._C._VariableFunctionsClass.max(x).cpu().detach().float().numpy().tolist()
            tensor_min = torch._C._VariableFunctionsClass.min(x).cpu().detach().float().numpy().tolist()
        # inf
        if tensor_max == float('inf') or tensor_min == float('-inf'):
            return True
        # nan
        elif tensor_max != tensor_max or tensor_min != tensor_min:
            return True
        else:
            return False
    elif isinstance(x, bool) or isinstance(x, int) or isinstance(x, float):
        if x == float('inf') or x == float('-inf') or x != x:
            return True
        else:
            return False
    else:
        return False


def check_data_overflow(x):
    if isinstance(x, (tuple, list)) and x:
        for i, item in enumerate(x):
            if check_data_overflow(item):
                return True
        return False
    else:
        return check_tensor_overflow(x)


def run_overflow_check(forward_file):
    print_info_log("start UT test")
    forward_content = get_json_contents(forward_file)
    for api_full_name, api_info_dict in tqdm(forward_content.items()):
        try:
            run_torch_api(api_full_name, api_info_dict)
        except Exception as err:
            api_name = api_full_name.split("_", 1)[1].rsplit("_", 2)[0]
            if "not implemented for 'Half'" in str(err):
                print_warn_log(f"API {api_name} not support half tensor in CPU, please add {api_name} to CONVERT_API "
                               f"'fp16_to_fp32' list in accuracy_tools/api_accuracy_check/common/utils.py file.")
            elif "expected scalar type Long" in str(err):
                print_warn_log(f"API {api_name} not support int32 tensor in CPU, please add {api_name} to CONVERT_API "
                               f"'int32_to_int64' list in accuracy_tools/api_accuracy_check/common/utils.py file.")
            else:
                print_error_log(f"Run {api_full_name} UT Error: %s" % str(err))


def run_torch_api(api_full_name, api_info_dict):
    torch.npu.clear_npu_overflow_flag()
    api_type = api_full_name.split("_")[0]
    api_name = api_full_name.split("_", 1)[1].rsplit("_", 2)[0]
    args, kwargs, need_grad = get_api_info(api_info_dict, api_name)
    if not need_grad:
        print_warn_log("%s function with out=... arguments don't support automatic differentiation, skip backward." % api_full_name)
    npu_args, npu_kwargs = generate_device_params(args, kwargs, False)
    if kwargs.get("device"):
        del kwargs["device"]
    out = exec_api(api_type, api_name, args, kwargs)
    npu_out = exec_api(api_type, api_name, npu_args, npu_kwargs)
    cpu_overflow = check_data_overflow(out)
    npu_overflow = torch_npu.npu.utils.npu_check_overflow(npu_out)
    if cpu_overflow == npu_overflow:
        print_warn_log("The %s overflow is a normal overflow." % api_full_name)
    else:
        print_warn_log("The %s overflow is an abnormal overflow." % api_full_name)
    return


def _run_ut_parser(parser):
    parser.add_argument("-forward", "--forward_input_file", dest="forward_input_file", default="",
                        help="<Required> The api param tool forward result file: generate from api param tool, "
                             "a json file.",
                        required=True)
    parser.add_argument("-j", "--jit_compile", dest="jit_compile", help="<optional> whether to turn on jit compile",
                        default=False, required=False)
    parser.add_argument("-d", "--device", dest="device_id", type=int, help="<optional> set NPU device id to run ut",
                        default=0, required=False)


def _run_overflow_check():
    parser = argparse.ArgumentParser()
    _run_ut_parser(parser)
    args = parser.parse_args(sys.argv[1:])
    torch.npu.set_compile_mode(jit_compile=args.jit_compile)
    npu_device = "npu:" + str(args.device_id)
    check_link(args.forward_input_file)
    forward_file = os.path.realpath(args.forward_input_file)
    try:
        torch.npu.set_device(npu_device)
    except Exception as error:
        print_error_log(f"Set NPU device id failed. device id is: {args.device_id}")
        raise NotImplementedError from error
    run_overflow_check(forward_file)


if __name__ == '__main__':
    _run_overflow_check()
    print_info_log("UT task completed.")
