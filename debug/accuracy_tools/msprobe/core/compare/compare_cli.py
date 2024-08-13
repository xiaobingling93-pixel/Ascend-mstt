import json
from msprobe.core.common.file_check import FileOpen, check_file_type
from msprobe.core.common.const import FileCheckConst
from msprobe.core.common.utils import CompareException
from msprobe.core.common.log import logger


def compare_cli(args,frame_name):
    with FileOpen(args.input_path, "r") as file:
        input_param = json.load(file)
    npu_path = input_param.get("npu_path", None)
    bench_path = input_param.get("bench_path", None)

    if check_file_type(npu_path) == FileCheckConst.FILE and check_file_type(bench_path) == FileCheckConst.FILE:
        input_param["npu_json_path"] = input_param.pop("npu_path")
        input_param["bench_json_path"] = input_param.pop("bench_path")
        input_param["stack_json_path"] = input_param.pop("stack_path")
        if frame_name == "pytorch":
            from msprobe.pytorch.compare.pt_compare import compare
            from msprobe.pytorch.compare.distributed_compare import compare_distributed
            compare(input_param, args.output_path, stack_mode=args.stack_mode, auto_analyze=args.auto_analyze,
                fuzzy_match=args.fuzzy_match)
        else:
            from msprobe.mindspore.compare.ms_compare import ms_compare
            from msprobe.mindspore.compare.distributed_compare import ms_compare_distributed
            ms_compare(input_param, args.output_path, stack_mode=args.stack_mode, auto_analyze=args.auto_analyze,
                fuzzy_match=args.fuzzy_match)
    elif check_file_type(npu_path) == FileCheckConst.DIR and check_file_type(bench_path) == FileCheckConst.DIR:
        kwargs = {"stack_mode": args.stack_mode, "auto_analyze": args.auto_analyze, "fuzzy_match": args.fuzzy_match}
        if frame_name == "pytorch":
            compare_distributed(npu_path, bench_path, args.output_path, **kwargs)
        else:
            ms_compare_distributed(npu_path, bench_path, args.output_path, **kwargs)
    else:
        logger.error("The npu_path and bench_path need to be of the same type.")
        raise CompareException(CompareException.INVALID_COMPARE_MODE)
