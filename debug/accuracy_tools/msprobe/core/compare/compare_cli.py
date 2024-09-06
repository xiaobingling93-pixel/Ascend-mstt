import json
from msprobe.core.common.file_utils import FileOpen, check_file_type
from msprobe.core.common.const import FileCheckConst, Const
from msprobe.core.common.utils import CompareException
from msprobe.core.common.log import logger


def compare_cli(args):
    with FileOpen(args.input_path, "r") as file:
        input_param = json.load(file)
    npu_path = input_param.get("npu_path", None)
    bench_path = input_param.get("bench_path", None)
    frame_name = args.framework
    auto_analyze = not args.compare_only
    if frame_name == Const.PT_FRAMEWORK:
        from msprobe.pytorch.compare.pt_compare import compare
        from msprobe.pytorch.compare.distributed_compare import compare_distributed
    else:
        from msprobe.mindspore.compare.ms_compare import ms_compare
        from msprobe.mindspore.compare.distributed_compare import ms_compare_distributed, ms_graph_compare
    if check_file_type(npu_path) == FileCheckConst.FILE and check_file_type(bench_path) == FileCheckConst.FILE:
        input_param["npu_json_path"] = input_param.pop("npu_path")
        input_param["bench_json_path"] = input_param.pop("bench_path")
        input_param["stack_json_path"] = input_param.pop("stack_path")
        if frame_name == Const.PT_FRAMEWORK:
            compare(input_param, args.output_path, stack_mode=args.stack_mode, auto_analyze=auto_analyze,
                    fuzzy_match=args.fuzzy_match)
        else:
            kwargs = {
                "stack_mode": args.stack_mode,
                "auto_analyze": auto_analyze,
                "fuzzy_match": args.fuzzy_match,
                "cell_mapping": args.cell_mapping,
                "api_mapping": args.api_mapping,
            }

            ms_compare(input_param, args.output_path, **kwargs)
    elif check_file_type(npu_path) == FileCheckConst.DIR and check_file_type(bench_path) == FileCheckConst.DIR:
        kwargs = {"stack_mode": args.stack_mode, "auto_analyze": auto_analyze, "fuzzy_match": args.fuzzy_match}
        if input_param.get("rank_id") is not None:
            ms_graph_compare(input_param, args.output_path)
            return
        if frame_name == Const.PT_FRAMEWORK:
            compare_distributed(npu_path, bench_path, args.output_path, **kwargs)
        else:
            ms_compare_distributed(npu_path, bench_path, args.output_path, **kwargs)
    else:
        logger.error("The npu_path and bench_path need to be of the same type.")
        raise CompareException(CompareException.INVALID_COMPARE_MODE)
