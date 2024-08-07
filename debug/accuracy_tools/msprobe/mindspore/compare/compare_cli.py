import json
from msprobe.core.common.file_check import FileOpen, check_file_type
from msprobe.core.common.const import FileCheckConst
from msprobe.core.common.utils import CompareException
from msprobe.core.common.log import logger
from msprobe.mindspore.compare.ms_compare import ms_compare


def compare_cli_ms(args):
    with FileOpen(args.input_path, "r") as file:
        input_param = json.load(file)
    npu_path = input_param.get("npu_path", None)
    bench_path = input_param.get("bench_path", None)
    
    if check_file_type(npu_path) == FileCheckConst.FILE and check_file_type(bench_path) == FileCheckConst.FILE:
        ms_compare(input_param, args.output_path, stack_mode=args.stack_mode, auto_analyze=args.auto_analyze,
                fuzzy_match=args.fuzzy_match)
    elif check_file_type(npu_path) == FileCheckConst.DIR and check_file_type(bench_path) == FileCheckConst.DIR:
        logger.error('This function is not supported at this time.')
        raise Exception("Mindspore Unsupport function compare_distributed.")
    else:
        logger.error("The npu_path and bench_path need to be of the same type.")
        raise CompareException(CompareException.INVALID_COMPARE_MODE)
