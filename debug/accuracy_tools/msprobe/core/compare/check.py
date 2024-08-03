from msprobe.core.compare.match import graph_mapping
from msprobe.core.common.log import logger
from msprobe.core.common.const import Const, CompareConst, FileCheckConst
from msprobe.core.compare.utils import rename_api 



def check_struct_match(npu_dict, bench_dict):
    npu_struct_in = npu_dict.get("input_struct")
    bench_struct_in = bench_dict.get("input_struct")
    npu_struct_out = npu_dict.get("output_struct")
    bench_struct_out = bench_dict.get("output_struct")
    is_match = npu_struct_in == bench_struct_in and npu_struct_out == bench_struct_out
    if not is_match:
        if len(npu_struct_in) == 0 or len(bench_struct_in) == 0 or len(npu_struct_in) != len(bench_struct_in):
            return False
        struct_in_is_match = check_type_shape_match(npu_struct_in, bench_struct_in)
        struct_out_is_match = check_type_shape_match(npu_struct_out, bench_struct_out)
        is_match = struct_in_is_match and struct_out_is_match
    return is_match

def check_type_shape_match(npu_struct, bench_struct):
    shape_type_match = False
    for npu_type_shape, bench_type_shape in zip(npu_struct, bench_struct):
        npu_type = npu_type_shape[0]
        npu_shape = npu_type_shape[1]
        bench_type = bench_type_shape[0]
        bench_shape = bench_type_shape[1]
        shape_match = npu_shape == bench_shape
        type_match = npu_type == bench_type
        if not type_match:
            if ([npu_type, bench_type] in [["Float16", "Float32"], ["Float32", "Float16"]] )or  ([npu_type, bench_type] in [["torch.float16", "torch.float32"], ["torch.float32", "torch.float16"],
                                ["torch.float16", "torch.bfloat16"], ["torch.bfloat16", "torch.float16"]]):                    
                type_match = True
            else:
                type_match = False
        shape_type_match = shape_match and type_match
        if not shape_type_match:
            return False
    return shape_type_match

def check_graph_mode(a_op_name, b_op_name):
    if "Aten" in a_op_name and "Aten" not in b_op_name:
        return True
    if "Aten" not in a_op_name and "Aten" in b_op_name:
        return True
    return False


def check_op(npu_dict, bench_dict, fuzzy_match):
    a_op_name = npu_dict["op_name"]
    b_op_name = bench_dict["op_name"]
    graph_mode = check_graph_mode(a_op_name[0], b_op_name[0])
    if graph_mode:
        return graph_mapping.match(a_op_name[0], b_op_name[0])
    struct_match = check_struct_match(npu_dict, bench_dict)
    if not fuzzy_match:
        return a_op_name == b_op_name and struct_match
    is_match = True
    try:
        is_match = fuzzy_check_op(a_op_name, b_op_name)
    except Exception as err:
        logger.warning("%s and %s can not fuzzy match." % (a_op_name, b_op_name))
        is_match = False
    return is_match and struct_match



def fuzzy_check_op(npu_name_list, bench_name_list):
    if len(npu_name_list) == 0 or len(bench_name_list) == 0 or len(npu_name_list) != len(bench_name_list):
        return False
    is_match = True
    for npu_name, bench_name in zip(npu_name_list, bench_name_list):
        is_match = fuzzy_check_name(npu_name, bench_name)
        if not is_match:
            break
    return is_match

def fuzzy_check_name(npu_name, bench_name):
    if "forward" in npu_name and "forward" in bench_name:
        is_match = rename_api(npu_name, "forward") == rename_api(bench_name, "forward")
    elif "backward" in npu_name and "backward" in bench_name:
        is_match = rename_api(npu_name, "backward") == rename_api(bench_name, "backward")
    else:
        is_match = npu_name == bench_name
    return is_match



def check_accuracy(cos, max_abs_err):
    if cos == CompareConst.SHAPE_UNMATCH:
        return CompareConst.ACCURACY_CHECK_UNMATCH
    if cos == CompareConst.NONE or max_abs_err == CompareConst.NONE:
        return CompareConst.NONE
    if cos == "N/A" or max_abs_err == "N/A":
        return CompareConst.ACCURACY_CHECK_NO
    try:
        cos, max_abs_err = float(cos), float(max_abs_err)
    except ValueError:
        logger.warning("Cosine or MaxAbsErr can not get float value.")
        return CompareConst.NONE
    if cos < CompareConst.COS_THRESHOLD and max_abs_err > CompareConst.MAX_ABS_ERR_THRESHOLD:
        return CompareConst.ACCURACY_CHECK_NO
    if cos < CompareConst.COS_MAX_THRESHOLD or max_abs_err > CompareConst.MAX_ABS_ERR_MAX_THRESHOLD:
        return CompareConst.ACCURACY_CHECK_NO
    return CompareConst.ACCURACY_CHECK_YES