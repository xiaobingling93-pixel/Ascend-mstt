from msprobe.core.common.log import logger
from msprobe.core.compare.utils import rename_api 


dtype_mapping = {
    "Int8": "torch.int8",
    "UInt8": "torch.uint8",
    "Int16": "torch.int16",
    "UInt16": "torch.uint16",
    "Int32": "torch.int32",
    "UInt32": "torch.uint32",
    "Int64": "torch.int64",
    "UInt64": "torch.uint64",
    "Float16": "torch.float16",
    "Float32": "torch.float32",
    "Float64": "torch.float64",
    "Bool": "torch.bool",
    "BFloat16": "torch.bfloat16",
    "Complex64": "torch.complex64",
    "Complex128": "torch.complex128"
    }


def check_struct_match(npu_dict, bench_dict, cross_frame=False):
    npu_struct_in = npu_dict.get("input_struct")
    bench_struct_in = bench_dict.get("input_struct")
    npu_struct_out = npu_dict.get("output_struct")
    bench_struct_out = bench_dict.get("output_struct")

    if cross_frame:
        npu_struct_in = [(dtype_mapping.get(item[0], item[0]), item[1]) for item in npu_struct_in]
        npu_struct_out = [(dtype_mapping.get(item[0], item[0]), item[1]) for item in npu_struct_out]
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
            ms_type=[["Float16", "Float32"], ["Float32", "Float16"],["Float16", "BFloat16"],["BFloat16", "Float16"]] 
            torch_type=[["torch.float16", "torch.float32"], ["torch.float32", "torch.float16"],
                                ["torch.float16", "torch.bfloat16"], ["torch.bfloat16", "torch.float16"]]
            if ([npu_type, bench_type] in ms_type)or  ([npu_type, bench_type] in torch_type):                    
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



