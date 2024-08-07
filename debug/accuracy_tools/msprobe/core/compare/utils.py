
import os
import numpy as np
from msprobe.core.common.const import Const, CompareConst


def rename_api(npu_name, process):
    npu_split = npu_name.split(process)
    torch_func_index, in_out = npu_split[0], npu_split[1]
    torch_func_split = torch_func_index.rsplit(Const.SEP, 2)
    torch_func = str(torch_func_split[0]) + str(in_out)
    return torch_func


def read_op(op_data, op_name):
    op_parsed_list = []
    if 'forward' in op_name:
        if 'input_args' in op_data:
            input_item = op_data['input_args']
            input_parsed_list = op_item_parse(input_item, op_name + '_input', None)
            op_parsed_list = input_parsed_list.copy()
            input_parsed_list.clear()
        if 'input_kwargs' in op_data:
            kwargs_item = op_data['input_kwargs']
            if isinstance(kwargs_item, dict) and "type" in kwargs_item or isinstance(kwargs_item, list):
                kwarg_parsed_list = op_item_parse(kwargs_item, op_name + '_input', None)
                op_parsed_list += kwarg_parsed_list
                kwarg_parsed_list.clear()
            elif kwargs_item:
                for kwarg in kwargs_item:
                    kwarg_parsed_list = op_item_parse(kwargs_item[kwarg], op_name + '_input.' + kwarg, None)
                    op_parsed_list += kwarg_parsed_list
                    kwarg_parsed_list.clear()
        if 'output' in op_data:
            output_item = op_data['output']
            output_parsed_list = op_item_parse(output_item, op_name + '_output', None)
            op_parsed_list += output_parsed_list
            output_parsed_list.clear()
    if 'backward' in op_name:
        if 'input' in op_data:
            input_item = op_data['input']
            input_parsed_list = op_item_parse(input_item, op_name + '_input', None)
            op_parsed_list = input_parsed_list.copy()
            input_parsed_list.clear()
        if 'output' in op_data:
            output_item = op_data['output']
            output_parsed_list = op_item_parse(output_item, op_name + '_output', None)
            op_parsed_list += output_parsed_list
            output_parsed_list.clear()
    return op_parsed_list


def op_item_parse(item, op_name, index, item_list=None, top_bool=True):
    if item_list is None:
        item_list = []
    if item is None or (isinstance(item, dict) and not item):
        if not top_bool:
            tmp = {'full_op_name': op_name + '.' + str(index), 'Max': None, 'Min': None, 'Mean': None, 'Norm': None,
                   'dtype': None, 'shape': None, 'md5': None, 'data_name': '-1'}
        else:
            tmp = {'full_op_name': op_name + '.0', 'Max': None, 'Min': None, 'Mean': None, 'Norm': None, 'dtype': None,
                   'shape': None, 'md5': None, 'data_name': '-1'}
        item_list.append(tmp)
        return item_list
    if index is None:
        if isinstance(item, dict):
            full_op_name = op_name + '.0'
        else:
            full_op_name = op_name
    else:
        full_op_name = op_name + Const.SEP + str(index)
    if isinstance(item, dict):
        if 'type' not in item:
            for kwarg in item:
                kwarg_parsed_list = op_item_parse(item[kwarg], op_name + Const.SEP + kwarg, None)
                item_list += kwarg_parsed_list
                kwarg_parsed_list.clear()
        elif 'dtype' in item:
            parsed_item = item
            parsed_item['full_op_name'] = full_op_name
            item_list.append(parsed_item)
        elif 'type' in item:
            parsed_item = {}
            if item['type'] == 'torch.Size':
                parsed_item['full_op_name'] = full_op_name
                parsed_item['dtype'] = 'torch.Size'
                parsed_item['shape'] = str(item['value'])
                parsed_item['md5'] = None
                parsed_item['Max'] = None
                parsed_item['Min'] = None
                parsed_item['Mean'] = None
                parsed_item['Norm'] = None
                parsed_item['data_name'] = '-1'
                item_list.append(parsed_item)
            elif item['type'] == 'slice':
                parsed_item['full_op_name'] = full_op_name
                parsed_item['dtype'] = 'slice'
                parsed_item['shape'] = str(np.shape(np.array(item['value'])))
                parsed_item['md5'] = None
                parsed_item['Max'] = None
                parsed_item['Min'] = None
                parsed_item['Mean'] = None
                parsed_item['Norm'] = None
                parsed_item['data_name'] = '-1'
                item_list.append(parsed_item)
            else:
                parsed_item['full_op_name'] = full_op_name
                parsed_item['dtype'] = str(type(item['value']))
                parsed_item['shape'] = '[]'
                parsed_item['md5'] = None
                parsed_item['Max'] = item['value']
                parsed_item['Min'] = item['value']
                parsed_item['Mean'] = item['value']
                parsed_item['Norm'] = item['value']
                parsed_item['data_name'] = '-1'
                item_list.append(parsed_item)
        else:
            resolve_api_special_parameters(item, full_op_name, item_list)
    else:
        for j, item_spec in enumerate(item):
            op_item_parse(item_spec, full_op_name, j, item_list=item_list, top_bool=False)
    return item_list


def resolve_api_special_parameters(data_dict, full_op_name, item_list):
    """
    Function Description:
        解析下面格式的数据, 是api参数的一种特殊格式
        {
         "last_hidden_state": {
          "type": "torch.Tensor",
          "dtype": "torch.bfloat16",
          ...
         },
         "loss": {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
          ...
         }
        }
    Parameter:
        data_dict: 字典格式的数据
        full_op_name: 参数的全名字符串
        item_list: 参数信息集合        
    """
    for key, value in data_dict.items():
        if isinstance(value, dict):
            parsed_item = value
            parts = full_op_name.split(".")
            parts.insert(-1, key)
            full_op_name_new = ".".join(parts)
            parsed_item['full_op_name'] = full_op_name_new
            item_list.append(parsed_item)


def get_accuracy(result, n_dict, b_dict, summary_compare=False, md5_compare=False):
    def get_accuracy_core(n_start, n_len, b_start, b_len, key):
        min_len = min(n_len, b_len)
        npu_stack_info = n_dict.get("stack_info", None)
        bench_stack_info = b_dict.get("stack_info", None)
        has_stack = npu_stack_info and bench_stack_info

        all_mode_bool = not (summary_compare or md5_compare)
        if all_mode_bool:
            npu_data_name = n_dict.get("data_name", None)
            bench_data_name = b_dict.get("data_name", None)

        for index in range(min_len):

            n_name = n_dict['op_name'][n_start + index]
            b_name = b_dict['op_name'][b_start + index]
            n_struct = n_dict[key][index]
            b_struct = b_dict[key][index]
            err_msg = ""
            if md5_compare:
                result_item = [n_name, b_name, n_struct[0], b_struct[0], n_struct[1], b_struct[1],
                               n_struct[2], b_struct[2],
                               CompareConst.PASS if n_struct[2] == b_struct[2] else CompareConst.DIFF]
                if has_stack and index == 0 and key == "input_struct":
                    result_item.extend(npu_stack_info)
                else:
                    result_item.append(CompareConst.NONE)
                result.append(result_item)
                continue

            if summary_compare:
                result_item = [n_name, b_name, n_struct[0], b_struct[0], n_struct[1], b_struct[1],
                               " ", " ", " ", " ", " ", " ", " ", " "]
            else:
                result_item = [n_name, b_name, n_struct[0], b_struct[0], n_struct[1], b_struct[1],
                               " ", " ", " ", " ", " "]

            npu_summary_data = n_dict.get("summary")[n_start + index]
            result_item.extend(npu_summary_data)
            bench_summary_data = b_dict.get("summary")[b_start + index]
            result_item.extend(bench_summary_data)

            if summary_compare:
                start_idx = CompareConst.SUMMARY_COMPARE_RESULT_HEADER.index(CompareConst.MAX_DIFF)
                warning_flag = False
                for i, (npu_val, bench_val) in enumerate(zip(npu_summary_data, bench_summary_data)):
                    if isinstance(npu_val, (float, int)) and isinstance(bench_val, (float, int)):
                        diff = npu_val - bench_val
                        if bench_val != 0:
                            relative = str(abs((diff / bench_val) * 100)) + '%'
                        else:
                            relative = "N/A"
                        result_item[start_idx + i] = diff
                        result_item[start_idx + i + 4] = relative
                        magnitude_diff = abs(diff) / (max(abs(npu_val), abs(bench_val)) + 1e-10)
                        if magnitude_diff > 0.5:
                            warning_flag = True
                    else:
                        result_item[start_idx + i] = CompareConst.NONE
                accuracy_check = CompareConst.WARNING if warning_flag else ""
                err_msg += "Need double check api accuracy." if warning_flag else ""
                for i in range(start_idx, len(result_item)):
                    if str(result_item[i]) in ('inf', '-inf', 'nan'):
                        result_item[i] = f'{result_item[i]}\t'

            result_item.append(accuracy_check if summary_compare else CompareConst.ACCURACY_CHECK_YES)
            result_item.append(err_msg)
            if has_stack and index == 0 and key == "input_struct":
                result_item.extend(npu_stack_info)
            else:
                result_item.append(CompareConst.NONE)
            if all_mode_bool:
                result_item.append(npu_data_name[n_start + index])

            result.append(result_item)

        if n_len > b_len:
            for index in range(b_len, n_len):
                n_name = n_dict['op_name'][n_start + index]
                n_struct = n_dict[key][index]
                if md5_compare:
                    result_item = [n_name, CompareConst.NAN, n_struct[0], CompareConst.NAN,
                                   n_struct[1], CompareConst.NAN, n_struct[2], CompareConst.NAN, CompareConst.NAN]
                    result.append(result_item)
                    continue
                result_item = [n_name, CompareConst.NAN, n_struct[0], CompareConst.NAN,
                               n_struct[1], CompareConst.NAN, " ", " ", " ", " ", " "]
                summary_data = n_dict.get("summary")[n_start + index]
                result_item.extend(summary_data)
                summary_data = [CompareConst.NAN for _ in range(len(n_dict.get("summary")[0]))]
                result_item.extend(summary_data)

                err_msg = ""
                result_item.append(CompareConst.ACCURACY_CHECK_YES)
                result_item.append(err_msg)

                if has_stack and index == 0 and key == "input_struct":
                    result_item.extend(npu_stack_info)
                else:
                    result_item.append(CompareConst.NONE)
                if all_mode_bool:
                    result_item.append(npu_data_name[n_start + index])

                result.append(result_item)

    n_num = len(n_dict['op_name'])
    b_num = len(b_dict['op_name'])
    n_num_input = len([name for name in n_dict['op_name'] if 'input' in name])
    b_num_input = len([name for name in b_dict['op_name'] if 'input' in name])
    n_num_kwarg = len([name for name in n_dict['op_name'] if 'kwarg' in name])
    b_num_kwarg = len([name for name in b_dict['op_name'] if 'kwarg' in name])
    n_num_output = n_num - n_num_input - n_num_kwarg
    b_num_output = b_num - b_num_input - b_num_kwarg
    get_accuracy_core(0, n_num_input, 0, b_num_input, 'input_struct')
    get_accuracy_core(n_num_input, n_num_kwarg, b_num_input, b_num_kwarg, "kwargs_struct")
    get_accuracy_core(n_num_input + n_num_kwarg, n_num_output, b_num_input + b_num_kwarg, b_num_output, 'output_struct')


def get_un_match_accuracy(result, n_dict, md5_compare, summary_compare):
    index_out = 0
    npu_stack_info = n_dict.get("stack_info", None)
    bench_name, bench_type, bench_shape = CompareConst.NAN, CompareConst.NAN, CompareConst.NAN
    err_msg = CompareConst.NO_BENCH
    accuracy_check_res = CompareConst.NAN
    for index, n_name in enumerate(n_dict["op_name"]):
        if n_name.find("input") != -1:
            n_struct = n_dict["input_struct"][index]
        else:
            n_struct = n_dict["output_struct"][index_out]
            index_out += 1

        result_item = [n_name, bench_name, n_struct[0], bench_type, n_struct[1], bench_shape]
        if md5_compare:
            result_item.extend([CompareConst.NAN] * 3)
            if npu_stack_info and index == 0:
                result_item.extend(npu_stack_info)
            result.append(result_item)
            continue
        if summary_compare:
            result_item.extend([CompareConst.NAN] * 8)
        else:
            result_item.extend([CompareConst.NAN] * 5)
        summary_data = n_dict.get("summary")[index]
        result_item.extend(summary_data)
        summary_data = [CompareConst.NAN] * 4
        result_item.extend(summary_data)
        result_item.append(accuracy_check_res)
        result_item.append(err_msg)
        if npu_stack_info and index == 0:
            result_item.extend(npu_stack_info)
        if not md5_compare and not summary_compare and result_item[1] == CompareConst.NAN:
            if index == 0:
                result_item.extend(["-1"])
            else:
                result_item.extend([CompareConst.NONE, "-1"])
        result.append(result_item)


def merge_tensor(tensor_list, summary_compare, md5_compare):
    op_dict = {}
    op_dict["op_name"] = []
    op_dict["input_struct"] = []
    op_dict["kwargs_struct"] = []
    op_dict["output_struct"] = []
    op_dict["summary"] = []
    op_dict["stack_info"] = []

    all_mode_bool = not (summary_compare or md5_compare)
    if all_mode_bool:
        op_dict["data_name"] = []

    for tensor in tensor_list:
        if len(tensor) == 2:
            op_dict['stack_info'].append(tensor['full_info'])
            break
        op_dict["op_name"].append(tensor['full_op_name'])
        if not md5_compare:
            if tensor['full_op_name'].find("input") != -1:
                op_dict["input_struct"].append((tensor['dtype'], tensor['shape']))
            elif tensor['full_op_name'].find("kwarg") != -1:
                op_dict["kwargs_struct"].append((tensor['dtype'], tensor['shape']))
            elif tensor['full_op_name'].find("output") != -1:
                op_dict["output_struct"].append((tensor['dtype'], tensor['shape']))
        else:
            if tensor['full_op_name'].find("input") != -1:
                op_dict["input_struct"].append((tensor['dtype'], tensor['shape'], tensor['md5']))
            elif tensor['full_op_name'].find("kwarg") != -1:
                op_dict["kwargs_struct"].append((tensor['dtype'], tensor['shape'], tensor['md5']))
            elif tensor['full_op_name'].find("output") != -1:
                op_dict["output_struct"].append((tensor['dtype'], tensor['shape'], tensor['md5']))

        op_dict["summary"].append([tensor['Max'], tensor['Min'], tensor['Mean'], tensor['Norm']])

        if all_mode_bool:
            op_dict["data_name"].append(tensor['data_name'])

    if not op_dict["kwargs_struct"]:
        del op_dict["kwargs_struct"]
    return op_dict if op_dict["op_name"] else {}


def _compare_parser(parser):
    parser.add_argument("-i", "--input_path", dest="input_path", type=str,
                        help="<Required> The compare input path, a dict json.",  required=True)
    parser.add_argument("-o", "--output_path", dest="output_path", type=str,
                        help="<Required> The compare task result out path.", required=True)
    parser.add_argument("-s", "--stack_mode", dest="stack_mode", action="store_true",
                        help="<optional> Whether to save stack info.", required=False)
    parser.add_argument("-a", "--auto_analyze", dest="auto_analyze", action="store_false",
                        help="<optional> Whether to give advisor.", required=False)
    parser.add_argument("-f", "--fuzzy_match", dest="fuzzy_match", action="store_true",
                        help="<optional> Whether to perform a fuzzy match on the api name.", required=False)


    
    
