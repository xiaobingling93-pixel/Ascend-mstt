import os
import json

from msprobe.core.common.const import Const
from msprobe.core.common.log import logger


def modify_mapping_with_stack(stack, construct):
    if not stack or not construct:
        return {}

    # 是否是mindspore的数据结构
    is_ms = any("Cell" in ii for ii in construct)
    # 调整后的mapping结构
    final_pres =  {}
    # 查看归属关系
    for key in construct:
        key_components = key.split('.')
        # 名称如果非标准开头，转为标准开头
        if key.startswith(("Functional", "Tensor", "Torch", "NPU", "Jit", "MintFunctional", "Mint", "Primitive")):
            code_list = stack.get(key, None)
            if not code_list:
                logger.info(f"{key} not found in code stack")
                continue

            if not construct.get(key, None):
                if not key.startswith(("Module", "Cell")):
                    # 将节点名字转为标准的Module或Cell
                    key_components[0] = "Cell" if is_ms else "Module"
                    # 重复该节点的名字作为类型 如add.add add在-3位置
                    if len(key_components) < 3 or key_components[-3].isdigit():
                        logger.warning("key in construct.json is shorter than 3 parts or not name valid.")
                        duplicated_components = key_components
                    else:
                        duplicated_components = key_components[:-2] + key_components[-3:]
                    modified_key = '.'.join(duplicated_components)
                else:
                    modified_key = key
                modified_key = modified_key.replace(".forward", "").replace(".backward", "")
                final_pres[modified_key] = {Const.ORIGIN_DATA: key, Const.SCOPE: None, Const.STACK: None}
                continue

            parent = construct[key].split('.')
            if len(parent) < 4:
                logger.warning(f"{construct[key]} in construct.json is not valid")
                continue
            # {name}.Class.count_number.X ward Or {name}.Class.count_number.X ward.ele_number
            parent_idx = -4 if not parent[-4].isdigit() else -5

            parent_name = parent[parent_idx] 
            if parent_name.endswith('s'):
                parent_name = parent_name[:-1]
            # {name}.count_number.X ward
            func_name = key_components[-3]

            # 找出 start_pos 和 end_pos
            start_pos = end_pos = -1
            for idx, ii in enumerate(code_list):
                if func_name in ii:
                    start_pos = idx
                elif parent_name in ii:
                    end_pos = idx

            # 获取指定范围的代码
            regard_scope = code_list[start_pos:end_pos]
            func_skip_list = ["construct", "__call__"]
            file_skip_list = ["site-packages/mindspore", "package/mindspore", "msprobe", "torch"]
            res_list = []

            # 过滤和处理 regard_scope
            for line in regard_scope:
                ele_list = line.split(',')
                file_ele = ele_list[0]
                if any(ii in file_ele for ii in file_skip_list):
                    continue

                func_ele = ele_list[2]
                if any(ii in func_ele for ii in func_skip_list):
                    continue
                in_func_name = func_ele.split()[1]
                res_list.append(in_func_name)
            
            # 反转res_list并生成final_name
            reversed_list = res_list[::-1]
            # 组合逻辑：parent的节点名（到节点名字为止）加上调用栈名[reversed_list]加上原来key重复key的节点名[key_components[1:-2] + key_components[-3:]]
            final_res_key = '.'.join(parent[:parent_idx + 1] + reversed_list + key_components[1:-2] + key_components[-3:])
            final_res_key = final_res_key.strip(".forward").strip(".backward")
        else:
            final_res_key = '.'.join(key_components[:-2] + [key_components[-1]])
            reversed_list = []
        final_pres[final_res_key] = {Const.ORIGIN_DATA:key, Const.SCOPE: construct[key], Const.STACK: '.'.join(reversed_list) if reversed_list else None}
    return final_pres
        