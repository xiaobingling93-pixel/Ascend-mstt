from msprobe.core.common.const import Const
from msprobe.core.common.log import logger

def find_regard_scope(lines, start_sign, end_sign):
    # 找出 start_pos 和 end_pos
    start_pos = end_pos = -1
    for idx, ii in enumerate(lines):
        if start_sign in ii:
            start_pos = idx
        elif end_sign in ii:
            end_pos = idx
            break
    return start_pos, end_pos


def find_stack_func_list(lines):
    res_list = []
    # 过滤和处理 regard_scope
    for line in lines:
        ele_list = line.split(',')
        file_ele = ele_list[Const.STACK_FILE_INDEX]
        if any(ii in file_ele for ii in Const.FILE_SKIP_LIST):
            continue

        func_ele = ele_list[Const.STACK_FUNC_INDEX]
        if any(ii in func_ele for ii in Const.FUNC_SKIP_LIST):
            continue

        in_func_name = func_ele.split()[Const.STACK_FUNC_ELE_INDEX]

        res_list.append(in_func_name)
    # 反转res_list并生成final_res
    reversed_list = res_list[::-1]
    return reversed_list


def get_duplicated_name(components):
    duplicated_components = components
    if len(components) < 3 or components[Const.CONSTRUCT_NAME_INDEX].isdigit():
        logger.warning("key in construct.json is shorter than 3 parts or not name valid.")
    else:
        # 重复name，如Functional.add.add.X ward
        duplicated_components = components[:Const.CONSTRUCT_NAME_INDEX + 1] + components[Const.CONSTRUCT_NAME_INDEX:]
    return duplicated_components


def modify_mapping_with_stack(stack, construct):
    if not stack or not construct:
        return {}

    # 是否是mindspore的数据结构
    is_ms = any("Cell" in ii for ii in construct)
    # 调整后的mapping结构
    final_pres = {}
    # 查看归属关系
    for key in construct:
        key_components = key.split(Const.SEP)
        code_list = stack.get(key, None)
        parent_node = construct.get(key, None)
        # 名称如果非标准开头，转为标准开头
        if not key.startswith(("Module", "Cell")):
            # 如果没有拿到父属scope name，默认顶级域名为Module或Cell
            if not parent_node:
                # 将节点名字转为标准的Module或Cell
                key_components[0] = "Cell" if is_ms else "Module"
                # 重复该节点的名字作为类型 如add.add add在-3位置
                duplicated_components = get_duplicated_name(key_components)
                modified_key = Const.SEP.join(duplicated_components)

                modified_key = modified_key.replace(".forward", "").replace(".backward", "")
                final_pres[modified_key] = {Const.ORIGIN_DATA: key, Const.SCOPE: None, Const.STACK: None}
                continue
            parent = parent_node.split(Const.SEP)
            if len(parent) < 4:
                logger.info(f"Parent name in construct.json is not valid")
                continue
            parent_idx = Const.NAME_FIRST_POSSIBLE_INDEX if not \
            parent[Const.NAME_FIRST_POSSIBLE_INDEX].isdigit() else Const.NAME_SECOND_POSSIBLE_INDEX
            parent_name = parent[parent_idx]

            if code_list:
                # {name}.Class.count_number.X ward Or {name}.Class.count_number.X ward.ele_number
                if parent_name.endswith('s'):
                    parent_name = parent_name[:-1]
                if len(key_components) < 3:
                    logger.info("The length of key in construct is less than 3, please check")
                    continue
                # {name}.count_number.X ward
                func_name = key_components[-3]
                start_pos, end_pos = find_regard_scope(code_list, func_name, parent_name)

                # 获取指定范围的代码
                regard_scope = code_list[start_pos:end_pos]

                func_stack_list = find_stack_func_list(regard_scope)
            else:
                func_stack_list = []
            # 组合逻辑：parent的节点名（到节点名字为止）加上调用栈名[reversed_list]加上原来key重复key的节点名[key_components[1:-2] + key_components[-3:]]
            final_res_key = Const.SEP.join(parent[:parent_idx + 1] + func_stack_list +
                                     key_components[1:Const.CONSTRUCT_NAME_INDEX + 1] + key_components[Const.CONSTRUCT_NAME_INDEX:])
            final_res_key = final_res_key.strip(".forward").strip(".backward")
        else:
            final_res_key = Const.SEP.join(key_components[:-2] + [key_components[-1]])
            func_stack_list = []
        final_pres[final_res_key] = {Const.ORIGIN_DATA: key, Const.SCOPE: parent_node,
                                     Const.STACK: Const.SEP.join(func_stack_list) if func_stack_list else None}
    return final_pres
