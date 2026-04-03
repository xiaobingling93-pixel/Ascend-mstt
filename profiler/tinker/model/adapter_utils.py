# Copyright (c) 2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import importlib
import inspect
import os
import sys
import textwrap
from typing import List, Tuple, Dict

import astor

from tinker.model.block_adapters import BlockAdapter, legacy_block_adapters, mcore_block_adapters
from tinker.utils.config import TINKER_DIR
from tinker.utils.logger import logger
from tinker.utils.utils import write_lines, project_root, find_keywords_line_idx, get_lines, read_file, path_to_package
from tinker.utils.constant import MODULE_NAME, PYTHON_STANDARD_INDENT

block_adapter_file_path = os.path.join(TINKER_DIR, f'model/{MODULE_NAME}.py')


def find_source_code(location_list: List[List[str]]) -> Tuple[List[str], List]:
    """
    根据输入的外部地址，返回生成 model adapter 所需条件
    :param location_list: 用于定位
    :return: source_code_list 前向所在源码列表 以及 模块对象列表
    """

    source_code_list = []
    module_obj_list = []
    for locations in location_list:
        for location in locations:
            last_dot_index = location.rfind('.')
            module_path = location[:last_dot_index]
            class_or_method_name = location[last_dot_index + 1:]
            try:
                # 动态导入包
                module_obj = importlib.import_module(module_path)
                class_or_method_obj = getattr(module_obj, class_or_method_name)
                method_obj = getattr(class_or_method_obj, 'forward') if inspect.isclass(
                    class_or_method_obj) else class_or_method_obj
                source_code = inspect.getsource(method_obj)
                source_code_list.append(source_code)
                module_obj_list.append(module_obj)
            except (ImportError, AttributeError):
                logger.debug('location: %s is error', location, exc_info=True)
            else:
                logger.info(f'Successfully find location {location}')
                break
        else:
            location_text = "\n".join(locations)
            raise RuntimeError(f'The package is not supported in the current version:\n{location_text}')
    return source_code_list, module_obj_list


def get_top_level_import(tinker_patch_dict: dict, module_source_code: str) -> List[str]:
    """
    从module_source_code中搜索顶层import或from ..import，若 from ...import xxx中的xxx命中tinker_patch_dict的key，则用value 替换此时的 ...
    :param tinker_patch_dict: 提供的写死的返回的参数
    :param module_source_code: 提供的写死的返回的参数
    :return: import 或 from import list
    """
    node = ast.parse(module_source_code)
    import_statements = []
    for n in node.body:  # 仅遍历模块顶层的节点
        if isinstance(n, ast.Import):
            for alias in n.names:
                import_statement = f"import {alias.name} as {alias.asname}" if alias.asname else f"import {alias.name}"
                import_statements.append(import_statement)
        elif isinstance(n, ast.ImportFrom):
            # 排除相对路径引用，相对路径的level大于0，相对路径尽可能在此出现
            if n.level != 0:
                continue
            module = n.module if n.module else ''
            for alias in n.names:
                real_module = tinker_patch_dict.get(alias.name, module)
                import_statement = f"from {real_module} import {alias.name} as {alias.asname}" if alias.asname else \
                    f"from {real_module} import {alias.name}"
                import_statements.append(import_statement)

    return import_statements


def scan_tinker_megatron_patch(scan_path: str):
    """
    扫描 scan_path 下的所有patch文件，返回 method -> module package 对
    :param scan_path: 是tinker根目录下的相对路径
    :return:
    """
    project_path = project_root()
    megatron_patch_path = os.path.join(project_path, scan_path)
    try:
        patch_files = os.listdir(megatron_patch_path)
    except FileNotFoundError:
        logger.error('Cannot find path: %s', megatron_patch_path)
        raise

    res = dict()
    for patch_file in patch_files:
        # 要求是.py文件，且略过init.py文件
        if patch_file == '__init__.py' or not patch_file.endswith('.py'):
            continue
        file = read_file(os.path.join(megatron_patch_path, patch_file))
        node = ast.parse(file)
        # 仅遍历模块顶层的节点
        for n in node.body:
            if isinstance(n, ast.FunctionDef):
                package_path = path_to_package(scan_path)
                base_name = os.path.splitext(patch_file)[0]
                res[n.name] = '.'.join([package_path, base_name])
    return res


class ImportTracker(ast.NodeVisitor):
    """
    统计所有import内容，存在self.imports中
    """

    def __init__(self):
        # {别名: 所在模块名}
        self.imports: Dict[str, str] = {}

    def visit_Import(self, node):
        for alias in node.names:
            mod_name = alias.name.split('.')[0]
            self.imports[alias.asname or mod_name] = mod_name

    def visit_ImportFrom(self, node):
        module = node.module.split('.')[0] if node.module else ''
        for alias in node.names:
            full_name = f"{module}.{alias.name}" if module else alias.name
            self.imports[alias.asname or alias.name] = full_name


class FuncUsageFinder(ast.NodeVisitor):
    def __init__(self, target_modules: Dict[str, str], imports: Dict[str, str]):
        self.targets = target_modules
        self.import_map = imports
        self.used_funcs_code = []

    def visit_Name(self, node):
        self._check_usage(node.id)

    def visit_Attribute(self, node):
        """处理属性链中的顶级对象"""
        if isinstance(node.value, ast.Name):
            self._check_usage(node.value.id)
        elif isinstance(node.value, ast.Attribute):
            # 递归处理嵌套属性如a.b.c -> 最终检测a
            self.visit(node.value)

    def _check_usage(self, name: str):
        """核心匹配逻辑"""
        # 直接匹配目标模块
        if name not in self.import_map and name in self.targets:
            self.used_funcs_code.append(self.targets.get(name))


def get_import_code_str(module_obj_list):
    """
    从源码中抽取并汇总 所有import 部分代码
    :param module_obj_list: 模块源码列表
    :return:
    """
    tinker_patch_dict = scan_tinker_megatron_patch('tinker/megatron_patch')
    all_imports = set()
    for module_obj in module_obj_list:
        module_source_code = inspect.getsource(module_obj)
        top_level_import = get_top_level_import(tinker_patch_dict, module_source_code)
        all_imports.update(top_level_import)
    return '\n'.join(all_imports)


def get_module_methods(module_obj):
    """
    获取模块对象所有方法和源码的组合
    :param module_obj: 模块对象
    :return:
    """
    functions = {
        name: inspect.getsource(getattr(module_obj, name))
        for name in dir(module_obj)
        if inspect.isfunction(getattr(module_obj, name))
    }
    return functions


def error_free_import():
    """确保各框架版本，megatron均被patch"""
    package_path = os.getenv('ML_PATH', None)
    if package_path:
        sys.path.append(package_path)
    module_names = ['modellink', 'mindspeed_llm', 'ascendspeed.megatron_adaptor']
    for module_name in module_names:
        try:
            logger.debug(f'try to import {module_name}')
            importlib.import_module(module_name)
            if module_name == 'mindspeed_llm':
                sys.modules['modellink'] = sys.modules['mindspeed_llm']
            logger.debug(f'import {module_name} success')
            break
        except ImportError:
            logger.debug(f'import {module_name} failed', exc_info=True)
    else:
        raise RuntimeError(f'No available patch framework')


def gen_block_adapter(use_mcore_models):
    """
    从用户提供的版本以及是否启用mcore_model，动态生成适配每个版本的 block_adapter
    :param version: 版本号
    :param use_mcore_models: 是否使用 mcore
    :return:
    """
    package_path = os.getenv('ML_PATH', None)
    if not package_path:
        raise RuntimeError("ML_PATH is not set")
    if not os.path.exists(package_path):
        raise RuntimeError(f'The package path is not exist: {package_path}')
    # 这里特殊处理一下1.0 版本的patch，以防止后续 MethodLocation 导入报错
    logger.info('The package_path is: %s', package_path)
    sys.path.append(package_path)
    error_free_import()
    block_adapters = mcore_block_adapters if use_mcore_models else legacy_block_adapters
    source_method_paths = [adapter.source_method_path for adapter in block_adapters]
    method_forward_source_code_list, module_obj_list = find_source_code(source_method_paths)
    # 1 import 部分
    import_code_str = get_import_code_str(module_obj_list)
    result = [import_code_str]

    for method_forward_source_code, block_adapter, module_obj in zip(method_forward_source_code_list,
                                                                     block_adapters,
                                                                     module_obj_list):
        # 2 生成前向代码
        method_forward_str = gen_method_forward(method_forward_source_code, block_adapter)

        # 3 检测 有些特殊情况，如1.2的代码中，可能forward中用到的内容，也不全是import，如定义在模块中的方法 post_language_model_processing
        used_funcs_code = find_used_top_func(import_code_str, method_forward_str, module_obj)
        result.extend(used_funcs_code)

        result.append(method_forward_str)
    file_content = '\n\n\n'.join(result)
    try:
        if os.path.exists(block_adapter_file_path):
            # 删除历史文件
            os.remove(block_adapter_file_path)
    except OSError:
        # 捕获删除文件时可能出现的异常
        logger.error("Error occurred when attempting to delete the historical old file during the preparation "
                     "for dynamic block forward method generation. Please check file permissions, "
                     "whether the file is occupied by other processes, or the disk status.")
        raise
    write_lines(file_content.splitlines(), block_adapter_file_path)


def find_used_top_func(import_code_str, method_forward_str, module_obj):
    """
    前向方法中用到，但import中没有，那么需要加入这个方法
    :param import_code_str:
    :param method_forward_str:
    :param module_obj: 当前操作的模块，用于获取该模块顶层方法
    :return: 被调用的、需要放进生成代码的方法定义
    """
    # 1. 获取所有顶层方法
    module_methods = get_module_methods(module_obj)
    # 2. 获取import内容
    temp_target_code = '\n'.join([import_code_str, method_forward_str])
    tree = ast.parse(temp_target_code)
    import_tracker = ImportTracker()
    import_tracker.visit(tree)
    # 3. 获取不在import中，且被forward使用的顶层方法
    usage_finder = FuncUsageFinder(module_methods, import_tracker.imports)
    usage_finder.visit(tree)
    used_funcs_code = usage_finder.used_funcs_code
    return used_funcs_code


def modify_method(method_forward_head_body, function_args: list, block_name: str) -> str:
    """
    修改 针对forward方法做修改
    :param method_forward_head_body: 给定识别条件
    :param function_args: 需要增加的参数列表
    :param block_name: block名称，用于生成forward函数名
    :return:
    """
    method_forward_head_body_tree = ast.parse(method_forward_head_body)
    # 树解析的第一个节点，就是方法节点
    function_node = method_forward_head_body_tree.body[0]
    change_func_name(function_node, block_name)
    set_method_param_default_none(function_node)
    add_params_if_not_exist(function_node, function_args)
    return astor.to_source(method_forward_head_body_tree)


def has_return_statement(func_code):
    """
    查询方法节点是否包含 return 子节点
    :param func_code: 方法代码
    :return:
    """
    func_node = ast.parse(func_code)
    for node in ast.walk(func_node):
        # 如果找到 Return 节点，说明该函数有返回语句
        if isinstance(node, ast.Return):
            return True
    return False


def gen_method_forward(source_code: str, block_adapter: BlockAdapter) -> str:
    """
    获取 forward 及 get_output_name 方法
    :param source_code: 给定识别条件
    :param block_adapter: 给定识别条件
    :return:
    """
    # 提取原函数指定范围的代码
    target_code = get_effective_part(block_adapter, source_code)
    target_code = textwrap.dedent(target_code)
    try:
        target_code_tree = ast.parse(target_code)
    except SyntaxError as e:
        logger.error(f'Cannot parse target forward method code for {block_adapter.block_name}, '
                     f'please check keywords and source code')
        raise RuntimeError('Cannot parse target code') from e
    first_node_of_tree = target_code_tree.body[0]
    # 检查head_body 部分是否已包含函数定义，若无，则需要重新加上 head；method_forward_head_body无缩进
    if isinstance(first_node_of_tree, ast.FunctionDef):
        method_forward_head_body = target_code
    else:
        # 从source_code中单独把方法头摘出来
        method_forward_head = get_function_header(textwrap.dedent(source_code))

        # 格式化，保证head 和 body 之间的相对缩进
        method_forward_head = textwrap.dedent(method_forward_head)
        method_forward_body = textwrap.indent(target_code, PYTHON_STANDARD_INDENT)
        method_forward_head_body = '\n'.join([method_forward_head, method_forward_body])

    # 处理方法名、注解以及参数等
    method_forward_head_body = modify_method(method_forward_head_body,
                                             block_adapter.append_method_signatures, block_adapter.block_name)

    # 处理方法的返回值
    has_return = has_return_statement(method_forward_head_body)
    # 这里可能为空，说明 method_forward_head_body 已经包含了函数的返回语句
    if has_return:
        method_forward_return = ''
    else:
        return_values = ', '.join(block_adapter.return_values)
        method_forward_return = f'return {return_values}'
        # 格式化，保证 return\head\body之间的相对缩进
        method_forward_return = textwrap.indent(method_forward_return, PYTHON_STANDARD_INDENT)

    # 格式化，保证与class的相对缩进
    method_forward = '\n'.join([method_forward_head_body, method_forward_return])

    # 为 post_process 块添加 output_weight 变量的定义
    if block_adapter.block_name == 'post_process':
        # 使用 AST 解析来确保正确插入变量定义
        try:
            tree = ast.parse(method_forward)
            function_def = tree.body[0]
            if isinstance(function_def, ast.FunctionDef):
                # 创建 output_weight 变量定义的 AST 节点
                assign_node = ast.Assign(
                    targets=[ast.Name(id='output_weight', ctx=ast.Store())],
                    value=ast.Constant(value=None)
                )
                # 在函数体的开始处插入变量定义
                function_def.body.insert(0, assign_node)
                # 重新生成代码
                method_forward = astor.to_source(tree).strip()
        except Exception as e:
            logger.debug(f'Error adding output_weight definition: {e}')

    return method_forward


def cut_lines(source_code: str, start_idx: int, end_idx: int):
    """
    将source_code扣掉 start_idx 和 end_idx 之间的部分
    :param source_code: 源代码
    :param start_idx: 起始行
    :param end_idx: 截止行
    :return:
    """
    lines = source_code.splitlines()
    res = list()
    res.extend(lines[: start_idx])
    res.extend(lines[end_idx + 1:])
    return '\n'.join(res)


def get_effective_part(block_adapter: BlockAdapter, source_code: str):
    """
    根据给定关键字，提取源码中对应的部分
    :param block_adapter: 存储block前向代码识别条件
    :param source_code: 目标代码所在源码
    :return: 匹配到的目标代码
    """
    start_key_word, end_key_word = block_adapter.key_words
    if start_key_word:
        # 如果有多个，取第一个关键字出现的地方
        start_line_idx = find_keywords_line_idx(source_code, start_key_word)[0]
    else:
        start_line_idx = 0

    if end_key_word:
        # 如果有多个，取最后一个关键字出现的地方
        end_line_idx = find_keywords_line_idx(source_code, end_key_word)[-1]
    else:
        end_line_idx = len(source_code.splitlines()) - 1
    if block_adapter.method_location.cut_mode:
        target_code = cut_lines(source_code, start_line_idx, end_line_idx + 1)
    else:
        target_code = get_lines(source_code, start_line_idx, end_line_idx + 1)
    return target_code


def get_function_header(function_code: str):
    """
    获取方法头
    :param function_code: 方法代码
    :return: 方法头
    """
    tree = ast.parse(function_code)
    first_node = tree.body[0]
    if isinstance(first_node, ast.FunctionDef):
        # 获取函数体中的第一个节点
        first_statement = first_node.body[0]

        # 获取第一个节点的行号
        first_statement_line = first_statement.lineno
        return get_lines(function_code, 0, first_statement_line - 1)
    else:
        raise RuntimeError('When parsing function head line, the first line should be func.')


def add_params_if_not_exist(function_node, function_args: list):
    """
    若参数不存在，则给函数节点添加参数
    :param function_node: 函数节点
    :param function_args: 要添加的参数
    :return:
    """
    if not function_args:
        return

    exist_args = [arg.arg for arg in function_node.args.args]

    for function_arg in function_args:
        # 已存在的参数，不必重复添加
        if function_arg in exist_args:
            continue
        # 创建新的参数节点
        new_arg = ast.arg(arg=function_arg, annotation=None)
        # 创建默认值节点
        default_value = ast.Constant(value=None)
        # 将新的参数和默认值添加到函数定义节点的参数列表中
        function_node.args.args.append(new_arg)
        function_node.args.defaults.append(default_value)


def set_method_param_default_none(function_node):
    """
    把函数的所有方法参数置为None
    :param function_node: 函数节点
    :return:
    """
    # 不包含 self 的参数的个数
    num_params_not_contains_self = sum(arg.arg != 'self' for arg in function_node.args.args)

    # 有默认值的参数的个数
    num_params_contains_defaults = len(function_node.args.defaults)

    # 要补的默认值为None的参数的个数
    num_default_none = num_params_not_contains_self - num_params_contains_defaults
    need_insert = [ast.Constant(value=None) for _ in range(0, num_default_none)]
    function_node.args.defaults[:0] = need_insert


def change_func_name(function_node, block_name):
    """
    改函数名
    :param function_node: 函数节点
    :param block_name: block名称，用于生成forward函数名
    :return:
    """
    function_node.name = get_forward_func_name(block_name)


def get_forward_func_name(block_name: str) -> str:
    return f'tinker_{block_name}_forward'