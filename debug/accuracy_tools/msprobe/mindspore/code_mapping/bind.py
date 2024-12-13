import os
import logging
import time
import glob
from typing import Dict, List
from pathlib import Path
import pandas as pd
from msprobe.mindspore.code_mapping.graph import GraphNode
from msprobe.core.common.const import Const, CompareConst, MsCompareConst
from msprobe.core.common.file_utils import (
    FileOpen,
    create_directory,
    write_csv,
    load_json,
    load_yaml
)
from msprobe.mindspore.common.log import logger


# 定义Trie节点
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_key = False
        self.value = None


# 定义Trie树
class Trie:
    def __init__(self):
        self.root = TrieNode()

    # 向Trie中插入一个键
    def insert(self, key, value):
        node = self.root
        for char in key:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        # 标记结束位置
        node.is_end_of_key = True
        node.value = value

    # 在name字符串中查找所有匹配的键
    def search_in_string(self, string):
        matched_values = []
        for i in range(len(string)):
            node = self.root
            j = i
            # 从字符串的每个字符开始，逐字符查找匹配
            while j < len(string) and string[j] in node.children:
                node = node.children[string[j]]
                if node.is_end_of_key:
                    matched_values.append(node.value)
                j += 1
        return matched_values


# 定义匹配函数
def match_codes(trie, name):
    matched_nodes = trie.search_in_string(name)
    matched_codes = ['\n'.join(ii.code_info) for ii in matched_nodes]
    return '\n'.join(matched_codes)


def match_names(trie, name):
    matched_nodes = trie.search_in_string(name)
    matched_names = [ii.scope for ii in matched_nodes]
    return '\n'.join(matched_names)


def complex_map(df, match_dict):
# 构建Trie树并插入所有键
    trie = Trie()
    for key, value in match_dict.items():
        trie.insert(key, value)

    df['Code Stack'] = df['Op Name'].apply(lambda name: match_codes(trie, name))
    df['Scope Name'] = df['Op Name'].apply(lambda name: match_names(trie, name))
    return df


def find_npy_files(npy_path):
    """
    查找指定路径下所有的.npy文件。

    Parameters:
        npy_path (str): 搜索的路径，可以是文件或目录。

    Returns:
        List[Path]: 找到的.npy文件路径列表。
    """
    npy_files = []
    npy_path_obj = Path(npy_path)

    # 检查当前路径是否是一个以 .npy 结尾的文件
    if npy_path_obj.suffix == '.npy' and npy_path_obj.is_file():
        try:
            # 使用FileOpen安全打开文件，虽然这里只是获取路径，但保持一致性
            with FileOpen(str(npy_path_obj), "rb") as f:
                # 如果需要，可以在这里进行额外的文件内容检查
                npy_files.append(npy_path_obj.resolve())
        except Exception as e:
            logger.error(f"无法打开.npy文件 {npy_path}: {e}")
        return npy_files

    # 如果是目录，使用Path.rglob查找所有.npy文件
    if npy_path_obj.is_dir():
        try:
            for file in npy_path_obj.rglob('*.npy'):
                npy_files.append(file.resolve())
        except Exception as e:
            logger.error(f"查找.npy文件失败: {e}")
    else:
        logger.warning(f"指定的路径既不是文件也不是目录: {npy_path}")

    return npy_files


def write_to_csv(param: Dict, output_dir: str, append=False):
    """
    将参数写入CSV文件。

    Parameters:
        param (Dict): 要写入的数据，格式为{文件名: (代码堆栈, 作用域名称)}。
        output_dir (str): 输出目录路径。
        append (bool): 是否以追加模式写入。
    """
    create_directory(output_dir)

    # 使用时间戳生成文件名
    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    file_path = Path(output_dir) / f"code_mapping{timestamp}.csv"

    data = [(name, res1, res2) for name, (res1, res2) in param.items()]
    df = pd.DataFrame(data, columns=['File Path', 'Code Stacks', 'Scope Name'])

    # 清洗数据，筛选掉空字符串
    df = df[(df['Code Stacks'] != '') | (df['Scope Name'] != '')]

    try:
        if append and file_path.exists():
            write_csv(df, str(file_path), mode="a", malicious_check=False)
        else:
            write_csv(df, str(file_path), mode="w", malicious_check=False)
    except Exception as e:
        logger.error(f"写入CSV文件失败: {file_path}, 错误: {e}")
        raise


def find_statistic_files(directory):
    if not os.path.isdir(directory):
        return []
    pattern = os.path.join(directory, '**', "statistic.csv")

    # 有问题 查找文件的方式
    statistic_files = list(glob.glob(pattern, recursive=True))
    return statistic_files


def check_and_fix_header(file_path: str):
    """
    检查 CSV 文件的表头是否以逗号结尾，如果没有则添加一个逗号。

    Parameters:
        file_path (str): CSV 文件的路径。

    Returns:
        bool: 如果表头被修改，返回 True；否则，返回 False。
    """

    with FileOpen(file_path, "r") as f:
        lines = f.readlines()

    if not lines:
        logger.warning(f"文件 {file_path} 是空的。")
        return False

    # 获取表头并去除末尾的换行符
    header = lines[0].rstrip('\n').rstrip('\r')

    if not header.endswith(','):
        logger.info(f"表头不以逗号结尾，正在为文件 {file_path} 添加逗号。")
        # 添加逗号并恢复换行符
        lines[0] = header + ',\n'

        # 写回修复后的内容到文件
        with FileOpen(file_path, "w") as f:
            f.writelines(lines)
        logger.info(f"已为文件 {file_path} 添加末尾的逗号。")
        return True
    else:
        logger.info(f"表头已以逗号结尾，无需修改。")
        return False


def bind_for_statistic(statistic_files: List[str], match_dict: Dict):
    """
    处理统计文件，绑定代码信息。

    Parameters:
        statistic_files (List[str]): 统计文件路径列表。
        match_dict (Dict): 匹配字典，用于复杂映射。
    """
    for statistic_file in statistic_files:
        # 使用FileOpen安全打开文件
        header_modified = check_and_fix_header(statistic_file)
        if header_modified:
            logger.info(f"文件 {statistic_file} 的表头已被修复。")

        with FileOpen(statistic_file, "r") as f:
            df = pd.read_csv(f)

        # 进行复杂映射
        df = complex_map(df, match_dict)

        # 使用write_csv安全写入文件
        df.to_csv(statistic_file, index=False)


def bind_code_info_for_data(input_dir: str, nodes: Dict[str, GraphNode]) -> Dict[str, str]:
    # 待重构后优化性能
    match_dict = {}
    for node in nodes.values():
        # 屏蔽子图节点
        if node.is_subgraph:
            continue
        # 获取规范化后的scope name
        scope_name = node.scope.replace("/", "_")
        match_dict[scope_name] = node
    npy_files = find_npy_files(input_dir)

    bind_result = {}
    if not npy_files:
        statistic_files = find_statistic_files(input_dir)
        if statistic_files:
            bind_for_statistic(statistic_files, match_dict)
        return bind_result

    for npy_file in npy_files:
        directory, file_name = os.path.split(npy_file)  # 拆分路径
        name_without_ext = os.path.splitext(file_name)[0]  # 提取文件名（去掉扩展名）
        if '.' not in name_without_ext:
            # 3. 读取find.csv文件
            csv_file_path = os.path.join(directory, 'mapping.csv')
            df = pd.read_csv(csv_file_path, header=None)

            # 4. 查找是否有与xxx.npy匹配的条目
            matching_row = df[df[0] == file_name]  # 假设A列存储文件名
            if not matching_row.empty:
                corresponding_name = matching_row[1].values[0]
            else:
                corresponding_name = None
            name_without_ext = os.path.splitext(corresponding_name)[0]
        npy_path = os.path.realpath(npy_file)
        node_scope = name_without_ext.split(".")[1]
        trie = Trie()
        for key, value in match_dict.items():
            trie.insert(key, value)
        bind_code = match_codes(trie, node_scope)
        bind_name = match_names(trie, node_scope)
        bind_result[npy_path] = (bind_code, bind_name)
    return bind_result
