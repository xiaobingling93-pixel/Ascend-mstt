import os
import logging
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
        for file in npy_path_obj.rglob('*.npy'):
            npy_files.append(file.resolve())
        return npy_files


def write_to_csv(param: Dict, output_dir: str, append=False):
    """
    将参数写入CSV文件。

    Parameters:
        param (Dict): 要写入的数据，格式为{文件名: (代码堆栈, 作用域名称)}。
        output_dir (str): 输出目录路径。
        append (bool): 是否以追加模式写入。
    """
    try:
        # 创建安全的输出目录
        create_directory(output_dir)
    except Exception as e:
        logger.error(f"无法创建目录 {output_dir}: {e}")
        raise

    file_path = Path(output_dir) / "code.csv"
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
        logger.warning(f"提供的路径不是一个有效的目录或目录不存在: {directory}")
        return []
    pattern = os.path.join(directory, '**', "statistic.csv")
    logger.info(f"使用的搜索模式: {pattern}")

    statistic_files = list(glob.glob(pattern, recursive=True))
    logger.info(f"使用的搜索模式: {pattern}")
    for file in statistic_files:
        logger.info(f"找到的文件: {file}")
    return statistic_files


def bind_for_statistic(statistic_files: List[str], match_dict: Dict):
    """
    处理统计文件，绑定代码信息。

    Parameters:
        statistic_files (List[str]): 统计文件路径列表。
        match_dict (Dict): 匹配字典，用于复杂映射。
    """
    for statistic_file in statistic_files:
        with FileOpen(statistic_file, "r") as f:
            df = pd.read_csv(f)
            logger.info(f"读取的 DataFrame（前5行）:\n{df.head()}")

        # 进行复杂映射
        df = complex_map(df, match_dict)
        logger.info(f"经过 complex_map 处理后的 DataFrame（前5行）:\n{df.head()}")

        logger.info(f"成功写入更新后的 DataFrame 到: {statistic_file}")

        # 使用write_csv安全写入文件
        df.to_csv(statistic_file, index=False)
        # write_csv(df, statistic_file, mode="w", malicious_check=False)
        logger.info(f"成功写入更新后的 DataFrame 到: {statistic_file}")


def bind_code_info_for_data(input_dir: str, nodes: Dict[str, 'GraphNode']) -> Dict[str, str]:
    logger.info(f"Starting bind_code_info_for_data with input_dir: {input_dir}")

    # 待重构后优化性能
    match_dict = {}
    logger.info("Initializing match_dict")
    for node in nodes.values():
        logger.info(f"Processing node: {node}")
        # 屏蔽子图节点
        if node.is_subgraph:
            logger.info(f"Skipping subgraph node: {node}")
            continue
        # 获取规范化后的scope name
        scope_name = node.scope.replace("/", "_")
        logger.info(f"Normalized scope name: {scope_name}")
        match_dict[scope_name] = node
    logger.info(f"Completed building match_dict with {len(match_dict)} entries")

    npy_files = find_npy_files(input_dir)

    bind_result = {}
    if not npy_files:
        logger.info("No npy_files found. Searching for statistic_files.")
        statistic_files = find_statistic_files(input_dir)
        logger.info(f"Found {len(statistic_files)} statistic_files")
        if statistic_files:
            logger.info("Binding for statistic files")
            bind_for_statistic(statistic_files, match_dict)
        logger.info("Exiting bind_code_info_for_data as no npy_files were found")
        return bind_result

    for idx, npy_file in enumerate(npy_files, start=1):
        logger.info(f"Processing npy_file {idx}/{len(npy_files)}: {npy_file}")
        try:
            directory, file_name = os.path.split(npy_file)  # 拆分路径
            logger.info(f"Directory: {directory}, File name: {file_name}")
            name_without_ext = os.path.splitext(file_name)[0]  # 提取文件名（去掉扩展名）
            logger.info(f"Name without extension: {name_without_ext}")

            if '.' not in name_without_ext:
                logger.info("'.' not found in name_without_ext. Processing mapping.csv")
                # 3. 读取find.csv文件
                csv_file_path = os.path.join(directory, 'mapping.csv')
                logger.info(f"Reading CSV file at: {csv_file_path}")
                df = pd.read_csv(csv_file_path, header=None)
                logger.info(f"CSV file read successfully with {len(df)} rows")

                # 4. 查找是否有与xxx.npy匹配的条目
                matching_row = df[df[0] == file_name]  # 假设A列存储文件名
                if not matching_row.empty:
                    corresponding_name = matching_row[1].values[0]
                    logger.info(f"The corresponding name in column B is: {corresponding_name}")
                else:
                    corresponding_name = None
                    logger.warning(f"No entry found for {file_name} in mapping.csv.")
                if corresponding_name:
                    name_without_ext = os.path.splitext(corresponding_name)[0]
                    logger.info(f"Updated name_without_ext after mapping: {name_without_ext}")
                else:
                    logger.error(f"corresponding_name is None for file: {file_name}")
                    continue  # 跳过当前循环
            else:
                logger.info("'.' found in name_without_ext. Skipping CSV mapping.")

            npy_path = os.path.realpath(npy_file)
            logger.info(f"Real path of npy_file: {npy_path}")

            split_parts = name_without_ext.split(".")
            if len(split_parts) < 2:
                logger.error(f"Unexpected format in name_without_ext: {name_without_ext}")
                continue  # 跳过当前循环
            node_scope = split_parts[1]
            logger.info(f"Extracted node_scope: {node_scope}")

            trie = Trie()
            logger.info("Initializing Trie and inserting match_dict entries")
            for key, value in match_dict.items():
                trie.insert(key, value)
            logger.info("Trie populated with match_dict entries")

            bind_code = match_codes(trie, node_scope)
            logger.info(f"Obtained bind_code: {bind_code} for node_scope: {node_scope}")

            bind_name = match_names(trie, node_scope)
            logger.info(f"Obtained bind_name: {bind_name} for node_scope: {node_scope}")

            bind_result[npy_path] = (bind_code, bind_name)
            logger.info(f"Added bind_result entry for: {npy_path}")
        except Exception as e:
            logger.exception(f"Exception occurred while processing npy_file: {npy_file}")

    logger.info(f"Completed bind_code_info_for_data with {len(bind_result)} bind_result entries")
    return bind_result