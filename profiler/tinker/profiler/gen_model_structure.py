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
"""
生成 profiler -> script -> model_xxx.sh
"""
import argparse
import logging
import os
import re
import shlex
import subprocess
import sys
import uuid

sys.path.append("./")
from tinker.utils.logger import logger, init_log
from tinker.utils.utils import project_root, extract_and_format_model_size

from tinker.utils.utils import read_file, extract_line, extract_between, del_line, write_lines

DEL_PARAM_IN_GPT_ARGS = [
    '--tensor-model-parallel-size ', '--pipeline-model-parallel-size ', '--sequence-parallel ',
    '--context-parallel-size ', '--num-layers-per-virtual-pipeline-stage ',
    '--use-distributed-optimizer ',
    '--overlap-param-gather ', '--num-layer-list ', '--load ', '--save ', '--recompute-granularity ',
    '--recompute-method ', '--recompute-num-layers ', '--context-parallel-size ', '--context-parallel-algo ',
    '--ulysses-degree-in-cp ', '--cp-attention-mask-type ', '--use-cp-send-recv-overlap ',
    '--kv-head-repeat-before-uly-alltoall '
]

DEL_PARAM_IN_MOE_ARGS = ['--expert-parallel-size ']


def replace_export(match):
    var_name = match.group(1)  # 获取变量名
    var_value = match.group(2) if match.group(2) else ''  # 获取赋值，如果没有则为空
    if var_value:  # 如果有赋值
        return f'tinker_export {var_name}={var_value}'
    else:  # 如果没有赋值
        return f'tinker_export {var_name}'


def _dump_modified_content(modified_content, dump_file_path):
    """
    存一个文件文件
    :param modified_content:
    :param dump_file_path:
    :return:
    """
    with open(dump_file_path, 'w') as file:
        # 写入字符串
        file.write(modified_content)


def gen_model_structure_version2(pretrain_script, dest_file, args):
    """
    基于运行时参数生成脚本
    :param pretrain_script: 用户的预训练脚本
    :param dest_file: 目标文件
    :param args: 用户参数
    :return:
    """
    new_script_lines = []
    pretrain_content = _hook_pretrain_script(pretrain_script)
    _add_export_content(new_script_lines, pretrain_content)
    cmd_dict_paris = _add_cmd_content(new_script_lines, pretrain_content)
    new_script_content = _del_params(new_script_lines)
    # 写文件
    if os.path.exists(dest_file) and not args.overwrite:
        raise RuntimeError(f'The file: {dest_file} already exists, if you want to overwrite, add param \'-o\'.')
    write_lines(new_script_content.splitlines(), dest_file)
    logger.info(f'successfully write file to {dest_file}')
    return cmd_dict_paris


def _del_params(new_script_lines):
    """
    删除以下参数
    :param new_script_lines:
    :return:
    """
    del_params = []
    del_params.extend(DEL_PARAM_IN_GPT_ARGS)
    del_params.extend(DEL_PARAM_IN_MOE_ARGS)
    new_script_content = del_line(del_params, '\n'.join(new_script_lines))
    return new_script_content


def _extract_nproc_per_node(content):
    search = re.search('--nproc_per_node *(\d+)', content)
    if not search:
        raise RuntimeError(f'Cannot find nproc_per_node in pretrain_script')
    nproc_per_node = search.group(1)
    if not nproc_per_node.isdigit():
        raise RuntimeError(f'Invalid nproc_per_node: {nproc_per_node}')
    return int(nproc_per_node)


def _add_cmd_content(new_script_lines, result):
    """
    增加 python 或 torchrun 部分的命令参数
    :param new_script_lines:
    :param result:
    :return:
    """
    # 提取 python 或 torchrun 后的参数
    keyword_pairs = [('python_cmd_start', 'python_cmd_end'), ('torchrun_cmd_start', 'torchrun_cmd_end')]
    for keyword_pair in keyword_pairs:
        cmd_content = extract_between(keyword_pair[0], keyword_pair[1], result.stdout)
        if cmd_content:
            break
    if not cmd_content:
        raise RuntimeError('Cannot find \'python\' or \'torchrun\' in script.')
    # 处理 cmd
    cmd_dict_pairs = []
    cmd_content = cmd_content.splitlines()[1]
    cmd_dict_pairs.append(['nproc_per_node', _extract_nproc_per_node(cmd_content)])
    split_commands = shlex.split(cmd_content)
    # 找到 '.py'，之后的参数才是需要加进来的
    index = next((idx for idx, item in enumerate(split_commands) if '.py' in item), -1)
    cmd_pairs = []
    cmd_pair = None
    for split_command in split_commands[index + 1:]:
        if '--' in split_command:
            # 把上一个加进去
            if cmd_pair is not None:
                cmd_pairs.append(cmd_pair)
            cmd_dict_pairs.append([split_command[2:].replace('-', '_'), ''])
            cmd_pair = split_command
        else:
            cmd_pair += f' {split_command}'
            cmd_dict_pairs[-1][1] += split_command
    # 加最后一个
    cmd_pairs.append(cmd_pair)
    # 处理完后格式化拼接
    cmd_pairs = [f'    {cmd_pair} \\' for cmd_pair in cmd_pairs]
    cmd_pairs.insert(0, "\nGPT_ARGS=\"")
    cmd_pairs.append("\"")
    new_script_lines.extend(cmd_pairs)
    return cmd_dict_pairs


def _add_export_content(new_script_lines, result):
    """
    添加  export 内容
    :param new_script_lines:
    :param result:
    :return:
    """
    export_content = extract_between('export_start', 'export_end', result.stdout)
    export_lines = export_content.splitlines()[1:-1]
    new_script_lines.extend(export_lines)


def _hook_pretrain_script(pretrain_script):
    """
    hook住 export、python、torchrun 函数
    :param pretrain_script: 用户的输入脚本
    :return:
    """
    try:
        pretrain_script = read_file(pretrain_script)
    except (FileNotFoundError, RuntimeError):
        logger.error(f'cannot find pretrain script: {pretrain_script}')
        raise

    root = project_root()
    copy_of_pretrain_file = f'{root}/tinker/profiler/pretrain_file_{uuid.uuid4()}.sh'
    # 存一份临时的用户脚本
    _dump_modified_content(pretrain_script, copy_of_pretrain_file)
    if not os.path.exists(copy_of_pretrain_file):
        raise RuntimeError(f'failed to store file: {copy_of_pretrain_file}')
    wrap_file_path = os.path.join(root, 'tinker/profiler/wrap_pretrain.sh')
    command = ['bash', wrap_file_path, copy_of_pretrain_file]
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=False)
        if result.returncode != 0:
            raise RuntimeError(f"Failed run with code: {result.returncode}, error message:\n{result.stderr}")
    except Exception as e:
        logger.error(f"An error occurred while executing the command, message: {e}")
        raise
    # 删除临时存的文件
    os.remove(copy_of_pretrain_file)
    return result


def main(from_sh=True):
    if from_sh:
        init_log('', logging.INFO)
    # 需要有两个参数 1 examples 下脚本路径 2 最终生成的脚本路径
    parser = argparse.ArgumentParser(description="A script argument parser.")
    parser.add_argument('-m', '--model-name', help='input the model name.', required=True)
    parser.add_argument('-s', '--model-size', help='input the model size.', required=True)
    parser.add_argument('-v', '--version', help='choose modellink version, modellink-1.0 for example')
    parser.add_argument('-d', '--dest-path', help='the dest path to store file.', default='./')
    parser.add_argument('-o', '--overwrite', action='store_true', help='use \'-o\' to overwrite the dest file.')
    parser.add_argument('-p', '--pretrain-script-path', type=str, default=None,
                        help='the path of pretrain script')

    args = parser.parse_args()

    # 处理 model_size
    model_size = extract_and_format_model_size(args.model_size)

    # version 和
    if args.pretrain_script_path is not None:
        # 用户提供了pretrain脚本
        pretrain_script = args.pretrain_script_path
    elif args.version is not None:
        # 去modellink-ref里找
        # 项目路径
        project_root_path = project_root()
        # pretrain 脚本所在文件夹路径
        pretrain_script_dir = os.path.join(project_root_path, 'modellink-ref', args.version, 'examples',
                                           args.model_name)
        # pretrain 脚本名
        file_names = os.listdir(pretrain_script_dir)
        pretrain_script_name = None

        key_words = ['pretrain_', f'_{args.model_name}', f'_{model_size}']
        for file_name in file_names:
            if key_words[0] in file_name and key_words[1] in file_name and key_words[2] in file_name.lower():
                pretrain_script_name = file_name
                break
        if pretrain_script_name is None:
            raise RuntimeError(
                f'Cannot find file in dir {pretrain_script_dir}, the file name should contains {key_words}')
        # pretrain 脚本路径
        pretrain_script = os.path.join(pretrain_script_dir, pretrain_script_name)
    else:
        raise RuntimeError('Either pretrain_script_path or version should be provided in user input.')

    # 存储目标路径 当前文件所在文件夹
    dest_dir = os.path.dirname(os.path.abspath(__file__))
    dest_file = f'{dest_dir}/model_{args.model_name}_{model_size}.sh'
    return gen_model_structure_version2(pretrain_script, dest_file, args)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        sys.exit(1)
    sys.exit(0)
