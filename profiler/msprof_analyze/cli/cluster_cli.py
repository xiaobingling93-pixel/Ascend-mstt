# -------------------------------------------------------------------------
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#    http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import click

from msprof_analyze.prof_common.constant import Constant
from msprof_analyze.cluster_analyse.cluster_analysis import ALL_FEATURE_LIST, Interface
from msprof_analyze.prof_common.path_manager import PathManager
from msprof_analyze.advisor.version import print_version_callback, cli_version

context_settings = dict(Constant.CONTEXT_SETTINGS)
context_settings['ignore_unknown_options'] = True


@click.command(context_settings=context_settings, name="cluster",
               short_help='Analyze cluster data to locate performance bottleneck')
@click.option('--profiling_path', '-d', type=click.Path(), required=True, callback=PathManager.expanduser_for_cli,
              help='path of the profiling data')
@click.option('--mode', '-m', type=click.Choice(ALL_FEATURE_LIST), default='all')
@click.option('--output_path', '-o', type=click.Path(), default='', callback=PathManager.expanduser_for_cli,
              help='Path of cluster analysis output')
@click.option('--force', is_flag=True,
              help="Indicates whether to skip verification of the owner, size, and permissions.")
@click.option("--parallel_mode", type=str, help="context mode", default="concurrent")
@click.option("--export_type", help="recipe export type", type=click.Choice(["db", "notebook"]), default="db")
@click.option("--rank_list", type=str, help="Rank id list", default='all')
@click.option("--step_id", type=int, help="Step id", default=Constant.VOID_STEP)
@click.option('--version', '-V', '-v', is_flag=True,
              callback=print_version_callback, expose_value=False,
              is_eager=True, help=cli_version())
@click.argument('args', nargs=-1)
def cluster_cli(**kwargs) -> None:
    """
    subcommand:\n
        compare    Compare the performance diffs between GPUs and NPUs. try 'msprof-analyze compare -h' for detail\n
        advisor    Analyze and give performance optimization suggestion.try 'msprof-analyze advisor -h' for detail\n
        cluster    default run as cluster subcommand
    """
    Interface(kwargs).run()

