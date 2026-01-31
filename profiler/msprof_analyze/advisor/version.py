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

import sys


def get_package_version(package_name) -> str:
    """
    Get package version info by importlib
    Args:
        package_name: package name

    Returns:
        version: version info string
    """
    if sys.version_info >= (3, 8):
        # Because importlib_metadata has been changed to importlib.metadata in py3.8
        from importlib import metadata
        from importlib.metadata import PackageNotFoundError
    else:
        import importlib_metadata as metadata
        from importlib_metadata import PackageNotFoundError

    try:
        version = metadata.version(package_name)
    except PackageNotFoundError:
        version = "UNKNOWN"
    return version


def print_version_callback(ctx, param, value):  # NOQA
    import click

    if not value or ctx.resilient_parsing:
        return
    click.echo('Version {}'.format(get_package_version("msprof-analyze")))
    ctx.exit()


def cli_version():
    return get_package_version("msprof-analyze")
