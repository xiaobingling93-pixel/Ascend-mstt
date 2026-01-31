# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
# `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------


from typing import Optional, Any, Tuple, Dict, Callable


class HandlerParams:
    """
    参数结合体

    """
    args: Optional[Tuple] = None
    kwargs: Optional[Dict] = None
    index: Optional[int] = None
    original_result: Optional[Any] = None
    fuzzed_result: Optional[Any] = None
    is_consistent: Optional[bool] = True
    fuzzed_value: Optional[Any] = None
    original_func: Optional[Callable] = None
