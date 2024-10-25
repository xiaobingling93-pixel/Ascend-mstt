# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
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

from abc import ABC, abstractmethod

from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import ScopeException


def build_scope(scope_class, scope=None, api_list=None):
    if not scope and not api_list:
        return None
    if scope is None:
        scope = []
    if api_list is None:
        api_list = []
    if scope_class:
        return scope_class(scope, api_list)
    return build_range_scope_according_to_scope_name(scope, api_list)


def build_range_scope_according_to_scope_name(scope, api_list):
    api_range_scope = APIRangeScope(scope, api_list)
    module_range_scope = ModuleRangeScope(scope, api_list)
    if not scope:  # 如果没有scope参数则用哪类scope都一样
        return api_range_scope
    if api_range_scope.is_valid and module_range_scope.is_valid:
        raise ScopeException(ScopeException.InvalidScope, f"scope={scope}.")
    elif api_range_scope.is_valid:
        return api_range_scope
    elif module_range_scope.is_valid:
        return module_range_scope
    else:
        raise ScopeException(ScopeException.InvalidScope, f"scope={scope}")


class BaseScope(ABC):
    Module_Type_Module = "Module"
    Module_Type_API = "api"
    module_type = ["Module", "Cell"]

    def __init__(self, scope, api_list):
        scope, api_list = self.rectify_args(scope, api_list)
        self.scope = scope
        self.api_list = api_list

    @staticmethod
    def rectify_args(scope, api_list):
        if not isinstance(api_list, list):
            raise ScopeException(ScopeException.InvalidApiStr,
                f"api_list参数须配置为列表，实际类型为{type(api_list)}.")
        for api in api_list:
            if not isinstance(api, str):
                raise ScopeException(ScopeException.InvalidApiStr,
                    f"api_list中的元素须配置为字符串，实际类型为{type(api)}.")
        if isinstance(scope, str):
            scope = [scope]
            return scope, api_list
        if not isinstance(scope, list):
            raise ScopeException(ScopeException.InvalidScope,
                f"scope参数须配置为字符串或列表，实际类型为{type(scope)}.")
        for s in scope:
            if not isinstance(s, str):
                raise ScopeException(ScopeException.InvalidScope,
                f"scope列表元素要求类型为字符串，实际类型为{type(s)}.")
        return scope, api_list

    @abstractmethod
    def check(self, name):
        pass

    def check_api_list(self, api_name):
        if not self.api_list:
            return True
        for api_str in self.api_list:
            if api_str in api_name:
                return True
        return False


class ListScope(BaseScope):
    @staticmethod
    def rectify_args(scope, api_list):
        if scope and api_list:
            raise ScopeException(ScopeException.ArgConflict,
                f"scope和api_list不可以同时配置，实际配置为scope={scope}, api_list={api_list}.")
        return super(ListScope, ListScope).rectify_args(scope, api_list)

    def check(self, name):
        if not self.scope or name in self.scope:
            return self.check_api_list(name)
        return False


class RangeScope(BaseScope, ABC):

    def __init__(self, *args):
        super().__init__(*args)
        self.in_scope = False
        self.is_valid = self.check_scope_is_valid()

    @staticmethod
    def rectify_args(scope, api_list):
        scope, api_list = super(RangeScope, RangeScope).rectify_args(scope, api_list)
        if isinstance(scope, list):
            if len(scope) == 1:
                scope.append(scope[0])
            elif len(scope) > 2:
                raise ScopeException(ScopeException.InvalidScope,
                    f"scope参数指定区间断点，须传入长度为1或2的列表，实际长度为{len(scope)}.")
        return scope, api_list

    @abstractmethod
    def check_scope_is_valid(self):
        pass

    def begin_module(self, module_name):
        pass

    def end_module(self, module_name):
        pass


class APIRangeScope(RangeScope):
    def check_scope_is_valid(self):
        if not self.scope:
            return True
        scope_start_type = self.scope[0].split(Const.SEP)[0]
        if scope_start_type in BaseScope.module_type:
            return False
        scope_stop_type = self.scope[1].split(Const.SEP)[0]
        if scope_stop_type in BaseScope.module_type:
            return False
        return True

    def check(self, name):
        if self.scope and name == self.scope[0]:
            self.in_scope = True

        if not self.scope or self.in_scope:
            result = self.check_api_list(name)
        else:
            result = False

        if self.scope and name == self.scope[1]:
            self.in_scope = False
        return result


class ModuleRangeScope(RangeScope):
    """
        模块与api不同的是，模块内部还有子结构需要dump，
        需要用pre_hook和full_backward_hook来精确控制module的开始和结束，
        在这些hook触发时调用begin_module和end_module做区间控制
    """

    def check_scope_is_valid(self):
        if not self.scope:
            return True
        scope_start_type = self.scope[0].split(Const.SEP)[0]
        scope_stop_type = self.scope[1].split(Const.SEP)[0]
        if scope_start_type in BaseScope.module_type and \
                scope_stop_type in BaseScope.module_type:
            return True
        return False

    def begin_module(self, module_name):
        if not self.scope:
            return
        if module_name == self.scope[0]:
            self.in_scope = True

    def end_module(self, module_name):
        if not self.scope:
            return
        if module_name == self.scope[1]:
            self.in_scope = False

    def check(self, name):
        if not self.scope or self.in_scope:
            return self.check_api_list(name)
        return False
