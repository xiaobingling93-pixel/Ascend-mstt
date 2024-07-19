import unittest
from unittest.mock import MagicMock

from atat.core.common.exceptions import ScopeException
from atat.core.data_dump.scope import (
    build_scope,
    build_range_scope_according_to_scope_name,
    BaseScope,
    ListScope,
    RangeScope,
    APIRangeScope,
    ModuleRangeScope
)


class TestBuildScope(unittest.TestCase):
    def test_build_scope(self):
        scope_class = MagicMock()
        result1 = build_scope(scope_class, None, None)
        self.assertEqual(result1, None)

        api_list = ['api1', 'api2']
        result2 = build_scope(scope_class, None, api_list)
        self.assertEqual(result2, scope_class.return_value)

    def test_build_range_scope_according_to_scope_name(self):
        result = build_range_scope_according_to_scope_name([], [])
        self.assertIsInstance(result, APIRangeScope)


class TestBaseScope(unittest.TestCase):
    def test_rectify_args(self):
        scope = []
        api_list = "invalid_api_list"
        with self.assertRaises(ScopeException) as context:
            BaseScope.rectify_args(scope, api_list)
        self.assertEqual(context.exception.code, ScopeException.InvalidApiStr)

        api_list = [1, 2, 3]
        with self.assertRaises(ScopeException) as context:
            BaseScope.rectify_args(scope, api_list)
        self.assertEqual(context.exception.code, ScopeException.InvalidApiStr)

        scope = "module1"
        api_list = []

        expected_scope = ["module1"]
        expected_api_list = []
        result_scope, result_api_list = BaseScope.rectify_args(scope, api_list)
        self.assertEqual(result_scope, expected_scope)
        self.assertEqual(result_api_list, expected_api_list)

        scope = 123
        api_list = []
        with self.assertRaises(ScopeException) as context:
            BaseScope.rectify_args(scope, api_list)
        self.assertEqual(context.exception.code, ScopeException.InvalidScope)

        scope = ["module1", 2, "module3"]
        api_list = []
        with self.assertRaises(ScopeException) as context:
            BaseScope.rectify_args(scope, api_list)
        self.assertEqual(context.exception.code, ScopeException.InvalidScope)


class TestListScope(unittest.TestCase):
    def test_rectify_args(self):
        scope = ["module1"]
        api_list = ["api1"]
        with self.assertRaises(ScopeException) as context:
            ListScope.rectify_args(scope, api_list)
        self.assertEqual(context.exception.code, ScopeException.ArgConflict)

    def test_check(self):
        list_scope = ListScope([], [])
        module_name = "module1"
        result = list_scope.check(module_name)
        self.assertTrue(result)

        list_scope = ListScope(["module1"], [])
        module_name = "module1"
        result = list_scope.check(module_name)
        self.assertTrue(result)

        list_scope = ListScope(["module1"], [])
        module_name = "module2"
        result = list_scope.check(module_name)
        self.assertFalse(result)


class TestRangeScope(unittest.TestCase):
    def test_rectify_args(self):
        scope = ["module1", "module2", "module3"]
        with self.assertRaises(ScopeException) as context:
            RangeScope.rectify_args(scope, [])
        self.assertEqual(context.exception.code, ScopeException.InvalidScope)

        scope = ["module1"]
        expected_scope = ["module1", "module1"]
        result_scope, result_api_list = RangeScope.rectify_args(scope, [])
        self.assertEqual(result_scope, expected_scope)


class TestAPIRangeScope(unittest.TestCase):
    def test_check_scope_is_valid(self):
        api_range_scope = APIRangeScope([], [])
        result = api_range_scope.check_scope_is_valid()
        self.assertTrue(result)

    def test_check(self):
        api_range_scope = APIRangeScope([], [])
        api_name = "api1"
        result = api_range_scope.check(api_name)
        self.assertTrue(result)


class TestModuleRangeScope(unittest.TestCase):
    def test_check_scope_is_valid(self):
        module_range_scope = ModuleRangeScope([], [])
        result = module_range_scope.check_scope_is_valid()
        self.assertTrue(result)

    def test_begin_module(self):
        module_range_scope = ModuleRangeScope(["module1", "module2"], [])
        module_name = "module1"
        module_range_scope.begin_module(module_name)
        self.assertTrue(module_range_scope.in_scope)

        module_range_scope = ModuleRangeScope(["module1", "module2"], [])
        module_name = "module3"
        module_range_scope.begin_module(module_name)
        self.assertFalse(module_range_scope.in_scope)

    def test_end_module(self):
        module_range_scope = ModuleRangeScope(["module1", "module2"], [])
        module_name = "module2"
        module_range_scope.in_scope = True
        module_range_scope.end_module(module_name)
        self.assertFalse(module_range_scope.in_scope)

        module_range_scope = ModuleRangeScope(["module1", "module2"], [])
        module_name = "module3"
        module_range_scope.in_scope = True
        module_range_scope.end_module(module_name)
        self.assertTrue(module_range_scope.in_scope)

    def test_check(self):
        module_range_scope = ModuleRangeScope([], [])
        module_name = "module1"
        result = module_range_scope.check(module_name)
        self.assertTrue(result)
