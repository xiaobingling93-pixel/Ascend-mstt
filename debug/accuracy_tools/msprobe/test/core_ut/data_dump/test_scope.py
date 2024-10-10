import unittest
from unittest.mock import MagicMock

from msprobe.core.common.exceptions import ScopeException
from msprobe.core.data_dump.scope import (
    build_scope,
    build_range_scope_according_to_scope_name,
    BaseScope,
    ListScope,
    RangeScope,
    APIRangeScope,
    ModuleRangeScope
)


class TestBuildScope(unittest.TestCase):

    def test_build_scope_with_no_scope_and_no_api_list(self):
        result = build_scope(None)
        self.assertIsNone(result)

    def test_build_scope_with_no_scope(self):
        result = build_scope(None, api_list=['api1', 'api2'])
        self.assertIsInstance(result, APIRangeScope)

    def test_build_scope_with_no_api_list(self):
        result = build_scope(None, scope=['scope1', 'scope2'])
        self.assertIsInstance(result, APIRangeScope)

    def test_build_scope_with_valid_scope_class(self):
        class DummyScope:
            def __init__(self, scope, api_list):
                self.scope = scope
                self.api_list = api_list

        result = build_scope(DummyScope, scope=['scope1', 'scope2'], api_list=['api1', 'api2'])
        self.assertIsInstance(result, DummyScope)
        self.assertEqual(result.scope, ['scope1', 'scope2'])
        self.assertEqual(result.api_list, ['api1', 'api2'])

    def test_build_scope_with_invalid_scope_class(self):
        with self.assertRaises(TypeError):
            build_scope("NotAScopeClass", scope=['scope1'], api_list=['api1'])

    def test_build_range_scope_with_valid_api_range_scope(self):
        result = build_range_scope_according_to_scope_name(['scope1'], ['api1'])
        self.assertIsInstance(result, APIRangeScope)
        self.assertTrue(result.is_valid)

    def test_build_range_scope_with_valid_module_range_scope(self):
        result = build_range_scope_according_to_scope_name(['Module.m1', 'Module.m2'], ['api1'])
        self.assertIsInstance(result, ModuleRangeScope)
        self.assertTrue(result.is_valid)

    def test_build_range_scope_with_invalid_scope(self):
        with self.assertRaises(ScopeException) as context:
            build_range_scope_according_to_scope_name(['Module.m1', 'scope1'], ['api1'])
        self.assertIn("scope=['Module.m1', 'scope1']", str(context.exception))

    def test_build_range_scope_with_empty_scope(self):
        result = build_range_scope_according_to_scope_name([], ['api1'])
        self.assertIsInstance(result, APIRangeScope)


class TestBaseScope(unittest.TestCase):

    def setUp(self):
        class BaseScopeImpl(BaseScope):
            def check(self, name):
                pass
        self.base_scope = BaseScopeImpl(scope=["TestScope"], api_list=[])

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

    def test_check_api_list_empty(self):
        self.base_scope.api_list = []
        self.assertTrue(self.base_scope.check_api_list(""))

    def test_check_api_list_match(self):
        self.base_scope.api_list = ["api1"]
        self.assertTrue(self.base_scope.check_api_list("api1"))

    def test_check_api_list_no_match(self):
        self.base_scope.api_list = ["api1"]
        self.assertFalse(self.base_scope.check_api_list("api2"))


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
    
    def setUp(self):
        self.scope_valid = ['api.1', 'api.2']
        self.scope_invalid = ['Module.1', 'api.2']
        self.api_range_scope_valid = APIRangeScope(self.scope_valid, [])
        self.api_range_scope_invalid = APIRangeScope(self.scope_invalid, [])

    def test_check_scope_is_valid_valid_scope(self):
        result = self.api_range_scope_valid.check_scope_is_valid()
        self.assertTrue(result)

    def test_check_scope_is_valid_invalid_scope(self):
        result = self.api_range_scope_invalid.check_scope_is_valid()
        self.assertFalse(result)

    def test_check_api_in_scope(self):
        self.api_range_scope_valid.check_api_list = lambda x: x == 'api.1'
        self.assertTrue(self.api_range_scope_valid.check('api.1'))
        self.assertFalse(self.api_range_scope_valid.check('api.unknown'))

    def test_check_api_out_of_scope(self):
        self.api_range_scope_valid.check_api_list = lambda x: x == 'api.2'
        self.api_range_scope_valid.check('api.1')
        self.assertTrue(self.api_range_scope_valid.check('api.2'))

    def test_check_api_transition(self):
        self.api_range_scope_valid.check_api_list = lambda x: True
        self.api_range_scope_valid.check('api.1')
        self.assertTrue(self.api_range_scope_valid.in_scope)
        self.api_range_scope_valid.check('api.2')
        self.assertFalse(self.api_range_scope_valid.in_scope)


class TestModuleRangeScope(unittest.TestCase):

    def test_check_scope_is_valid(self):
        module_range_scope = ModuleRangeScope([], [])
        result = module_range_scope.check_scope_is_valid()
        self.assertTrue(result)

        module_range_scope = ModuleRangeScope(["Module.1"], ["Module.2"])
        self.assertTrue(module_range_scope.check_scope_is_valid())

    def test_begin_module(self):
        module_range_scope = ModuleRangeScope([], [])
        self.assertIsNone(module_range_scope.begin_module(""))

        module_range_scope = ModuleRangeScope(["module1", "module2"], [])
        module_name = "module1"
        module_range_scope.begin_module(module_name)
        self.assertTrue(module_range_scope.in_scope)

        module_range_scope = ModuleRangeScope(["module1", "module2"], [])
        module_name = "module3"
        module_range_scope.begin_module(module_name)
        self.assertFalse(module_range_scope.in_scope)

    def test_end_module(self):
        module_range_scope = ModuleRangeScope([], [])
        self.assertIsNone(module_range_scope.end_module(""))

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

        module_range_scope = ModuleRangeScope(["Module.1"], [])
        self.assertFalse(module_range_scope.check(""))
