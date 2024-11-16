import unittest
from unittest.mock import patch

from msprobe.core.overflow_check.api_info import APIInfo
from msprobe.core.overflow_check.filter import IgnoreFilter, Rule, IgnoreItem


class TestIgnoreItem(unittest.TestCase):
    def setUp(self):
        self.ignore_item = IgnoreItem()

    def test_add_index(self):
        self.ignore_item.add_index(0)
        self.ignore_item.add_index(1)
        self.assertEqual(self.ignore_item.index, {0, 1})

    def test_add_name(self):
        self.ignore_item.add_name("param1")
        self.ignore_item.add_name("param2")
        self.assertEqual(self.ignore_item.name, {"param1", "param2"})

    def test_has_index(self):
        self.ignore_item.add_index(0)
        self.assertTrue(self.ignore_item.has_index(0))
        self.assertFalse(self.ignore_item.has_index(1))

    def test_has_name(self):
        self.ignore_item.add_name("param1")
        self.assertTrue(self.ignore_item.has_name("param1"))
        self.assertFalse(self.ignore_item.has_name("param2"))


class TestRule(unittest.TestCase):
    def setUp(self):
        self.rule = Rule(
            api_name="test_api",
            desc="Test description",
            input_ignore=[{"index": 0}, {"name": "param1"}],
            output_ignore=[{"index": 0}]
        )

    def test_init(self):
        self.assertEqual(self.rule.api_name, "test_api")
        self.assertEqual(self.rule.desc, "Test description")
        self.assertTrue(self.rule.input_ignore.has_index(0))
        self.assertTrue(self.rule.input_ignore.has_name("param1"))
        self.assertTrue(self.rule.output_ignore.has_index(0))

    def test_verify_field_valid(self):
        self.assertTrue(self.rule.verify_field())

    def test_verify_field_invalid(self):
        # Test empty api_name
        invalid_rule = Rule(api_name="")
        self.assertFalse(invalid_rule.verify_field())

        # Test no ignore rules
        invalid_rule = Rule(api_name="test_api")
        self.assertFalse(invalid_rule.verify_field())

    def test_match_valid_api_info(self):
        api_info = APIInfo()
        self.assertTrue(self.rule.match(api_info))

    def test_match_invalid_api_name(self):
        api_info = APIInfo()
        self.assertFalse(self.rule.match(api_info))

    def test_match_unignored_nan_inf(self):
        api_info = APIInfo()
        self.assertFalse(self.rule.match(api_info))


class TestIgnoreFilter(unittest.TestCase):
    def setUp(self):
        self.yaml_data = {
            "ignore_nan_inf": [
                {
                    "api_name": "test_api",
                    "description": "Test description",
                    "input_ignore": [{"index": 0}, {"name": "param1"}],
                    "output_ignore": [{"index": 0}]
                }
            ]
        }

    @patch('msprobe.core.common.file_utils.load_yaml')
    def test_load_rules(self, mock_load_yaml):
        mock_load_yaml.return_value = self.yaml_data

        ignore_filter = IgnoreFilter()
        ignore_filter._load_rules("dummy_path")

        self.assertTrue(ignore_filter.has_api_rule("test_api"))
        self.assertEqual(len(ignore_filter.rules), 1)

    @patch('msprobe.core.common.file_utils.load_yaml')
    def test_load_rules_duplicate_api(self, mock_load_yaml):
        # Add duplicate API rule
        self.yaml_data["ignore_nan_inf"].append({
            "api_name": "test_api",
            "description": "Duplicate API",
            "input_ignore": [{"index": 1}],
            "output_ignore": [{"index": 1}]
        })

        mock_load_yaml.return_value = self.yaml_data

        ignore_filter = IgnoreFilter()
        ignore_filter._load_rules("dummy_path")

        # Should only keep the first rule
        self.assertEqual(len(ignore_filter.rules), 1)

    def test_has_api_rule(self):
        ignore_filter = IgnoreFilter()
        ignore_filter.rules = {"test_api": Rule("test_api")}

        self.assertTrue(ignore_filter.has_api_rule("test_api"))
        self.assertFalse(ignore_filter.has_api_rule("non_existent_api"))

    def test_apply_filter(self):
        ignore_filter = IgnoreFilter()
        ignore_filter.rules = {
            "test_api": Rule(
                api_name="test_api",
                input_ignore=[{"index": 0}],
                output_ignore=[{"index": 0}]
            )
        }

        api_info = APIInfo()
        self.assertTrue(ignore_filter.apply_filter(api_info))

        api_info = APIInfo()
        self.assertFalse(ignore_filter.apply_filter(api_info))

    def test_apply_filter_no_rule(self):
        ignore_filter = IgnoreFilter()
        api_info = APIInfo()
        self.assertFalse(ignore_filter.apply_filter(api_info))

    @patch('msprobe.core.common.file_utils.load_yaml')
    def test_load_rules_invalid_data(self, mock_load_yaml):
        mock_load_yaml.return_value = {"ignore_nan_inf": [
            {
                "api_name": "",  # Invalid empty API name
                "input_ignore": [{"index": 0}]
            }
        ]}

        ignore_filter = IgnoreFilter()
        ignore_filter._load_rules("dummy_path")

        self.assertEqual(len(ignore_filter.rules), 0)


if __name__ == '__main__':
    unittest.main()
