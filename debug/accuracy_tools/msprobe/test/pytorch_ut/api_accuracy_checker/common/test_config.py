import unittest
import os
from unittest.mock import patch

from msprobe.pytorch.api_accuracy_checker.common.config import Config


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.base_test_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
        self.input_dir = os.path.join(self.base_test_dir, 'resources')
        self.yaml_file = os.path.join(self.input_dir, "config.yaml")
        self.cfg = Config(self.yaml_file)

    def test_validate_valid_data(self):
        for key, val in self.cfg.config.items():
            validated_type = self.cfg.validate(key, val)
            self.assertEqual(validated_type, val)

    def test_validate_should_raise_when_invalid_type(self):
        with self.assertRaises(ValueError):
            self.cfg.validate('error_data_path', True)

    def test_validate_should_raise_when_invalid_key(self):
        with self.assertRaises(ValueError):
            self.cfg.validate('invalid_key', 'mock_value')

    def test_validate_precision(self):
        self.assertEqual(self.cfg.validate('precision', 1), 1)

        with self.assertRaises(ValueError):
            self.cfg.validate('precision', -1)

    def test_validate_white_list(self):
        validate_white_list = ['conv1d', 'max_pool1d', 'dropout', '__add__']
        self.assertEqual(self.cfg.validate('white_list', validate_white_list), validate_white_list)

        with self.assertRaises(Exception):
            self.cfg.validate('white_list', ['invalid_api1', 'invalid_api2'])
