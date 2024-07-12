import unittest
import os
from unittest.mock import patch

from api_accuracy_checker.common.config import Config


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
            self.cfg.validate('real_data', 'true')

    def test_validate_should_raise_when_invalid_key(self):
        with self.assertRaises(ValueError):
            self.cfg.validate('invalid_key', 'mock_value')

    def test_validate_target_iter(self):
        valid_target_iter = [0, 1, 2]
        self.assertEqual(self.cfg.validate('target_iter', valid_target_iter), valid_target_iter)

        # int
        with self.assertRaises(ValueError):
            self.cfg.validate('target_iter', [-1, 2, 3])

        # bool
        with self.assertRaises(ValueError):
            self.cfg.validate('target_iter', [1, True, 3])

    def test_validate_precision(self):
        self.assertEqual(self.cfg.validate('precision', 1), 1)

        with self.assertRaises(ValueError):
            self.cfg.validate('precision', -1)

    def test_validate_white_list(self):
        validate_white_list = ['conv1d', 'max_pool1d', 'dropout', '__add__']
        self.assertEqual(self.cfg.validate('white_list', validate_white_list), validate_white_list)

        with self.assertRaises(ValueError):
            self.cfg.validate('white_list', ['invalid_api1', 'invalid_api2'])

    @patch('os.path.exists', return_value=True)
    def test_validate_nfs_path(self, mock_path_exists):
        with self.assertRaises(ValueError):
            self.cfg.validate('nsf_path', '/invalid/path')

    def test_update_config(self):
        self.cfg.update_config(dump_path='/new/path/to/dump', real_data=False, port=9000)
        self.assertEqual(self.cfg.dump_path, '/new/path/to/dump')
        self.assertEqual(self.cfg.real_data, False)
        self.assertEqual(self.cfg.port, 9000)
