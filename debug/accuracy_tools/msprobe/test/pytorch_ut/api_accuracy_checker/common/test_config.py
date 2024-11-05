import unittest
import os
from unittest.mock import patch

from msprobe.pytorch.api_accuracy_checker.common.config import Config, CheckerConfig, OnlineConfig, msCheckerConfig


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.base_test_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
        self.input_dir = os.path.join(self.base_test_dir, 'resources')
        self.yaml_file = os.path.join(self.input_dir, "config.yaml")
        self.cfg = Config(self.yaml_file)
        self.task_config = {
            'white_list': ['api1', 'api2'],
            'black_list': ['api3'],
            'error_data_path' : '/path/to/error_data',
            'is_online': True,
            'nfs_path': '/path/to/nfs',
            'host': 'localhost',
            'port': 8080,
            'rank_list': [0, 1, 2],
            'tls_path': '/path/to/tls'
        }

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

    def test_CheckerConfig_init_with_defaults(self):
        checker_config = CheckerConfig()
        self.assertEqual(checker_config.white_list, msCheckerConfig.white_list)
        self.assertEqual(checker_config.black_list, msCheckerConfig.black_list)
        self.assertEqual(checker_config.error_data_path, msCheckerConfig.error_data_path)
        self.assertEqual(checker_config.is_online, msCheckerConfig.is_online)
        self.assertEqual(checker_config.nfs_path, msCheckerConfig.nfs_path)
        self.assertEqual(checker_config.host, msCheckerConfig.host)
        self.assertEqual(checker_config.port, msCheckerConfig.port)
        self.assertEqual(checker_config.rank_list, msCheckerConfig.rank_list)
        self.assertEqual(checker_config.tls_path, msCheckerConfig.tls_path)


    def test_init_with_task_config(self):
        checker_config = CheckerConfig(self.task_config)
        self.assertEqual(checker_config.white_list, self.task_config.get('white_list'))
        self.assertEqual(checker_config.black_list, self.task_config.get('black_list'))
        self.assertEqual(checker_config.error_data_path, self.task_config.get('error_data_path'))
        self.assertEqual(checker_config.is_online, self.task_config.get('is_online'))
        self.assertEqual(checker_config.nfs_path, self.task_config.get('nfs_path'))
        self.assertEqual(checker_config.host, self.task_config.get('host'))
        self.assertEqual(checker_config.port, self.task_config.get('port'))
        self.assertEqual(checker_config.rank_list, self.task_config.get('rank_list'))
        self.assertEqual(checker_config.tls_path, self.task_config.get('tls_path'))

    def test_load_config(self):
        checker_config = CheckerConfig()
        checker_config.load_config(self.task_config)
        self.assertEqual(checker_config.is_online, self.task_config.get('is_online'))
        self.assertEqual(checker_config.nfs_path, self.task_config.get('nfs_path'))
        self.assertEqual(checker_config.host, self.task_config.get('host'))
        self.assertEqual(checker_config.port, self.task_config.get('port'))
        self.assertEqual(checker_config.rank_list, self.task_config.get('rank_list'))
        self.assertEqual(checker_config.tls_path, self.task_config.get('tls_path'))

    def test_get_online_config(self):
        checker_config = CheckerConfig()
        checker_config.load_config(self.task_config)
        online_config = checker_config.get_online_config()
        self.assertIsInstance(online_config, OnlineConfig)
        self.assertEqual(online_config.is_online, self.task_config['is_online'])
        self.assertEqual(online_config.nfs_path, self.task_config['nfs_path'])
        self.assertEqual(online_config.host, self.task_config.get('host'))
        self.assertEqual(online_config.port, self.task_config.get('port'))
        self.assertEqual(online_config.rank_list, self.task_config.get('rank_list'))
        self.assertEqual(online_config.tls_path, self.task_config.get('tls_path'))
