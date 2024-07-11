import unittest
from atat.pytorch.api_accuracy_checker.common.config import Config


class TestConfig(unittest.TestCase):
    def setUp(self):
        yaml_path = "./config.yaml"
        self.yaml_file = yaml_path
        self.config = Config(self.yaml_file)

    def test_validate(self):
        self.assertEqual(self.config.validate('error_data_path', '/path/to/dump'), '/path/to/dump')
        self.assertEqual(self.config.validate('white_list', ['conv2d']), ['conv2d'])
        self.assertEqual(self.config.validate('precision', 10), 10)

        with self.assertRaises(ValueError):
            self.config.validate('error_data_path', 123)
            self.config.validate('white_list', 123)
            self.config.validate('precision', '14')
