import os
import unittest
from msprobe.visualization.mapping_config import MappingConfig


class TestMappingConfig(unittest.TestCase):

    def setUp(self):
        self.yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mapping.yaml")

    def test_validate(self):
        with self.assertRaises(ValueError):
            MappingConfig.validate(123, "some value")
        with self.assertRaises(ValueError):
            MappingConfig.validate("some key", 456)
        self.assertEqual(MappingConfig.validate("key", "value"), "value")

    def test_convert_to_regex(self):
        regex = MappingConfig.convert_to_regex("hello{world}")
        self.assertEqual(regex, ".*hello\\{world\\}.*")

    def test_replace_parts(self):
        result = MappingConfig._replace_parts('hello world', 'world', 'everyone')
        self.assertEqual(result, 'hello everyone')
        result = MappingConfig._replace_parts('radio_model.layers.0.input_norm', 'radio_model.layers.{}.input_norm',
                                              'radio_model.transformer.layers.{}.input_layernorm')
        self.assertEqual(result, 'radio_model.transformer.layers.0.input_layernorm')

    def test_get_mapping_string(self):
        mc = MappingConfig(self.yaml_path)
        mc.classify_config = {
            'category1': [('category1.key1', 'replacement1')],
            'category2': [('category2.key1', 'replacement2')]
        }
        result = mc.get_mapping_string("some category1.key1 text")
        self.assertEqual(result, "some replacement1 text")

    def test_long_string(self):
        long_string = "x" * (MappingConfig.MAX_STRING_LEN + 1)
        mc = MappingConfig(self.yaml_path)
        result = mc.get_mapping_string(long_string)
        self.assertEqual(result, long_string)

    def test__classify_and_sort_keys(self):
        mc = MappingConfig(self.yaml_path)
        result = mc._classify_and_sort_keys()
        self.assertEqual(result, {'vision_model': [('vision_model', 'language_model.vision_encoder')],
                                  'vision_projection': [('vision_projection', 'language_model.projection')]})


if __name__ == '__main__':
    unittest.main()
