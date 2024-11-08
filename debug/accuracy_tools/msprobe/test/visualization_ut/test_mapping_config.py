import os
import unittest
from unittest.mock import patch
from msprobe.visualization.mapping_config import MappingConfig, MappingInfo, DATA_MAPPING


class TestMappingConfig(unittest.TestCase):

    def setUp(self):
        self.yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mapping.yaml")

    def test_validate(self):
        with self.assertRaises(ValueError):
            MappingConfig.validate(123, "some value")
        with self.assertRaises(ValueError):
            MappingConfig.validate("some key", 456)
        self.assertEqual(MappingConfig.validate("key", "value"), "value")

    @patch('msprobe.core.compare.layer_mapping.generate_api_mapping_by_layer_mapping')
    def test_get_mapping_string(self, mock_generate_api_mapping_by_layer_mapping):
        mock_generate_api_mapping_by_layer_mapping.return_value = {'Mint.where.0.forward': 'Torch.where.0.forward',
                                                                   'Functional.flash_attention_score.0.forward':
                                                                       'NPU.npu_fusion_attention.0.forward'}

        mc = MappingConfig(self.yaml_path, MappingInfo())
        result = mc.get_mapping_string('Mint.where.0.forward')
        self.assertEqual(result, 'Mint.where.0.forward')

        mc = MappingConfig(self.yaml_path, MappingInfo(mapping_type=DATA_MAPPING))
        result = mc.get_mapping_string('NPU.npu_fusion_attention.4.forward.input.0')
        self.assertEqual(result, 'Function.attention.4.forward.input.0')


if __name__ == '__main__':
    unittest.main()
