import unittest
from unittest import TestCase

from msprobe.pytorch.monitor.anomaly_detect import BaseWriterWithAD


class TestBaseWriterWithAD(TestCase):
    def test_update_tag2scalars(self):
        writer = BaseWriterWithAD('', None, None)
        writer._update_tag2scalars('tag1', 1.0)
        self.assertEqual(writer.tag2scalars['tag1']['avg'], 1.0)
        self.assertEqual(writer.tag2scalars['tag1']['count'], 1)
        writer._update_tag2scalars('tag1', 2.0)
        self.assertEqual(writer.tag2scalars['tag1']['avg'], 1.5)
        self.assertEqual(writer.tag2scalars['tag1']['count'], 2)


if __name__ == '__main__':
    unittest.main()
