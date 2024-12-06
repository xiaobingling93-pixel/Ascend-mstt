# coding=utf-8
import argparse
import unittest
from msprobe.core.compare.utils import _compare_parser


class TestUtilsMethods(unittest.TestCase):
    def test_compare_parser(self):
        parser = argparse.ArgumentParser()
        _compare_parser(parser)
        self.assertEqual(parser.parse_args('-i aaa -o'.split(' ')).output_path, './output')
        self.assertEqual(parser.parse_args('-i aaa'.split(' ')).output_path, './output')
        self.assertEqual(parser.parse_args('-i aaa -o ./aaa/output'.split(' ')).output_path, './aaa/output')