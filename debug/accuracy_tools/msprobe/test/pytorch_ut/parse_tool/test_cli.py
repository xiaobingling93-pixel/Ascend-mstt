import unittest
from unittest.mock import patch

from msprobe.pytorch.parse_tool.cli import parse

class TestCli(unittest.TestCase):
    @patch('msprobe.pytorch.parse_tool.cli._run_interactive_cli')
    def test_parse(self, mock_run_interactive_cli):
        parse()

        mock_run_interactive_cli.assert_called_once()
