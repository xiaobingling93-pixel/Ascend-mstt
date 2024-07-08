from unittest import TestCase
from unittest.mock import patch

from atat.core.utils import check_seed_all, Const, CompareException


class TestUtils(TestCase):
    @patch("atat.core.utils.print_error_log")
    def test_check_seed_all(self, mock_print_error_log):
        self.assertIsNone(check_seed_all(1234, True))
        self.assertIsNone(check_seed_all(0, True))
        self.assertIsNone(check_seed_all(Const.MAX_SEED_VALUE, True))

        with self.assertRaises(CompareException) as context:
            check_seed_all(-1, True)
        self.assertEqual(context.exception.code, CompareException.INVALID_PARAM_ERROR)
        mock_print_error_log.assert_called_with(f"Seed must be between 0 and {Const.MAX_SEED_VALUE}.")

        with self.assertRaises(CompareException) as context:
            check_seed_all(Const.MAX_SEED_VALUE + 1, True)
        self.assertEqual(context.exception.code, CompareException.INVALID_PARAM_ERROR)
        mock_print_error_log.assert_called_with(f"Seed must be between 0 and {Const.MAX_SEED_VALUE}.")

        with self.assertRaises(CompareException) as context:
            check_seed_all("1234", True)
        self.assertEqual(context.exception.code, CompareException.INVALID_PARAM_ERROR)
        mock_print_error_log.assert_called_with("Seed must be integer.")

        with self.assertRaises(CompareException) as context:
            check_seed_all(1234, 1)
        self.assertEqual(context.exception.code, CompareException.INVALID_PARAM_ERROR)
        mock_print_error_log.assert_called_with("seed_all mode must be bool.")
