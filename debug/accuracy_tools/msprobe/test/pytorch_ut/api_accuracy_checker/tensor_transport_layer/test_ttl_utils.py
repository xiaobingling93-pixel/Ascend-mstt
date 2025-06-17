import unittest
from unittest.mock import Mock, patch

from OpenSSL import crypto, SSL

from msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.utils import verify_callback, is_certificate_revoked


class TestVerifyCallback(unittest.TestCase):
    """
    Test for verify_callback and is_certificate_revoked.
    """

    def setUp(self):
        self.conn = Mock(spec=SSL.Connection)
        self.cert = Mock(spec=crypto.X509)
        self.crl = [Mock()]
        self.crl[0].serial_number = 89981275109692867917699502952114227065605526936

    @patch('msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.utils.is_certificate_revoked')
    def test_preverify_ok(self, mock_is_certificate_revoked):
        mock_is_certificate_revoked.return_value = False
        self.assertTrue(verify_callback(self.conn, self.cert, 0, 0, 1, self.crl))

    @patch('msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.utils.is_certificate_revoked')
    def test_preverify_not_ok(self, mock_is_certificate_revoked):
        self.assertFalse(verify_callback(self.conn, self.cert, 0, 0, 0, None))

        mock_is_certificate_revoked.return_value = False
        self.assertEqual(verify_callback(self.conn, self.cert, 0, 0, 1, self.crl), 1)

    def test_is_certificate_revoked_true(self):
        self.cert.get_serial_number.return_value = 89981275109692867917699502952114227065605526936
        result = is_certificate_revoked(self.cert, self.crl)
        self.assertTrue(result)

    def test_is_certificate_revoked_false(self):
        self.cert.get_serial_number.return_value = 89981275109692867917699502952114227065605526937
        result = is_certificate_revoked(self.cert, self.crl)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
