# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import os
from datetime import datetime, timezone

from OpenSSL import crypto
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from dateutil import parser

from msprobe.core.common.file_utils import FileOpen
from msprobe.core.common.log import logger

cipher_list = ":".join(
    ["TLS_DHE_RSA_WITH_AES_128_GCM_SHA256",
     "TLS_DHE_RSA_WITH_AES_256_GCM_SHA384",
     "TLS_DHE_DSS_WITH_AES_128_GCM_SHA256",
     "TLS_DHE_DSS_WITH_AES_256_GCM_SHA384",
     "TLS_DHE_PSK_WITH_AES_128_GCM_SHA256",
     "TLS_DHE_PSK_WITH_AES_256_GCM_SHA384",
     "TLS_DHE_PSK_WITH_CHACHA20_POLY1305_SHA256",
     "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256",
     "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384",
     "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
     "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
     "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256",
     "TLS_ECDHE_PSK_WITH_CHACHA20_POLY1305_SHA256",
     "TLS_ECDHE_PSK_WITH_AES_128_GCM_SHA256",
     "TLS_ECDHE_PSK_WITH_AES_256_GCM_SHA384",
     "TLS_ECDHE_PSK_WITH_AES_128_CCM_SHA256",
     "TLS_DHE_RSA_WITH_AES_128_CCM",
     "TLS_DHE_RSA_WITH_AES_256_CCM",
     "TLS_DHE_RSA_WITH_CHACHA20_POLY1305_SHA256",
     "TLS_DHE_PSK_WITH_AES_128_CCM",
     "TLS_DHE_PSK_WITH_AES_256_CCM",
     "TLS_ECDHE_ECDSA_WITH_AES_128_CCM",
     "TLS_ECDHE_ECDSA_WITH_AES_256_CCM",
     "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256"]
).encode()

STRUCT_UNPACK_MODE = "!Q"
STR_TO_BYTES_ORDER = "big"


def is_certificate_revoked(cert, crl):
    # 获取证书的序列号
    cert_serial_number = cert.get_serial_number()

    # 检查证书是否在CRL中
    revoked_serials = [revoked_cert.serial_number for revoked_cert in crl]
    if cert_serial_number in revoked_serials:
        logger.error(f"证书已吊销:{cert_serial_number:020x}")
        return True

    return False


def verify_callback(conn, cert, errno, depth, preverify_ok, crl=None):
    """
    验证对端证书的有效性
    :param conn: OpenSSL.SSL.Connection, SSL 连接对象
    :param cert: OpenSSL.crypto.X509, 当前证书
    :param errno: int, OpenSSL错误代码, 0:无错误 | 9:证书过期 | 18: 自签名证书
    :param depth: int, 当前证书在证书链中的深度 (0=叶子节点), 1:中间CA证书 -1:根CA证书 2+:更高级别CA证书
    :param preverify_ok: int, 验证结果 (1=通过， 0=失败)
    :param crl: _CRLInternal, CRL证书对象
    :return: bool, True表示接受证书, False表示拒绝
    """

    if not preverify_ok:
        from OpenSSL import SSL
        error_str = SSL._ffi.string(SSL._lib.X509_verify_cert_error_string(errno)).decode()
        logger.error(f"证书验证失败 (depth={depth}, err={errno}): {error_str}")
        return False

    if crl and is_certificate_revoked(cert, crl):
        return False

    return preverify_ok


def load_ssl_pem(key_file, cert_file, ca_file, crl_file):
    """
    Load SSL PEM files.

    Args:
        key_file (str): The path to the private key file.
        cert_file (str): The path to the certificate file.
        ca_file (str): The path to the CA certificate file.
        crl_file (str): The path to the CRL file.

    Returns:
        tuple: (key, crt, ca_crt, crl)

    Raises:
        Exception: If the file paths are invalid or the file contents are incorrect, exceptions may be thrown.
    """

    try:
        # your_private_key_password
        import pwinput
        passphrase = pwinput.pwinput("Enter your password: ")
        with FileOpen(key_file, "rb") as f:
            key = crypto.load_privatekey(crypto.FILETYPE_PEM, f.read(), passphrase.encode())
            del passphrase
            gc.collect()
        with FileOpen(cert_file, "rb") as f:
            crt = crypto.load_certificate(crypto.FILETYPE_PEM, f.read())
            check_crt_valid(crt)

            crt_serial_number = hex(crt.get_serial_number())[2:]
            logger.info(f"crt_serial_number: {crt_serial_number}")

        check_certificate_match(crt, key)

        with FileOpen(ca_file, "rb") as f:
            ca_crt = crypto.load_certificate(crypto.FILETYPE_PEM, f.read())
            check_crt_valid(ca_crt)

            ca_serial_number = hex(ca_crt.get_serial_number())[2:]
            logger.info(f"ca_serial_number: {ca_serial_number}")
        crl = None
        if os.path.exists(crl_file):
            with FileOpen(crl_file, "rb") as f:
                crl = x509.load_pem_x509_crl(f.read(), default_backend())
                check_crl_valid(crl, ca_crt)
            for revoked_cert in crl:
                logger.info(f"Serial Number: {revoked_cert.serial_number}, "
                            f"Revocation Date: {revoked_cert.revocation_date_utc}")

    except Exception as e:
        raise RuntimeError(f"The SSL certificate is invalid") from e

    return key, crt, ca_crt, crl


def check_crt_valid(pem):
    """
    Check the validity of the SSL certificate.

    Raises:
    RuntimeError: If the SSL certificate is invalid or expired.
    """
    try:
        pem_start = parser.parse(pem.get_notBefore().decode("UTF-8"))
        pem_end = parser.parse(pem.get_notAfter().decode("UTF-8"))
        logger.info(f"The SSL certificate passes the verification and the validity period "
                    f"starts from {pem_start} ends at {pem_end}.")
    except Exception as e:
        raise RuntimeError(f"The SSL certificate is invalid") from e

    now_utc = datetime.now(tz=timezone.utc)
    if pem.has_expired() or not (pem_start <= now_utc <= pem_end):
        raise RuntimeError(f"The SSL certificate has expired.")


def check_certificate_match(certificate, private_key):
    """
    Check certificate and private_key is match or not. if mismatched, an exception is thrown.
    :param certificate:
    :param private_key:
    :return:
    """
    test_data = os.urandom(256)
    try:
        signature = crypto.sign(private_key, test_data, "sha256")
        crypto.verify(
            certificate,  # 包含公钥的证书
            signature,  # 生成的签名
            test_data,  # 原始数据
            "sha256",  # 哈希算法
        )
        logger.info("公钥和私钥匹配")
    except Exception as e:
        raise RuntimeError("公钥和私钥不匹配") from e


def check_crl_valid(crl, ca_crt):
    # 验证CRL签名（确保CRL未被篡改）
    if not crl.is_signature_valid(ca_crt.get_pubkey().to_cryptography_key()):
        raise RuntimeError("CRL签名无效！")

    # 检查CRL有效期
    if not (crl.last_update <= datetime.utcnow() <= crl.next_update):
        raise RuntimeError("CRL已过期或尚未生效!")
