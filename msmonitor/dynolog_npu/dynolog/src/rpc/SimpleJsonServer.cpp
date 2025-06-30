// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "dynolog/src/rpc/SimpleJsonServer.h"
#include <arpa/inet.h>
#include <fmt/format.h>
#include <glog/logging.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/x509.h>
#include <openssl/x509_vfy.h>
#include <openssl/x509v3.h>
#include <string>
#include <iostream>
#include <termios.h>
#include <algorithm>

DEFINE_string(certs_dir, "", "TLS crets dir");

constexpr int CLIENT_QUEUE_LEN = 50;
const std::string NO_CERTS_MODE = "NO_CERTS";
const size_t MIN_RSA_KEY_LENGTH = 3072;
constexpr char BACKSPACE_ASCII = 8;
constexpr char DEL_ASCII = 127;

namespace dynolog {

SimpleJsonServerBase::SimpleJsonServerBase(int port) : port_(port)
{
    try {
        initSocket();
        if (FLAGS_certs_dir != NO_CERTS_MODE) {
        init_openssl();
        ctx_ = create_context();
        configure_context(ctx_);
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to initialize server: " << e.what();
        if (sock_fd_ != -1) {
            close(sock_fd_);
            sock_fd_ = -1;
        }
        throw;
    }
}

SimpleJsonServerBase::~SimpleJsonServerBase()
{
    if (thread_) {
        stop();
    }
    close(sock_fd_);
    if (FLAGS_certs_dir != NO_CERTS_MODE && ctx_) {
        SSL_CTX_free(ctx_);
    }
}

void SimpleJsonServerBase::initSocket()
{
    struct sockaddr_in6 server_addr;

    /* Create socket for listening (client requests). */
    sock_fd_ = ::socket(AF_INET6, SOCK_STREAM, 0);
    if (sock_fd_ == -1) {
        std::perror("socket()");
        return;
    }

    /* Set socket to reuse address in case server is restarted. */
    int flag = 1;
    int ret =
        ::setsockopt(sock_fd_, SOL_SOCKET, SO_REUSEADDR, &flag, sizeof(flag));
    if (ret == -1) {
        std::perror("setsockopt()");
        return;
    }

    // in6addr_any allows us to bind to both IPv4 and IPv6 clients.
    server_addr.sin6_addr = in6addr_any;
    server_addr.sin6_family = AF_INET6;
    server_addr.sin6_port = htons(port_);

    /* Bind address and socket together */
    ret = ::bind(sock_fd_, (struct sockaddr*)&server_addr, sizeof(server_addr));
    if (ret == -1) {
        std::perror("bind()");
        close(sock_fd_);
        return;
    }

    /* Create listening queue (client requests) */
    ret = ::listen(sock_fd_, CLIENT_QUEUE_LEN);
    if (ret == -1) {
        std::perror("listen()");
        close(sock_fd_);
        return;
    }

    /* Get port if assigned 0 */
    if (port_ == 0) {
        socklen_t len_out = sizeof(server_addr);
        ret = ::getsockname(sock_fd_, (struct sockaddr*)&server_addr, &len_out);
        if (ret < 0 || len_out != sizeof(server_addr)) {
            std::perror("getsockname()");
        } else {
        port_ = ntohs(server_addr.sin6_port);
        LOG(INFO) << "System assigned port = " << ntohs(server_addr.sin6_port);
        }
    }

    LOG(INFO) << "Listening to connections on port " << port_;
    initSuccess_ = true;
}

/* A simple wrapper to accept connections and read data
 *
 *  Messages are prefixed using the length so we know how long a message
 *  to actually read.
 *     : int32_t len
 *     : char json[]
 */
class ClientSocketWrapper {
public:
    ~ClientSocketWrapper()
    {
        if (FLAGS_certs_dir != NO_CERTS_MODE && ssl_) {
        SSL_shutdown(ssl_);
        SSL_free(ssl_);
        }
        if (client_sock_fd_ != -1) {
            ::close(client_sock_fd_);
        }
    }

    bool accept(int server_socket, SSL_CTX* ctx)
    {
        struct sockaddr_in6 client_addr;
        socklen_t client_addr_len = sizeof(client_addr);
        std::array<char, INET6_ADDRSTRLEN> client_addr_str;

        client_sock_fd_ = ::accept(
            server_socket, (struct sockaddr*)&client_addr, &client_addr_len);
        if (client_sock_fd_ == -1) {
            std::perror("accept()");
            return false;
        }

        inet_ntop(
            AF_INET6,
            &(client_addr.sin6_addr),
            client_addr_str.data(),
            client_addr_str.size());
        LOG(INFO) << "Received connection from " << client_addr_str.data();

        if (FLAGS_certs_dir == NO_CERTS_MODE) {
            LOG(INFO) << "No certs mode";
            return true;
        }

        ssl_ = SSL_new(ctx);
        SSL_set_fd(ssl_, client_sock_fd_);
        if (SSL_accept(ssl_) <= 0) {
            ERR_print_errors_fp(stderr);
            return false;
        }
        LOG(INFO) << "SSL handshake success";
        return true;
    }

    std::string get_message()
    {
        int32_t msg_size = -1;
        if (!read_helper((uint8_t*)&msg_size, sizeof(msg_size)) || msg_size <= 0) {
            LOG(ERROR) << "Invalid message size = " << msg_size;
            return "";
        }

        std::string message;
        message.resize(msg_size);
        int recv = 0;
        int ret = 1;
        while (recv < msg_size && ret > 0) {
            ret = read_helper((uint8_t*)&message[recv], msg_size - recv);
            recv += ret > 0 ? ret : 0;
        }

        if (recv != msg_size) {
            LOG(ERROR) << "Received partial message, expected size " << msg_size
                << " found : " << recv;
            LOG(ERROR) << "Message received = " << message;
            return "";
        }

        return message;
    }

    bool send_response(const std::string& response)
    {
        int32_t size = response.size();
        int ret;
        if (FLAGS_certs_dir == NO_CERTS_MODE) {
            ret = ::write(client_sock_fd_, (void*)&size, sizeof(size));
            if (ret == -1) {
                std::perror("write()");
                return false;
            }
        } else {
            ret = SSL_write(ssl_, (void*)&size, sizeof(size));
            if (ret <= 0) {
                ERR_print_errors_fp(stderr);
                return false;
            }
        }
        int sent = 0;
        while (sent < size && ret > 0) {
            if (FLAGS_certs_dir == NO_CERTS_MODE) {
                ret = ::write(client_sock_fd_, (void*)&response[sent], size - sent);
                if (ret == -1) {
                    std::perror("write()");
                } else {
                    sent += ret;
                }
            } else {
                ret = SSL_write(ssl_, (void*)&response[sent], size - sent);
                if (ret <= 0) {
                    ERR_print_errors_fp(stderr);
                } else {
                    sent += ret;
                }
            }
        }

        if (sent < response.size()) {
            LOG(ERROR) << "Unable to write full response";
            return false;
        }
        return ret > 0;
    }

private:
    int read_helper(uint8_t* buf, int size)
    {
        if (FLAGS_certs_dir == NO_CERTS_MODE) {
            int ret = ::read(client_sock_fd_, (void*)buf, size);
            if (ret == -1) {
                std::perror("read()");
            }
            return ret;
        }
        int ret = SSL_read(ssl_, (void*)buf, size);
        if (ret <= 0) {
            ERR_print_errors_fp(stderr);
        }
        return ret;
    }
        int client_sock_fd_ = -1;
        SSL* ssl_ = nullptr;
};

/* Accepts socket connections and processes the payloads.
 * This will inturn call the Handler functions */
void SimpleJsonServerBase::loop() noexcept
{
    if (sock_fd_ == -1 || !initSuccess_) {
        return;
    }

    while (run_) {
        processOne();
    }
}

void SimpleJsonServerBase::processOne() noexcept
{
    LOG(INFO) << "Waiting for connection.";
    ClientSocketWrapper client;
    if (!client.accept(sock_fd_, ctx_)) {
        return;
    }
    std::string request_str = client.get_message();
    LOG(INFO) << "RPC message received = " << request_str;
    auto response_str = processOneImpl(request_str);
    if (response_str.empty()) {
        return;
    }
    if (!client.send_response(response_str)) {
        LOG(ERROR) << "Failed to send response";
    }
}

void SimpleJsonServerBase::run()
{
    LOG(INFO) << "Launching RPC thread";
    thread_ = std::make_unique<std::thread>([this]() { this->loop(); });
}

void SimpleJsonServerBase::init_openssl()
{
    SSL_load_error_strings();
    OpenSSL_add_ssl_algorithms();
}

SSL_CTX* SimpleJsonServerBase::create_context()
{
    const SSL_METHOD* method = TLS_server_method();
    SSL_CTX* ctx = SSL_CTX_new(method);
    if (!ctx) {
        ERR_print_errors_fp(stderr);
        throw std::runtime_error("Unable to create SSL context");
    }
    return ctx;
}

static bool is_cert_revoked(X509* cert, X509_STORE* store)
{
    if (!cert || !store) {
        LOG(ERROR) << "Invalid certificate or store pointer";
        return false;
    }
    // 获取证书的颁发者名称
    X509_NAME* issuer = X509_get_issuer_name(cert);
    if (!issuer) {
        LOG(ERROR) << "Failed to get certificate issuer";
        return false;
    }
    // 获取证书的序列号
    const ASN1_INTEGER* serial = X509_get_serialNumber(cert);
    if (!serial) {
        LOG(ERROR) << "Failed to get certificate serial number";
        return false;
    }
    // 创建证书验证上下文
    X509_STORE_CTX* ctx = X509_STORE_CTX_new();
    if (!ctx) {
        LOG(ERROR) << "Failed to create certificate store context";
        return false;
    }
    bool is_revoked = false;
    try {
        // 初始化证书验证上下文
        if (!X509_STORE_CTX_init(ctx, store, cert, nullptr)) {
            LOG(ERROR) << "Failed to initialize certificate store context";
            X509_STORE_CTX_free(ctx);
            return false;
        }
        // 获取CRL列表
        STACK_OF(X509_CRL)* crls = X509_STORE_CTX_get1_crls(ctx, issuer);
        if (!crls) {
            LOG(INFO) << "No CRLs found for issuer";
            X509_STORE_CTX_free(ctx);
            return false;
        }
        time_t current_time = time(nullptr);
        for (int i = 0; i < sk_X509_CRL_num(crls); i++) {
            X509_CRL* crl = sk_X509_CRL_value(crls, i);
            if (!crl) {
                LOG(ERROR) << "Invalid CRL at index " << i;
                continue;
            }
            // 检查 CRL 的有效期
            const ASN1_TIME* crl_this_update = X509_CRL_get0_lastUpdate(crl);
            const ASN1_TIME* crl_next_update = X509_CRL_get0_nextUpdate(crl);
            if (!crl_this_update) {
                LOG(ERROR) << "Failed to get CRL this_update time";
                continue;
            }
            // 检查 CRL 是否已生效
            if (X509_cmp_time(crl_this_update, &current_time) > 0) {
                LOG(INFO) << "CRL is not yet valid";
                continue;
            }
            // 检查 CRL 是否过期
            if (crl_next_update) {
                if (X509_cmp_time(crl_next_update, &current_time) < 0) {
                    LOG(INFO) << "CRL has expired";
                    continue;
                }
            }
            // 检查证书是否在 CRL 中
            STACK_OF(X509_REVOKED)* revoked = X509_CRL_get_REVOKED(crl);
            if (revoked) {
                for (int j = 0; j < sk_X509_REVOKED_num(revoked); j++) {
                    X509_REVOKED* rev = sk_X509_REVOKED_value(revoked, j);
                    if (rev) {
                        const ASN1_INTEGER* rev_serial = X509_REVOKED_get0_serialNumber(rev);
                        if (rev_serial && ASN1_INTEGER_cmp(serial, rev_serial) == 0) {
                            LOG(INFO) << "Certificate is found in CRL";
                            is_revoked = true;
                            break;
                        }
                    }
                }
            }
            if (is_revoked) {
                break;
            }
        }
        if (crls) {
            sk_X509_CRL_pop_free(crls, X509_CRL_free);
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "Exception while checking CRL: " << e.what();
        is_revoked = false;
    }
    X509_STORE_CTX_free(ctx);
    return is_revoked;
}

// 禁用终端回显的函数，但显示星号
int get_password_with_stars(char* buf, size_t bufsize)
{
    struct termios old_flags;
    struct termios new_flags;
    size_t idx = 0;

    tcgetattr(fileno(stdin), &old_flags);
    new_flags = old_flags;
    new_flags.c_lflag &= ~ECHO;
    tcsetattr(fileno(stdin), TCSANOW, &new_flags);

    char ch;
    while ((ch = getchar()) != '\n' && idx + 1 < bufsize) {
        if (ch == DEL_ASCII || ch == BACKSPACE_ASCII) {
            if (idx > 0) {
                idx--;
                printf("\b \b");
                fflush(stdout);
            }
        } else {
            buf[idx++] = ch;
            printf("*");
            fflush(stdout);
        }
    }
    buf[idx] = '\0';
    tcsetattr(fileno(stdin), TCSANOW, &old_flags);
    return idx;
}

// 验证证书版本和签名算法
void SimpleJsonServerBase::verify_certificate_version_and_algorithm(X509* cert)
{
    // 1. 检查证书版本是否为 X.509v3
    if (X509_get_version(cert) != 2) {  // 2 表示 X.509v3
        throw std::runtime_error("Certificate is not X.509v3");
    }

    // 2. 检查证书签名算法
    const X509_ALGOR* sig_alg = X509_get0_tbs_sigalg(cert);
    if (!sig_alg) {
        throw std::runtime_error("Failed to get signature algorithm");
    }

    int sig_nid = OBJ_obj2nid(sig_alg->algorithm);
    // 检查是否使用不安全的算法
    if (sig_nid == NID_md2WithRSAEncryption ||
        sig_nid == NID_md5WithRSAEncryption ||
        sig_nid == NID_sha1WithRSAEncryption) {
        throw std::runtime_error("Certificate uses insecure signature algorithm: " + std::to_string(sig_nid));
    }
}

// 验证 RSA 密钥长度
void SimpleJsonServerBase::verify_rsa_key_length(EVP_PKEY* pkey)
{
    if (EVP_PKEY_base_id(pkey) == EVP_PKEY_RSA) {
        size_t key_length = 0;
#if OPENSSL_VERSION_NUMBER >= 0x30000000L
        // OpenSSL 3.0 及以上版本
        key_length = EVP_PKEY_get_size(pkey) * 8;  // 转换为位数
#else
        // OpenSSL 1.1.1 及以下版本
        RSA* rsa = EVP_PKEY_get0_RSA(pkey);
        if (!rsa) {
            throw std::runtime_error("Failed to get RSA key");
        }
        
        const BIGNUM* n = nullptr;
        RSA_get0_key(rsa, &n, nullptr, nullptr);
        if (!n) {
            throw std::runtime_error("Failed to get RSA modulus");
        }
        
        key_length = BN_num_bits(n);
#endif
        if (key_length < MIN_RSA_KEY_LENGTH) {
            throw std::runtime_error("RSA key length " + std::to_string(key_length) + " bits is less than required " + std::to_string(MIN_RSA_KEY_LENGTH) + " bits");
        }
    }
}

// 验证证书有效期
void SimpleJsonServerBase::verify_certificate_validity(X509* cert)
{
    ASN1_TIME* not_before = X509_get_notBefore(cert);
    ASN1_TIME* not_after = X509_get_notAfter(cert);
    if (!not_before || !not_after) {
        throw std::runtime_error("Failed to get certificate validity period");
    }

    time_t current_time = time(nullptr);
    struct tm tm_before = {};
    struct tm tm_after = {};
    if (!ASN1_TIME_to_tm(not_before, &tm_before) ||
        !ASN1_TIME_to_tm(not_after, &tm_after)) {
        throw std::runtime_error("Failed to convert certificate dates");
    }
    
    time_t not_before_time = mktime(&tm_before);
    time_t not_after_time = mktime(&tm_after);
    
    // 检查证书是否已生效
    if (current_time < not_before_time) {
        BIO* bio = BIO_new(BIO_s_mem());
        if (bio) {
            ASN1_TIME_print(bio, not_before);
            char* not_before_str = nullptr;
            long len = BIO_get_mem_data(bio, &not_before_str);
            if (len > 0) {
                std::string time_str(not_before_str, len);
                BIO_free(bio);
                throw std::runtime_error("Server certificate is not yet valid. Valid from: " + time_str);
            }
            BIO_free(bio);
        }
        throw std::runtime_error("Server certificate is not yet valid");
    }

    // 检查证书是否已过期
    if (current_time > not_after_time) {
        BIO* bio = BIO_new(BIO_s_mem());
        if (bio) {
            ASN1_TIME_print(bio, not_after);
            char* not_after_str = nullptr;
            long len = BIO_get_mem_data(bio, &not_after_str);
            if (len > 0) {
                std::string time_str(not_after_str, len);
                BIO_free(bio);
                throw std::runtime_error("Server certificate has expired. Expired at: " + time_str);
            }
            BIO_free(bio);
        }
        throw std::runtime_error("Server certificate has expired");
    }
}

// 验证证书扩展域
void SimpleJsonServerBase::verify_certificate_extensions(X509* cert)
{
    bool has_ca_constraint = false;
    bool has_key_usage = false;
    bool has_cert_sign = false;
    bool has_crl_sign = false;

    const STACK_OF(X509_EXTENSION)* exts = X509_get0_extensions(cert);
    if (exts) {
        for (int i = 0; i < sk_X509_EXTENSION_num(exts); i++) {
            X509_EXTENSION* ext = sk_X509_EXTENSION_value(exts, i);
            ASN1_OBJECT* obj = X509_EXTENSION_get_object(ext);
            
            if (OBJ_obj2nid(obj) == NID_basic_constraints) {
                BASIC_CONSTRAINTS* constraints = (BASIC_CONSTRAINTS*)X509V3_EXT_d2i(ext);
                if (constraints) {
                    has_ca_constraint = constraints->ca;
                    BASIC_CONSTRAINTS_free(constraints);
                }
            } else if (OBJ_obj2nid(obj) == NID_key_usage) {
                ASN1_BIT_STRING* usage = (ASN1_BIT_STRING*)X509V3_EXT_d2i(ext);
                if (usage) {
                    has_key_usage = true;
                    has_cert_sign = (usage->data[0] & KU_KEY_CERT_SIGN) != 0;
                    has_crl_sign = (usage->data[0] & KU_CRL_SIGN) != 0;
                    ASN1_BIT_STRING_free(usage);
                }
            }
        }
    }

    if (has_ca_constraint) {
        throw std::runtime_error("Client certificate should not have CA constraint");
    }
    if (!has_key_usage) {
        throw std::runtime_error("Client certificate must have key usage extension");
    }
}

// 加载私钥
void SimpleJsonServerBase::load_private_key(SSL_CTX* ctx, const std::string& server_key)
{
    FILE* key_file = fopen(server_key.c_str(), "r");
    if (!key_file) {
        throw std::runtime_error("Failed to open server key file");
    }

    bool is_encrypted = false;
    char line[256];
    while (fgets(line, sizeof(line), key_file)) {
        if (strstr(line, "ENCRYPTED")) {
            is_encrypted = true;
            break;
        }
    }
    rewind(key_file);

    if (is_encrypted) {
        char password[256] = {0};
        std::cout << "Please enter the certificate password: ";
        get_password_with_stars(password, sizeof(password));
        std::cout << std::endl;

        EVP_PKEY* pkey = PEM_read_PrivateKey(
            key_file,
            nullptr,
            [](char* buf, int size, int rwflag, void* userdata) -> int {
                const char* password = static_cast<const char*>(userdata);
                int pwlen = strlen(password);
                if (pwlen > size) return 0;
                std::copy(password, password + pwlen, buf);
                return pwlen;
            },
            password);

        fclose(key_file);
        // 直接清空 char[] 密码
        std::fill(std::begin(password), std::end(password), 0);

        if (!pkey) {
            ERR_print_errors_fp(stderr);
            throw std::runtime_error("Failed to load encrypted server private key");
        }

        if (SSL_CTX_use_PrivateKey(ctx, pkey) <= 0) {
            EVP_PKEY_free(pkey);
            ERR_print_errors_fp(stderr);
            throw std::runtime_error("Failed to use server private key");
        }

        EVP_PKEY_free(pkey);
    } else {
        fclose(key_file);
        if (SSL_CTX_use_PrivateKey_file(ctx, server_key.c_str(), SSL_FILETYPE_PEM) <= 0) {
            ERR_print_errors_fp(stderr);
            throw std::runtime_error("Failed to load server private key");
        }
    }
}

// 加载和验证 CRL
void SimpleJsonServerBase::load_and_verify_crl(SSL_CTX* ctx, const std::string& crl_file)
{
    X509_STORE* store = SSL_CTX_get_cert_store(ctx);
    if (!store) {
        throw std::runtime_error("Failed to get certificate store");
    }

    if (access(crl_file.c_str(), F_OK) != -1) {
        FILE* crl_file_ptr = fopen(crl_file.c_str(), "r");
        if (!crl_file_ptr) {
            LOG(WARNING) << "Failed to open CRL file: " << crl_file;
            return;
        }

        X509_CRL* crl = PEM_read_X509_CRL(crl_file_ptr, nullptr, nullptr, nullptr);
        fclose(crl_file_ptr);

        if (!crl) {
            LOG(WARNING) << "Failed to read CRL from file: " << crl_file;
            return;
        }

        if (X509_STORE_add_crl(store, crl) != 1) {
            LOG(WARNING) << "Failed to add CRL to certificate store";
            X509_CRL_free(crl);
            return;
        }

        X509* cert = SSL_CTX_get0_certificate(ctx);
        if (!cert) {
            X509_CRL_free(crl);
            throw std::runtime_error("Failed to get server certificate");
        }

        if (is_cert_revoked(cert, store)) {
            X509_CRL_free(crl);
            throw std::runtime_error("Server certificate is revoked");
        }

        X509_CRL_free(crl);
    }
}

void SimpleJsonServerBase::configure_context(SSL_CTX* ctx)
{
    if (FLAGS_certs_dir.empty()) {
        throw std::runtime_error("--certs-dir must be specified!");
    }

    std::string certs_dir = FLAGS_certs_dir;
    if (!certs_dir.empty() && certs_dir.back() != '/')
        certs_dir += '/';

    std::string server_cert = certs_dir + "server.crt";
    std::string server_key  = certs_dir + "server.key";
    std::string ca_cert     = certs_dir + "ca.crt";
    std::string crl_file    = certs_dir + "ca.crl";

    LOG(INFO) << "Loading server cert: " << server_cert;
    LOG(INFO) << "Loading server key: " << server_key;
    LOG(INFO) << "Loading CA cert: " << ca_cert;

    // 1. 加载并验证服务器证书
    if (SSL_CTX_use_certificate_file(ctx, server_cert.c_str(), SSL_FILETYPE_PEM) <= 0) {
        ERR_print_errors_fp(stderr);
        throw std::runtime_error("Failed to load server certificate");
    }

    X509* cert = SSL_CTX_get0_certificate(ctx);
    if (!cert) {
        throw std::runtime_error("Failed to get server certificate");
    }

    // 2. 验证证书版本和签名算法
    verify_certificate_version_and_algorithm(cert);

    // 3. 验证 RSA 密钥长度
    EVP_PKEY* pkey = X509_get_pubkey(cert);
    if (!pkey) {
        throw std::runtime_error("Failed to get public key");
    }
    verify_rsa_key_length(pkey);
    EVP_PKEY_free(pkey);

    // 4. 验证证书有效期
    verify_certificate_validity(cert);

    // 5. 验证证书扩展域
    verify_certificate_extensions(cert);

    // 6. 加载私钥
    load_private_key(ctx, server_key);

    // 7. 加载 CA 证书
    if (SSL_CTX_load_verify_locations(ctx, ca_cert.c_str(), NULL) <= 0) {
        ERR_print_errors_fp(stderr);
        throw std::runtime_error("Failed to load CA certificate");
    }

    // 8. 加载和验证 CRL
    load_and_verify_crl(ctx, crl_file);

    // 9. 设置证书验证选项
    SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, NULL);
}

} // namespace dynolog