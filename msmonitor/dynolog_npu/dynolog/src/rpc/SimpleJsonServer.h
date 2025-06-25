// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <gflags/gflags.h>
#include <openssl/ssl.h>
#include "dynolog/src/ServiceHandler.h"

DECLARE_string(certs_dir);

namespace dynolog {
// This is a simple service built using UNIX Sockets
// with remote procedure calls implemented via JSON string.
class SimpleJsonServerBase {
public:
    explicit SimpleJsonServerBase(int port);
    virtual ~SimpleJsonServerBase();

    int getPort() const
    {
        return port_;
    }

    bool initSuccessful() const
    {
        return initSuccess_;
    }
    // spin up a new thread to process requets
    void run();

    void stop()
    {
        run_ = 0;
        thread_->join();
    }

    // synchronously processes a request
    void processOne() noexcept;

protected:
    void initSocket();
    void init_openssl();
    SSL_CTX *create_context();
    void configure_context(SSL_CTX *ctx);

    // process requests in a loop
    void loop() noexcept;

    // implement processing of request using the handler
    virtual std::string processOneImpl(const std::string &request_str)
    {
        return "";
    }

    void verify_certificate_version_and_algorithm(X509 *cert);
    void verify_rsa_key_length(EVP_PKEY *pkey);
    void verify_certificate_validity(X509 *cert);
    void verify_certificate_extensions(X509 *cert);
    void load_private_key(SSL_CTX *ctx, const std::string &server_key);
    void load_and_verify_crl(SSL_CTX *ctx, const std::string &crl_file);

    int port_;
    int sock_fd_{-1};
    bool initSuccess_{false};

    std::atomic<bool> run_{true};
    std::unique_ptr<std::thread> thread_;

    SSL_CTX *ctx_{nullptr};
};

} // namespace dynolog