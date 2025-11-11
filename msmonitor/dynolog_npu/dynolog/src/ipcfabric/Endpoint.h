// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <glog/logging.h>
#include "dynolog/src/utils.h"

namespace dynolog {
namespace ipcfabric {
// Define a type for fds to improve readibility.
using fd_t = int;
constexpr int SOCK_PATH_PERMISSION = 0600;

struct Payload {
    explicit Payload(size_t size, void *data) : size(size), data(data) {}
    size_t size{0};
    void *data{nullptr};
};

struct EndPointCtxt {
    explicit EndPointCtxt(size_t n) : iov{std::vector<struct iovec>(n)} {}
    struct sockaddr_un msg_name{};
    size_t msg_namelen{0};
    struct msghdr msghdr{};
    std::vector<struct iovec> iov;
};

class EndPoint final {
public:
    // Maximum defined in man unix, but minus 2 because abstract sockets, first and last are '\0'.
    constexpr static size_t kAbsSockFixLen = 2;
    constexpr static size_t kMaxNameLen = 108 - kAbsSockFixLen;

    explicit EndPoint(std::string address)
    {
        socket_fd_ = socket(AF_UNIX, SOCK_DGRAM, 0);
        if (socket_fd_ == -1) {
            throw std::runtime_error(std::strerror(errno));
        }
        if (!checkAndSetSocketPath_(address)) {
            throw std::runtime_error("Invalid socket path: " + address);
        }
        struct sockaddr_un addr;
        size_t addrlen = setAddress_(address, addr, false);
        if (addr.sun_path[0] != '\0') {
            // delete domain socket file just in case before binding
            if (PathUtils::Exist(addr.sun_path)) {
                if (!(PathUtils::IsFile(addr.sun_path) && PathUtils::IsOwner(addr.sun_path))) {
                    throw std::runtime_error(
                        std::string("Permission denied to delete existing socket file: ") + addr.sun_path);
                }
                if (unlink(addr.sun_path) != 0) {
                    throw std::runtime_error(
                        std::string("Failed to delete existing socket file: ") + std::strerror(errno));
                }
            }
        }

        int ret = bind(socket_fd_, (const struct sockaddr *)&addr, addrlen);
        if (ret == -1) {
            throw std::runtime_error(std::strerror(errno));
        }
        if (addr.sun_path[0] != '\0') {
            // set correct permissions for the socket. We avoid using umask because
            // of multithreaded nature. A short window exists between bind and chmod
            // where the permissions are wrong but it's ok for our use case.
            if (chmod(addr.sun_path, SOCK_PATH_PERMISSION) == -1) {
                throw std::runtime_error(
                        std::string("Failed to set socket path permission: ") + std::strerror(errno));
            }
        }
    }

    ~EndPoint()
    {
        if (socket_fd_ != -1) {
            close(socket_fd_);
        }
        if (!socket_path_.empty()) {
            unlink(socket_path_.c_str());
        }
    }

    [[nodiscard]] auto buildSendCtxt(const std::string &dest_name, const std::vector<Payload> &payload)
    {
        if (dest_name.empty()) {
            throw std::invalid_argument("Cannot send to empty socket name");
        }

        auto ctxt = buildCtxt_(payload);
        ctxt->msghdr.msg_namelen = setAddress_(dest_name, ctxt->msg_name, true);
        return ctxt;
    }

    // non-blocking. The error ECONNREFUSED may be caused by socket not being yet
    // initialized. See man unix.
    [[nodiscard]] bool trySendMsg(EndPointCtxt const &ctxt, bool retryOnConnRefused = true)
    {
        ssize_t ret = sendmsg(socket_fd_, &ctxt.msghdr, MSG_DONTWAIT);
        if (ret > 0) { // XXX: Enforce correct number of bytes sent.
            return true;
        }
        if (ret == -1 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            return false;
        }
        if (ret == -1 && retryOnConnRefused && errno == ECONNREFUSED) {
            return false;
        }
        throw std::runtime_error(std::strerror(errno));
    }

    [[nodiscard]] auto buildRcvCtxt(const std::vector<Payload> &payload)
    {
        return buildCtxt_(payload);
    }

    // If false, must retry. Only enabled for bound sockets.
    [[nodiscard]] bool tryRcvMsg(EndPointCtxt &ctxt) noexcept
    {
        ssize_t ret = recvmsg(socket_fd_, &ctxt.msghdr, MSG_DONTWAIT);

        if (ret > 0) { // XXX: Enforce correct number of bytes sent.
            return true;
        }
        if (ret == 0) {
            return false; // Receiver is down.
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return false;
        }

        throw std::runtime_error("tryRcvMsg() got " + std::string(std::strerror(errno)));
    }

    [[nodiscard]] bool tryPeekMsg(EndPointCtxt &ctxt) noexcept
    {
        ssize_t ret = recvmsg(socket_fd_, &ctxt.msghdr, MSG_DONTWAIT | MSG_PEEK);
        if (ret > 0) { // XXX: Enforce correct number of bytes sent.
            return true;
        }
        if (ret == 0) {
            return false; // Receiver is down.
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return false;
        }
        throw std::runtime_error("tryPeekMsg() got " + std::string(std::strerror(errno)));
    }

    const char *getName(EndPointCtxt const &ctxt, bool is_abstract) const
    {
        if (is_abstract) {
            if (ctxt.msg_name.sun_path[0] != '\0') {
                throw std::runtime_error(
                    "Expected abstract socket, got " +
                    std::string(ctxt.msg_name.sun_path));
            }
            return ctxt.msg_name.sun_path + 1;
        } else {
            auto home_path = GetCurrentUserHomePath();
            if (!home_path.empty()) {
                if (strncmp(home_path.c_str(), ctxt.msg_name.sun_path, home_path.size()) != 0 ||
                    ctxt.msg_name.sun_path[home_path.size()] != '/') {
                    throw std::runtime_error(
                        "Unexpected socket name: " + std::string(ctxt.msg_name.sun_path) +
                        ". Expected to start with " + std::string(home_path));
                }
                return ctxt.msg_name.sun_path + home_path.size() + 1;
            }
        }
        return nullptr;
    }

protected:
    fd_t socket_fd_;
    std::string socket_path_;

    // Initialize <dest> with address provided in <src>.
    size_t setAddress_(const std::string& src, struct sockaddr_un& dest, bool is_abstract)
    {
        dest.sun_family = AF_UNIX;
        if (is_abstract) {
            if (src.size() > kMaxNameLen) {
                throw std::invalid_argument(
                    "Abstract UNIX Socket path cannot be larger than kMaxNameLen");
            }
            dest.sun_path[0] = '\0';
            if (src.empty()) {
                return sizeof(sa_family_t); // autobind socket.
            }
            src.copy(dest.sun_path + 1, src.size());
            dest.sun_path[src.size() + 1] = '\0';
            return sizeof(sa_family_t) + src.size() + kAbsSockFixLen;
        } else {
            if (!socket_path_.empty()) {
                socket_path_.copy(dest.sun_path, socket_path_.size());
                dest.sun_path[socket_path_.size()] = '\0';
                return sizeof(sa_family_t) + socket_path_.size() + 1;
            }
        }
        return sizeof(sa_family_t);
    }

    std::unique_ptr<EndPointCtxt> buildCtxt_(const std::vector<Payload> &payload)
    {
        auto ctxt = std::make_unique<EndPointCtxt>(payload.size());
        for (size_t i = 0; i < payload.size(); i++) {
            ctxt->iov[i] = {payload[i].data, payload[i].size};
        }
        ctxt->msghdr.msg_name = &ctxt->msg_name;
        ctxt->msghdr.msg_namelen = sizeof(decltype(ctxt->msg_name));
        ctxt->msghdr.msg_iov = ctxt->iov.data();
        ctxt->msghdr.msg_iovlen = payload.size();
        ctxt->msghdr.msg_control = nullptr;
        ctxt->msghdr.msg_controllen = 0;
        return ctxt;
    }

    bool checkAndSetSocketPath_(const std::string& address)
    {
        if (address.empty()) {
            return false;
        }
        auto home_path = GetCurrentUserHomePath();
        if (home_path.empty() || !PathUtils::DirPathCheck(home_path)) {
            LOG(ERROR) << "Invalid home directory: " << home_path;
            return false;
        }
        auto socket_path = home_path + "/" + address + ".sock";
        if (socket_path.size() > kMaxNameLen) {
            LOG(ERROR) << "Socket path " << socket_path << " is too long";
            return false;
        }
        socket_path_ = socket_path;
        return true;
    }
};
} // namespace ipcfabric
} // namespace dynolog
