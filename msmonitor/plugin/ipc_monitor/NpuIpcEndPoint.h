/*
 * Copyright (C) 2025-2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef NPU_IPC_ENDPOINT_H
#define NPU_IPC_ENDPOINT_H

#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>
#include <stdexcept>
#include "utils.h"
#include "securec.h"

namespace dynolog_npu {
namespace ipc_monitor {

using fileDesT = int;
constexpr const char STR_END_CHAR = '\0';
constexpr int SOCKET_FD_CHMOD = 0666;

struct NpuPayLoad {
    size_t size;
    void *data;
    NpuPayLoad(size_t size, void *data) : size(size), data(data) {}
};

template <size_t MaxNumFileDes = 0> struct NpuIpcEndPointCtxt {
    struct sockaddr_un messageName;
    size_t messageLen;
    fileDesT *fileDesPtr;
    struct msghdr msghdr;
    std::vector<struct iovec> iov;
    char ancillaryBuf[CMSG_SPACE(MaxNumFileDes * sizeof(fileDesT))];
    explicit NpuIpcEndPointCtxt(size_t num) : iov(std::vector<struct iovec>(num)){};
};

template <size_t MaxNumFileDes = 0> class NpuIpcEndPoint final {
    using Ctxt = NpuIpcEndPointCtxt<MaxNumFileDes>;

public:
    constexpr static size_t addressMaxLen = 108 - 2; // Max unix socket path length
    explicit NpuIpcEndPoint(const std::string &addressName)
    {
        socketFd = socket(AF_UNIX, SOCK_DGRAM, 0);
        if (socketFd == -1) {
            throw std::runtime_error(std::strerror(errno) + IPC_ERROR(ErrCode::PARAM));
        }
        int ret = 0;
        struct sockaddr_un address;
        size_t addressLen = SetSocketAdress(addressName, address);
        if (address.sun_path[0] != STR_END_CHAR) {
            ret = unlink(address.sun_path);
        }
        if (ret == -1) {
            throw std::runtime_error("Unlink failed, error is " + std::string(strerror(errno)) + IPC_ERROR(ErrCode::PARAM));
        }

        ret = bind(socketFd, ReinterpretConvert<const struct sockaddr *>(&address), addressLen);
        if (ret == -1) {
            throw std::runtime_error("Bind socket failed." + IPC_ERROR(ErrCode::PARAM));
        }

        if (address.sun_path[0] != STR_END_CHAR) {
            ret = chmod(address.sun_path, SOCKET_FD_CHMOD);
        }
        if (ret == -1) {
            throw std::runtime_error("Chmod failed, error is " + std::string(strerror(errno)) + IPC_ERROR(ErrCode::PARAM));
        }
    }

    ~NpuIpcEndPoint()
    {
        close(socketFd);
    }

    [[nodiscard]] auto BuildSendNpuCtxt(const std::string &desAddrName, const std::vector<NpuPayLoad> &npuPayLoad,
        const std::vector<fileDesT> &fileDes)
    {
        if (fileDes.size() > MaxNumFileDes) {
            throw std::runtime_error("Request to fill more than max connections " + IPC_ERROR(ErrCode::PARAM));
        }
        if (desAddrName.empty()) {
            throw std::runtime_error("Can not send to dest point, because dest socket name is empty " +
                IPC_ERROR(ErrCode::PARAM));
        }
        auto ctxt = BuildNpuCtxt_(npuPayLoad, fileDes.size());
        ctxt->msghdr.msg_namelen = SetSocketAdress(desAddrName, ctxt->messageName);
        if (!fileDes.empty()) {
            if (fileDes.size() * sizeof(fileDesT) > sizeof(ctxt->fileDesPtr)) {
                throw std::runtime_error("Memcpy failed when fileDes size large than ctxt fileDesPtr " +
                    IPC_ERROR(ErrCode::PARAM));
            }
            if (memcpy_s(ctxt->fileDesPtr, sizeof(ctxt->fileDesPtr),
                         fileDes.data(), fileDes.size() * sizeof(fileDesT)) != EOK) {
                throw std::runtime_error("Memcpy failed when fileDes size large than ctxt fileDesPtr " +
                    IPC_ERROR(ErrCode::MEMORY));
            }
        }
        return ctxt;
    }

    [[nodiscard]] bool TrySendMessage(Ctxt const & ctxt, bool retryOnConnRefused = true)
    {
        ssize_t retCode = sendmsg(socketFd, &ctxt.msghdr, MSG_DONTWAIT);
        if (retCode > 0) {
            return true;
        }
        if ((errno == EAGAIN || errno == EWOULDBLOCK) && retCode == -1) {
            return false;
        }
        if (retryOnConnRefused && errno == ECONNREFUSED && retCode == -1) {
            return false;
        }
        throw std::runtime_error("TrySendMessage occur " + std::string(std::strerror(errno)) + " " +
            IPC_ERROR(ErrCode::PARAM));
    }

    [[nodiscard]] auto BuildNpuRcvCtxt(const std::vector<NpuPayLoad> &npuPayLoad)
    {
        return BuildNpuCtxt_(npuPayLoad, MaxNumFileDes);
    }

    [[nodiscard]] bool TryRcvMessage(Ctxt &ctxt) noexcept
    {
        auto retCode = recvmsg(socketFd, &ctxt.msghdr, MSG_DONTWAIT);
        if (retCode > 0) {
            return true;
        }
        if (retCode == 0) {
            return false;
        }
        if (errno == EWOULDBLOCK || errno == EAGAIN) {
            return false;
        }
        throw std::runtime_error("TryRcvMessage occur " + std::string(std::strerror(errno)) + " " +
            IPC_ERROR(ErrCode::PARAM));
    }

    [[nodiscard]] bool TryPeekMessage(Ctxt &ctxt)
    {
        ssize_t ret = recvmsg(socketFd, &ctxt.msghdr, MSG_DONTWAIT | MSG_PEEK);
        if (ret > 0) {
            return true;
        }
        if (ret == 0) {
            return false;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return false;
        }
        throw std::runtime_error("TryPeekMessage occur " + std::string(std::strerror(errno)));
    }

    const char *GetName(Ctxt const & ctxt) const
    {
        if (ctxt.messageName.sun_path[0] != STR_END_CHAR) {
            throw std::runtime_error("GetName() want to got abstract socket, but got " +
                std::string(ctxt.messageName.sun_path));
        }
        return ctxt.messageName.sun_path + 1;
    }

    std::vector<fileDesT> GetFileDes(const Ctxt &ctxt) const
    {
        struct cmsghdr *cmg = CMSG_FIRSTHDR(&ctxt.msghdl);
        unsigned numFileDes = (cmg->cmsg_len - sizeof(struct cmsghdr)) / sizeof(fileDesT);
        return { ctxt.fileDesPtr, ctxt.fileDesPtr + numFileDes };
    }

protected:
    fileDesT socketFd;
    size_t SetSocketAdress(const std::string &srcSocket, struct sockaddr_un &destSocket)
    {
        if (srcSocket.size() > addressMaxLen) {
            throw std::runtime_error("Abstract UNIX Socket path cannot be larger than addressMaxLen");
        }
        destSocket.sun_family = AF_UNIX;
        destSocket.sun_path[0] = STR_END_CHAR;
        if (srcSocket.empty()) {
            return sizeof(sa_family_t);
        }
        srcSocket.copy(destSocket.sun_path + 1, srcSocket.size());
        destSocket.sun_path[srcSocket.size() + 1] = STR_END_CHAR;
        return sizeof(sa_family_t) + srcSocket.size() + 2;  // 2
    }

    auto BuildNpuCtxt_(const std::vector<NpuPayLoad> &npuPayLoad, unsigned numFileDes)
    {
        auto ctxt = std::make_unique<Ctxt>(npuPayLoad.size());
        if (memset_s(&ctxt->msghdr, sizeof(ctxt->msghdr), 0, sizeof(ctxt->msghdr)) != EOK) {
            throw std::runtime_error("Memset failed when build ctxt " + IPC_ERROR(ErrCode::MEMORY));
        }
        for (size_t i = 0; i < npuPayLoad.size(); i++) {
            ctxt->iov[i] = {npuPayLoad[i].data, npuPayLoad[i].size};
        }
        ctxt->msghdr.msg_name = &ctxt->messageName;
        ctxt->msghdr.msg_namelen = sizeof(decltype(ctxt->messageName));
        ctxt->msghdr.msg_iov = ctxt->iov.data();
        ctxt->msghdr.msg_iovlen = npuPayLoad.size();
        ctxt->fileDesPtr = nullptr;
        if (numFileDes == 0) {
            return ctxt;
        }
        const size_t fileDesSize = sizeof(fileDesT) * numFileDes;
        ctxt->msghdr.msg_control = ctxt->ancillaryBuf;
        ctxt->msghdr.msg_controllen = CMSG_SPACE(fileDesSize);

        struct cmsghdr *cmsg = CMSG_FIRSTHDR(&ctxt->msghdr);
        cmsg->cmsg_level = SOL_SOCKET;
        cmsg->cmsg_type = SCM_RIGHTS;
        cmsg->cmsg_len = CMSG_LEN(fileDesSize);
        ctxt->fileDesPtr = ReinterpretConvert<fileDesT *>(CMSG_DATA(cmsg));
        return ctxt;
    }
};
} // namespace ipc_monitor
} // namespace dynolog_npu

#endif // NPU_IPC_ENDPOINT_H
