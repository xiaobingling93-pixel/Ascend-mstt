// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <deque>
#include <exception>
#include <mutex>
#include <algorithm>
#include "dynolog/src/ipcfabric/Endpoint.h"
#include "dynolog/src/ipcfabric/Utils.h"

// If building inside Kineto, use its logger, otherwise use glog
#if defined USE_GOOGLE_LOG
#include <glog/logging.h>
#endif // USE_GOOGLE_LOG

namespace dynolog::ipcfabric {

constexpr size_t TYPE_SIZE = 32;
constexpr size_t MAX_MSG_SIZE = 4096;

struct Metadata {
    size_t size = 0;
    char type[TYPE_SIZE] = "";
};

struct Message {
    template <class T>
    static std::unique_ptr<Message> constructMessage(const T &data, const std::string &type)
    {
        if (type.size() >= TYPE_SIZE) {
            LOG(ERROR) << "type size exceeds TYPE_SIZE: " << type;
            return nullptr;
        }
        std::unique_ptr<Message> msg = std::make_unique<Message>(Message());
        std::copy_n(type.begin(), type.size(), msg->metadata.type);
        msg->metadata.type[type.size()] = '\0';
#if __cplusplus >= 201703L
        if constexpr (std::is_same<std::string, T>::value == true)
        {
            msg->metadata.size = data.size();
            msg->buf = std::make_unique<unsigned char[]>(msg->metadata.size);
            std::copy_n(data.begin(), msg->metadata.size, msg->buf.get());
        }
        else
        {
#endif
            static_assert(std::is_trivially_copyable<T>::value);
            msg->metadata.size = sizeof(data);
            msg->buf = std::make_unique<unsigned char[]>(msg->metadata.size);
            std::copy_n((const unsigned char*)&data, msg->metadata.size, msg->buf.get());
#if __cplusplus >= 201703L
        }
#endif
        return msg;
    }

    template <class T, class U>
    static std::unique_ptr<Message> constructMessage(const T &data, const std::string &type, int n)
    {
        if (type.size() >= TYPE_SIZE) {
            LOG(ERROR) << "type size exceeds TYPE_SIZE: " << type;
            return nullptr;
        }
        std::unique_ptr<Message> msg = std::make_unique<Message>(Message());
        std::copy_n(type.begin(), type.size(), msg->metadata.type);
        msg->metadata.type[type.size()] = '\0';
        static_assert(std::is_trivially_copyable<T>::value);
        static_assert(std::is_trivially_copyable<U>::value);
        msg->metadata.size = sizeof(data) + sizeof(U) * n;
        msg->buf = std::make_unique<unsigned char[]>(msg->metadata.size);
        std::copy_n((const unsigned char*)&data, msg->metadata.size, msg->buf.get());
        return msg;
    }

    Metadata metadata;
    std::unique_ptr<unsigned char[]> buf{nullptr};
    // endpoint name of the sender
    std::string src;
};

class FabricManager {
public:
    explicit FabricManager(std::string endpoint_name = "") : ep_{std::move(endpoint_name)} {}
    FabricManager(const FabricManager &) = delete;
    FabricManager &operator=(const FabricManager &) = delete;

    static std::unique_ptr<FabricManager> factory(std::string endpoint_name = "")
    {
        try {
            return std::make_unique<FabricManager>(endpoint_name);
        } catch (std::exception &e) {
            LOG(ERROR) << "Error when initializing FabricManager: " << e.what();
        }
        return nullptr;
    }

    // warning: this will block for user passed in time with exponential increase
    // if send keeps failing
    bool sync_send(
        const Message &msg,
        const std::string &dest_name,
        int num_retries = 10,
        int sleep_time_us = 10000)
    {
        if (dest_name.empty()) {
            LOG(ERROR) << "Cannot send to empty socket name";
            return false;
        }

        std::vector<Payload> payload{
            Payload(sizeof(struct Metadata), (void *)&msg.metadata),
            Payload(msg.metadata.size, msg.buf.get())};
        int i = 0;
        try {
            auto ctxt = ep_.buildSendCtxt(dest_name, payload);
            while (!ep_.trySendMsg(*ctxt) && i < num_retries)
            {
                i++;
                /* sleep override */
                usleep(sleep_time_us);
                sleep_time_us += sleep_time_us;
            }
        } catch (std::exception &e) {
            LOG(ERROR) << "Error when sync_send(): " << e.what();
            return false;
        }
        return i < num_retries;
    }

    bool recv()
    {
        try {
            Metadata receive_metadata;
            std::vector<Payload> peek_payload{Payload(sizeof(struct Metadata), &receive_metadata)};
            auto peek_ctxt = ep_.buildRcvCtxt(peek_payload);
            // unix socket only fills the data for the iov that have a non NULL
            // buffer. Leverage that to read metadata to find buffer size by:
            //   1) FabricManager assumes metadata in first iov, data in second
            //   2) peek with only metadata buffer in iov
            //   3) read metadata
            //   4) use metadata to find the desired size for the buffer to allocate.
            //   5) read metadata + data with allocated buffer
            bool success{false};
            try {
                success = ep_.tryPeekMsg(*peek_ctxt);
            } catch (std::exception &e) {
                LOG(ERROR) << "Error when tryPeekMsg(): " << e.what();
                return false;
            }
            if (success) {
                receive_metadata.size = std::min(receive_metadata.size, MAX_MSG_SIZE);
                std::unique_ptr<Message> msg = std::make_unique<Message>(Message());
                msg->metadata = receive_metadata;
                msg->buf = std::make_unique<unsigned char[]>(receive_metadata.size);
                auto src = ep_.getName(*peek_ctxt, true);
                if (src == nullptr) {
                    LOG(ERROR) << "Failed to get source name from peek context";
                    return false;
                }
                msg->src = std::string(src);
                std::vector<Payload> payload{Payload(sizeof(struct Metadata), (void *)&msg->metadata),
                    Payload(receive_metadata.size, msg->buf.get())};
                auto recv_ctxt = ep_.buildRcvCtxt(payload);
                try {
                    success = ep_.tryRcvMsg(*recv_ctxt);
                } catch (std::exception &e) {
                    LOG(ERROR) << "Error when tryRcvMsg(): " << e.what();
                    return false;
                }
                if (recv_ctxt->msghdr.msg_flags & MSG_CTRUNC) {
                    LOG(ERROR) << "Received message with truncated data, message will be ingored.";
                    return false;
                }
                if (msg->metadata.size > MAX_MSG_SIZE) {
                    LOG(ERROR) << "Received message size: " << msg->metadata.size
                               << ", exceeds max size, message will be ingored.";
                    return false;
                }
                if (success) {
                    std::lock_guard<std::mutex> wguard(dequeLock_);
                    message_deque_.emplace_back(std::move(msg));
                    return true;
                }
            }
        } catch (std::exception &e) {
            LOG(ERROR) << "Error in recv(): " << e.what();
        }
        return false;
    }

    std::unique_ptr<Message> retrieve_msg()
    {
        std::lock_guard<std::mutex> wguard(dequeLock_);
        if (message_deque_.empty()) {
            return nullptr;
        }
        std::unique_ptr<Message> msg = std::move(message_deque_.front());
        message_deque_.pop_front();
        return msg;
    }

    std::unique_ptr<Message> poll_recv(int max_retries, int sleep_time_us)
    {
        for (int i = 0; i < max_retries; i++) {
            if (recv()) {
                return retrieve_msg();
            }
            /* sleep override */
            usleep(sleep_time_us);
        }
        return nullptr;
    }

private:
    // message LIFO deque with oldest message at front
    std::deque<std::unique_ptr<Message>> message_deque_;
    EndPoint ep_;
    std::mutex dequeLock_;
};
} // namespace dynolog::ipcfabric
