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
#include "NpuIpcClient.h"
#include <glog/logging.h>

namespace dynolog_npu {
namespace ipc_monitor {
bool IpcClient::Init()
{
    pids_ = GetPids();
    return true;
}

bool IpcClient::RegisterInstance(int32_t npu)
{
    NpuContext context{
        .npu = npu,
        .pid = getpid(),
        .jobId = JOB_ID,
    };
    std::unique_ptr<Message> message = Message::ConstructMessage<decltype(context)>(context, MSG_TYPE_CONTEXT);
    try {
        if (!SyncSendMessage(*message, DYNO_IPC_NAME)) {
            LOG(WARNING) << "Failed to send register ctxt for pid " << context.pid << " with dyno";
            return false;
        }
    } catch (const std::exception &e) {
        LOG(WARNING) << "Error when SyncSendMessage: " << e.what();
        return false;
    }
    LOG(INFO) << "Resigter pid " << context.pid << " for dynolog success!";
    return true;
}

std::string IpcClient::IpcClientNpuConfig()
{
    auto size = pids_.size();
    auto *req = ReinterpretConvert<NpuRequest *>(malloc(sizeof(NpuRequest) + sizeof(int32_t) * size));
    if (req == nullptr) {
        LOG(ERROR) << " Malloc for NpuRequest failed !";
        return "";
    }
    req->type = DYNO_IPC_TYPE;
    req->pidSize = size;
    req->jobId = JOB_ID;
    for (size_t i = 0; i < size; i++) {
        req->pids[i] = pids_[i];
    }
    std::unique_ptr<Message> message;
    try{
        message = Message::ConstructMessage<NpuRequest, int32_t>(*req, MSG_TYPE_REQUEST, size);
    }
    catch (const std::exception &e) {
        LOG(ERROR) << "ConstructMessage failed: " << e.what();
        free(req);
        req = nullptr;
        throw;
    }
    if (!message || !SyncSendMessage(*message, DYNO_IPC_NAME)) {
        LOG(WARNING) << "Failed to send config to dyno server";
        free(req);
        req = nullptr;
        return "";
    }
    free(req);
    req = nullptr;
    message = PollRecvMessage(MAX_IPC_RETRIES, MAX_SLEEP_US);
    if (!message) {
        LOG(WARNING) << "Failed to receive on-demand config";
        return "";
    }
    std::string res = std::string(ReinterpretConvert<char *>(message->buf.get()), message->metadata.size);
    return res;
}

std::unique_ptr<Message> IpcClient::ReceiveMessage()
{
    std::lock_guard<std::mutex> wguard(dequeLock_);
    if (msgDynoDeque_.empty()) {
        return nullptr;
    }
    std::unique_ptr<Message> message = std::move(msgDynoDeque_.front());
    msgDynoDeque_.pop_front();
    return message;
}

bool IpcClient::SyncSendMessage(const Message &message, const std::string &destName, int numRetry, int seepTimeUs)
{
    if (destName.empty()) {
        LOG(WARNING) << "Can not send to empty socket name!";
        return false;
    }
    int i = 0;
    std::vector<NpuPayLoad> npuPayLoad{ NpuPayLoad(sizeof(struct Metadata), (void *)&message.metadata),
        NpuPayLoad(message.metadata.size, message.buf.get()) };
    try {
        auto ctxt = ep_.BuildSendNpuCtxt(destName, npuPayLoad, std::vector<int>());
        while (!ep_.TrySendMessage(*ctxt) && i < numRetry) {
            i++;
            usleep(seepTimeUs);
            seepTimeUs *= 2;  // 2: double sleep time
        }
    } catch (const std::exception &e) {
        LOG(ERROR) << "Error when SyncSendMessage: " << e.what();
        return false;
    }
    return i < numRetry;
}

bool IpcClient::Recv()
{
    try {
        Metadata recvMetadata;
        std::vector<NpuPayLoad> PeekNpuPayLoad{ NpuPayLoad(sizeof(struct Metadata), &recvMetadata) };
        auto peekCtxt = ep_.BuildNpuRcvCtxt(PeekNpuPayLoad);
        bool successFlag = false;
        try {
            successFlag = ep_.TryPeekMessage(*peekCtxt);
        } catch (std::exception &e) {
            LOG(ERROR) << "Error when TryPeekMessage: " << e.what();
            return false;
        }
        if (successFlag) {
            std::unique_ptr<Message> npuMessage = std::make_unique<Message>(Message());
            npuMessage->metadata = recvMetadata;
            npuMessage->buf = std::make_unique<unsigned char[]>(recvMetadata.size);
            npuMessage->src = std::string(ep_.GetName(*peekCtxt));
            std::vector<NpuPayLoad> npuPayLoad{ NpuPayLoad(sizeof(struct Metadata), (void *)&npuMessage->metadata),
                NpuPayLoad(recvMetadata.size, npuMessage->buf.get()) };
            auto recvCtxt = ep_.BuildNpuRcvCtxt(npuPayLoad);
            try {
                successFlag = ep_.TryRcvMessage(*recvCtxt);
            } catch (std::exception &e) {
                LOG(ERROR) << "Error when TryRecvMsg: " << e.what();
                return false;
            }
            if (successFlag) {
                std::lock_guard<std::mutex> wguard(dequeLock_);
                msgDynoDeque_.push_back(std::move(npuMessage));
                return true;
            }
        }
    } catch (std::exception &e) {
        LOG(ERROR) << "Error in Recv(): " << e.what();
        return false;
    }
    return false;
}

std::unique_ptr<Message> IpcClient::PollRecvMessage(int maxRetry, int sleeTimeUs)
{
    for (int i = 0; i < maxRetry; i++) {
        if (Recv()) {
            return ReceiveMessage();
        }
        usleep(sleeTimeUs);
    }
    return nullptr;
}
} // namespace ipc_monitor
} // namespace dynolog_npu
