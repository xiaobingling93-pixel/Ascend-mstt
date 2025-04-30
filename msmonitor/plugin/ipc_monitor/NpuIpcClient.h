#ifndef NPU_IPC_CLIENT_H
#define NPU_IPC_CLIENT_H

#include <deque>
#include <memory>
#include <mutex>
#include "NpuIpcEndPoint.h"
#include "utils.h"

namespace dynolog_npu {
namespace ipc_monitor {

constexpr int TYPE_SIZE = 32;
constexpr int JOB_ID = 0;
constexpr const int DYNO_IPC_TYPE = 3;
constexpr const int MAX_IPC_RETRIES = 5;
constexpr const int MAX_SLEEP_US = 10000;
const std::string DYNO_IPC_NAME = "dynolog";
const std::string MSG_TYPE_REQUEST = "req";
const std::string MSG_TYPE_CONTEXT = "ctxt";
const std::string MSG_TYPE_DATA = "data";

struct NpuRequest {
    int type;
    int pidSize;
    int64_t jobId;
    int32_t pids[0];
};

struct NpuContext {
    int32_t npu;
    pid_t pid;
    int64_t jobId;
};

struct Metadata {
    size_t size = 0;
    char type[TYPE_SIZE] = "";
};

struct Message {
    Metadata metadata;
    std::unique_ptr<unsigned char[]> buf;
    std::string src;
    template <class T> static std::unique_ptr<Message> ConstructMessage(const T &data, const std::string &type)
    {
        std::unique_ptr<Message> ipcNpuMessage = std::make_unique<Message>(Message());
        if (type.size() + 1 > sizeof(ipcNpuMessage->metadata.type)) {
            throw std::runtime_error("Type string is too long to fit in metadata.type" + IPC_ERROR(ErrCode::PARAM));
        }
        memcpy(ipcNpuMessage->metadata.type, type.c_str(), type.size() + 1);
#if __cplusplus >= 201703L
        if constexpr (std::is_same<std::string, T>::value == true) {
            ipcNpuMessage->metadata.size = data.size();
            ipcNpuMessage->buf = std::make_unique<unsigned char[]>(ipcNpuMessage->metadata.size);
            memcpy(ipcNpuMessage->buf.get(), data.c_str(), sizeof(data));
            return ipcNpuMessage;
        }
#endif
        static_assert(std::is_trivially_copyable<T>::value);
        ipcNpuMessage->metadata.size = sizeof(data);
        ipcNpuMessage->buf = std::make_unique<unsigned char[]>(ipcNpuMessage->metadata.size);
        memcpy(ipcNpuMessage->buf.get(), &data, sizeof(data));
        return ipcNpuMessage;
    }

    template <class T, class U>
    static std::unique_ptr<Message> ConstructMessage(const T &data, const std::string &type, int n)
    {
        std::unique_ptr<Message> ipcNpuMessage = std::make_unique<Message>(Message());
        if (type.size() + 1 > sizeof(ipcNpuMessage->metadata.type)) {
            throw std::runtime_error("Type string is too long to fit in metadata.type" + IPC_ERROR(ErrCode::PARAM));
        }
        memcpy(ipcNpuMessage->metadata.type, type.c_str(), type.size() + 1);
        static_assert(std::is_trivially_copyable<T>::value);
        static_assert(std::is_trivially_copyable<U>::value);
        ipcNpuMessage->metadata.size = sizeof(data) + sizeof(U) * n;
        ipcNpuMessage->buf = std::make_unique<unsigned char[]>(ipcNpuMessage->metadata.size);
        memcpy(ipcNpuMessage->buf.get(), &data, ipcNpuMessage->metadata.size);
        return ipcNpuMessage;
    }

    static std::unique_ptr<Message> ConstructStrMessage(const std::string &data, const std::string &type)
    {
        std::unique_ptr<Message> ipcNpuMessage = std::make_unique<Message>(Message());
        if (type.size() + 1 > sizeof(ipcNpuMessage->metadata.type)) {
            throw std::runtime_error("Type string is too long to fit in metadata.type" + IPC_ERROR(ErrCode::PARAM));
        }
        memcpy(ipcNpuMessage->metadata.type, type.c_str(), type.size() + 1);
        ipcNpuMessage->metadata.size = data.size();
        ipcNpuMessage->buf = std::make_unique<unsigned char[]>(ipcNpuMessage->metadata.size);
        memcpy(ipcNpuMessage->buf.get(), data.c_str(), ipcNpuMessage->metadata.size);
        return ipcNpuMessage;
    }
};

class IpcClient {
public:
    IpcClient(const IpcClient &) = delete;
    IpcClient &operator = (const IpcClient &) = delete;
    IpcClient() = default;
    bool RegisterInstance(int32_t npu);
    std::string IpcClientNpuConfig();
    bool SyncSendMessage(const Message &message, const std::string &destName, int numRetry = 10,
        int seepTimeUs = 10000);

private:
    std::vector<int32_t> pids_ = GetPids();
    NpuIpcEndPoint<0> ep_{ "dynoconfigclient" + GenerateUuidV4() };
    std::mutex dequeLock_;
    std::deque<std::unique_ptr<Message>> msgDynoDeque_;
    std::unique_ptr<Message> ReceiveMessage();
    bool Recv();
    std::unique_ptr<Message> PollRecvMessage(int maxRetry, int sleeTimeUs);
};
} // namespace ipc_monitor
} // namespace dynolog_npu

#endif // NPU_IPC_CLIENT_H
