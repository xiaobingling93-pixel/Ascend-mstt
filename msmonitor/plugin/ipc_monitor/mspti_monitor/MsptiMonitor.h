#ifndef MSPTI_MONITOR_H
#define MSPTI_MONITOR_H

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <set>
#include "mspti.h"
#include "thread.h"


namespace dynolog_npu {
namespace ipc_monitor {
class MsptiMonitor : public Thread {
public:
    explicit MsptiMonitor();
    virtual ~MsptiMonitor();
    void Start();
    void Stop();
    void EnableActivity(msptiActivityKind kind);
    void DisableActivity(msptiActivityKind kind);
    void SetFlushInterval(uint32_t interval);
    bool IsStarted();
    std::set<msptiActivityKind> GetEnabledActivities();
    void Uninit();

private:
    static void BufferRequest(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
    static void BufferComplete(uint8_t *buffer, size_t size, size_t validSize);
    static void BufferConsume(msptiActivity *record);
    static std::atomic<uint32_t> allocCnt;

private:
    void Run() override;

private:
    std::atomic<bool> start_;
    std::mutex cvMtx_;
    std::condition_variable cv_;
    msptiSubscriberHandle subscriber_;
    std::mutex activityMtx_;
    std::set<msptiActivityKind> enabledActivities_;
    std::atomic<bool> checkFlush_;
    std::atomic<uint32_t> flushInterval_;
};
} // namespace ipc_monitor
} // namespace dynolog_npu
#endif // MSPTI_MONITOR_H
