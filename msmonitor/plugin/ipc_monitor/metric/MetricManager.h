#ifndef METRIC_MANAGER_H
#define METRIC_MANAGER_H

#include <vector>
#include <atomic>

#include "utils.h"
#include "singleton.h"
#include "mspti.h"
#include "TimerTask.h"
#include "MetricProcessBase.h"

namespace dynolog_npu {
namespace ipc_monitor {
namespace metric {
class MetricManager: public ipc_monitor::Singleton<MetricManager>, public TimerTask
{
public:
    MetricManager();
    ~MetricManager() = default;
    ErrCode ConsumeMsptiData(msptiActivity *record);
    void SetReportInterval(uint32_t intervalTimes);
    void SendMetricMsg();
    void ExecuteTask() override;
    void EnableKindSwitch_(msptiActivityKind kind, bool flag);
    void ReleaseResource() override;
private:
    std::vector<std::atomic<bool>> kindSwitchs_;
    std::vector<std::atomic<bool>> consumeStatus_;
    std::atomic<uint32_t> reportInterval_;
    std::vector<std::shared_ptr<MetricProcessBase>> metrics;
};
}
}
}
#endif