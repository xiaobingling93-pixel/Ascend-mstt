#ifndef METRIC_KERNEL_PROCESS_H
#define METRIC_KERNEL_PROCESS_H 

#include <vector>
#include "MetricProcessBase.h"


namespace dynolog_npu {
namespace ipc_monitor{
namespace metric {

struct KernelMetric {
    uint64_t duration;
    uint64_t timestamp;
    uint32_t deviceId;
public: 
    std::string seriesToJson();
};

class MetricKernelProcess: public MetricProcessBase
{
public:
    MetricKernelProcess() = default;
    void ConsumeMsptiData(msptiActivity *record) override;
    std::vector<KernelMetric> AggregatedData();
    void SendProcessMessage() override;
    void Clear() override;
private:
    std::mutex dataMutex;
    std::vector<std::shared_ptr<msptiActivityKernel>> records;
};
}
}
}

#endif