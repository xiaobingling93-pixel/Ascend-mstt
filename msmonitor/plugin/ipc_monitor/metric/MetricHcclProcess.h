#ifndef METRIC_HCCL_PROCESS_H
#define METRIC_HCCL_PROCESS_H 

#include <vector>
#include <memory>
#include "MetricProcessBase.h"


namespace dynolog_npu {
namespace ipc_monitor{
namespace metric {

struct HcclMetric {
    std::string kindName;
    uint64_t duration;
    uint64_t timestamp;
    uint32_t deviceId;
public:
    std::string seriesToJson();
};

class MetricHcclProcess: public MetricProcessBase
{
public:
    MetricHcclProcess() = default;
    void ConsumeMsptiData(msptiActivity *record) override;
    std::vector<HcclMetric> AggregatedData();
    void SendProcessMessage() override;
    void Clear() override;
private:
    std::mutex dataMutex;
    std::vector<std::shared_ptr<msptiActivityHccl>> records;
};
}
}
}

#endif