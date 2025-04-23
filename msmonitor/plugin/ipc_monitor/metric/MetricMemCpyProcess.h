#ifndef METRIC_MEMCPY_PROCESS_H
#define METRIC_MEMCPY_PROCESS_H 

#include <vector>
#include "MetricProcessBase.h"


namespace dynolog_npu {
namespace ipc_monitor{
namespace metric {

struct MemCpyMetric {
    uint64_t duration;
    uint64_t timestamp;
    uint32_t deviceId;
public: 
    std::string seriesToJson();
};

class MetricMemCpyProcess: public MetricProcessBase
{
public:
    MetricMemCpyProcess() = default;
    void ConsumeMsptiData(msptiActivity *record) override;
    std::vector<MemCpyMetric> AggregatedData();
    void SendProcessMessage() override;
    void Clear() override;
private:
    std::mutex dataMutex;
    std::vector<std::shared_ptr<msptiActivityMemcpy>> records;
};
}
}
}

#endif