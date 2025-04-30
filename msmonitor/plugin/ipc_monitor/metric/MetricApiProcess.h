#ifndef METRIC_API_PROCESS_H
#define METRIC_API_PROCESS_H 

#include <vector>
#include <memory>
#include "MetricProcessBase.h"


namespace dynolog_npu {
namespace ipc_monitor{
namespace metric {

struct ApiMetric {
    uint64_t duration;
    uint64_t timestamp;
    uint32_t deviceId;
public:
    std::string seriesToJson();
};

class MetricApiProcess: public MetricProcessBase
{
public:
    MetricApiProcess() = default;
    void ConsumeMsptiData(msptiActivity *record) override;
    std::vector<ApiMetric> AggregatedData();
    void SendProcessMessage() override;
    void Clear() override;
private:
    std::mutex dataMutex;
    std::vector<std::shared_ptr<msptiActivityApi>> records;
};
}
}
}

#endif