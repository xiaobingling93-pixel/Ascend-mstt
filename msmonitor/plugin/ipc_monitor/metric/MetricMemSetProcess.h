#ifndef METRIC_MEM_SET_PROCESS_H
#define METRIC_MEM_SET_PROCESS_H 

#include <vector>
#include "metric/MetricProcessBase.h"


namespace dynolog_npu {
namespace ipc_monitor{
namespace metric {

struct MemSetMetric {
    std::string name;
    uint64_t duration;
    uint64_t timestamp;
    uint32_t deviceId;
public: 
    std::string seriesToJson();
};

class MetricMemSetProcess: public MetricProcessBase
{
public:
    MetricMemSetProcess() = default;
    void ConsumeMsptiData(msptiActivity *record) override;
    std::vector<MemSetMetric> AggregatedData();
    void SendProcessMessage() override;
    void Clear() override;
private:
    std::mutex dataMutex;
    std::vector<std::shared_ptr<msptiActivityMemset>> records;
};
}
}
}

#endif