#ifndef METRIC_MARK_PROCESS_H
#define METRIC_MARK_PROCESS_H 

#include <vector>
#include <memory>
#include "MetricProcessBase.h"


namespace dynolog_npu {
namespace ipc_monitor{
namespace metric {

struct MarkMetric {
    std::string name;
    std::string domain;
    uint64_t duration;
    uint64_t timestamp;
    uint32_t deviceId;
public:
    std::string seriesToJson();
};

struct RangeMarkData
{
    std::string domain;
    uint64_t duration;
    uint64_t start{0};
    uint64_t end{0};
    uint64_t deviceStart{0};
    uint64_t deviceEnd{0};
    msptiActivitySourceKind sourceKind;
    uint32_t deviceId;
};


class MetricMarkProcess: public MetricProcessBase
{
public:
    MetricMarkProcess() = default;
    void ConsumeMsptiData(msptiActivity *record) override;
    std::vector<MarkMetric> AggregatedData();
    void SendProcessMessage() override;
    void Clear() override;
private:
    bool TransMarkData2Range(const std::vector<std::shared_ptr<msptiActivityMarker>>& markDatas, 
        RangeMarkData& rangemarkData);
private:
    std::mutex dataMutex;
    std::unordered_map<uint64_t, std::shared_ptr<std::string>> domainMsg;
    std::vector<std::shared_ptr<msptiActivityMarker>> records;
    std::map<uint64_t, std::vector<std::shared_ptr<msptiActivityMarker>>> id2Marker;
};
}
}
}

#endif