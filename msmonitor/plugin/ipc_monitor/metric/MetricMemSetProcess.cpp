#include "MetricMemSetProcess.h"

#include <numeric>

namespace dynolog_npu {
namespace ipc_monitor{
namespace metric {

std::string MemSetMetric::seriesToJson()
{
    nlohmann::json jsonMsg;
    jsonMsg["kind"] = "MemSet";
    jsonMsg["deviceId"] = deviceId;
    jsonMsg["duration"] = duration;
    jsonMsg["timestamp"] = timestamp;
    return jsonMsg.dump();
}

void MetricMemSetProcess::ConsumeMsptiData(msptiActivity *record) 
{
    msptiActivityMemset* memSet = ReinterpretConvert<msptiActivityMemset*>(record);
    msptiActivityMemset* ptr = ReinterpretConvert<msptiActivityMemset*>(MsptiMalloc(sizeof(msptiActivityMemset), ALIGN_SIZE));
    memcpy(ptr, memSet, sizeof(msptiActivityMemset));
    {
        std::unique_lock<std::mutex> lock(dataMutex);
        records.emplace_back(ptr);
    }
}

std::vector<MemSetMetric> MetricMemSetProcess::AggregatedData()
{
    std::vector<std::shared_ptr<msptiActivityMemset>> copyRecords;
    {
        std::unique_lock<std::mutex> lock(dataMutex);
        copyRecords = std::move(records);
        records.clear();
    }
    if (copyRecords.empty()) {
        return {};
    }
    auto deviceId = copyRecords[0]->deviceId;
    MemSetMetric memSetMetric{};
    auto ans = std::accumulate(copyRecords.begin(), copyRecords.end(), 0ULL,
            [](uint64_t acc, std::shared_ptr<msptiActivityMemset> memSet) {
                return acc + memSet->end - memSet->start;
            });
    memSetMetric.duration = ans;
    memSetMetric.deviceId = deviceId;
    memSetMetric.timestamp = getCurrentTimestamp64();
    return {memSetMetric};
}

void MetricMemSetProcess::SendProcessMessage()
{
    auto afterAggregated = AggregatedData();
    for (auto& metric: afterAggregated) {
        SendMessage(metric.seriesToJson());
    }
}

void MetricMemSetProcess::Clear()
{
    records.clear();
}
}
}
}