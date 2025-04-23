#include "MetricMemCpyProcess.h"

#include <numeric>

namespace dynolog_npu {
namespace ipc_monitor{
namespace metric {

std::string MemCpyMetric::seriesToJson()
{
    nlohmann::json jsonMsg;
    jsonMsg["kind"] = "MemCpy";
    jsonMsg["deviceId"] = deviceId;
    jsonMsg["duration"] = duration;
    jsonMsg["timestamp"] = timestamp;
    return jsonMsg.dump();
}

void MetricMemCpyProcess::ConsumeMsptiData(msptiActivity *record) 
{
    msptiActivityMemcpy* kernel = ReinterpretConvert<msptiActivityMemcpy*>(record);
    msptiActivityMemcpy* ptr = ReinterpretConvert<msptiActivityMemcpy*>(MsptiMalloc(sizeof(msptiActivityMemcpy), ALIGN_SIZE));
    memcpy(ptr, kernel, sizeof(msptiActivityMemcpy));
    {
        std::unique_lock<std::mutex> lock(dataMutex);
        records.emplace_back(ptr);
    }
}

std::vector<MemCpyMetric> MetricMemCpyProcess::AggregatedData()
{
    std::vector<std::shared_ptr<msptiActivityMemcpy>> copyRecords;
    {
        std::unique_lock<std::mutex> lock(dataMutex);
        copyRecords = std::move(records);
        records.clear();
    }
    if (copyRecords.empty()) {
        return {};
    }
    auto deviceId = copyRecords[0]->deviceId;
    MemCpyMetric memCpyMetric{};
    auto ans = std::accumulate(copyRecords.begin(), copyRecords.end(), 0ULL,
            [](uint64_t acc, std::shared_ptr<msptiActivityMemcpy> memcpy) {
                return acc + memcpy->end - memcpy->start;
            });
    memCpyMetric.duration = ans;
    memCpyMetric.deviceId = deviceId;
    memCpyMetric.timestamp = getCurrentTimestamp64();
    return {memCpyMetric};
}

void MetricMemCpyProcess::SendProcessMessage()
{
    auto afterAggregated = AggregatedData();
    for (auto& metric: afterAggregated) {
        SendMessage(metric.seriesToJson());
    }
}

void MetricMemCpyProcess::Clear()
{
    records.clear();
}
}
}
}