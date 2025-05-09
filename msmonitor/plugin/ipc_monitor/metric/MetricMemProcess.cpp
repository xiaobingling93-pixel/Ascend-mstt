#include "MetricMemProcess.h"

#include <numeric>

namespace dynolog_npu {
namespace ipc_monitor{
namespace metric {

std::string MemMetric::seriesToJson()
{
    nlohmann::json jsonMsg;
    jsonMsg["kind"] = "Memory";
    jsonMsg["deviceId"] = deviceId;
    jsonMsg["duration"] = duration;
    jsonMsg["timestamp"] = timestamp;
    return jsonMsg.dump();
}

void MetricMemProcess::ConsumeMsptiData(msptiActivity *record) 
{
    msptiActivityMemory* mem = ReinterpretConvert<msptiActivityMemory*>(record);
    msptiActivityMemory* ptr = ReinterpretConvert<msptiActivityMemory*>(MsptiMalloc(sizeof(msptiActivityMemory), ALIGN_SIZE));
    memcpy(ptr, mem, sizeof(msptiActivityMemory));
    {
        std::unique_lock<std::mutex> lock(dataMutex);
        records.emplace_back(ptr);
    }
}

std::vector<MemMetric> MetricMemProcess::AggregatedData()
{
    std::vector<std::shared_ptr<msptiActivityMemory>> copyRecords;
    {
        std::unique_lock<std::mutex> lock(dataMutex);
        copyRecords = std::move(records);
        records.clear();
    }
    if (copyRecords.empty()) {
        return {};
    }
    auto deviceId = copyRecords[0]->deviceId;
    MemMetric memMetric{};
    auto ans = std::accumulate(copyRecords.begin(), copyRecords.end(), 0ULL,
            [](uint64_t acc, std::shared_ptr<msptiActivityMemory> mem) {
                return acc + mem->end - mem->start;
            });
    memMetric.duration = ans;
    memMetric.deviceId = deviceId;
    memMetric.timestamp = getCurrentTimestamp64();
    return {memMetric};
}

void MetricMemProcess::SendProcessMessage()
{
    auto afterAggregated = AggregatedData();
    for (auto& metric: afterAggregated) {
        SendMessage(metric.seriesToJson());
    }
}

void MetricMemProcess::Clear()
{
    records.clear();
}
}
}
}