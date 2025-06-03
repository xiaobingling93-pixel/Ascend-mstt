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
    if (memcpy_s(ptr, sizeof(msptiActivityMemory), mem, sizeof(msptiActivityMemory)) != EOK) {
        MsptiFree(ReinterpretConvert<uint8_t*>(ptr));
        LOG(ERROR) << "memcpy_s failed" << IPC_ERROR(ErrCode::MEMORY);
        return;
    }
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
    std::unordered_map<uint32_t, std::vector<std::shared_ptr<msptiActivityMemory>>> deviceId2MemData =
    groupby(copyRecords, [](const std::shared_ptr<msptiActivityMemory>& data) -> std::uint32_t {
        return data->deviceId;
    });
    std::vector<MemMetric> ans;
    auto curTimestamp = getCurrentTimestamp64();
    for (auto& pair: deviceId2MemData) {
        auto deviceId = pair.first;
        auto& memDatas = pair.second;
        MemMetric memMetric{};
        memMetric.duration = std::accumulate(memDatas.begin(), memDatas.end(), 0ULL,
            [](uint64_t acc, std::shared_ptr<msptiActivityMemory> mem) {
                return acc + mem->end - mem->start;
            });
        memMetric.deviceId = deviceId;
        memMetric.timestamp = curTimestamp;
        ans.emplace_back(memMetric);
    }
    return ans;
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