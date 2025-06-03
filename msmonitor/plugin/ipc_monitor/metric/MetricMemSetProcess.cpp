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
    if (memcpy_s(ptr, sizeof(msptiActivityMemset), memSet, sizeof(msptiActivityMemset)) != EOK) {
        MsptiFree(ReinterpretConvert<uint8_t*>(ptr));
        LOG(ERROR) << "memcpy_s failed" << IPC_ERROR(ErrCode::MEMORY);
        return;
    }
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
    std::unordered_map<uint32_t, std::vector<std::shared_ptr<msptiActivityMemset>>> deviceId2MemsetData =
    groupby(copyRecords, [](const std::shared_ptr<msptiActivityMemset>& data) -> std::uint32_t {
        return data->deviceId;
    });
    std::vector<MemSetMetric> ans;
    auto curTimestamp = getCurrentTimestamp64();
    for (auto& pair: deviceId2MemsetData) {
        MemSetMetric memSetMetric{};
        auto deviceId = pair.first;
        auto& memSetDatas = pair.second;
        memSetMetric.duration = std::accumulate(memSetDatas.begin(), memSetDatas.end(), 0ULL,
            [](uint64_t acc, std::shared_ptr<msptiActivityMemset> memSet) {
                return acc + memSet->end - memSet->start;
            });
        memSetMetric.deviceId = deviceId;
        memSetMetric.timestamp = curTimestamp;
        ans.emplace_back(memSetMetric);
    }
    return ans;
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