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
    if (memcpy_s(ptr, sizeof(msptiActivityMemcpy), kernel, sizeof(msptiActivityMemcpy)) != EOK) {
        MsptiFree(ReinterpretConvert<uint8_t*>(ptr));
        LOG(ERROR) << "memcpy_s failed" << IPC_ERROR(ErrCode::MEMORY);
        return;
    }
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
    std::unordered_map<uint32_t, std::vector<std::shared_ptr<msptiActivityMemcpy>>> deviceId2Memcpy =
    groupby(copyRecords, [](const std::shared_ptr<msptiActivityMemcpy>& data) -> std::uint32_t {
        return data->deviceId;
    });
    std::vector<MemCpyMetric> ans;
    auto curTimestamp = getCurrentTimestamp64();
    for (auto& pair: deviceId2Memcpy) {
        auto deviceId = pair.first;
        MemCpyMetric memCpyMetric{};
        auto& memCpyDatas = pair.second;
        memCpyMetric.duration = std::accumulate(memCpyDatas.begin(), memCpyDatas.end(), 0ULL,
            [](uint64_t acc, std::shared_ptr<msptiActivityMemcpy> memcpy) {
                return acc + memcpy->end - memcpy->start;
            });
        memCpyMetric.deviceId = deviceId;
        memCpyMetric.timestamp = curTimestamp;
        ans.emplace_back(memCpyMetric);
    }
    return ans;
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
