#include "MetricApiProcess.h"

#include <numeric>
#include <nlohmann/json.hpp>

#include "utils.h"

namespace dynolog_npu {
namespace ipc_monitor{
namespace metric {

std::string ApiMetric::seriesToJson()
{
    nlohmann::json jsonMsg;
    jsonMsg["kind"] = "API";
    jsonMsg["deviceId"] = -1;
    jsonMsg["duration"] = duration;
    jsonMsg["timestamp"] = timestamp;
    return jsonMsg.dump();
}

void MetricApiProcess::ConsumeMsptiData(msptiActivity *record)
{
    msptiActivityApi* apiData = ReinterpretConvert<msptiActivityApi*>(record);
    msptiActivityApi* tmp = ReinterpretConvert<msptiActivityApi*>(MsptiMalloc(sizeof(msptiActivityApi), ALIGN_SIZE));
    if (memcpy_s(tmp, sizeof(msptiActivityApi), apiData, sizeof(msptiActivityApi)) != EOK) {
        MsptiFree(ReinterpretConvert<uint8_t*>(tmp));
        LOG(ERROR) << "memcpy_s failed" << IPC_ERROR(ErrCode::MEMORY);
        return;
    }
    {
        std::unique_lock<std::mutex> lock(dataMutex);
        records.emplace_back(tmp);
    }
}

std::vector<ApiMetric> MetricApiProcess::AggregatedData()
{
    std::vector<std::shared_ptr<msptiActivityApi>> copyRecords;
    {
        std::unique_lock<std::mutex> lock(dataMutex);
        copyRecords = std::move(records);
        records.clear();
    }
    ApiMetric apiMetric{};
    auto ans = std::accumulate(copyRecords.begin(), copyRecords.end(), 0ULL,
                [](uint64_t acc, std::shared_ptr<msptiActivityApi> api) {
                    return acc + api->end - api->start;
                });
    apiMetric.duration = ans;
    apiMetric.deviceId = -1;
    apiMetric.timestamp = getCurrentTimestamp64();
    return {apiMetric};
}

void MetricApiProcess::SendProcessMessage()
{
    auto afterAggregated = AggregatedData();
    for (auto& metric: afterAggregated) {
        SendMessage(metric.seriesToJson());
    }
}

void MetricApiProcess::Clear()
{
    records.clear();
}
}
}
}
