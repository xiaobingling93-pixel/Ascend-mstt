#include "MetricHcclProcess.h"
#include <numeric>
#include <nlohmann/json.hpp>
#include "utils.h"

namespace dynolog_npu {
namespace ipc_monitor{
namespace metric {

std::string HcclMetric::seriesToJson()
{
    nlohmann::json jsonMsg;
    jsonMsg["kind"] = "Hccl";
    jsonMsg["deviceId"] = deviceId;
    jsonMsg["duration"] = duration;
    jsonMsg["timestamp"] = timestamp;
    return jsonMsg.dump();
}

void MetricHcclProcess::ConsumeMsptiData(msptiActivity *record) 
{
    msptiActivityHccl* hcclData = ReinterpretConvert<msptiActivityHccl*>(record);
    msptiActivityHccl* tmp = ReinterpretConvert<msptiActivityHccl*>(MsptiMalloc(sizeof(msptiActivityHccl), ALIGN_SIZE));
    memcpy(tmp, hcclData, sizeof(msptiActivityHccl));
    {
        std::unique_lock<std::mutex> lock(dataMutex);
        records.emplace_back(tmp);
    }
    
}

std::vector<HcclMetric> MetricHcclProcess::AggregatedData()
{   
    std::vector<std::shared_ptr<msptiActivityHccl>> copyRecords;
    {
        std::unique_lock<std::mutex> lock(dataMutex);
        copyRecords = std::move(records);
        records.clear();
    }
    if (copyRecords.empty()) {
        return {};
    }
    HcclMetric hcclMetric{};
    auto ans = std::accumulate(copyRecords.begin(), copyRecords.end(), 0ULL,
                [](uint64_t acc, std::shared_ptr<msptiActivityHccl> hccl) {
                    return acc + hccl->end - hccl->start;
                });
    hcclMetric.duration = ans;
    hcclMetric.deviceId = copyRecords[0]->ds.deviceId;
    hcclMetric.timestamp = getCurrentTimestamp64();
    return {hcclMetric};
}

void MetricHcclProcess::SendProcessMessage()
{
    auto afterAggregated = AggregatedData();
    for (auto& metric: afterAggregated) {
        SendMessage(metric.seriesToJson());
    }
}

void MetricHcclProcess::Clear()
{
    records.clear();
}
}
}
}