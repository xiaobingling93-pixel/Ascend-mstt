#include "MetricKernelProcess.h"

#include <numeric>

namespace dynolog_npu {
namespace ipc_monitor{
namespace metric {

std::string KernelMetric::seriesToJson()
{
    nlohmann::json jsonMsg;
    jsonMsg["kind"] = "Kernel";
    jsonMsg["deviceId"] = deviceId;
    jsonMsg["duration"] = duration;
    jsonMsg["timestamp"] = timestamp;
    return jsonMsg.dump();
}

void MetricKernelProcess::ConsumeMsptiData(msptiActivity *record) 
{
    msptiActivityKernel* kernel = ReinterpretConvert<msptiActivityKernel*>(record);
    msptiActivityKernel* ptr = ReinterpretConvert<msptiActivityKernel*>(MsptiMalloc(sizeof(msptiActivityKernel), ALIGN_SIZE));
    memcpy(ptr, kernel, sizeof(msptiActivityKernel));
    {
        std::unique_lock<std::mutex> lock(dataMutex);
        records.emplace_back(ptr);
    }
}

std::vector<KernelMetric> MetricKernelProcess::AggregatedData()
{
    std::vector<std::shared_ptr<msptiActivityKernel>> copyRecords;
    {
        std::unique_lock<std::mutex> lock(dataMutex);
        copyRecords = std::move(records);
        records.clear();
    }
    if (copyRecords.empty()) {
        return {};
    }
    std::unordered_map<uint32_t, std::vector<std::shared_ptr<msptiActivityKernel>>> deviceId2KernelData = 
        groupby(copyRecords, [](const std::shared_ptr<msptiActivityKernel>& data) -> std::uint32_t {
            return data->ds.deviceId;
        });
    std::vector<KernelMetric> ans;
    auto curTimestamp = getCurrentTimestamp64();
    for (auto& pair: deviceId2KernelData) {
        auto deviceId = pair.first;
        auto& kernelDatas = pair.second;
        KernelMetric kernelMetric{};
        kernelMetric.duration = std::accumulate(kernelDatas.begin(), kernelDatas.end(), 0ULL,
            [](uint64_t acc, std::shared_ptr<msptiActivityKernel> kernel) {
                return acc + kernel->end - kernel->start;
            });
        kernelMetric.deviceId = deviceId;
        kernelMetric.timestamp = curTimestamp;
        ans.emplace_back(kernelMetric);
    }

    return ans;
}

void MetricKernelProcess::SendProcessMessage()
{
    auto afterAggregated = AggregatedData();
    for (auto& metric: afterAggregated) {
        SendMessage(metric.seriesToJson());
    }
}

void MetricKernelProcess::Clear()
{
    records.clear();
}
}
}
}