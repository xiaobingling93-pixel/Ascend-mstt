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
    auto deviceId = copyRecords[0]->ds.deviceId;
    KernelMetric kernelMetric{};
    auto ans = std::accumulate(copyRecords.begin(), copyRecords.end(), 0ULL,
            [](uint64_t acc, std::shared_ptr<msptiActivityKernel> kernel) {
                return acc + kernel->end - kernel->start;
            });
    kernelMetric.duration = ans;
    kernelMetric.deviceId = deviceId;
    kernelMetric.timestamp = getCurrentTimestamp64();
    return {kernelMetric};
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