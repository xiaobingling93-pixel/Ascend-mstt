#include "MetricManager.h"
#include "MetricKernelProcess.h"
#include "MetricApiProcess.h"
#include "MetricMemCpyProcess.h"
#include "MetricHcclProcess.h"
#include "MetricMarkProcess.h"
#include "MetricMemSetProcess.h"
#include "MetricMemProcess.h"

namespace dynolog_npu {
namespace ipc_monitor{
namespace metric {

MetricManager::MetricManager(): TimerTask("MetricManager", 0),
kindSwitchs_(MSPTI_ACTIVITY_KIND_COUNT), consumeStatus_(MSPTI_ACTIVITY_KIND_COUNT){
    metrics.resize(MSPTI_ACTIVITY_KIND_COUNT);
    metrics[MSPTI_ACTIVITY_KIND_KERNEL] = std::make_shared<MetricKernelProcess>();
    metrics[MSPTI_ACTIVITY_KIND_API] = std::make_shared<MetricApiProcess>();
    metrics[MSPTI_ACTIVITY_KIND_MEMCPY] = std::make_shared<MetricMemCpyProcess>();
    metrics[MSPTI_ACTIVITY_KIND_MARKER] = std::make_shared<MetricMarkProcess>();
    metrics[MSPTI_ACTIVITY_KIND_MEMSET] = std::make_shared<MetricMemSetProcess>();
    metrics[MSPTI_ACTIVITY_KIND_HCCL] = std::make_shared<MetricHcclProcess>();
    metrics[MSPTI_ACTIVITY_KIND_MEMORY] = std::make_shared<MetricMemProcess>();
}

void MetricManager::ReleaseResource()
{
    for (int i = 0; i < MSPTI_ACTIVITY_KIND_COUNT; i++) {
        if (kindSwitchs_[i].load()) {
            kindSwitchs_[i] = false;
            metrics[i]->Clear();
        }
    } 
}

ErrCode MetricManager::ConsumeMsptiData(msptiActivity *record) 
{
    if (!kindSwitchs_[record->kind]) {
        return ErrCode::PERMISSION;
    }
    auto metricProcess = metrics[record->kind];
    consumeStatus_[record->kind] = true;
    metricProcess->ConsumeMsptiData(record);
    consumeStatus_[record->kind] = false;
    return ErrCode::SUC;
}

void MetricManager::SetReportInterval(uint32_t intervalTimes)
{
    if (reportInterval_.load() != intervalTimes) {
        SendMetricMsg();
        SetInterval(intervalTimes);
        reportInterval_.store(intervalTimes);
    }
}

void MetricManager::ExecuteTask()
{
    SendMetricMsg();
}

void MetricManager::SendMetricMsg()
{
    for (int i = 0; i < MSPTI_ACTIVITY_KIND_COUNT; i++) {
        if (kindSwitchs_[i].load()) {
            metrics[i]->SendProcessMessage();
        }
    }
}

void MetricManager::EnableKindSwitch_(msptiActivityKind kind, bool flag) 
{
    kindSwitchs_[kind] = flag;
}
}
}
}