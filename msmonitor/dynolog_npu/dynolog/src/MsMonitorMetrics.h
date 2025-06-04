#ifndef DYNOLOG_NPU_MSMONITOR_METRICS_H
#define DYNOLOG_NPU_MSMONITOR_METRICS_H

#include <string>
#include <unordered_map>
#include <unordered_set>

namespace dynolog {

const std::unordered_set<std::string> validDynamicMetrics_ {
  {"deviceId", "kind", "domain"}
};

const std::unordered_map<std::string, std::unordered_set<std::string>> validMetrics_ {
  {"Marker", {"duration"}},
  {"Kernel", {"duration"}},
  {"API", {"duration"}},
  {"Hccl", {"duration"}},
  {"Memory", {"duration"}},
  {"MemSet", {"duration"}},
  {"MemCpy", {"duration"}}
};

struct MsptiMetricDesc {
  std::string device_id_;
  std::string kind_;
  std::string tag_;
  std::string metric_name_;
  double val_;
};
} // namespace dynolog

#endif