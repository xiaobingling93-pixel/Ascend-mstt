// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "dynolog/src/Metrics.h"

#include <fmt/format.h>
#include <map>

namespace dynolog {

const std::vector<MetricDesc> getAllMetrics()
{
    static std::vector<MetricDesc> metrics_ = {
        {.name = "kindName",
         .type = MetricType::Instant,
         .desc = "Report data kind name"},
        {.name = "duration",
         .type = MetricType::Delta,
         .desc = "Total execution time for corresponding kind"},
        {.name = "timestamp",
         .type = MetricType::Instant,
         .desc = "The timestamp of the reported data"},
        {.name = "deviceId",
         .type = MetricType::Instant,
         .desc = "The ID of the device for reporting data"},
    };
    return metrics_;
}

// These metrics are dynamic per network drive
const std::vector<MetricDesc> getNetworkMetrics()
{
    static std::vector<MetricDesc> metrics_ = {};
    return metrics_;
}

} // namespace dynolog