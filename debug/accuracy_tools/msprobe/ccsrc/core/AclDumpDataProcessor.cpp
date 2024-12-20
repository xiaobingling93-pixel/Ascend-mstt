/*
 * Copyright (C) 2024-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstring>
#include <thread>
#include <sys/wait.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/types.h>

#include "include/Macro.hpp"
#include "utils/FileUtils.hpp"
#include "utils/FileOperation.hpp"
#include "utils/DataUtils.hpp"
#include "utils/MathUtils.hpp"
#include "core/AclTensor.hpp"
#include "base/ErrorInfos.hpp"
#include "proto/AclDumpMsg.pb.h"
#include "AclDumpDataProcessor.hpp"

namespace MindStudioDebugger {

namespace AclDumpMsg = toolkit::dumpdata;

constexpr size_t kDhaAtomicAddInfoSize = 128;
constexpr size_t kL2AtomicAddInfoSize = 128;
constexpr size_t kAiCoreInfoSize = 256;
constexpr size_t kDhaAtomicAddStatusSize = 256;
constexpr size_t kL2AtomicAddStatusSize = 256;
constexpr size_t kUint64Size = sizeof(uint64_t);
constexpr const char* debugFileSign = "Opdebug.Node_OpDebug.";

constexpr const char* kStatsHeaderInout = "Input/Output";
constexpr const char* kStatsHeaderId = "Index";
constexpr const char* kStatsHeaderDataSize = "Data Size";
constexpr const char* kStatsHeaderDataType = "Data Type";
constexpr const char* kStatsHeaderFormat = "Format";
constexpr const char* kStatsHeaderShape = "Shape";
constexpr const char* kStatsHeaderMax = "Max Value";
constexpr const char* kStatsHeaderMin = "Min Value";
constexpr const char* kStatsHeaderAvg = "Avg Value";
constexpr const char* kStatsHeaderL2Norm = "L2 Norm Value";
constexpr const char* kStatsHeaderMD5 = "MD5 Value";
constexpr const char* kStatsHeaderNan = "Nan Count";
constexpr const char* kStatsHeaderNegInf = "Negative Inf Count";
constexpr const char* kStatsHeaderPosInf = "Positive Inf Count";
constexpr const char* kRankId = "RANK_ID";
constexpr const char* kDigitalNumbers = "0123456789";

static const std::map<DebuggerSummaryOption, std::string> summaryOptionHeaderStrMap = {
    {DebuggerSummaryOption::MAX, kStatsHeaderMax},
    {DebuggerSummaryOption::MIN, kStatsHeaderMin},
    {DebuggerSummaryOption::MEAN, kStatsHeaderAvg},
    {DebuggerSummaryOption::L2NORM, kStatsHeaderL2Norm},
    {DebuggerSummaryOption::NAN_CNT, kStatsHeaderNan},
    {DebuggerSummaryOption::NEG_INF_CNT, kStatsHeaderNegInf},
    {DebuggerSummaryOption::POS_INF_CNT, kStatsHeaderPosInf},
    {DebuggerSummaryOption::MD5, kStatsHeaderMD5},
};

class AclTensorStats {
public:
    AclTensorStats() = default;
    explicit AclTensorStats(const std::map<DebuggerSummaryOption, std::string>& input) : stats(input) {}
    ~AclTensorStats() = default;

    std::string& operator[](DebuggerSummaryOption opt) { return stats[opt]; }
    std::string GetCsvHeader() const;
    std::string GetCsvValue() const;

private:
    std::map<DebuggerSummaryOption, std::string> stats;
};

std::string AclTensorStats::GetCsvHeader() const
{
    std::string ret("");
    for (auto it = stats.begin(); it != stats.end(); it++) {
        ret.append(summaryOptionHeaderStrMap.at(it->first));
        ret.append(",");
    }

    if (!ret.empty()) {
        ret.pop_back();
    }
    return ret;
}

std::string AclTensorStats::GetCsvValue() const
{
    std::string ret("");
    for (auto it = stats.begin(); it != stats.end(); it++) {
        ret.append(it->second);
        ret.append(",");
    }

    if (!ret.empty()) {
        ret.pop_back();
    }
    return ret;
}

AclDumpDataProcessor::~AclDumpDataProcessor()
{
    while (!buffer.empty()) {
        delete buffer.front();
        buffer.pop();
    }
}

std::string AclDumpDataProcessor::ToString() const
{
    return "AclDumpDataProcessor(path=" + dumpPath + ",completed=" + std::to_string(completed) + ",len=" +
           std::to_string(totalLen) + ")";
}

DebuggerErrno AclDumpDataProcessor::PushData(const acldumpChunk *chunk)
{
    DEBUG_FUNC_TRACE();
    if (completed) {
        LOG_WARNING(DebuggerErrno::ERROR_INVALID_OPERATION,
                    ToString() + " receive data when completed. Some errors may occur.");
        return DebuggerErrno::ERROR_INVALID_OPERATION;
    }

    /* 防止最后一包处理出错导致processor残留，此处先设置完成标记位 */
    if (chunk->isLastChunk) {
        completed = true;
    }

    size_t len = chunk->bufLen;
    /* 防止正负翻转 */
    if (SIZE_MAX - len < totalLen || totalLen + len > kMaxDataLen || len == 0) {
        LOG_ERROR(DebuggerErrno::ERROR_BUFFER_OVERFLOW, ToString() + ": buffer overflow(cached size " +
                  std::to_string(totalLen) + ", receiving size " + std::to_string(len) + ").");
        errorOccurred = true;
        return DebuggerErrno::ERROR_BUFFER_OVERFLOW;
    }

    std::vector<uint8_t> *p = new std::vector<uint8_t>(len);
    if (p == nullptr) {
        LOG_ERROR(DebuggerErrno::ERROR_NO_MEMORY, "Acl dump data processor(" +  dumpPath + "): Alloc failed(" +
                  std::to_string(len) + " bytes).");
        errorOccurred = true;
        return DebuggerErrno::ERROR_NO_MEMORY;
    }

    if (memcpy(p->data(), chunk->dataBuf, len) == nullptr) {
        LOG_ERROR(DebuggerErrno::ERROR_SYSCALL_FAILED, ToString() + ": Failed to copy data;");
        delete p;
        errorOccurred = true;
        return DebuggerErrno::ERROR_SYSCALL_FAILED;
    }

    buffer.push(p);
    totalLen += len;
    if (!chunk->isLastChunk) {
        return DebuggerErrno::OK;
    }

    completed = true;
    DebuggerErrno ret = ConcatenateData();
    if (ret != DebuggerErrno::OK) {
        LOG_ERROR(ret, "Acl dump data processor(" +  dumpPath + "): Failed to concatenate data.");
        errorOccurred = true;
        return ret;
    }
    LOG_DEBUG(ToString() + " is completed.");

    return DebuggerErrno::OK;
}

DebuggerErrno AclDumpDataProcessor::ConcatenateData()
{
    DEBUG_FUNC_TRACE();
    if (!completed) {
        LOG_ERROR(DebuggerErrno::ERROR_INVALID_OPERATION, "Acl dump data processor(" +  dumpPath +
                  "): Data is incomplete.");
        return DebuggerErrno::ERROR_INVALID_OPERATION;
    }

    if (buffer.empty()) {
        LOG_ERROR(DebuggerErrno::ERROR_INVALID_VALUE, "Data processor(" +  dumpPath + "): No data.");
        return DebuggerErrno::ERROR_INVALID_VALUE;
    }

    /* 为了减少数据重复拷贝，此处只整合一次，不再剥数据头，用偏移来取数据段 */
    if (buffer.size() > 1) {
        std::vector<uint8_t> *p = new std::vector<uint8_t>(totalLen);
        if (p == nullptr) {
            LOG_ERROR(DebuggerErrno::ERROR_NO_MEMORY, "Alloc failed(" + std::to_string(totalLen) + ").");
            return DebuggerErrno::ERROR_NO_MEMORY;
        }

        size_t offset = 0;
        uint8_t* msg = p->data();
        while (!buffer.empty()) {
            if (memcpy(msg + offset, buffer.front()->data(), buffer.front()->size()) == nullptr) {
                delete p;
                LOG_ERROR(DebuggerErrno::ERROR_SYSCALL_FAILED, "Data processor(" +  dumpPath + "): Failed to copy.");
                return DebuggerErrno::ERROR_SYSCALL_FAILED;
            }
            offset += buffer.front()->size();
            delete buffer.front();
            buffer.pop();
        }
        buffer.push(p);
    }

    if (FileUtils::GetFileSuffix(dumpPath) == CSV_SUFFIX) {
        dataSegOffset = 0;
        dataSegLen = totalLen;
        return DebuggerErrno::OK;
    }

    headerSegOffset = sizeof(uint64_t);
    if (totalLen < headerSegOffset) {
        LOG_ERROR(DebuggerErrno::ERROR_INVALID_FORMAT, "Acl dump data processor(" +  dumpPath +
                  "): Invalid data length " + std::to_string(totalLen) + ".");
        return DebuggerErrno::ERROR_INVALID_FORMAT;
    }

    headerSegLen = *(reinterpret_cast<const uint64_t *>(buffer.front()->data()));
    if (totalLen < headerSegOffset + headerSegLen) {
        LOG_ERROR(DebuggerErrno::ERROR_INVALID_FORMAT, "Acl dump data processor(" +  dumpPath +
                  "): Invalid header len " + std::to_string(headerSegLen) + "/" + std::to_string(totalLen) + ".");
        return DebuggerErrno::ERROR_INVALID_FORMAT;
    }

    dataSegOffset = headerSegOffset + headerSegLen;
    dataSegLen = totalLen - dataSegOffset;
    return DebuggerErrno::OK;
}

static nlohmann::json ParseOverflowInfo(const uint8_t* data)
{
    DEBUG_FUNC_TRACE();
    uint32_t index = 0;
    nlohmann::json overflowInfo;
    uint64_t modelId = DataUtils::UnpackUint64Value_Le(data);
    index += kUint64Size;
    uint64_t streamId = DataUtils::UnpackUint64Value_Le(data + index);
    index += kUint64Size;
    uint64_t taskId = DataUtils::UnpackUint64Value_Le(data + index);
    index += kUint64Size;
    uint64_t taskType = DataUtils::UnpackUint64Value_Le(data + index);
    index += kUint64Size;
    uint64_t pcStart = DataUtils::UnpackUint64Value_Le(data + index);
    index += kUint64Size;
    uint64_t paraBase = DataUtils::UnpackUint64Value_Le(data + index);

    overflowInfo["model_id"] = modelId;
    overflowInfo["stream_id"] = streamId;
    overflowInfo["task_id"] = taskId;
    overflowInfo["task_type"] = taskType;
    overflowInfo["pc_start"] = DataUtils::U64ToHexString(pcStart);
    overflowInfo["para_base"] = DataUtils::U64ToHexString(paraBase);
    return overflowInfo;
}

static DebuggerErrno DumpOpDebugDataToDisk(const std::string& dumpPath, AclDumpMsg::DumpData& dumpData,
                                           const uint8_t* data, size_t dataLen)
{
    DEBUG_FUNC_TRACE();
    std::string outPath = dumpPath + ".output.";
    uint32_t num = dumpData.output().size();
    for (uint32_t slot = 0; slot < num; slot++) {
        uint32_t offset = 0;
        // parse DHA Atomic Add info
        nlohmann::json dhaAtomicAddInfo = ParseOverflowInfo(data + offset);
        offset += kDhaAtomicAddInfoSize;
        // parse L2 Atomic Add info
        nlohmann::json l2AtomicAddInfo = ParseOverflowInfo(data + offset);
        offset += kL2AtomicAddInfoSize;
        // parse AICore info
        nlohmann::json aiCoreInfo = ParseOverflowInfo(data + offset);
        offset += kAiCoreInfoSize;
        // parse DHA Atomic Add status
        dhaAtomicAddInfo["status"] = DataUtils::UnpackUint64Value_Le(data + offset);
        offset += kDhaAtomicAddStatusSize;
        // parse L2 Atomic Add status
        l2AtomicAddInfo["status"] = DataUtils::UnpackUint64Value_Le(data + offset);
        offset += kL2AtomicAddStatusSize;
        // parse AICore status
        uint64_t kernelCode = DataUtils::UnpackUint64Value_Le(data + offset);
        offset += kUint64Size;
        uint64_t blockIdx = DataUtils::UnpackUint64Value_Le(data + offset);
        offset += kUint64Size;
        uint64_t status = DataUtils::UnpackUint64Value_Le(data + offset);
        aiCoreInfo["kernel_code"] = DataUtils::U64ToHexString(kernelCode);
        aiCoreInfo["block_idx"] = blockIdx;
        aiCoreInfo["status"] = status;

        nlohmann::json opdebugData;
        opdebugData["DHA Atomic Add"] = dhaAtomicAddInfo;
        opdebugData["L2 Atomic Add"] = l2AtomicAddInfo;
        opdebugData["AI Core"] = aiCoreInfo;

        // save json to file
        std::string filePath = outPath + std::to_string(slot) + "." + JSON_SUFFIX;
        DebuggerErrno ret = FileOperation::DumpJson(filePath, opdebugData);
        if (ret != DebuggerErrno::OK) {
            LOG_ERROR(ret, "Failed to dump data to " + filePath + ".");
            return ret;
        }
    }
    return DebuggerErrno::OK;
}

static DebuggerErrno ConvertFormatDeviceToHost(AclTensorInfo& tensor)
{
    DEBUG_FUNC_TRACE();
    if (tensor.deviceFmt == tensor.hostFmt || AclTensor::SizeOfTensor(tensor) == 0) {
        LOG_DEBUG(tensor + ": No need to convert format.");
        return DebuggerErrno::OK;
    }

    DebuggerErrno ret = AclTensor::TransFormatD2H(tensor);
    if (ret == DebuggerErrno::ERROR_UNKNOWN_TRANS) {
        LOG_INFO("Do not support convert format from " +
                    std::to_string(tensor.deviceFmt) + " to " + std::to_string(tensor.hostFmt) + ".");
        tensor.hostFmt = tensor.deviceFmt;
        return DebuggerErrno::OK;
    }

    if (ret != DebuggerErrno::OK) {
        LOG_ERROR(ret, tensor + ": Failed to convert format.");
        return ret;
    }

    LOG_DEBUG(tensor + ": Convert format successfully.");
    return DebuggerErrno::OK;
}

static std::string MappingFilePath(const std::string& originPath)
{
    /* adump一次最多传10个tensor数据，输入输出数超过10的算子会分包，但是时序上是连续的，此处缓存上一次的映射 */
    static std::string lastOriName;
    static std::string lastMappingPath;

    if (lastOriName == originPath && !lastMappingPath.empty()) {
        return lastMappingPath;
    }

    std::string dir = FileUtils::GetParentDir(originPath);
    std::string suffix = FileUtils::GetFileSuffix(originPath);
    std::string mappingName;
    uint32_t retry = 10;
    constexpr uint32_t randFileNameLen = 32;
    do {
        mappingName = MathUtils::RandomString(randFileNameLen, '0', '9');
        if (!suffix.empty()) {
            mappingName.append(".").append(suffix);
        }
        if (!FileUtils::IsPathExist(dir + "/" + mappingName)) {
            break;
        }
    } while (--retry);

    if (retry == 0) {
        LOG_ERROR(DebuggerErrno::ERROR, "Failed to map path " + originPath + ".");
        return std::string();
    }

    DebuggerErrno ret;
    FileUtils::CreateDir(dir);
    std::ofstream ofs;
    constexpr const char* mapFileName = "mapping.csv";

    ret = FileUtils::OpenFile(dir + "/" + mapFileName, ofs, std::ofstream::app);
    if (ret != DebuggerErrno::OK) {
        LOG_ERROR(DebuggerErrno::ERROR, "Failed to open mapping file " + dir + "/" + mapFileName + ".");
        return std::string();
    }

    ofs << mappingName << "," << FileUtils::GetFileName(originPath) << "\n";
    if (ofs.fail()) {
        LOG_ERROR(DebuggerErrno::ERROR_FAILED_TO_WRITE_FILE, "Failed to write file " + dir + "/" + mapFileName + ".");
        ofs.close();
        return std::string();
    }
    ofs.close();
    lastOriName = originPath;
    lastMappingPath = dir + "/" + mappingName;
    return lastMappingPath;
}

static DebuggerErrno StandardizedDumpPath(std::string& originPath)
{
    std::string filename = FileUtils::GetFileName(originPath);
    if (filename.length() <= FileUtils::FILE_NAME_MAX) {
        return DebuggerErrno::OK;
    }

    std::string mappingPath = MappingFilePath(originPath);
    if (mappingPath.empty()) {
        LOG_ERROR(DebuggerErrno::ERROR, "Failed to open mapping file " + originPath + ".");
        return DebuggerErrno::ERROR;
    }

    originPath = std::move(mappingPath);
    return DebuggerErrno::OK;
}

static std::string GenDataPath(const std::string& path) {
    LOG_DEBUG("Original acl data path is " + path);
    std::string outputPath = DebuggerConfig::GetInstance().GetOutputPath();
    std::string dataPath;
    if (path.compare(0, outputPath.length(), outputPath) != 0) {
        return path;
    }
    dataPath = path.substr(outputPath.length());
    const std::vector<std::string> items = FileUtils::SplitPath(dataPath);
    constexpr const size_t expectSegLen = 9;
    constexpr const size_t rankIdPos = 0;
    constexpr const size_t timeStampPos = 1;
    constexpr const size_t stepIdPos = 2;
    constexpr const size_t dataNamePos = 8;

    if (items.size() >= expectSegLen) {
        dataPath = outputPath;
        if (dataPath.at(dataPath.length() - 1) != '/') {
            dataPath.append("/");
        }
        /*
        * ACL 接口返回数据的路径格式如下
        * {dump_path}/rank_{rank_id}/{time stamp}/step_{step_id}/{time}/{device_id}/{model_name}/{model_id}/{iteration_id}/{data name}
        * items[0] 表示 rank_{rank_id}
        * items[1] 表示 {time stamp}
        * items[2] 表示 step_{step_id}
        * items[8] 表示 {data name}
        */
        dataPath.append(items[rankIdPos] + "/");
        dataPath.append(items[timeStampPos] + "/");
        dataPath.append(items[stepIdPos] + "/");
        dataPath.append(items[dataNamePos]);
        return dataPath;
    }
    return path;
}

inline std::string GetTensorInfoSuffix(AclTensorInfo& tensor)
{
    return "." + tensor.inout + "." + std::to_string(tensor.slot) +
           "." + DataUtils::GetFormatString(tensor.hostFmt) + "." + DataUtils::GetDTypeString(tensor.dtype);
}

static DebuggerErrno DumpOneAclTensorFmtBin(AclTensorInfo& tensor)
{
    DebuggerErrno ret;
    std::string dumpPathSlot = tensor.dumpPath + GetTensorInfoSuffix(tensor);
    if (StandardizedDumpPath(dumpPathSlot) != DebuggerErrno::OK) {
        LOG_ERROR(DebuggerErrno::ERROR, "Failed to standardize path " + dumpPathSlot + ".");
        return DebuggerErrno::ERROR;
    }

    std::ofstream ofs;
    ret = FileUtils::OpenFile(dumpPathSlot, ofs, std::ios::out | std::ios::binary);
    if (ret != DebuggerErrno::OK) {
        LOG_ERROR(ret, "Failed to open file " + dumpPathSlot + ".");
        return ret;
    }

    ofs.write(reinterpret_cast<const char*>(tensor.aclData), tensor.dataSize);
    if (ofs.fail()) {
        LOG_ERROR(DebuggerErrno::ERROR_FAILED_TO_WRITE_FILE, "Failed to write file " + dumpPathSlot + ".");
        ret = DebuggerErrno::ERROR_FAILED_TO_WRITE_FILE;
    }
    ofs.close();
    return ret;
}


static DebuggerErrno DumpOneAclTensorFmtNpy(AclTensorInfo& tensor)
{
    DEBUG_FUNC_TRACE();
    DebuggerErrno ret;
    if (tensor.dataSize == 0) {
        LOG_INFO(tensor + ": Data size is 0. No need to dump.");
        return DebuggerErrno::OK;
    }

    if (tensor.dtype == AclDtype::DT_BF16) {
        ret = AclTensor::TransDtype(tensor, AclDtype::DT_FLOAT);
        if (ret != DebuggerErrno::OK) {
            LOG_ERROR(ret, tensor + ": Failed to transform dtype from bf16 to fp32.");
            return ret;
        }
    }

    // dump_path: dump_dir/op_type.op_name.task_id.stream_id.timestamp
    std::string dumpPathSlot = tensor.dumpPath + GetTensorInfoSuffix(tensor) +  "." + NPY_SUFFIX;

    if (StandardizedDumpPath(dumpPathSlot) != DebuggerErrno::OK) {
        LOG_ERROR(DebuggerErrno::ERROR, "Failed to standardize path " + dumpPathSlot + ".");
        return DebuggerErrno::ERROR;
    }

    if (tensor.transBuf.empty()) {
        ret = FileOperation::DumpNpy(dumpPathSlot, tensor.aclData, tensor.dataSize, tensor.dtype, tensor.hostShape);
    } else {
        ret = FileOperation::DumpNpy(dumpPathSlot, tensor.transBuf.data(), tensor.transBuf.size(), tensor.dtype,
                                     tensor.hostShape);
    }

    if (ret != DebuggerErrno::OK) {
        LOG_ERROR(ret, tensor + ": Failed to dump as npy.");
        return ret;
    }

    LOG_DEBUG(tensor + ": dump successfully.");

    return ret;
}

static DebuggerErrno WriteOneTensorStatToDisk(const AclTensorInfo& tensor, const AclTensorStats& stat)
{
    DEBUG_FUNC_TRACE();
    static constexpr auto csvHeaderComm = "Input/Output,Index,Data Size,Data Type,Format,Shape";
    std::string dumpPath = tensor.dumpPath;
    std::string csvHeader;
    std::ofstream ofs;
    DebuggerErrno ret;

    if (FileUtils::GetFileSuffix(dumpPath) != CSV_SUFFIX) {
        dumpPath.append(".").append(CSV_SUFFIX);
    }

    if (StandardizedDumpPath(dumpPath) != DebuggerErrno::OK) {
        LOG_ERROR(DebuggerErrno::ERROR, "Failed to standardize path " + dumpPath + ".");
        return DebuggerErrno::ERROR;
    }

    if (FileUtils::IsPathExist(dumpPath)) {
        if (!FileUtils::IsRegularFile(dumpPath)) {
            LOG_ERROR(DebuggerErrno::ERROR_ILLEGAL_FILE_TYPE, dumpPath + " exists and is not a regular file.");
            return DebuggerErrno::ERROR_ILLEGAL_FILE_TYPE;
        }
        ret = FileUtils::OpenFile(dumpPath, ofs, std::ofstream::app);
    } else {
        csvHeader = csvHeaderComm;
        csvHeader.append(",");
        csvHeader.append(stat.GetCsvHeader());
        ret = FileUtils::OpenFile(dumpPath, ofs);
    }

    if (ret != DebuggerErrno::OK) {
        LOG_ERROR(ret, tensor + ": Failed to open file " + dumpPath + ".");
        return ret;
    }

    /* map会根据键值自动排序，此处可以保障头和值的顺序，直接追加写即可 */
    if (!csvHeader.empty()) {
        ofs << csvHeader << '\n';
    }

    ofs << tensor.inout << ',';
    ofs << tensor.slot << ',';
    ofs << tensor.dataSize << ',';
    ofs << DataUtils::GetDTypeString(tensor.dtype) << ',';
    ofs << DataUtils::GetFormatString(tensor.hostFmt) << ',';
    ofs << DataUtils::GetShapeString(tensor.hostShape) << ',';
    ofs << stat.GetCsvValue() << '\n';

    if (ofs.fail()) {
        LOG_ERROR(DebuggerErrno::ERROR_FAILED_TO_WRITE_FILE, tensor + ": Failed to write file " + dumpPath + ".");
        ret = DebuggerErrno::ERROR_FAILED_TO_WRITE_FILE;
    }
    ofs.close();
    return ret;
}

static AclTensorStats CalTensorSummary(AclTensorInfo& tensor, std::vector<DebuggerSummaryOption>& opt)
{
    DEBUG_FUNC_TRACE();
    AclTensorStats stat;
    if (ELE_IN_VECTOR(opt, DebuggerSummaryOption::MD5)) {
        const uint8_t* data = tensor.transBuf.empty() ?  tensor.aclData : tensor.transBuf.data();
        stat[DebuggerSummaryOption::MD5] = MathUtils::CalculateMD5(data, tensor.dataSize);
    }
    return stat;
}

static DebuggerErrno DumpOneAclTensor(AclTensorInfo& tensor, std::vector<DebuggerSummaryOption>& opt)
{
    DEBUG_FUNC_TRACE();
    if (tensor.dumpOriginData || !FileOperation::IsDtypeSupportByNpy(tensor.dtype)) {
        return DumpOneAclTensorFmtBin(tensor);
    }

    DebuggerErrno ret = ConvertFormatDeviceToHost(tensor);
    if (ret != DebuggerErrno::OK) {
        LOG_ERROR(ret, tensor + ": Failed to convert format to host.");
        return ret;
    }

    if (!opt.empty()) {
        AclTensorStats stat = CalTensorSummary(tensor, opt);
        return WriteOneTensorStatToDisk(tensor, stat);
    }

    return DumpOneAclTensorFmtNpy(tensor);
}

static void DumpAclTensor(std::vector<AclTensorInfo>::iterator begin, std::vector<AclTensorInfo>::iterator end,
                          std::vector<DebuggerSummaryOption> opt)
{
    DEBUG_FUNC_TRACE();
    DebuggerErrno ret = DebuggerErrno::OK;
    for (auto it = begin; it != end; it++) {
        ret = DumpOneAclTensor(*it, opt);
        if (ret != DebuggerErrno::OK) {
            LOG_WARNING(ret, *it + ": Failed to dump to disk.");
            break;
        }
    }
    return;
}

static DebuggerErrno DumpTensorDataToDisk(const std::string& dumpPath, AclDumpMsg::DumpData& dumpData,
                                          const uint8_t* data, size_t dataLen, std::vector<DebuggerSummaryOption>& opt)
{
    DEBUG_FUNC_TRACE();
    std::vector<AclTensorInfo> aclTensorInfos;
    uint64_t offset = 0;
    uint32_t slot = 0;
    for (auto& tensor : dumpData.input()) {
        aclTensorInfos.push_back(AclTensor::ParseAttrsFromDumpData(dumpPath, data + offset, tensor, "input", slot));
        offset += tensor.size();
        slot++;
    }

    slot = 0;
    for (auto& tensor : dumpData.output()) {
        aclTensorInfos.push_back(AclTensor::ParseAttrsFromDumpData(dumpPath, data + offset, tensor, "output", slot));
        offset += tensor.size();
        slot++;
    }

    if (aclTensorInfos.empty()) {
        return DebuggerErrno::OK;
    }

    if (offset > dataLen) {
        LOG_ERROR(DebuggerErrno::ERROR_VALUE_OVERFLOW, dumpPath + ": offset overflow " + std::to_string(offset) + "/" +
                  std::to_string(dataLen) + ".");
        return DebuggerErrno::ERROR_VALUE_OVERFLOW;
    }

    /* 根据tensor的数据量，1MB以下串行，1MB以上多线程并发，最大并发量为 最大线程数/4 */
    constexpr int kMaxTensorSize = 1024 * 1024;
    if (offset < kMaxTensorSize) {
        DumpAclTensor(aclTensorInfos.begin(), aclTensorInfos.end(), opt);
    } else {
        size_t concurrent = std::max<size_t>(1, std::thread::hardware_concurrency() / 4);
        concurrent = std::min(concurrent, aclTensorInfos.size());
        size_t total = aclTensorInfos.size();
        size_t batch = MathUtils::DivCeil(total, concurrent);
        size_t cur = 0;
        std::vector<std::thread> threads;
        std::vector<AclTensorInfo>::iterator begin = aclTensorInfos.begin();

        threads.reserve(concurrent);
        while (cur < total) {
            threads.emplace_back(std::thread(&DumpAclTensor, begin + cur, begin + std::min(total, cur + batch), opt));
            cur += batch;
        }

        for (auto& t : threads) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    DebuggerErrLevel err = ErrorInfosManager::GetTopErrLevelInDuration();
    return err >= DebuggerErrLevel::LEVEL_ERROR ? DebuggerErrno::ERROR : DebuggerErrno::OK;
}

static DebuggerErrno DumpStatsDataToDisk(const std::string& dumpPath, const uint8_t* data, size_t dataLen)
{
    DEBUG_FUNC_TRACE();
    std::ofstream ofs;
    DebuggerErrno ret;

    std::string path = dumpPath;
    if (StandardizedDumpPath(path) != DebuggerErrno::OK) {
        LOG_ERROR(DebuggerErrno::ERROR, "Failed to standardize path " + path + ".");
        return DebuggerErrno::ERROR;
    }

    if (FileUtils::IsPathExist(path)) {
        if (!FileUtils::IsRegularFile(path)) {
            LOG_ERROR(DebuggerErrno::ERROR_ILLEGAL_FILE_TYPE, path + " exists and is not a regular file.");
            return DebuggerErrno::ERROR_ILLEGAL_FILE_TYPE;
        }
        ret = FileUtils::OpenFile(path, ofs, std::ofstream::app);
    } else {
        ret = FileUtils::OpenFile(path, ofs);
    }
    if (ret != DebuggerErrno::OK) {
        LOG_ERROR(ret, "Failed to open file " + path + ".");
        return ret;
    }

    /* 统计量模式adump返回的数据就是csv格式的字符流，直接落盘即可 */
    ofs.write(reinterpret_cast<const char*>(data), dataLen);
    if (ofs.fail()) {
        LOG_ERROR(DebuggerErrno::ERROR_FAILED_TO_WRITE_FILE, "Failed to write file " + path + ".");
        ret = DebuggerErrno::ERROR_FAILED_TO_WRITE_FILE;
    }
    ofs.close();
    return ret;
}

DebuggerErrno AclDumpDataProcessor::DumpToDisk()
{
    DEBUG_FUNC_TRACE();
    if (!completed) {
        LOG_ERROR(DebuggerErrno::ERROR_INVALID_OPERATION, ToString() + ": Data is incomplete.");
        return DebuggerErrno::ERROR_INVALID_OPERATION;
    }

    uint8_t* msg = buffer.front()->data();
    AclDumpMsg::DumpData dumpData;
    if (headerSegLen > 0) {
        if (!dumpData.ParseFromArray(msg + headerSegOffset, headerSegLen)) {
            LOG_ERROR(DebuggerErrno::ERROR_INVALID_FORMAT, ToString() + ": Failed to parse header.");
            return DebuggerErrno::ERROR_INVALID_FORMAT;
        }
    }

    const std::string dataPath = GenDataPath(dumpPath);
    DebuggerErrno ret;
    if (FileUtils::GetFileName(dumpPath).find(debugFileSign) == 0 &&
        DebuggerConfig::GetInstance().GetOverflowCheckCfg() != nullptr) {
        ret = DumpOpDebugDataToDisk(dataPath, dumpData, msg + dataSegOffset, dataSegLen);
    } else if (DebuggerConfig::GetInstance().GetStatisticsCfg() != nullptr &&
               hostAnalysisOpts.empty()) {
        ret = DumpStatsDataToDisk(dataPath, msg + dataSegOffset, dataSegLen);
    } else {
        ret = DumpTensorDataToDisk(dataPath, dumpData, msg + dataSegOffset, dataSegLen, hostAnalysisOpts);
    }

    if (ret != DebuggerErrno::OK) {
        LOG_ERROR(DebuggerErrno::ERROR_OPERATION_FAILED, ToString() + ": Failed to dump to disk.");
    }

    return ret;
}

}
