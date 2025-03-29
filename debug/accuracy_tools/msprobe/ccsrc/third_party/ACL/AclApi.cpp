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

#include <dlfcn.h>
#include <map>
#include <stdexcept>

#include "base/ErrorInfos.hpp"
#include "AclApi.hpp"

namespace MindStudioDebugger {
namespace AscendCLApi {

using namespace MindStudioDebugger;

constexpr const char* kLibAscendclName = "libascendcl.so";
constexpr const char* kLibMSAscendName = "libmindspore_ascend.so.2";

using aclInitFuncType = aclError (*)(const char *);
using aclmdlInitDumpFuncType = aclError (*)();
using aclmdlSetDumpFuncType = aclError (*)(const char *);
using aclmdlFinalizeDumpFuncType = aclError (*)();
using acldumpRegCallbackFuncType = aclError (*)(AclDumpCallbackFuncType, int32_t);
using aclrtSynchronizeDeviceFuncType = aclError (*)();

static aclInitFuncType aclInitFunc = nullptr;
static aclmdlInitDumpFuncType aclmdlInitDumpFunc = nullptr;
static aclmdlSetDumpFuncType aclmdlSetDumpFunc = nullptr;
static aclmdlFinalizeDumpFuncType aclmdlFinalizeDumpFunc = nullptr;
static acldumpRegCallbackFuncType acldumpRegCallbackFunc = nullptr;
static aclrtSynchronizeDeviceFuncType aclrtSynchronizeDeviceFunc = nullptr;

DebuggerErrno LoadAclApi()
{
    static void* hLibAscendcl = nullptr;

    if (hLibAscendcl != nullptr) {
        LOG_INFO("No need to load acl api again.");
        return DebuggerErrno::OK;
    }

    hLibAscendcl = dlopen(kLibAscendclName, RTLD_LAZY | RTLD_NOLOAD);
    if (hLibAscendcl == nullptr) {
        LOG_ERROR(DebuggerErrno::ERROR_DEPENDENCY_NOT_FIND,
                  "Failed to search libascendcl.so." + std::string(dlerror()));
        return DebuggerErrno::ERROR_DEPENDENCY_NOT_FIND;
    }

    static const std::map<const char*, void**> functionMap = {
        {"aclInit", reinterpret_cast<void**>(&aclInitFunc)},
        {"aclmdlInitDump", reinterpret_cast<void**>(&aclmdlInitDumpFunc)},
        {"aclmdlSetDump", reinterpret_cast<void**>(&aclmdlSetDumpFunc)},
        {"aclmdlFinalizeDump", reinterpret_cast<void**>(&aclmdlFinalizeDumpFunc)},
        {"aclrtSynchronizeDevice", reinterpret_cast<void**>(&aclrtSynchronizeDeviceFunc)},
    };

    for (auto& iter : functionMap) {
        if (*(iter.second) != nullptr) {
            continue;
        }
        *(iter.second) = dlsym(hLibAscendcl, iter.first);
        if (*(iter.second) == nullptr) {
            LOG_ERROR(DebuggerErrno::ERROR_DEPENDENCY_NOT_FIND, "Failed to load function " +
                      std::string(iter.first) + " from libascendcl.so." + std::string(dlerror()));
            dlclose(hLibAscendcl);
            hLibAscendcl = nullptr;
            return DebuggerErrno::ERROR_DEPENDENCY_NOT_FIND;
        }
        LOG_DEBUG("Load function " + std::string(iter.first) + " from libascendcl.so.");
    }

    /* 规避adump的bug，mindspore场景优先使用libmindspore_ascend.so中的符号 */
    void* handler = dlopen(kLibMSAscendName, RTLD_LAZY | RTLD_NOLOAD);
    std::string libName = kLibMSAscendName;
    if (handler == nullptr) {
        handler = hLibAscendcl;
        libName = kLibAscendclName;
    }

    acldumpRegCallbackFunc = reinterpret_cast<acldumpRegCallbackFuncType>(dlsym(handler, "acldumpRegCallback"));
    if (acldumpRegCallbackFunc == nullptr) {
        LOG_ERROR(DebuggerErrno::ERROR_DEPENDENCY_NOT_FIND, "Failed to load function acldumpRegCallback from " +
                  libName + ".");
    }
    LOG_DEBUG("Load function acldumpRegCallback from " + libName);

    if (handler != hLibAscendcl) {
        dlclose(handler);
    }

    return DebuggerErrno::OK;
}

aclError ACLAPI_aclInit(const char* cfg)
{
    if (aclInitFunc == nullptr) {
        throw std::runtime_error("API aclInit does not have a definition.");
    }
    return aclInitFunc(cfg);
}

aclError ACLAPI_aclmdlInitDump()
{
    if (aclmdlInitDumpFunc == nullptr) {
        throw std::runtime_error("API aclmdlInitDump does not have a definition.");
    }
    return aclmdlInitDumpFunc();
}

aclError ACLAPI_aclmdlSetDump(const char* cfg)
{
    if (aclmdlSetDumpFunc == nullptr) {
        throw std::runtime_error("API aclmdlSetDump does not have a definition.");
    }
    return aclmdlSetDumpFunc(cfg);
}

aclError ACLAPI_aclmdlFinalizeDump()
{
    if (aclmdlFinalizeDumpFunc == nullptr) {
        throw std::runtime_error("API aclmdlFinalizeDump does not have a definition.");
    }
    return aclmdlFinalizeDumpFunc();
}

aclError ACLAPI_acldumpRegCallback(AclDumpCallbackFuncType messageCallback, int32_t flag)
{
    if (acldumpRegCallbackFunc == nullptr) {
        throw std::runtime_error("API acldumpRegCallback does not have a definition.");
    }
    return acldumpRegCallbackFunc(messageCallback, flag);
}

aclError ACLAPI_aclrtSynchronizeDevice()
{
    if (aclrtSynchronizeDeviceFunc == nullptr) {
        throw std::runtime_error("API aclrtSynchronizeDevice does not have a definition.");
    }
    return aclrtSynchronizeDeviceFunc();
}

} 
}
