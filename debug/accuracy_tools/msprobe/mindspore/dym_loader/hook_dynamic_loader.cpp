/* -------------------------------------------------------------------------
 *  This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          `http://license.coscl.org.cn/MulanPSL2`
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * ------------------------------------------------------------------------- */


#include "hook_dynamic_loader.h"
#include <sys/stat.h>
#include <cstdlib>
#include <cstring>
#include <pybind11/embed.h>
#include "utils/log_adapter.h"

namespace py = pybind11;

HookDynamicLoader &HookDynamicLoader::GetInstance()
{
    static HookDynamicLoader instance;
    return instance;
}

bool HookDynamicLoader::LoadFunction(void *handle, const std::string &functionName)
{
    void *func = dlsym(handle, functionName.c_str());
    if (!func) {
        MS_LOG(WARNING) << "Could not load function: " << functionName << ", error: " << dlerror();
        return false;
    }
    funcMap_[functionName] = func;
    return true;
}

bool HookDynamicLoader::LoadLibrary()
{
    std::string msprobePath = "";
    // 获取gil锁
    py::gil_scoped_acquire acquire;
    try {
        py::module msprobeMod = py::module::import("msprobe.lib._msprobe_c");
        if (!py::hasattr(msprobeMod, "__file__")) {
            MS_LOG(WARNING) << "Adump mod not found";
            return false;
    }
    msprobePath = msprobeMod.attr("__file__").cast<std::string>();
    } catch (const std::exception& e) {
    MS_LOG(WARNING) << "Adump mod path unable to get: " << e.what();
    return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (handle_) {
        MS_LOG(WARNING) << "Hook library already loaded!";
        return false;
    }
    if (msprobePath == "") {
        MS_LOG(WARNING) << "Adump path not loaded";
        return false;
    }
    handle_ = dlopen(msprobePath.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (!handle_) {
        MS_LOG(WARNING) << "Failed to load Hook library: " << dlerror();
        return false;
    }

    for (const auto &functionName : functionList_) {
        if (!LoadFunction(handle_, functionName)) {
            MS_LOG(WARNING) << "Failed to load adump function";
            dlclose(handle_);
            handle_ = nullptr;
            return false;
        }
    }

    MS_LOG(INFO) << "Hook library loaded successfully.";
    return true;
}

bool HookDynamicLoader::UnloadLibrary()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!handle_) {
        MS_LOG(WARNING) << "Hook library hasn't been loaded.";
        return false;
    }

    dlclose(handle_);
    handle_ = nullptr;
    funcMap_.clear();
    MS_LOG(INFO) << "Library unloaded successfully.";
    return true;
}

void *HookDynamicLoader::GetHooker(const std::string &funcName)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = funcMap_.find(funcName);
    if (iter == funcMap_.end()) {
        MS_LOG(WARNING) << "Function not found: " << funcName;
        return nullptr;
    }
    return iter->second;
}
