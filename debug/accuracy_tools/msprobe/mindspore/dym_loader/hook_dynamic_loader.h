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


#ifndef HOOK_DYNAMIC_LOADER_H
#define HOOK_DYNAMIC_LOADER_H

#include <dlfcn.h>
#include <string>
#include <vector>
#include <map>
#include <mutex>

constexpr auto kHookBegin = "MS_DbgOnStepBegin";
constexpr auto kHookEnd = "MS_DbgOnStepEnd";

class HookDynamicLoader {
public:
    static HookDynamicLoader &GetInstance();

    HookDynamicLoader(const HookDynamicLoader &) = delete;
    HookDynamicLoader &operator=(const HookDynamicLoader &) = delete;

    bool LoadLibrary();
    bool UnloadLibrary();
    void *GetHooker(const std::string &funcName);

private:
    // Helper functions
    bool LoadFunction(void *handle, const std::string &functionName);

    HookDynamicLoader() = default;

    void *handle_ = nullptr;
    std::vector<std::string> functionList_ = {kHookBegin, kHookEnd};
    std::map<std::string, void *> funcMap_;
    std::mutex mutex_;
};

#endif  // HOOK_DYNAMIC_LOADER_H
