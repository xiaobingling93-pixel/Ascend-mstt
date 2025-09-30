/*
 * Copyright (C) 2025-2025. Huawei Technologies Co., Ltd. All rights reserved.
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

#ifndef DYNOLOG_UTIL_H
#define DYNOLOG_UTIL_H

#include <string>

namespace dynolog {
std::string Rstrip(const std::string &str1, const std::string &str2);
struct PathUtils {
    static bool DirPathCheck(const std::string &path);
    static bool Access(const std::string &path, const int &mode);
    static bool Exist(const std::string &path);
    static bool IsSoftLink(const std::string &path);
    static bool IsFile(const std::string &path);
};
}
#endif
