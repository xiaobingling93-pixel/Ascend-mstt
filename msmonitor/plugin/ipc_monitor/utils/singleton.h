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
#ifndef SINGLETON_H
#define SINGLETON_H
#include <memory>

namespace dynolog_npu {
namespace ipc_monitor {

template<typename T>
class Singleton {
public:
    static T *GetInstance() noexcept(std::is_nothrow_constructible<T>::value)
    {
        static T instance;
        return &instance;
    }

    virtual ~Singleton() = default;

protected:
    explicit Singleton() = default;

private:
    explicit Singleton(const Singleton &obj) = delete;
    Singleton& operator=(const Singleton &obj) = delete;
    explicit Singleton(Singleton &&obj) = delete;
    Singleton& operator=(Singleton &&obj) = delete;
};

} // ipc_monitor
} // dynolog_npu

#endif