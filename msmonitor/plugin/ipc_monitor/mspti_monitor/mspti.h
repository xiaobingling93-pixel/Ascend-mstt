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
#ifndef MSPTI_STUB_H
#define MSPTI_STUB_H

constexpr int ACTIVITY_STRUCT_ALIGNMENT = 8;
#if defined(_WIN32)
#define START_PACKED_ALIGNMENT __pragma(pack(push, 1))
#define PACKED_ALIGNMENT __declspec(align(ACTIVITY_STRUCT_ALIGNMENT))
#define END_PACKED_ALIGNMENT __pragma(pack(pop))
#elif defined(__GNUC__)
#define START_PACKED_ALIGNMENT
#define PACKED_ALIGNMENT __attribute__((__packed__)) __attribute__((aligned(ACTIVITY_STRUCT_ALIGNMENT)))
#define END_PACKED_ALIGNMENT
#else
#define START_PACKED_ALIGNMENT
#define PACKED_ALIGNMENT
#define END_PACKED_ALIGNMENT
#endif

#include <cstdint>
#include <cstddef>

#define MSPTI_INVALID_DEVICE_ID ((uint32_t) 0xFFFFFFFFU)
#define MSPTI_INVALID_STREAM_ID ((uint32_t) 0xFFFFFFFFU)
#define MSPTI_INVALID_CORRELATION_ID ((uint64_t) 0)
using msptiCallbackId = uint32_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef enum {
    MSPTI_SUCCESS = 0,
    MSPTI_ERROR_INVALID_PARAMETER = 1,
    MSPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED = 2,
    MSPTI_ERROR_MAX_LIMIT_REACHED = 3,
    MSPTI_ERROR_DEVICE_OFFLINE = 4,
    MSPTI_ERROR_QUERY_EMPTY = 5,
    MSPTI_ERROR_INNER = 999,
    MSPTI_ERROR_FOECE_INT = 0x7fffffff
} msptiResult;

typedef enum {
    MSPTI_CB_DOMAIN_INVALID = 0,
    MSPTI_CB_DOMAIN_RUNTIME = 1,
    MSPTI_CB_DOMAIN_HCCL = 2,
    MSPTI_CB_DOMAIN_SIZE,
    MSPTI_CB_DOMAIN_FORCE_INT = 0x7fffffff
} msptiCallbackDomain;

typedef enum {
    MSPTI_API_ENTER = 0,
    MSPTI_API_EXIT = 1,
    MSPTI_API_CBSITE_FORCE_INT = 0x7fffffff
} msptiApiCallbackSite;

typedef struct {
    msptiApiCallbackSite callbackSite;
    const char *functionName;
    const void *functionParams;
    const void *functionReturnValue;
    const char *symbolName;
    uint64_t correlationId;
    uint64_t reserved1;
    uint64_t reserved2;
    uint64_t *correlationData;
} msptiCallbackData;

typedef enum {
    MSPTI_ACTIVITY_KIND_INVALID = 0,
    MSPTI_ACTIVITY_KIND_MARKER = 1,
    MSPTI_ACTIVITY_KIND_KERNEL = 2,
    MSPTI_ACTIVITY_KIND_API = 3,
    MSPTI_ACTIVITY_KIND_HCCL = 4,
    MSPTI_ACTIVITY_KIND_MEMORY = 5,
    MSPTI_ACTIVITY_KIND_MEMSET = 6,
    MSPTI_ACTIVITY_KIND_MEMCPY = 7,
    MSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION = 8,
    MSPTI_ACTIVITY_KIND_COMMUNICATION = 9,
    MSPTI_ACTIVITY_KIND_COUNT,
    MSPTI_ACTIVITY_KIND_FORCE_INT = 0x7fffffff
} msptiActivityKind;

typedef enum {
    MSPTI_ACTIVITY_FLAG_NONE = 0,
    MSPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS = 1 << 0,
    MSPTI_ACTIVITY_FLAG_MARKER_START = 1 << 1,
    MSPTI_ACTIVITY_FLAG_MARKER_END = 1 << 2,
    MSPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS_WITH_DEVICE = 1 << 3,
    MSPTI_ACTIVITY_FLAG_MARKER_START_WITH_DEVICE = 1 << 4,
    MSPTI_ACTIVITY_FLAG_MARKER_END_WITH_DEVICE = 1 << 5
} msptiActivityFlag;

typedef enum {
    MSPTI_ACTIVITY_SOURCE_KIND_HOST = 0,
    MSPTI_ACTIVITY_SOURCE_KIND_DEVICE = 1
} msptiActivitySourceKind;

typedef enum {
    MSPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATATION = 0,
    MSPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE = 1
} msptiActivityMemoryOperationType;

typedef enum {
    MSPTI_ACTIVITY_MEMORY_KIND_UNKNOWN = 0,
    MSPTI_ACTIVITY_MEMORY_KIND_DEVICE = 1
} msptiActivityMemoryKind;

typedef enum {
    MSPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN = 0,
    MSPTI_ACTIVITY_MEMCPY_KIND_HTOH = 1,
    MSPTI_ACTIVITY_MEMCPY_KIND_HTOD = 2,
    MSPTI_ACTIVITY_MEMCPY_KIND_DTOH = 3,
    MSPTI_ACTIVITY_MEMCPY_KIND_DTOD = 4,
    MSPTI_ACTIVITY_MEMCPY_KIND_DEFAULT = 5
} msptiActivityMemcpyKind;

typedef enum {
    MSPTI_ACTIVITY_COMMUNICATION_INT8 = 0,
    MSPTI_ACTIVITY_COMMUNICATION_INT16 = 1,
    MSPTI_ACTIVITY_COMMUNICATION_INT32 = 2,
    MSPTI_ACTIVITY_COMMUNICATION_FP16 = 3,
    MSPTI_ACTIVITY_COMMUNICATION_FP32 = 4,
    MSPTI_ACTIVITY_COMMUNICATION_INT64 = 5,
    MSPTI_ACTIVITY_COMMUNICATION_UINT64 = 6,
    MSPTI_ACTIVITY_COMMUNICATION_UINT8 = 7,
    MSPTI_ACTIVITY_COMMUNICATION_UINT16 = 8,
    MSPTI_ACTIVITY_COMMUNICATION_UINT32 = 9,
    MSPTI_ACTIVITY_COMMUNICATION_FP64 = 10,
    MSPTI_ACTIVITY_COMMUNICATION_BFP16 = 11,
    MSPTI_ACTIVITY_COMMUNICATION_INT128 = 12,
    MSPTI_ACTIVITY_COMMUNICATION_INVALID_TYPE = 0x0000FFFF
} msptiCommunicationDataType;

START_PACKED_ALIGNMENT

typedef union PACKED_ALIGNMENT {
    struct {
        uint32_t processId;
        uint32_t threadId;
    } pt;
    struct {
        uint32_t deviceId;
        uint32_t streamId;
    } ds;
} msptiObjectId;

typedef struct PACKED_ALIGNMENT {
    msptiActivityKind kind;
} msptiActivity;

typedef struct PACKED_ALIGNMENT {
    msptiActivityKind kind;
    uint64_t start;
    uint64_t end;
    struct {
        uint32_t processId;
        uint32_t threadId;
    } pt;
    uint64_t correlationId;
    const char* name;
} msptiActivityApi;

typedef struct PACKED_ALIGNMENT {
    msptiActivityKind kind;
    uint64_t start;
    uint64_t end;
    struct {
        uint32_t deviceId;
        uint32_t streamId;
    } ds;
    uint64_t correlationId;
    const char *type;
    const char *name;
} msptiActivityKernel;

typedef struct PACKED_ALIGNMENT {
    msptiActivityKind kind;
    msptiActivityFlag flag;
    msptiActivitySourceKind sourceKind;
    uint64_t timestamp;
    uint64_t id;
    msptiObjectId objectId;
    const char *name;
    const char *domain;
} msptiActivityMarker;

typedef struct PACKED_ALIGNMENT {
    msptiActivityKind kind;
    uint64_t start;
    uint64_t end;
    struct {
        uint32_t deviceId;
        uint32_t streamId;
    } ds;
    double bandWidth;
    const char *name;
    const char *commName;
} msptiActivityHccl;

typedef struct PACKED_ALIGNMENT {
    msptiActivityKind kind;
    msptiActivityMemoryOperationType memoryOperationType;
    msptiActivityMemoryKind memoryKind;
    uint64_t correlationId;
    uint64_t start;
    uint64_t end;
    uint64_t address;
    uint64_t bytes;
    uint32_t processId;
    uint32_t deviceId;
    uint32_t streamId;
} msptiActivityMemory;

typedef struct PACKED_ALIGNMENT {
    msptiActivityKind kind;
    uint32_t value;
    uint64_t bytes;
    uint64_t start;
    uint64_t end;
    uint32_t deviceId;
    uint32_t streamId;
    uint64_t correlationId;
    uint8_t isAsync;
} msptiActivityMemset;

typedef struct PACKED_ALIGNMENT {
    msptiActivityKind kind;
    msptiActivityMemcpyKind copyKind;
    uint64_t bytes;
    uint64_t start;
    uint64_t end;
    uint32_t deviceId;
    uint32_t streamId;
    uint64_t correlationId;
    uint8_t isAsync;
} msptiActivityMemcpy;

typedef struct PACKED_ALIGNMENT {
    msptiActivityKind kind;
    msptiCommunicationDataType dataType;
    uint64_t count;
    struct {
        uint32_t deviceId;
        uint32_t streamId;
    } ds;
    uint64_t start;
    uint64_t end;
    const char* algType;
    const char* name;
    const char* commName;
    uint64_t correlationId;
} msptiActivityCommunication;

END_PACKED_ALIGNMENT

typedef void(*msptiCallbackFunc)(void* userdata, msptiCallbackDomain domain, msptiCallbackId cbid, const msptiCallbackData *cbdata);
typedef void(*msptiBuffersCallbackRequestFunc)(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
typedef void(*msptiBuffersCallbackCompleteFunc)(uint8_t *buffer, size_t size, size_t validSize);

struct msptiSubscriber_st {
    msptiCallbackFunc callback;
    void *userdata;
};

typedef struct msptiSubscriber_st *msptiSubscriberHandle;

msptiResult msptiSubscribe(msptiSubscriberHandle *subscriber, msptiCallbackFunc callback, void *userdata);
msptiResult msptiUnsubscribe(msptiSubscriberHandle subscriber);
msptiResult msptiActivityRegisterCallbacks(msptiBuffersCallbackRequestFunc funcBufferRequested, msptiBuffersCallbackCompleteFunc funcBufferCompleted);
msptiResult msptiActivityEnable(msptiActivityKind kind);
msptiResult msptiActivityDisable(msptiActivityKind kind);
msptiResult msptiActivityGetNextRecord(uint8_t *buffer, size_t validBufferSizeBytes, msptiActivity **record);
msptiResult msptiActivityFlushAll(uint32_t flag);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // MSPTI_STUB_H
