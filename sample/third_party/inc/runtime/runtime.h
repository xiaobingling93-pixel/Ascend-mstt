#ifndef INC_EXTERNAL_RUNTIME_RUNTIME_H_
#define INC_EXTERNAL_RUNTIME_RUNTIME_H_

#include <cstdint>
#include "stdlib.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t rtError_t;
static const int32_t RT_ERROR_NONE = 0;

typedef struct tagRtSmData {
 uint64_t L2_mirror_addr;
 uint32_t L2_data_section_size;
 uint8_t L2_preload;
 uint8_t modified;
 uint8_t priority;
 int8_t prev_L2_page_offset_base;
 uint8_t L2_page_offset_base;
 uint8_t L2_load_to_ddr;
 uint8_t reserved[2];
} rtSmData_t;

typedef struct tagRtSmCtrl {
 rtSmData_t data[8];
 uint64_t size;
 uint8_t remap[64];
 uint8_t l2_in_main;
 uint8_t reserved[3];
} rtSmDesc_t;

typedef enum rtEventStatus {
 RT_EVENT_INIT = 0,
 RT_EVENT_RECORDED = 1,
} rtEventStatus_t;

typedef struct tagRtDevBinary {
 uint32_t magic;
 uint32_t version;
 const void* data;
 uint64_t length;
} rtDevBinary_t;

typedef void* rtStream_t;

rtError_t rtSetupArgument(const void* args, uint32_t size, uint32_t offset);
rtError_t rtConfigureCall(uint32_t numBlocks, rtSmDesc_t *smDesc, rtStream_t stm);
rtError_t rtDevBinaryUnRegister(void *handle);
rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **handle);
rtError_t rtFunctionRegister(void* binHandle, const void* stubFunc, const char* stubName, const void* devFunc, uint32_t funcMode);
rtError_t rtLaunch(const void* stubFunc);

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_RUNTIME_RUNTIME_H_