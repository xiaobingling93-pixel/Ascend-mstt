#include "runtime/runtime.h"

rtError_t rtSetupArgument(const void* args, uint32_t size, uint32_t offset) {return 0;}
rtError_t rtConfigureCall(uint32_t numBlocks, rtSmDesc_t *smDesc, rtStream_t stm){return 0;}
rtError_t rtDevBinaryUnRegister(void *handle){return 0;}
rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **handle){return 0;}
rtError_t rtFunctionRegister(void* binHandle, const void* stubFunc, const char* stubName, const void* devFunc, uint32_t funcMode){return 0;}
rtError_t rtLaunch(const void* stubFunc){return 0;}