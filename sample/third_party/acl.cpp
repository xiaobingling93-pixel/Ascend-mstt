#include "acl/acl.h"

aclError aclInit(const char* configPath){return 0;}
aclError aclrtSetDevice(int32_t deviceId){return 0;}
aclError aclrtCreateContext(aclrtContext *context, int32_t deviceId){return 0;}
aclError aclrtCreateStream(aclrtStream* stream){return 0;}
aclError aclrtMallocHost(void** hostPtr, size_t size){return 0;}
aclError aclrtMalloc(void** devPtr, size_t size, aclrtMemMallocPolicy policy){return 0;}
aclError aclrtMemcpy(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind){return 0;}
aclError aclrtFree(void* devPtr){return 0;}
aclError aclrtFreeHost(void* hostPtr){return 0;}
aclError aclrtDestroyStream(aclrtStream stream){return 0;}
aclError aclrtDestroyContext(aclrtContext context){return 0;}
aclError aclrtResetDevice(int32_t deviceId){return 0;}
aclError aclFinalize(){return 0;}
aclError aclrtSynchronizeStream(aclrtStream stream){return 0;}
aclFloat16 aclFloatToFloat16(float value){return 0;}
float aclFloat16ToFloat(aclFloat16 value){return 0;}