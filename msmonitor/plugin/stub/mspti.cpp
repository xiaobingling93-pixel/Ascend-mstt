#include "mspti.h"

msptiResult msptiSubscribe(msptiSubscriberHandle *subscriber, msptiCallbackFunc callback, void *userdata)
{
    return MSPTI_SUCCESS;
}

msptiResult msptiUnsubscribe(msptiSubscriberHandle subscriber)
{
    return MSPTI_SUCCESS;
}

msptiResult msptiActivityRegisterCallbacks(msptiBuffersCallbackRequestFunc funcBufferRequested, msptiBuffersCallbackCompleteFunc funcBufferCompleted)
{
    return MSPTI_SUCCESS;
}

msptiResult msptiActivityEnable(msptiActivityKind kind)
{
    return MSPTI_SUCCESS;
}

msptiResult msptiActivityDisable(msptiActivityKind kind)
{
    return MSPTI_SUCCESS;
}

msptiResult msptiActivityGetNextRecord(uint8_t *buffer, size_t validBufferSizeBytes, msptiActivity **record)
{
    return MSPTI_SUCCESS;
}

msptiResult msptiActivityFlushAll(uint32_t flag)
{
    return MSPTI_SUCCESS;
}
