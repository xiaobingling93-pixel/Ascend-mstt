/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2014-2021. All rights reserved.
 * Licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * Description: The user of this secure c library should include this header file in you source code.
 *              This header file declare all supported API prototype of the library,
 *              such as memcpy_s, strcpy_s, wcscpy_s,strcat_s, strncat_s, sprintf_s, scanf_s, and so on.
 * Create: 2014-02-25
 * Notes: Do not modify this file by yourself.
 */

#ifndef SECUREC_H_5D13A042_DC3F_4ED9_A8D1_882811274C27
#define SECUREC_H_5D13A042_DC3F_4ED9_A8D1_882811274C27

#include "securectype.h"
#ifndef SECUREC_HAVE_STDARG_H
#define SECUREC_HAVE_STDARG_H 1
#endif

#if SECUREC_HAVE_STDARG_H
#include <stdarg.h>
#endif

#ifndef SECUREC_HAVE_ERRNO_H
#define SECUREC_HAVE_ERRNO_H 1
#endif

/* EINVAL ERANGE may defined in errno.h */
#if SECUREC_HAVE_ERRNO_H
#if SECUREC_IN_KERNEL
#include <linux/errno.h>
#else
#include <errno.h>
#endif
#endif

/* Define error code */
#if defined(SECUREC_NEED_ERRNO_TYPE) || !defined(__STDC_WANT_LIB_EXT1__) || \
    (defined(__STDC_WANT_LIB_EXT1__) && (!__STDC_WANT_LIB_EXT1__))
#ifndef SECUREC_DEFINED_ERRNO_TYPE
#define SECUREC_DEFINED_ERRNO_TYPE
/* Just check whether macrodefinition exists. */
#ifndef errno_t
typedef int errno_t;
#endif
#endif
#endif

/* Success */
#ifndef EOK
#define EOK 0
#endif

#ifndef EINVAL
/* The src buffer is not correct and destination buffer can not be reset */
#define EINVAL 22
#endif

#ifndef EINVAL_AND_RESET
/* Once the error is detected, the dest buffer must be reset! Value is 22 or 128 */
#define EINVAL_AND_RESET 150
#endif

#ifndef ERANGE
/* The destination buffer is not long enough and destination buffer can not be reset */
#define ERANGE 34
#endif

#ifndef ERANGE_AND_RESET
/* Once the error is detected, the dest buffer must be reset! Value is 34 or 128 */
#define ERANGE_AND_RESET  162
#endif

#ifndef EOVERLAP_AND_RESET
/* Once the buffer overlap is detected, the dest buffer must be reset! Value is 54 or 128 */
#define EOVERLAP_AND_RESET 182
#endif

/* If you need export the function of this library in Win32 dll, use __declspec(dllexport) */
#ifndef SECUREC_API
#if defined(SECUREC_DLL_EXPORT)
#if defined(_MSC_VER)
#define SECUREC_API __declspec(dllexport)
#else /* build for linux */
#define SECUREC_API __attribute__((visibility("default")))
#endif /* end of _MSC_VER and SECUREC_DLL_EXPORT */
#elif defined(SECUREC_DLL_IMPORT)
#if defined(_MSC_VER)
#define SECUREC_API __declspec(dllimport)
#else
#define SECUREC_API
#endif /* end of _MSC_VER and SECUREC_DLL_IMPORT */
#else
/*
 * Standardized function declaration. If a security function is declared in the your code,
 * it may cause a compilation alarm,Please delete the security function you declared.
 * Adding extern under windows will cause the system to have inline functions to expand,
 * so do not add the extern in default
 */
#if defined(_MSC_VER)
#define SECUREC_API
#else
#define SECUREC_API extern
#endif
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif
/*
 * Description: The GetHwSecureCVersion function get SecureC Version string and version number.
 * Parameter: verNumber - to store version number (for example value is 0x500 | 0xa)
 * Return:   version string
 */
SECUREC_API const char *GetHwSecureCVersion(unsigned short *verNumber);

#if SECUREC_ENABLE_MEMSET
/*
 * Description: The memset_s function copies the value of c (converted to an unsigned char) into each of
 * the first count characters of the object pointed to by dest.
 * Parameter: dest - destination address
 * Parameter: destMax - The maximum length of destination buffer
 * Parameter: c - the value to be copied
 * Parameter: count - copies count bytes of value to dest
 * Return:    EOK if there was no runtime-constraint violation
 */
SECUREC_API errno_t memset_s(void *dest, size_t destMax, int c, size_t count);
#endif

#ifndef SECUREC_ONLY_DECLARE_MEMSET
#define SECUREC_ONLY_DECLARE_MEMSET     0
#endif

#if !SECUREC_ONLY_DECLARE_MEMSET

#if SECUREC_ENABLE_MEMCPY
/*
 * Description: The memcpy_s function copies n characters from the object pointed to
 * by src into the object pointed to by dest.
 * Parameter: dest - destination  address
 * Parameter: destMax - The maximum length of destination buffer
 * Parameter: src - source address
 * Parameter: count - copies count bytes from the  src
 * Return:    EOK if there was no runtime-constraint violation
 */
SECUREC_API errno_t memcpy_s(void *dest, size_t destMax, const void *src, size_t count);
#endif

#endif

#ifdef __cplusplus
}
#endif
#endif
