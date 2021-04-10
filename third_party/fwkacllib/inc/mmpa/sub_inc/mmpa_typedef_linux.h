/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MMPA_TYPEDEF_LINUX_H
#define MMPA_TYPEDEF_LINUX_H

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif // __cpluscplus
#endif // __cpluscplus

#ifndef FALSE
#define FALSE 0
#endif

#ifndef TRUE
#define TRUE 1
#endif

typedef unsigned char UINT8;
typedef signed char INT8;
typedef unsigned short UINT16;
typedef signed short INT16;
typedef unsigned int UINT32;
typedef signed int INT32;
typedef unsigned long long UINT64;
typedef signed long long INT64;
typedef float FLOAT;
typedef double DOUBLE;
typedef void VOID;
typedef unsigned char UCHAR;
typedef char CHAR;
typedef unsigned short USHORT;
typedef short SHORT;
typedef unsigned int UINT;
typedef int INT;
typedef unsigned long ULONG;
typedef unsigned long long ULONGLONG;

typedef long LONG;

#define HANDLE_INVALID_VALUE (-1)
#define MMPA_MEM_MAX_LEN (0x7fffffff)
#define MMPA_PROCESS_ERROR (0x7fffffff)
#define PATH_SIZE 256
#define MAX_IOVEC_SIZE 32
#define MMPA_MAX_SLEEP_MILLSECOND 4294967
#define MAX_PIPE_COUNT 2
#define MMPA_PIPE_COUNT 2
#define MMPA_THREADNAME_SIZE 16
#define MMPA_MIN_OS_NAME_SIZE 64
#define MMPA_MIN_OS_VERSION_SIZE 128

#define MMPA_ONE_THOUSAND 1000
#define MMPA_ONE_BILLION 1000000000
#define MMPA_COMPUTER_BEGIN_YEAR 1900
#define MMPA_ZERO 0
#define MMPA_MAX_THREAD_PIO 99
#define MMPA_MIN_THREAD_PIO 1
#define MMPA_DEFAULT_PIPE_PERMISSION 0777
#define MMPA_DEFAULT_MSG_TYPE 1

#define MMPA_THREAD_SCHED_RR SCHED_RR
#define MMPA_THREAD_SCHED_FIFO SCHED_FIFO
#define MMPA_THREAD_SCHED_OTHER SCHED_OTHER
#define MMPA_THREAD_MIN_STACK_SIZE PTHREAD_STACK_MIN

#define MM_MUTEX_INITIALIZER PTHREAD_MUTEX_INITIALIZER

#define MMPA_MAX_NI 19
#define MMPA_MIN_NI (-20)

#define EN_OK 0
#define EN_ERR 1
#define EN_ERROR (-1)
#define EN_INVALID_PARAM (-2)
#define EN_TIMEOUT (-3)

#ifdef __cplusplus
#if __cplusplus
}
#endif // __cpluscplus
#endif // __cpluscplus
#endif // MMPA_TYPEDEF_LINUX_H_
