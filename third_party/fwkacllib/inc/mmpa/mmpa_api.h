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

#ifndef _MMPA_API_H_
#define _MMPA_API_H_

#define LINUX 0
#define WIN 1

#if(OS_TYPE == LINUX) //lint !e553

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifdef FUNC_VISIBILITY
#define MMPA_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define MMPA_FUNC_VISIBILITY
#endif

#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include <semaphore.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <signal.h>
#include <pthread.h>
#include <syslog.h>
#include <dirent.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <string.h>
#include <poll.h>
#include <net/if.h>
#include <stdarg.h>
#include <limits.h>
#include <ctype.h>
#include <stddef.h>
#include <dirent.h>
#include <getopt.h>
#include <libgen.h>

#include <linux/types.h>
#include <linux/hdreg.h>
#include <linux/fs.h>
#include <linux/limits.h>

#include <sys/stat.h>
#include <sys/time.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/resource.h>
#include <sys/uio.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/shm.h>
#include <sys/un.h>
#include <sys/utsname.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#include <sys/shm.h>
#include <sys/msg.h>
#include <sys/wait.h>
#include <sys/statvfs.h>
#include <sys/prctl.h>
#include <sys/inotify.h>

#include "securec.h"

#include "./sub_inc/mmpa_typedef_linux.h"
#include "./sub_inc/mmpa_linux.h"

#endif


#if(OS_TYPE == WIN) //lint !e553

#ifdef FUNC_VISIBILITY
#define MMPA_FUNC_VISIBILITY _declspec(dllexport)
#else
#define MMPA_FUNC_VISIBILITY
#endif

#include <winsock2.h>
#include <winsock.h>
#include "Windows.h"
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <io.h>
#include <stdio.h>
#include <ws2tcpip.h>
#include <winioctl.h>
#include <WinBase.h>
#include <mswsock.h>
#include <strsafe.h>
#include <signal.h>
#include <time.h>
#include <stdarg.h>
#include "shlwapi.h"
#include <direct.h>
#include <VersionHelpers.h>
#include <processthreadsapi.h>
#include <Wbemidl.h>
#include <iphlpapi.h>
#include <synchapi.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "securec.h"

#include "sub_inc/mmpa_typedef_win.h"
#include "sub_inc/mmpa_win.h"

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "mswsock.lib")
#pragma comment(lib, "Kernel32.lib")
#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "wbemuuid.lib")
#pragma comment(lib, "Iphlpapi.lib")
#endif

#endif // MMPA_API_H_

