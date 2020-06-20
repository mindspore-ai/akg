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

#ifndef MMPA_LINUX_MMPA_LINUX_H
#define MMPA_LINUX_MMPA_LINUX_H

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif  // __cpluscplus
#endif  // __cpluscplus

#define MMPA_MACINFO_DEFAULT_SIZE 18
#define MMPA_CPUDESC_DEFAULT_SIZE 64

typedef pthread_t mmThread;
typedef pthread_mutex_t mmMutex_t;
typedef pthread_cond_t mmCond;
typedef pthread_mutex_t mmMutexFC;
typedef signed int mmProcess;
typedef int mmPollHandle;
typedef int mmPipeHandle;
typedef int mmComPletionKey;
typedef int mmCompletionHandle;

typedef VOID *mmExitCode;
typedef key_t mmKey_t;
typedef int mmMsgid;
typedef struct dirent mmDirent;
typedef int (*mmFilter)(const mmDirent *entry);
typedef int (*mmSort)(const mmDirent **a, const mmDirent **b);

typedef VOID *(*userProcFunc)(VOID *pulArg);

typedef struct {
  userProcFunc procFunc;  // Callback function pointer
  VOID *pulArg;           // Callback function parameters
} mmUserBlock_t;

typedef struct {
  int wSecond;             // Seconds. [0-60] (1 leap second)
  int wMinute;             // Minutes. [0-59]
  int wHour;               // Hours. [0-23]
  int wDay;                // Day. [1-31]
  int wMonth;              // Month. [1-12]
  int wYear;               // Year
  int wDayOfWeek;          // Day of week. [0-6]
  int tm_yday;             // Days in year.[0-365]
  int tm_isdst;            // DST. [-1/0/1]
  long int wMilliseconds;  // milliseconds
} mmSystemTime_t;

typedef sem_t mmSem_t;
typedef struct sockaddr mmSockAddr;
typedef socklen_t mmSocklen_t;
typedef int mmSockHandle;
typedef timer_t mmTimer;
typedef pthread_key_t mmThreadKey;

typedef int mmOverLap;

typedef ssize_t mmSsize_t;

typedef struct {
  UINT32 createFlag;
  INT32 oaFlag;
} mmCreateFlag;

typedef struct {
  VOID *sendBuf;
  INT32 sendLen;
} mmIovSegment;
typedef struct in_addr mmInAddr;

typedef struct {
  VOID *inbuf;
  INT32 inbufLen;
  VOID *outbuf;
  INT32 outbufLen;
  mmOverLap *oa;
} mmIoctlBuf;

typedef int mmAtomicType;

typedef enum {
  pollTypeRead = 1,  // pipe read
  pollTypeRecv,      // socket recv
  pollTypeIoctl,     // ioctl
} mmPollType;

typedef struct {
  mmPollHandle handle;            // The file descriptor or handle of poll is required
  mmPollType pollType;            // Operation type requiring poll
                                  // read or recv or ioctl
  INT32 ioctlCode;                // IOCTL operation code, dedicated to IOCTL
  mmComPletionKey completionKey;  // The default value is blank, which is used in windows
                                  // The data used to receive the difference between which handle is readable
} mmPollfd;

typedef struct {
  VOID *priv;              // User defined private content
  mmPollHandle bufHandle;  // Value of handle corresponding to buf
  mmPollType bufType;      // Data types polled to
  VOID *buf;               // Data used in poll
  UINT32 bufLen;           // Data length used in poll
  UINT32 bufRes;           // Actual return length
} mmPollData, *pmmPollData;

typedef VOID (*mmPollBack)(pmmPollData);

typedef struct {
  INT32 tz_minuteswest;  // How many minutes is it different from Greenwich
  INT32 tz_dsttime;      // type of DST correction
} mmTimezone;

typedef struct {
  LONG tv_sec;
  LONG tv_usec;
} mmTimeval;

typedef struct {
  LONG tv_sec;
  LONG tv_nsec;
} mmTimespec;

typedef struct {
  ULONGLONG totalSize;
  ULONGLONG freeSize;
  ULONGLONG availSize;
} mmDiskSize;

#define mmTLS __thread
typedef struct stat mmStat_t;
typedef struct stat64 mmStat64_t;
typedef mode_t mmMode_t;

typedef struct option mmStructOption;

typedef struct {
  char addr[MMPA_MACINFO_DEFAULT_SIZE];  // ex:aa-bb-cc-dd-ee-ff\0
} mmMacInfo;

typedef struct {
  char **argv;
  INT32 argvCount;
  char **envp;
  INT32 envpCount;
} mmArgvEnv;

typedef struct {
  char arch[MMPA_CPUDESC_DEFAULT_SIZE];
  char manufacturer[MMPA_CPUDESC_DEFAULT_SIZE];  // vendor
  char version[MMPA_CPUDESC_DEFAULT_SIZE];       // modelname
  INT32 frequency;                               // cpu frequency
  INT32 maxFrequency;                            // max speed
  INT32 ncores;                                  // cpu cores
  INT32 nthreads;                                // cpu thread count
  INT32 ncounts;                                 // logical cpu nums
} mmCpuDesc;

typedef mode_t MODE;

typedef struct {
  INT32 detachFlag;    // Determine whether to set separation property 0, not to separate 1
  INT32 priorityFlag;  // Determine whether to set priority 0 and not set 1
  INT32 priority;      // Priority value range to be set 1-99
  INT32 policyFlag;    // Set scheduling policy or not 0 do not set 1 setting
  INT32 policy;        // Scheduling policy value value
                       //  MMPA_THREAD_SCHED_RR
                       //  MMPA_THREAD_SCHED_OTHER
                       //  MMPA_THREAD_SCHED_FIFO
  INT32 stackFlag;     // Set stack size or not: 0 does not set 1 setting
  UINT32 stackSize;    // The stack size unit bytes to be set cannot be less than MMPA_THREAD_STACK_MIN
} mmThreadAttr;

#ifdef __ANDROID__
#define S_IREAD S_IRUSR
#define S_IWRITE S_IWUSR
#endif

#define M_FILE_RDONLY O_RDONLY
#define M_FILE_WRONLY O_WRONLY
#define M_FILE_RDWR O_RDWR
#define M_FILE_CREAT O_CREAT

#define M_RDONLY O_RDONLY
#define M_WRONLY O_WRONLY
#define M_RDWR O_RDWR
#define M_CREAT O_CREAT
#define M_BINARY O_RDONLY

#define M_IREAD S_IREAD
#define M_IRUSR S_IRUSR
#define M_IWRITE S_IWRITE
#define M_IWUSR S_IWUSR
#define M_IXUSR S_IXUSR
#define FDSIZE 64
#define M_MSG_CREAT IPC_CREAT
#define M_MSG_EXCL (IPC_CREAT | IPC_EXCL)
#define M_MSG_NOWAIT IPC_NOWAIT

#define M_WAIT_NOHANG WNOHANG  // Non blocking waiting
#define M_WAIT_UNTRACED \
  WUNTRACED  // If the subprocess enters the suspended state, it will return immediately
             // But the end state of the subprocess is ignored
#define M_UMASK_USRREAD S_IRUSR
#define M_UMASK_GRPREAD S_IRGRP
#define M_UMASK_OTHREAD S_IROTH

#define M_UMASK_USRWRITE S_IWUSR
#define M_UMASK_GRPWRITE S_IWGRP
#define M_UMASK_OTHWRITE S_IWOTH

#define M_UMASK_USREXEC S_IXUSR
#define M_UMASK_GRPEXEC S_IXGRP
#define M_UMASK_OTHEXEC S_IXOTH

#define mmConstructor(x) __attribute__((constructor)) VOID x()
#define mmDestructor(x) __attribute__((destructor)) VOID x()

#define MMPA_NO_ARGUMENT 0
#define MMPA_REQUIRED_ARGUMENT 1
#define MMPA_OPTIONAL_ARGUMENT 2

#define MMPA_MAX_PATH PATH_MAX

#define M_F_OK F_OK
#define M_R_OK R_OK
#define M_W_OK W_OK

#define MMPA_RTLD_NOW RTLD_NOW
#define MMPA_RTLD_GLOBAL RTLD_GLOBAL

#define MMPA_DL_EXT_NAME ".so"

extern INT32 mmCreateTask(mmThread *threadHandle, mmUserBlock_t *funcBlock);
extern INT32 mmJoinTask(mmThread *threadHandle);
extern INT32 mmMutexInit(mmMutex_t *mutex);
extern INT32 mmMutexLock(mmMutex_t *mutex);
extern INT32 mmMutexUnLock(mmMutex_t *mutex);
extern INT32 mmMutexDestroy(mmMutex_t *mutex);
extern INT32 mmCondInit(mmCond *cond);
extern INT32 mmCondLockInit(mmMutexFC *mutex);
extern INT32 mmCondLock(mmMutexFC *mutex);
extern INT32 mmCondUnLock(mmMutexFC *mutex);
extern INT32 mmCondLockDestroy(mmMutexFC *mutex);
extern INT32 mmCondWait(mmCond *cond, mmMutexFC *mutex);
extern INT32 mmCondTimedWait(mmCond *cond, mmMutexFC *mutex, UINT32 milliSecond);
extern INT32 mmCondNotify(mmCond *cond);
extern INT32 mmCondNotifyAll(mmCond *cond);
extern INT32 mmCondDestroy(mmCond *cond);
extern INT32 mmGetPid();
extern INT32 mmGetTid();
extern INT32 mmGetPidHandle(mmProcess *processHandle);
extern INT32 mmGetLocalTime(mmSystemTime_t *sysTime);

extern INT32 mmSemInit(mmSem_t *sem, UINT32 value);
extern INT32 mmSemWait(mmSem_t *sem);
extern INT32 mmSemPost(mmSem_t *sem);
extern INT32 mmSemDestroy(mmSem_t *sem);
extern INT32 mmOpen(const CHAR *pathName, INT32 flags);
extern INT32 mmOpen2(const CHAR *pathName, INT32 flags, MODE mode);
extern INT32 mmClose(INT32 fd);
extern mmSsize_t mmWrite(INT32 fd, VOID *buf, UINT32 bufLen);
extern mmSsize_t mmRead(INT32 fd, VOID *buf, UINT32 bufLen);
extern mmSockHandle mmSocket(INT32 sockFamily, INT32 type, INT32 protocol);
extern INT32 mmBind(mmSockHandle sockFd, mmSockAddr *addr, mmSocklen_t addrLen);
extern INT32 mmListen(mmSockHandle sockFd, INT32 backLog);
extern mmSockHandle mmAccept(mmSockHandle sockFd, mmSockAddr *addr, mmSocklen_t *addrLen);
extern INT32 mmConnect(mmSockHandle sockFd, mmSockAddr *addr, mmSocklen_t addrLen);
extern INT32 mmCloseSocket(mmSockHandle sockFd);
extern mmSsize_t mmSocketSend(mmSockHandle sockFd, VOID *sendBuf, INT32 sendLen, INT32 sendFlag);
extern mmSsize_t mmSocketRecv(mmSockHandle sockFd, VOID *recvBuf, INT32 recvLen, INT32 recvFlag);
extern INT32 mmSAStartup();
extern INT32 mmSACleanup();
extern VOID *mmDlopen(const CHAR *fileName, INT32 mode);
extern VOID *mmDlsym(VOID *handle, CHAR *funcName);
extern INT32 mmDlclose(VOID *handle);
extern CHAR *mmDlerror();
extern INT32 mmCreateAndSetTimer(mmTimer *timerHandle, mmUserBlock_t *timerBlock, UINT milliSecond, UINT period);
extern INT32 mmDeleteTimer(mmTimer timerHandle);
extern INT32 mmStatGet(const CHAR *path, mmStat_t *buffer);
extern INT32 mmStat64Get(const CHAR *path, mmStat64_t *buffer);
extern INT32 mmMkdir(const CHAR *pathName, mmMode_t mode);
extern INT32 mmSleep(UINT32 milliSecond);

extern INT32 mmCreateTaskWithAttr(mmThread *threadHandle, mmUserBlock_t *funcBlock);
extern INT32 mmGetProcessPrio(mmProcess pid);
extern INT32 mmSetProcessPrio(mmProcess pid, INT32 processPrio);
extern INT32 mmGetThreadPrio(mmThread *threadHandle);
extern INT32 mmSetThreadPrio(mmThread *threadHandle, INT32 threadPrio);
extern INT32 mmAccess(const CHAR *pathName);
extern INT32 mmAccess2(const CHAR *pathName, INT32 mode);
extern INT32 mmRmdir(const CHAR *pathName);

extern INT32 mmIoctl(mmProcess fd, INT32 ioctlCode, mmIoctlBuf *bufPtr);
extern INT32 mmSemTimedWait(mmSem_t *sem, INT32 timeout);
extern mmSsize_t mmWritev(mmProcess fd, mmIovSegment *iov, INT32 iovcnt);
extern VOID mmMb();
extern INT32 mmInetAton(const CHAR *addrStr, mmInAddr *addr);

extern mmProcess mmOpenFile(const CHAR *fileName, UINT32 access, mmCreateFlag fileFlag);
extern mmSsize_t mmReadFile(mmProcess fileId, VOID *buffer, INT32 len);
extern mmSsize_t mmWriteFile(mmProcess fileId, VOID *buffer, INT32 len);
extern INT32 mmCloseFile(mmProcess fileId);

extern mmAtomicType mmSetData(mmAtomicType *ptr, mmAtomicType value);
extern mmAtomicType mmValueInc(mmAtomicType *ptr, mmAtomicType value);
extern mmAtomicType mmValueSub(mmAtomicType *ptr, mmAtomicType value);
extern INT32 mmCreateTaskWithDetach(mmThread *threadHandle, mmUserBlock_t *funcBlock);

// The following 3 interfaces are to be deleted
extern INT32 mmCreateNamedPipe(mmPipeHandle pipe[], CHAR *pipeName[], INT32 waitMode);
extern INT32 mmOpenNamePipe(mmPipeHandle pipe[], CHAR *pipeName[], INT32 waitMode);
extern VOID mmCloseNamedPipe(mmPipeHandle namedPipe[]);

extern INT32 mmCreatePipe(mmPipeHandle pipe[], CHAR *pipeName[], UINT32 pipeCount, INT32 waitMode);
extern INT32 mmOpenPipe(mmPipeHandle pipe[], CHAR *pipeName[], UINT32 pipeCount, INT32 waitMode);
extern VOID mmClosePipe(mmPipeHandle pipe[], UINT32 pipeCount);

// Poll related interface
extern mmCompletionHandle mmCreateCompletionPort();
extern VOID mmCloseCompletionPort(mmCompletionHandle handle);
extern INT32 mmPoll(mmPollfd *fds, INT32 fdCount, INT32 timeout, mmCompletionHandle handleIOCP, pmmPollData polledData,
                    mmPollBack pollBack);
extern INT32 mmGetErrorCode();
extern INT32 mmGetTimeOfDay(mmTimeval *timeVal, mmTimezone *timeZone);
extern mmTimespec mmGetTickCount();
extern INT32 mmGetRealPath(CHAR *path, CHAR *realPath);
extern INT32 mmRealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen);

extern INT32 mmDup2(INT32 oldFd, INT32 newFd);

extern INT32 mmUnlink(const CHAR *filename);

extern INT32 mmChmod(const CHAR *filename, INT32 mode);

extern INT32 mmFileno(FILE *stream);

extern INT32 mmScandir(const CHAR *path, mmDirent ***entryList, mmFilter filterFunc, mmSort sort);

extern VOID mmScandirFree(mmDirent **entryList, INT32 count);

extern mmMsgid mmMsgCreate(mmKey_t key, INT32 msgFlag);

extern mmMsgid mmMsgOpen(mmKey_t key, INT32 msgFlag);

extern INT32 mmMsgSnd(mmMsgid msqid, VOID *buf, INT32 bufLen, INT32 msgFlag);

extern INT32 mmMsgRcv(mmMsgid msqid, VOID *buf, INT32 bufLen, INT32 msgFlag);

extern INT32 mmMsgClose(mmMsgid msqid);

extern INT32 mmLocalTimeR(const time_t *timep, struct tm *result);

extern INT32 mmGetOpt(INT32 argc, char *const *argv, const char *opts);
extern INT32 mmGetOptLong(INT32 argc, char *const *argv, const char *opts, const mmStructOption *longOpts,
                          INT32 *longIndex);

extern LONG mmLseek(INT32 fd, INT64 offset, INT32 seekFlag);
extern INT32 mmFtruncate(mmProcess fd, UINT32 length);

extern INT32 mmTlsCreate(mmThreadKey *key, VOID (*destructor)(VOID *));
extern INT32 mmTlsSet(mmThreadKey key, const VOID *value);
extern VOID *mmTlsGet(mmThreadKey key);
extern INT32 mmTlsDelete(mmThreadKey key);
extern INT32 mmGetOsType();

extern INT32 mmFsync(mmProcess fd);
extern INT32 mmChdir(const CHAR *path);
extern INT32 mmUmask(INT32 pmode);
extern INT32 mmThreadKill(mmThread id);
extern INT32 mmWaitPid(mmProcess pid, INT32 *status, INT32 options);

extern INT32 mmGetCwd(CHAR *buffer, INT32 maxLen);
extern INT32 mmGetEnv(const CHAR *name, CHAR *value, UINT32 len);
extern INT32 mmSetEnv(const CHAR *name, const CHAR *value, INT32 overwrite);
extern CHAR *mmStrTokR(CHAR *str, const CHAR *delim, CHAR **saveptr);
extern CHAR *mmDirName(CHAR *path);
extern CHAR *mmBaseName(CHAR *path);
extern INT32 mmGetDiskFreeSpace(const char *path, mmDiskSize *diskSize);

/*
 * Function: set the thread name created by mmcreatetask
 * Input: pstThreadHandle: thread ID
 *  name: thread name, the actual length of name must be < MMPA_THREADNAME_SIZE
 * The input parameter error returns EN_INVALID_PARAM, the execution success returns EN_OK, and the
 * execution failure returns EN_ERROR
 */
extern INT32 mmSetThreadName(mmThread *threadHandle, const CHAR *name);

/*
 * Function: get thread name
 * Input: pstThreadHandle: thread ID
 *      size: Cache length of thread name
 *  name:User allocated cache for thread name, Cache length must be >= MMPA_THREADNAME_SIZE
 * The input parameter error returns EN_INVALID_PARAM, the execution success returns EN_OK, and the
 * execution failure returns EN_ERROR
 */
extern INT32 mmGetThreadName(mmThread *threadHandle, CHAR *name, INT32 size);
/*
 * Function:Set the thread name of the currently executing thread - call inside the thread body
 * Input:name:Thread name to be set
 * The input parameter error returns EN_INVALID_PARAM, the execution success returns EN_OK, and the
 * execution failure returns EN_ERROR
 */
extern INT32 mmSetCurrentThreadName(const CHAR *name);
/*
 * Function:Get the thread name of the currently executing thread - in body call
 * Input:name:The name of the thread to get, and the cache is allocated by the user，size>=MMPA_THREADNAME_SIZE
 * The input parameter error returns EN_INVALID_PARAM, the execution success returns EN_OK, and the
 * execution failure returns EN_ERROR
 */
extern INT32 mmGetCurrentThreadName(CHAR *name, INT32 size);
extern INT32 mmGetFileSize(const CHAR *fileName, ULONGLONG *length);
extern INT32 mmIsDir(const CHAR *fileName);
extern INT32 mmGetOsName(CHAR *name, INT32 nameSize);
extern INT32 mmGetOsVersion(CHAR *versionInfo, INT32 versionLength);
extern INT32 mmGetMac(mmMacInfo **list, INT32 *count);
extern INT32 mmGetMacFree(mmMacInfo *list, INT32 count);
extern INT32 mmGetCpuInfo(mmCpuDesc **cpuInfo, INT32 *count);
extern INT32 mmCpuInfoFree(mmCpuDesc *cpuInfo, INT32 count);
extern INT32 mmCreateProcess(const CHAR *fileName, const mmArgvEnv *env, const char *stdoutRedirectFile, mmProcess *id);

extern INT32 mmCreateTaskWithThreadAttr(mmThread *threadHandle, const mmUserBlock_t *funcBlock,
                                        const mmThreadAttr *threadAttr);
#define MMPA_DLL_API

#ifdef __cplusplus
#if __cplusplus
}
#endif /* __cpluscplus */
#endif // __cpluscplus

#endif // MMPA_LINUX_MMPA_LINUX_H_
