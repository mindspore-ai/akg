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

#ifndef MMPA_WIN_MMPA_WIN_H
#define MMPA_WIN_MMPA_WIN_H
#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif  // __cpluscplus
#endif  // __cpluscplus
#ifdef MMPA_DLL
#define MMPA_DLL_API __declspec(dllexport)
#else
#define MMPA_DLL_API __declspec(dllimport)
#endif

#define MMPA_MACINFO_DEFAULT_SIZE 18
#define MMPA_CPUDESC_DEFAULT_SIZE 64

MMPA_DLL_API extern char *optarg;
MMPA_DLL_API extern int opterr;
MMPA_DLL_API extern int optind;
MMPA_DLL_API extern int optopt;

#pragma section(".CRT$XCU", long, read)
#pragma section(".CRT$XPU", long, read)

typedef HANDLE mmMutex_t;
typedef HANDLE mmThread;
typedef HANDLE mmProcess;
typedef HANDLE mmPollHandle;
typedef HANDLE mmPipeHandle;
typedef HANDLE mmCompletionHandle;

typedef CRITICAL_SECTION mmMutexFC;
typedef CONDITION_VARIABLE mmCond;

typedef VOID *(*userProcFunc)(VOID *pulArg);
typedef struct {
  userProcFunc procFunc;
  VOID *pulArg;
} mmUserBlock_t;

typedef DWORD mmThreadKey;
typedef SYSTEMTIME mmSystemTime_t;

typedef HANDLE mmSem_t;
typedef SOCKET mmSockHandle;
typedef struct sockaddr mmSockAddr;
typedef int mmSocklen_t;
typedef int mmSemTimeout_t;
typedef long mmAtomicType;
typedef DWORD mmExitCode;
typedef int mmKey_t;
typedef HANDLE mmMsgid;

typedef INT32 mmSsize_t;

typedef enum {
  DT_DIR = FILE_ATTRIBUTE_DIRECTORY,
} mmDtype;

typedef struct {
  unsigned char d_type;
  char d_name[MAX_PATH];  // file name
} mmDirent;

typedef int (*mmFilter)(const mmDirent *entry);
typedef int (*mmSort)(const mmDirent **a, const mmDirent **b);

typedef struct {
  VOID *sendBuf;
  INT32 sendLen;
} mmIovSegment;
typedef PVOID mmInAddr;

typedef enum {
  pollTypeRead = 1,  // pipeline reading
  pollTypeRecv,      // socket receive
  pollTypeIoctl,     // ioctl read
} mmPollType;

typedef struct {
  HANDLE completionHandle;
  mmPollType overlapType;
  OVERLAPPED oa;
} mmComPletionKey, *pmmComPletionKey;

typedef struct {
  VOID *priv;              // User defined private content
  mmPollHandle bufHandle;  // Value of handle corresponding to buf
  mmPollType bufType;      // Data types polled to
  VOID *buf;
  UINT32 bufLen;
  UINT32 bufRes;
} mmPollData, *pmmPollData;

typedef VOID (*mmPollBack)(pmmPollData);
typedef struct {
  mmPollHandle handle;            // The file descriptor or handle of poll is required
  mmPollType pollType;            // Operation type requiring poll，read or recv or ioctl
  INT32 ioctlCode;                // IOCTL operation code, dedicated to IOCTL
  mmComPletionKey completionKey;  // The default value is blank, which will be used in windows to receive the data with
                                  // different handle
} mmPollfd;

typedef struct {
  OVERLAPPED oa;
  HANDLE completionHandle;
  WSABUF DataBuf;
} PRE_IO_DATA, *PPRE_IO_DATA;

typedef OVERLAPPED mmOverLap;

typedef struct {
  UINT32 createFlag;
  INT32 oaFlag;  // Overlap operation is supported if it is not 0
} mmCreateFlag;

typedef struct {
  VOID *inbuf;
  INT32 inbufLen;
  VOID *outbuf;
  INT32 outbufLen;
  mmOverLap *oa;
} mmIoctlBuf;

typedef struct {
  HANDLE timerQueue;
  HANDLE timerHandle;
} mmTimerHandle;

typedef struct {
  LONG tv_sec;
  LONG tv_usec;
} mmTimeval;

typedef struct {
  INT32 tz_minuteswest;  // How many minutes is it different from Greenwich
  INT32 tz_dsttime;      // DST correction type
} mmTimezone;

typedef struct {
  LONG tv_sec;
  LONG tv_nsec;
} mmTimespec;

typedef mmTimerHandle mmTimer;

#define mmTLS __declspec(thread)

typedef struct stat mmStat_t;
typedef struct _stat64 mmStat64_t;
typedef int mmMode_t;

typedef int MODE;

typedef struct {
  const char *name;
  int has_arg;
  int *flag;
  int val;
} mmStructOption;

typedef struct {
  ULONGLONG totalSize;
  ULONGLONG freeSize;
  ULONGLONG availSize;
} mmDiskSize;

typedef struct {
  char addr[MMPA_MACINFO_DEFAULT_SIZE];  // ex:aa-bb-cc-dd-ee-ff\0
} mmMacInfo;

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

typedef struct {
  char **argv;
  INT32 argvCount;
  char **envp;
  INT32 envpCount;
} mmArgvEnv;

// Windows currently does not support properties other than thread separation properties
typedef struct {
  INT32 detachFlag;  // Thread detach property: 0 do not detach 1 detach
  INT32 priorityFlag;
  INT32 priority;
  INT32 policyFlag;
  INT32 policy;
  INT32 stackFlag;
  UINT32 stackSize;
} mmThreadAttr;

typedef VOID (*mmPf)(VOID);
#define M_FILE_RDONLY GENERIC_READ
#define M_FILE_WRONLY GENERIC_WRITE
#define M_FILE_RDWR (GENERIC_READ | GENERIC_WRITE)
#define M_FILE_CREAT OPEN_ALWAYS

#define M_RDONLY _O_RDONLY
#define M_WRONLY _O_WRONLY
#define M_RDWR _O_RDWR
#define M_CREAT _O_CREAT
#define M_BINARY _O_BINARY

#define M_IREAD _S_IREAD
#define M_IRUSR _S_IREAD
#define M_IWRITE _S_IWRITE
#define M_IWUSR _S_IWRITE
#define M_IXUSR 0

#define M_MSG_CREAT 1
#define M_MSG_EXCL 2
#define M_MSG_NOWAIT 3

#define M_WAIT_NOHANG 1
#define M_WAIT_UNTRACED 2

#define M_UMASK_USRREAD _S_IREAD
#define M_UMASK_GRPREAD _S_IREAD
#define M_UMASK_OTHREAD _S_IREAD

#define M_UMASK_USRWRITE _S_IWRITE
#define M_UMASK_GRPWRITE _S_IWRITE
#define M_UMASK_OTHWRITE _S_IWRITE

#define M_UMASK_USREXEC 0
#define M_UMASK_GRPEXEC 0
#define M_UMASK_OTHEXEC 0

#define mmConstructor(x) __declspec(allocate(".CRT$XCU")) mmPf con = x
#define mmDestructor(x) __declspec(allocate(".CRT$XPU")) mmPf de = x

#define MMPA_PRINT_ERROR ((opterr) && (*options != ':'))
#define MMPA_FLAG_PERMUTE 0x01   // permute non-options to the end of argv
#define MMPA_FLAG_ALLARGS 0x02   // treat non-options as args to option "-1"
#define MMPA_FLAG_LONGONLY 0x04  // operate as getopt_long_only
// return values
#define MMPA_BADCH (INT32)'?'
#define MMPA_BADARG ((*options == ':') ? (INT32)':' : (INT32)'?')
#define MMPA_INORDER (INT32)1

#define MMPA_NO_ARGUMENT 0
#define MMPA_REQUIRED_ARGUMENT 1
#define MMPA_OPTIONAL_ARGUMENT 2

#define MMPA_EMSG ""
#define MMPA_MAX_PATH MAX_PATH

#define M_F_OK 0
#define M_W_OK 2
#define M_R_OK 4

#define MMPA_RTLD_NOW 0
#define MMPA_RTLD_GLOBAL 0

#define MMPA_DL_EXT_NAME ".dll"

#define __attribute__(v)

_declspec(dllexport) INT32 mmCreateTask(mmThread *threadHandle, mmUserBlock_t *funcBlock);
_declspec(dllexport) INT32 mmJoinTask(mmThread *threadHandle);
_declspec(dllexport) INT32 mmMutexInit(mmMutex_t *mutex);
_declspec(dllexport) INT32 mmMutexLock(mmMutex_t *mutex);
_declspec(dllexport) INT32 mmMutexUnLock(mmMutex_t *mutex);
_declspec(dllexport) INT32 mmMutexDestroy(mmMutex_t *mutex);
_declspec(dllexport) INT32 mmCondInit(mmCond *cond);
_declspec(dllexport) INT32 mmCondLockInit(mmMutexFC *mutex);
_declspec(dllexport) INT32 mmCondLock(mmMutexFC *mutex);
_declspec(dllexport) INT32 mmCondUnLock(mmMutexFC *mutex);
_declspec(dllexport) INT32 mmCondLockDestroy(mmMutexFC *mutex);
_declspec(dllexport) INT32 mmCondWait(mmCond *cond, mmMutexFC *mutex);
_declspec(dllexport) INT32 mmCondTimedWait(mmCond *cond, mmMutexFC *mutex, UINT32 milliSecond);

_declspec(dllexport) INT32 mmCondNotify(mmCond *cond);
_declspec(dllexport) INT32 mmCondNotifyAll(mmCond *cond);
_declspec(dllexport) INT32 mmCondDestroy(mmCond *cond);
_declspec(dllexport) INT32 mmGetPid(VOID);
_declspec(dllexport) INT32 mmGetTid(VOID);
_declspec(dllexport) INT32 mmGetPidHandle(mmProcess *processHandle);
_declspec(dllexport) INT32 mmGetLocalTime(mmSystemTime_t *sysTime);
_declspec(dllexport) INT32 mmSemInit(mmSem_t *sem, UINT32 value);
_declspec(dllexport) INT32 mmSemWait(mmSem_t *sem);
_declspec(dllexport) INT32 mmSemPost(mmSem_t *sem);
_declspec(dllexport) INT32 mmSemDestroy(mmSem_t *sem);
_declspec(dllexport) INT32 mmOpen(const CHAR *pathName, INT32 flags);
_declspec(dllexport) INT32 mmOpen2(const CHAR *pathName, INT32 flags, MODE mode);
_declspec(dllexport) INT32 mmClose(INT32 fd);
_declspec(dllexport) mmSsize_t mmWrite(INT32 fd, VOID *buf, UINT32 bufLen);
_declspec(dllexport) mmSsize_t mmRead(INT32 fd, VOID *buf, UINT32 bufLen);
_declspec(dllexport) mmSockHandle mmSocket(INT32 sockFamily, INT32 type, INT32 protocol);
_declspec(dllexport) INT32 mmBind(mmSockHandle sockFd, mmSockAddr *addr, mmSocklen_t addrLen);
_declspec(dllexport) INT32 mmListen(mmSockHandle sockFd, INT32 backLog);
_declspec(dllexport) mmSockHandle mmAccept(mmSockHandle sockFd, mmSockAddr *addr, mmSocklen_t *addrLen);
_declspec(dllexport) INT32 mmConnect(mmSockHandle sockFd, mmSockAddr *addr, mmSocklen_t addrLen);
_declspec(dllexport) INT32 mmCloseSocket(mmSockHandle sockFd);
_declspec(dllexport) mmSsize_t mmSocketRecv(mmSockHandle sockFd, VOID *recvBuf, INT32 recvLen, INT32 recvFlag);
_declspec(dllexport) mmSsize_t mmSocketSend(mmSockHandle sockFd, VOID *sendBuf, INT32 sendLen, INT32 sendFlag);
_declspec(dllexport) INT32 mmSAStartup(VOID);
_declspec(dllexport) INT32 mmSACleanup(VOID);
_declspec(dllexport) VOID *mmDlopen(const CHAR *fileName, INT mode);
_declspec(dllexport) VOID *mmDlsym(VOID *handle, CHAR *fileName);
_declspec(dllexport) INT32 mmDlclose(VOID *handle);
_declspec(dllexport) CHAR *mmDlerror(VOID);
_declspec(dllexport) INT32
    mmCreateAndSetTimer(mmTimer *timerHandle, mmUserBlock_t *timerBlock, UINT milliSecond, UINT period);
_declspec(dllexport) INT32 mmDeleteTimer(mmTimer timerHandle);
_declspec(dllexport) INT32 mmStatGet(const CHAR *path, mmStat_t *buffer);
_declspec(dllexport) INT32 mmStat64Get(const CHAR *path, mmStat64_t *buffer);
_declspec(dllexport) INT32 mmMkdir(const CHAR *pathName, mmMode_t mode);
_declspec(dllexport) INT32 mmSleep(UINT32 milliSecond);
_declspec(dllexport) INT32 mmCreateTaskWithAttr(mmThread *threadHandle, mmUserBlock_t *funcBlock);
_declspec(dllexport) INT32 mmGetProcessPrio(mmProcess pid);
_declspec(dllexport) INT32 mmSetProcessPrio(mmProcess pid, INT32 processPrio);
_declspec(dllexport) INT32 mmGetThreadPrio(mmThread *threadHandle);
_declspec(dllexport) INT32 mmSetThreadPrio(mmThread *threadHandle, INT32 threadPrio);
_declspec(dllexport) INT32 mmAccess(const CHAR *pathName);
_declspec(dllexport) INT32 mmAccess2(const CHAR *pathName, INT32 mode);
_declspec(dllexport) INT32 mmRmdir(const CHAR *pathName);

_declspec(dllexport) INT32 mmIoctl(mmProcess fd, INT32 ioctlCode, mmIoctlBuf *bufPtr);
_declspec(dllexport) INT32 mmSemTimedWait(mmSem_t *sem, INT32 timeout);
_declspec(dllexport) mmSsize_t mmWritev(mmSockHandle fd, mmIovSegment *iov, INT32 iovcnt);
_declspec(dllexport) VOID mmMb();
_declspec(dllexport) INT32 mmInetAton(const CHAR *addrStr, mmInAddr *addr);

_declspec(dllexport) mmProcess mmOpenFile(const CHAR *fileName, UINT32 access, mmCreateFlag fileFlag);
_declspec(dllexport) mmSsize_t mmReadFile(mmProcess fileId, VOID *buffer, INT32 len);
_declspec(dllexport) mmSsize_t mmWriteFile(mmProcess fileId, VOID *buffer, INT32 len);
_declspec(dllexport) INT32 mmCloseFile(mmProcess fileId);

_declspec(dllexport) mmAtomicType mmSetData(mmAtomicType *ptr, mmAtomicType value);
_declspec(dllexport) mmAtomicType mmValueInc(mmAtomicType *ptr, mmAtomicType value);
_declspec(dllexport) mmAtomicType mmValueSub(mmAtomicType *ptr, mmAtomicType value);
_declspec(dllexport) INT32 mmCreateTaskWithDetach(mmThread *threadHandle, mmUserBlock_t *funcBlock);

_declspec(dllexport) INT32 mmCreateNamedPipe(mmPipeHandle pipe[], CHAR *pipeName[], INT32 waitMode);
_declspec(dllexport) INT32 mmOpenNamePipe(mmPipeHandle pipe[], CHAR *pipeName[], INT32 waitMode);
_declspec(dllexport) VOID mmCloseNamedPipe(mmPipeHandle namedPipe[]);

_declspec(dllexport) INT32 mmCreatePipe(mmPipeHandle pipe[], CHAR *pipeName[], UINT32 pipeCount, INT32 waitMode);
_declspec(dllexport) INT32 mmOpenPipe(mmPipeHandle pipe[], CHAR *pipeName[], UINT32 pipeCount, INT32 waitMode);
_declspec(dllexport) VOID mmClosePipe(mmPipeHandle pipe[], UINT32 pipeCount);

_declspec(dllexport) mmCompletionHandle mmCreateCompletionPort();
_declspec(dllexport) VOID mmCloseCompletionPort(mmCompletionHandle handle);
_declspec(dllexport) INT32 mmPoll(mmPollfd *fds, INT32 fdCount, INT32 timeout, mmCompletionHandle handleIOCP,
                                  pmmPollData polledData, mmPollBack pollBack);

_declspec(dllexport) INT32 mmGetErrorCode();
_declspec(dllexport) INT32 mmGetTimeOfDay(mmTimeval *timeVal, mmTimezone *timeZone);
_declspec(dllexport) mmTimespec mmGetTickCount();
_declspec(dllexport) INT32 mmGetRealPath(CHAR *path, CHAR *realPath);

_declspec(dllexport) INT32 mmRealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen);

_declspec(dllexport) INT32 mmDup2(INT32 oldFd, INT32 newFd);
_declspec(dllexport) INT32 mmUnlink(const CHAR *filename);
_declspec(dllexport) INT32 mmChmod(const CHAR *filename, INT32 mode);
_declspec(dllexport) INT32 mmFileno(FILE *stream);
_declspec(dllexport) INT32 mmScandir(const CHAR *path, mmDirent ***entryList, mmFilter filterFunc, mmSort sort);
_declspec(dllexport) VOID mmScandirFree(mmDirent **entryList, INT32 count);

_declspec(dllexport) mmMsgid mmMsgCreate(mmKey_t key, INT32 msgFlag);
_declspec(dllexport) mmMsgid mmMsgOpen(mmKey_t key, INT32 msgFlag);
_declspec(dllexport) INT32 mmMsgRcv(mmMsgid msqid, VOID *buf, INT32 bufLen, INT32 msgFlag);
_declspec(dllexport) INT32 mmMsgSnd(mmMsgid msqid, VOID *buf, INT32 bufLen, INT32 msgFlag);

_declspec(dllexport) INT32 mmMsgClose(mmMsgid msqid);

_declspec(dllexport) INT32 mmLocalTimeR(const time_t *timep, struct tm *result);
_declspec(dllexport) INT32 mmGetOpt(INT32 argc, char *const *argv, const char *opts);
_declspec(dllexport) INT32
    mmGetOptLong(INT32 argc, CHAR *const *argv, const CHAR *opts, const mmStructOption *longopts, INT32 *longindex);

_declspec(dllexport) LONG mmLseek(INT32 fd, INT64 offset, INT32 seekFlag);
_declspec(dllexport) INT32 mmFtruncate(mmProcess fd, UINT32 length);

_declspec(dllexport) INT32 mmTlsCreate(mmThreadKey *key, VOID (*destructor)(VOID *));
_declspec(dllexport) INT32 mmTlsSet(mmThreadKey key, const VOID *value);
_declspec(dllexport) VOID *mmTlsGet(mmThreadKey key);
_declspec(dllexport) INT32 mmTlsDelete(mmThreadKey key);
_declspec(dllexport) INT32 mmGetOsType();

_declspec(dllexport) INT32 mmFsync(mmProcess fd);

_declspec(dllexport) INT32 mmChdir(const CHAR *path);
_declspec(dllexport) INT32 mmUmask(INT32 pmode);
_declspec(dllexport) INT32 mmWaitPid(mmProcess pid, INT32 *status, INT32 options);

_declspec(dllexport) INT32 mmGetCwd(CHAR *buffer, INT32 maxLen);
_declspec(dllexport) CHAR *mmStrTokR(CHAR *str, const CHAR *delim, CHAR **saveptr);

_declspec(dllexport) INT32 mmGetEnv(const CHAR *name, CHAR *value, UINT32 len);
_declspec(dllexport) INT32 mmSetEnv(const CHAR *name, const CHAR *value, INT32 overwrite);
_declspec(dllexport) CHAR *mmDirName(CHAR *path);
_declspec(dllexport) CHAR *mmBaseName(CHAR *path);
_declspec(dllexport) INT32 mmGetDiskFreeSpace(const char *path, mmDiskSize *diskSize);

_declspec(dllexport) INT32 mmSetThreadName(mmThread *threadHandle, const CHAR *name);
_declspec(dllexport) INT32 mmGetThreadName(mmThread *threadHandle, CHAR *name, INT32 size);

/*
 * Function: set the thread name of the currently executing thread - internal call of thread, which is not supported
 * under Windows temporarily, and is null.
 * Input: name: the thread name to be set
 * The input parameter error returns EN_INVALID_PARAM, the execution success returns EN_OK, and the
 * execution failure returns EN_ERROR
 */
_declspec(dllexport) INT32 mmSetCurrentThreadName(const CHAR *name);

/*
 * Function: Get the thread name of the currently executing thread - thread body call, not supported under windows, null
 * implementation.
 * Input:name:The name of the thread to get, and the cache is allocated by the user，size>=MMPA_THREADNAME_SIZE.
 * The input parameter error returns EN_INVALID_PARAM, the execution success returns
 * EN_OK, and the execution failure returns EN_ERROR
 */
_declspec(dllexport) INT32 mmGetCurrentThreadName(CHAR *name, INT32 size);

_declspec(dllexport) INT32 mmGetFileSize(const CHAR *fileName, ULONGLONG *length);
_declspec(dllexport) INT32 mmIsDir(const CHAR *fileName);
_declspec(dllexport) INT32 mmGetOsName(CHAR *name, INT32 nameSize);
_declspec(dllexport) INT32 mmGetOsVersion(CHAR *versionInfo, INT32 versionLength);
_declspec(dllexport) INT32 mmGetMac(mmMacInfo **list, INT32 *count);
_declspec(dllexport) INT32 mmGetMacFree(mmMacInfo *list, INT32 count);
_declspec(dllexport) INT32 mmGetCpuInfo(mmCpuDesc **cpuInfo, INT32 *count);
_declspec(dllexport) INT32 mmCpuInfoFree(mmCpuDesc *cpuInfo, INT32 count);
_declspec(dllexport) INT32
    mmCreateProcess(const CHAR *fileName, const mmArgvEnv *env, const char *stdoutRedirectFile, mmProcess *id);

_declspec(dllexport) INT32
    mmCreateTaskWithThreadAttr(mmThread *threadHandle, const mmUserBlock_t *funcBlock, const mmThreadAttr *threadAttr);

#ifdef __cplusplus
#if __cplusplus
}
#endif /* __cpluscplus */
#endif // __cpluscplus

#endif // MMPA_WIN_MMPA_WIN_H_
