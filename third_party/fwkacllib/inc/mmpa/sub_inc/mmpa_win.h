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

#pragma section(".CRT$XCU", long, read)
#pragma section(".CRT$XPU", long, read)

typedef HANDLE mmMutex_t;
typedef HANDLE mmThread;
typedef HANDLE mmProcess;
typedef HANDLE mmPollHandle;
typedef HANDLE mmPipeHandle;
typedef HANDLE mmFileHandle;
typedef HANDLE mmCompletionHandle;
typedef HANDLE mmFd_t;
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
typedef SRWLOCK mmRWLock_t;
typedef struct sockaddr mmSockAddr;
typedef int mmSocklen_t;
typedef int mmSemTimeout_t;
typedef long mmAtomicType;
typedef long long mmAtomicType64;
typedef DWORD mmExitCode;
typedef DWORD  mmErrorMsg;
typedef int mmKey_t;
typedef HANDLE mmMsgid;
typedef long int mmOfft_t;
typedef int mmPid_t;

typedef INT32 mmSsize_t;
typedef int mmSize; // size
typedef size_t mmSize_t;
typedef VOID mmshmId_ds;
typedef long long MM_LONG;

typedef enum {
  DT_DIR = FILE_ATTRIBUTE_DIRECTORY,
} mmDtype;

typedef struct {
  unsigned char d_type;
  char d_name[MAX_PATH];  // file name
} mmDirent;

typedef struct {
  unsigned long d_type;
  char d_name[MAX_PATH];  // file name
} mmDirent2;

typedef int (*mmFilter)(const mmDirent *entry);
typedef int (*mmFilter2)(const mmDirent2 *entry);
typedef int (*mmSort)(const mmDirent **a, const mmDirent **b);
typedef int (*mmSort2)(const mmDirent2 **a, const mmDirent2 **b);

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
  MM_LONG tv_sec;
  MM_LONG tv_nsec;
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
  const char *dli_fname;
  void *dli_fbase;
  const char *dli_sname;
  void *dli_saddr;
  size_t dli_size; /* ELF only */
  int dli_bind; /* ELF only */
  int dli_type;
} mmDlInfo;

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

#define mm_no_argument        0
#define mm_required_argument  1
#define mm_optional_argument  2

#define M_FILE_RDONLY GENERIC_READ
#define M_FILE_WRONLY GENERIC_WRITE
#define M_FILE_RDWR (GENERIC_READ | GENERIC_WRITE)
#define M_FILE_CREAT OPEN_ALWAYS

#define M_RDONLY _O_RDONLY
#define M_WRONLY _O_WRONLY
#define M_RDWR _O_RDWR
#define M_IRWXU _O_RDWR
#define M_CREAT _O_CREAT
#define M_BINARY _O_BINARY
#define M_TRUNC _O_TRUNC
#define M_APPEND _O_APPEND

#define M_IREAD _S_IREAD
#define M_IRUSR _S_IREAD
#define M_IWRITE _S_IWRITE
#define M_IWUSR _S_IWRITE
#define M_IXUSR 0

#define M_IN_CREATE FILE_NOTIFY_CHANGE_FILE_NAME | FILE_NOTIFY_CHANGE_DIR_NAME
#define M_IN_CLOSE_WRITE FILE_NOTIFY_CHANGE_LAST_WRITE
#define M_IN_IGNORED FILE_NOTIFY_CHANGE_FILE_NAME | FILE_NOTIFY_CHANGE_DIR_NAME

#define M_OUT_CREATE 0x00000100
#define M_OUT_CLOSE_WRITE 0x00000008
#define M_OUT_IGNORED 0x00008000
#define M_OUT_ISDIR 0x40000000

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

#define DT_UNKNOWN 0
#define DT_FIFO 1
#define DT_CHR 2
#define DT_BLK 6
#define DT_REG 8
#define DT_LNK 10
#define DT_SOCK 12
#define DT_WHT 14
#define MM_DT_DIR 16
#define MM_DT_REG 32

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
#define M_NAME_MAX  _MAX_FNAME

#define M_F_OK 0
#define M_X_OK 1
#define M_W_OK 2
#define M_R_OK 4

#define MMPA_STDIN stdin
#define MMPA_STDOUT stdout
#define MMPA_STDERR stderr

#define MMPA_RTLD_NOW 0
#define MMPA_RTLD_GLOBAL 0
#define MMPA_RTLD_LAZY 0
#define MMPA_RTLD_NODELETE 0

#define MMPA_DL_EXT_NAME ".dll"

#define __attribute__(v)

MMPA_FUNC_VISIBILITY INT32 mmCreateTask(mmThread *threadHandle, mmUserBlock_t *funcBlock);
MMPA_FUNC_VISIBILITY INT32 mmJoinTask(mmThread *threadHandle);
MMPA_FUNC_VISIBILITY INT32 mmMutexInit(mmMutex_t *mutex);
MMPA_FUNC_VISIBILITY INT32 mmMutexLock(mmMutex_t *mutex);
MMPA_FUNC_VISIBILITY INT32 mmMutexTryLock(mmMutex_t *mutex);
MMPA_FUNC_VISIBILITY INT32 mmMutexUnLock(mmMutex_t *mutex);
MMPA_FUNC_VISIBILITY INT32 mmMutexDestroy(mmMutex_t *mutex);
MMPA_FUNC_VISIBILITY INT32 mmCondInit(mmCond *cond);
MMPA_FUNC_VISIBILITY INT32 mmCondLockInit(mmMutexFC *mutex);
MMPA_FUNC_VISIBILITY INT32 mmCondLock(mmMutexFC *mutex);
MMPA_FUNC_VISIBILITY INT32 mmCondUnLock(mmMutexFC *mutex);
MMPA_FUNC_VISIBILITY INT32 mmCondLockDestroy(mmMutexFC *mutex);
MMPA_FUNC_VISIBILITY INT32 mmRWLockInit(mmRWLock_t *rwLock);
MMPA_FUNC_VISIBILITY INT32 mmRWLockRDLock(mmRWLock_t *rwLock);
MMPA_FUNC_VISIBILITY INT32 mmRWLockTryRDLock(mmRWLock_t *rwLock);
MMPA_FUNC_VISIBILITY INT32 mmRWLockWRLock(mmRWLock_t *rwLock);
MMPA_FUNC_VISIBILITY INT32 mmRWLockTryWRLock(mmRWLock_t *rwLock);
MMPA_FUNC_VISIBILITY INT32 mmRDLockUnLock(mmRWLock_t *rwLock);
MMPA_FUNC_VISIBILITY INT32 mmWRLockUnLock(mmRWLock_t *rwLock);
MMPA_FUNC_VISIBILITY INT32 mmRWLockDestroy(mmRWLock_t *rwLock);
MMPA_FUNC_VISIBILITY INT32 mmCondWait(mmCond *cond, mmMutexFC *mutex);
MMPA_FUNC_VISIBILITY INT32 mmCondTimedWait(mmCond *cond, mmMutexFC *mutex, UINT32 milliSecond);

MMPA_FUNC_VISIBILITY INT32 mmCondNotify(mmCond *cond);
MMPA_FUNC_VISIBILITY INT32 mmCondNotifyAll(mmCond *cond);
MMPA_FUNC_VISIBILITY INT32 mmCondDestroy(mmCond *cond);
MMPA_FUNC_VISIBILITY INT32 mmGetPid(VOID);
MMPA_FUNC_VISIBILITY INT32 mmGetTid(VOID);
MMPA_FUNC_VISIBILITY INT32 mmGetPidHandle(mmProcess *processHandle);
MMPA_FUNC_VISIBILITY INT32 mmGetLocalTime(mmSystemTime_t *sysTime);
MMPA_FUNC_VISIBILITY INT32 mmGetSystemTime(mmSystemTime_t *sysTime);
MMPA_FUNC_VISIBILITY INT32 mmSemInit(mmSem_t *sem, UINT32 value);
MMPA_FUNC_VISIBILITY INT32 mmSemWait(mmSem_t *sem);
MMPA_FUNC_VISIBILITY INT32 mmSemPost(mmSem_t *sem);
MMPA_FUNC_VISIBILITY INT32 mmSemDestroy(mmSem_t *sem);
MMPA_FUNC_VISIBILITY INT32 mmOpen(const CHAR *pathName, INT32 flags);
MMPA_FUNC_VISIBILITY INT32 mmOpen2(const CHAR *pathName, INT32 flags, MODE mode);
MMPA_FUNC_VISIBILITY FILE *mmPopen(CHAR *command, CHAR *type);
MMPA_FUNC_VISIBILITY INT32 mmClose(INT32 fd);
MMPA_FUNC_VISIBILITY INT32 mmPclose(FILE *stream);
MMPA_FUNC_VISIBILITY mmSsize_t mmWrite(INT32 fd, VOID *buf, UINT32 bufLen);
MMPA_FUNC_VISIBILITY mmSsize_t mmRead(INT32 fd, VOID *buf, UINT32 bufLen);
MMPA_FUNC_VISIBILITY mmSockHandle mmSocket(INT32 sockFamily, INT32 type, INT32 protocol);
MMPA_FUNC_VISIBILITY INT32 mmBind(mmSockHandle sockFd, mmSockAddr *addr, mmSocklen_t addrLen);
MMPA_FUNC_VISIBILITY INT32 mmListen(mmSockHandle sockFd, INT32 backLog);
MMPA_FUNC_VISIBILITY mmSockHandle mmAccept(mmSockHandle sockFd, mmSockAddr *addr, mmSocklen_t *addrLen);
MMPA_FUNC_VISIBILITY INT32 mmConnect(mmSockHandle sockFd, mmSockAddr *addr, mmSocklen_t addrLen);
MMPA_FUNC_VISIBILITY INT32 mmCloseSocket(mmSockHandle sockFd);
MMPA_FUNC_VISIBILITY mmSsize_t mmSocketRecv(mmSockHandle sockFd, VOID *recvBuf, INT32 recvLen, INT32 recvFlag);
MMPA_FUNC_VISIBILITY mmSsize_t mmSocketSend(mmSockHandle sockFd, VOID *sendBuf, INT32 sendLen, INT32 sendFlag);
MMPA_FUNC_VISIBILITY INT32 mmSocketSendTo(mmSockHandle sockFd,
                                          VOID *sendMsg,
                                          INT32 sendLen,
                                          UINT32 sendFlag,
                                          const mmSockAddr* addr,
                                          INT32 tolen);
MMPA_FUNC_VISIBILITY mmSsize_t mmSocketRecvFrom(mmSockHandle sockFd,
                                                VOID *recvBuf,
                                                mmSize recvLen,
                                                UINT32 recvFlag,
                                                mmSockAddr* addr,
                                                mmSocklen_t *FromLen);
MMPA_FUNC_VISIBILITY INT32 mmSAStartup(VOID);
MMPA_FUNC_VISIBILITY INT32 mmSACleanup(VOID);
MMPA_FUNC_VISIBILITY VOID *mmDlopen(const CHAR *fileName, INT mode);
MMPA_FUNC_VISIBILITY INT32 mmDladdr(VOID *addr, mmDlInfo *info);
MMPA_FUNC_VISIBILITY VOID *mmDlsym(VOID *handle, const CHAR *fileName);
MMPA_FUNC_VISIBILITY INT32 mmDlclose(VOID *handle);
MMPA_FUNC_VISIBILITY CHAR *mmDlerror(VOID);
MMPA_FUNC_VISIBILITY INT32
    mmCreateAndSetTimer(mmTimer *timerHandle, mmUserBlock_t *timerBlock, UINT milliSecond, UINT period);
MMPA_FUNC_VISIBILITY INT32 mmDeleteTimer(mmTimer timerHandle);
MMPA_FUNC_VISIBILITY INT32 mmStatGet(const CHAR *path, mmStat_t *buffer);
MMPA_FUNC_VISIBILITY INT32 mmStat64Get(const CHAR *path, mmStat64_t *buffer);
MMPA_FUNC_VISIBILITY INT32 mmFStatGet(INT32 fd, mmStat_t *buffer);
MMPA_FUNC_VISIBILITY INT32 mmMkdir(const CHAR *pathName, mmMode_t mode);
MMPA_FUNC_VISIBILITY INT32 mmSleep(UINT32 milliSecond);
MMPA_FUNC_VISIBILITY INT32 mmCreateTaskWithAttr(mmThread *threadHandle, mmUserBlock_t *funcBlock);
MMPA_FUNC_VISIBILITY INT32 mmGetProcessPrio(mmProcess pid);
MMPA_FUNC_VISIBILITY INT32 mmSetProcessPrio(mmProcess pid, INT32 processPrio);
MMPA_FUNC_VISIBILITY INT32 mmGetThreadPrio(mmThread *threadHandle);
MMPA_FUNC_VISIBILITY INT32 mmSetThreadPrio(mmThread *threadHandle, INT32 threadPrio);
MMPA_FUNC_VISIBILITY INT32 mmAccess(const CHAR *pathName);
MMPA_FUNC_VISIBILITY INT32 mmAccess2(const CHAR *pathName, INT32 mode);
MMPA_FUNC_VISIBILITY INT32 mmRmdir(const CHAR *pathName);

MMPA_FUNC_VISIBILITY INT32 mmIoctl(mmProcess fd, INT32 ioctlCode, mmIoctlBuf *bufPtr);
MMPA_FUNC_VISIBILITY INT32 mmSemTimedWait(mmSem_t *sem, INT32 timeout);
MMPA_FUNC_VISIBILITY mmSsize_t mmWritev(mmSockHandle fd, mmIovSegment *iov, INT32 iovcnt);
MMPA_FUNC_VISIBILITY VOID mmMb();
MMPA_FUNC_VISIBILITY INT32 mmInetAton(const CHAR *addrStr, mmInAddr *addr);

MMPA_FUNC_VISIBILITY mmProcess mmOpenFile(const CHAR *fileName, UINT32 access, mmCreateFlag fileFlag);
MMPA_FUNC_VISIBILITY mmSsize_t mmReadFile(mmProcess fileId, VOID *buffer, INT32 len);
MMPA_FUNC_VISIBILITY mmSsize_t mmWriteFile(mmProcess fileId, VOID *buffer, INT32 len);
MMPA_FUNC_VISIBILITY INT32 mmCloseFile(mmProcess fileId);

MMPA_FUNC_VISIBILITY mmAtomicType mmSetData(mmAtomicType *ptr, mmAtomicType value);
MMPA_FUNC_VISIBILITY mmAtomicType mmValueInc(mmAtomicType *ptr, mmAtomicType value);
MMPA_FUNC_VISIBILITY mmAtomicType mmValueSub(mmAtomicType *ptr, mmAtomicType value);
MMPA_FUNC_VISIBILITY mmAtomicType64 mmSetData64(mmAtomicType64 *ptr, mmAtomicType64 value);
MMPA_FUNC_VISIBILITY mmAtomicType64 mmValueInc64(mmAtomicType64 *ptr, mmAtomicType64 value);
MMPA_FUNC_VISIBILITY mmAtomicType64 mmValueSub64(mmAtomicType64 *ptr, mmAtomicType64 value);
MMPA_FUNC_VISIBILITY INT32 mmCreateTaskWithDetach(mmThread *threadHandle, mmUserBlock_t *funcBlock);

MMPA_FUNC_VISIBILITY INT32 mmCreateNamedPipe(mmPipeHandle pipe[], CHAR *pipeName[], INT32 waitMode);
MMPA_FUNC_VISIBILITY INT32 mmOpenNamePipe(mmPipeHandle pipe[], CHAR *pipeName[], INT32 waitMode);
MMPA_FUNC_VISIBILITY VOID mmCloseNamedPipe(mmPipeHandle namedPipe[]);

MMPA_FUNC_VISIBILITY INT32 mmCreatePipe(mmPipeHandle pipe[], CHAR *pipeName[], UINT32 pipeCount, INT32 waitMode);
MMPA_FUNC_VISIBILITY INT32 mmOpenPipe(mmPipeHandle pipe[], CHAR *pipeName[], UINT32 pipeCount, INT32 waitMode);
MMPA_FUNC_VISIBILITY VOID mmClosePipe(mmPipeHandle pipe[], UINT32 pipeCount);

MMPA_FUNC_VISIBILITY mmCompletionHandle mmCreateCompletionPort();
MMPA_FUNC_VISIBILITY VOID mmCloseCompletionPort(mmCompletionHandle handle);
MMPA_FUNC_VISIBILITY INT32 mmPoll(mmPollfd *fds, INT32 fdCount, INT32 timeout, mmCompletionHandle handleIOCP,
                                  pmmPollData polledData, mmPollBack pollBack);

MMPA_FUNC_VISIBILITY INT32 mmGetErrorCode();
MMPA_FUNC_VISIBILITY CHAR *mmGetErrorFormatMessage(mmErrorMsg errnum, CHAR *buf, mmSize size);
MMPA_FUNC_VISIBILITY INT32 mmGetTimeOfDay(mmTimeval *timeVal, mmTimezone *timeZone);
MMPA_FUNC_VISIBILITY mmTimespec mmGetTickCount();
MMPA_FUNC_VISIBILITY INT32 mmGetRealPath(CHAR *path, CHAR *realPath);

MMPA_FUNC_VISIBILITY INT32 mmRealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen);

MMPA_FUNC_VISIBILITY INT32 mmDup2(INT32 oldFd, INT32 newFd);
MMPA_FUNC_VISIBILITY INT32 mmDup(INT32 fd);
MMPA_FUNC_VISIBILITY INT32 mmUnlink(const CHAR *filename);
MMPA_FUNC_VISIBILITY INT32 mmChmod(const CHAR *filename, INT32 mode);
MMPA_FUNC_VISIBILITY INT32 mmFileno(FILE *stream);
MMPA_FUNC_VISIBILITY INT32 mmScandir(const CHAR *path, mmDirent ***entryList, mmFilter filterFunc, mmSort sort);
MMPA_FUNC_VISIBILITY INT32 mmScandir2(const CHAR *path, mmDirent2 ***entryList, mmFilter2 filterFunc, mmSort2 sort);
MMPA_FUNC_VISIBILITY VOID mmScandirFree(mmDirent **entryList, INT32 count);
MMPA_FUNC_VISIBILITY VOID mmScandirFree2(mmDirent2 **entryList, INT32 count);

MMPA_FUNC_VISIBILITY mmMsgid mmMsgCreate(mmKey_t key, INT32 msgFlag);
MMPA_FUNC_VISIBILITY mmMsgid mmMsgOpen(mmKey_t key, INT32 msgFlag);
MMPA_FUNC_VISIBILITY INT32 mmMsgRcv(mmMsgid msqid, VOID *buf, INT32 bufLen, INT32 msgFlag);
MMPA_FUNC_VISIBILITY INT32 mmMsgSnd(mmMsgid msqid, VOID *buf, INT32 bufLen, INT32 msgFlag);

MMPA_FUNC_VISIBILITY INT32 mmMsgClose(mmMsgid msqid);

MMPA_FUNC_VISIBILITY INT32 mmLocalTimeR(const time_t *timep, struct tm *result);
MMPA_FUNC_VISIBILITY INT32 mmGetOptErr();
MMPA_FUNC_VISIBILITY VOID mmSetOptErr(INT32 mmOptErr);
MMPA_FUNC_VISIBILITY INT32 mmGetOptInd();
MMPA_FUNC_VISIBILITY VOID mmSetOptInd(INT32 mmOptInd);
MMPA_FUNC_VISIBILITY INT32 mmGetOptOpt();
MMPA_FUNC_VISIBILITY VOID mmSetOpOpt(INT32 mmOptOpt);
MMPA_FUNC_VISIBILITY CHAR *mmGetOptArg();
MMPA_FUNC_VISIBILITY VOID mmSetOptArg(CHAR *mmOptArg);
MMPA_FUNC_VISIBILITY INT32 mmGetOpt(INT32 argc, char *const *argv, const char *opts);
MMPA_FUNC_VISIBILITY INT32
    mmGetOptLong(INT32 argc, CHAR *const *argv, const CHAR *opts, const mmStructOption *longopts, INT32 *longindex);

MMPA_FUNC_VISIBILITY LONG mmLseek(INT32 fd, INT64 offset, INT32 seekFlag);
MMPA_FUNC_VISIBILITY INT32 mmFtruncate(mmProcess fd, UINT32 length);

MMPA_FUNC_VISIBILITY INT32 mmTlsCreate(mmThreadKey *key, VOID (*destructor)(VOID *));
MMPA_FUNC_VISIBILITY INT32 mmTlsSet(mmThreadKey key, const VOID *value);
MMPA_FUNC_VISIBILITY VOID *mmTlsGet(mmThreadKey key);
MMPA_FUNC_VISIBILITY INT32 mmTlsDelete(mmThreadKey key);
MMPA_FUNC_VISIBILITY INT32 mmGetOsType();

MMPA_FUNC_VISIBILITY INT32 mmFsync(mmProcess fd);
MMPA_FUNC_VISIBILITY INT32 mmFsync2(INT32 fd);
MMPA_FUNC_VISIBILITY INT32 mmChdir(const CHAR *path);
MMPA_FUNC_VISIBILITY INT32 mmUmask(INT32 pmode);
MMPA_FUNC_VISIBILITY INT32 mmWaitPid(mmProcess pid, INT32 *status, INT32 options);

MMPA_FUNC_VISIBILITY INT32 mmGetCwd(CHAR *buffer, INT32 maxLen);
MMPA_FUNC_VISIBILITY CHAR *mmStrTokR(CHAR *str, const CHAR *delim, CHAR **saveptr);

MMPA_FUNC_VISIBILITY INT32 mmGetEnv(const CHAR *name, CHAR *value, UINT32 len);
MMPA_FUNC_VISIBILITY INT32 mmSetEnv(const CHAR *name, const CHAR *value, INT32 overwrite);
MMPA_FUNC_VISIBILITY CHAR *mmDirName(CHAR *path);
MMPA_FUNC_VISIBILITY CHAR *mmBaseName(CHAR *path);
MMPA_FUNC_VISIBILITY INT32 mmGetDiskFreeSpace(const char *path, mmDiskSize *diskSize);

MMPA_FUNC_VISIBILITY INT32 mmSetThreadName(mmThread *threadHandle, const CHAR *name);
MMPA_FUNC_VISIBILITY INT32 mmGetThreadName(mmThread *threadHandle, CHAR *name, INT32 size);

/*
 * Function: set the thread name of the currently executing thread - internal call of thread, which is not supported
 * under Windows temporarily, and is null.
 * Input: name: the thread name to be set
 * The input parameter error returns EN_INVALID_PARAM, the execution success returns EN_OK, and the
 * execution failure returns EN_ERROR
 */
MMPA_FUNC_VISIBILITY INT32 mmSetCurrentThreadName(const CHAR *name);

/*
 * Function: Get the thread name of the currently executing thread - thread body call, not supported under windows, null
 * implementation.
 * Input:name:The name of the thread to get, and the cache is allocated by the user，size>=MMPA_THREADNAME_SIZE.
 * The input parameter error returns EN_INVALID_PARAM, the execution success returns
 * EN_OK, and the execution failure returns EN_ERROR
 */
MMPA_FUNC_VISIBILITY INT32 mmGetCurrentThreadName(CHAR *name, INT32 size);

MMPA_FUNC_VISIBILITY INT32 mmGetFileSize(const CHAR *fileName, ULONGLONG *length);
MMPA_FUNC_VISIBILITY INT32 mmIsDir(const CHAR *fileName);
MMPA_FUNC_VISIBILITY INT32 mmGetOsName(CHAR *name, INT32 nameSize);
MMPA_FUNC_VISIBILITY INT32 mmGetOsVersion(CHAR *versionInfo, INT32 versionLength);
MMPA_FUNC_VISIBILITY INT32 mmGetMac(mmMacInfo **list, INT32 *count);
MMPA_FUNC_VISIBILITY INT32 mmGetMacFree(mmMacInfo *list, INT32 count);
MMPA_FUNC_VISIBILITY INT32 mmGetCpuInfo(mmCpuDesc **cpuInfo, INT32 *count);
MMPA_FUNC_VISIBILITY INT32 mmCpuInfoFree(mmCpuDesc *cpuInfo, INT32 count);
MMPA_FUNC_VISIBILITY INT32
    mmCreateProcess(const CHAR *fileName, const mmArgvEnv *env, const char *stdoutRedirectFile, mmProcess *id);

MMPA_FUNC_VISIBILITY INT32
    mmCreateTaskWithThreadAttr(mmThread *threadHandle, const mmUserBlock_t *funcBlock, const mmThreadAttr *threadAttr);
MMPA_FUNC_VISIBILITY mmFileHandle mmShmOpen(const CHAR *name, INT32 oflag, mmMode_t mode);
MMPA_FUNC_VISIBILITY INT32 mmShmUnlink(const CHAR *name);
MMPA_FUNC_VISIBILITY VOID *mmMmap(mmFd_t fd, mmSize_t size, mmOfft_t offset, mmFd_t *extra, INT32 prot, INT32 flags);
MMPA_FUNC_VISIBILITY INT32 mmMunMap(VOID *data, mmSize_t size, mmFd_t *extra);
#ifdef __cplusplus
#if __cplusplus
}
#endif /* __cpluscplus */
#endif // __cpluscplus

#endif // MMPA_WIN_MMPA_WIN_H_
