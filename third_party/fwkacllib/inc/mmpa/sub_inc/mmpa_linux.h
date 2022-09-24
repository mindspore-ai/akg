/*
* @file mmpa_linux.h
*
* Copyright (C) Huawei Technologies Co., Ltd. 2019-2021. All Rights Reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
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
typedef pthread_rwlock_t mmRWLock_t;
typedef signed int mmProcess;
typedef int mmPollHandle;
typedef int mmPipeHandle;
typedef int mmFileHandle;
typedef int mmComPletionKey;
typedef int mmCompletionHandle;
typedef int mmErrorMsg;
typedef int mmFd_t;

typedef VOID *mmExitCode;
typedef key_t mmKey_t;
typedef int mmMsgid;
typedef struct dirent mmDirent;
typedef struct dirent mmDirent2;
typedef struct shmid_ds mmshmId_ds;
typedef int (*mmFilter)(const mmDirent *entry);
typedef int (*mmFilter2)(const mmDirent2 *entry);
typedef int (*mmSort)(const mmDirent **a, const mmDirent **b);
typedef int (*mmSort2)(const mmDirent2 **a, const mmDirent2 **b);
typedef size_t mmSize_t; //lint !e410 !e1051
typedef off_t mmOfft_t;
typedef pid_t mmPid_t;
typedef long MM_LONG;

typedef VOID *(*userProcFunc)(VOID *pulArg);

typedef struct {
    userProcFunc procFunc;  // Callback function pointer
    VOID *pulArg;           // Callback function parameters
} mmUserBlock_t;

typedef struct {
    const CHAR *dli_fname;
    VOID *dli_fbase;
    const CHAR *dli_sname;
    VOID *dli_saddr;
    size_t dli_size; /* ELF only */
    INT32 dli_bind; /* ELF only */
    INT32 dli_type;
} mmDlInfo;

typedef struct {
    INT32 wSecond;             // Seconds. [0-60] (1 leap second)
    INT32 wMinute;             // Minutes. [0-59]
    INT32 wHour;               // Hours. [0-23]
    INT32 wDay;                // Day. [1-31]
    INT32 wMonth;              // Month. [1-12]
    INT32 wYear;               // Year
    INT32 wDayOfWeek;          // Day of week. [0-6]
    INT32 tm_yday;             // Days in year.[0-365]
    INT32 tm_isdst;            // DST. [-1/0/1]
    LONG wMilliseconds;        // milliseconds
} mmSystemTime_t;

typedef sem_t mmSem_t;
typedef struct sockaddr mmSockAddr;
typedef socklen_t mmSocklen_t;
typedef int mmSockHandle;
typedef timer_t mmTimer;
typedef pthread_key_t mmThreadKey;

typedef int mmOverLap;

typedef ssize_t mmSsize_t;
typedef size_t mmSize; // size

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
typedef int mmAtomicType64;

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
    MM_LONG tv_sec;
    MM_LONG tv_nsec;
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
    CHAR addr[MMPA_MACINFO_DEFAULT_SIZE];  // ex:aa-bb-cc-dd-ee-ff\0
} mmMacInfo;

typedef struct {
    CHAR **argv;
    INT32 argvCount;
    CHAR **envp;
    INT32 envpCount;
} mmArgvEnv;

typedef struct {
    CHAR arch[MMPA_CPUDESC_DEFAULT_SIZE];
    CHAR manufacturer[MMPA_CPUDESC_DEFAULT_SIZE];  // vendor
    CHAR version[MMPA_CPUDESC_DEFAULT_SIZE];       // modelname
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

#define mm_no_argument        no_argument
#define mm_required_argument  required_argument
#define mm_optional_argument  optional_argument

#define M_FILE_RDONLY O_RDONLY
#define M_FILE_WRONLY O_WRONLY
#define M_FILE_RDWR O_RDWR
#define M_FILE_CREAT O_CREAT

#define M_RDONLY O_RDONLY
#define M_WRONLY O_WRONLY
#define M_RDWR O_RDWR
#define M_CREAT O_CREAT
#define M_BINARY O_RDONLY
#define M_TRUNC O_TRUNC
#define M_IRWXU S_IRWXU
#define M_APPEND O_APPEND

#define M_IN_CREATE IN_CREATE
#define M_IN_CLOSE_WRITE IN_CLOSE_WRITE
#define M_IN_IGNORED IN_IGNORED

#define M_OUT_CREATE IN_CREATE
#define M_OUT_CLOSE_WRITE IN_CLOSE_WRITE
#define M_OUT_IGNORED IN_IGNORED
#define M_OUT_ISDIR IN_ISDIR

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
#define M_NAME_MAX MAX_FNAME

#define M_F_OK F_OK
#define M_X_OK X_OK
#define M_W_OK W_OK
#define M_R_OK R_OK


#define MM_DT_DIR DT_DIR
#define MM_DT_REG DT_REG

#define MMPA_STDIN STDIN_FILENO
#define MMPA_STDOUT STDOUT_FILENO
#define MMPA_STDERR STDERR_FILENO

#define MMPA_RTLD_NOW RTLD_NOW
#define MMPA_RTLD_GLOBAL RTLD_GLOBAL
#define MMPA_RTLD_LAZY RTLD_LAZY
#define MMPA_RTLD_NODELETE RTLD_NODELETE

#define MMPA_DL_EXT_NAME ".so"

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
MMPA_FUNC_VISIBILITY INT32 mmGetPid();
MMPA_FUNC_VISIBILITY INT32 mmGetTid();
MMPA_FUNC_VISIBILITY INT32 mmGetPidHandle(mmProcess *processHandle);
MMPA_FUNC_VISIBILITY INT32 mmGetLocalTime(mmSystemTime_t *sysTimePtr);
MMPA_FUNC_VISIBILITY INT32 mmGetSystemTime(mmSystemTime_t *sysTimePtr);

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
MMPA_FUNC_VISIBILITY mmSsize_t mmSocketSend(mmSockHandle sockFd, VOID *sendBuf, INT32 sendLen, INT32 sendFlag);
MMPA_FUNC_VISIBILITY mmSsize_t mmSocketRecv(mmSockHandle sockFd, VOID *recvBuf, INT32 recvLen, INT32 recvFlag);
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
MMPA_FUNC_VISIBILITY INT32 mmSAStartup();
MMPA_FUNC_VISIBILITY INT32 mmSACleanup();
MMPA_FUNC_VISIBILITY VOID *mmDlopen(const CHAR *fileName, INT32 mode);
MMPA_FUNC_VISIBILITY INT32 mmDladdr(VOID *addr, mmDlInfo *info);
MMPA_FUNC_VISIBILITY VOID *mmDlsym(VOID *handle, const CHAR *funcName);
MMPA_FUNC_VISIBILITY INT32 mmDlclose(VOID *handle);
MMPA_FUNC_VISIBILITY CHAR *mmDlerror();
MMPA_FUNC_VISIBILITY INT32 mmCreateAndSetTimer(mmTimer *timerHandle,
                                               mmUserBlock_t *timerBlock,
                                               UINT milliSecond,
                                               UINT period);
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
MMPA_FUNC_VISIBILITY mmSsize_t mmWritev(mmProcess fd, mmIovSegment *iov, INT32 iovcnt);
MMPA_FUNC_VISIBILITY VOID mmMb();
MMPA_FUNC_VISIBILITY INT32 mmInetAton(const CHAR *addrStr, mmInAddr *addr);

MMPA_FUNC_VISIBILITY mmProcess mmOpenFile(const CHAR *fileName, UINT32 accessFlag, mmCreateFlag fileFlag);
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

// The following 3 interfaces are to be deleted
MMPA_FUNC_VISIBILITY INT32 mmCreateNamedPipe(mmPipeHandle pipeHandle[], CHAR *pipeName[], INT32 waitMode);
MMPA_FUNC_VISIBILITY INT32 mmOpenNamePipe(mmPipeHandle pipeHandle[], CHAR *pipeName[], INT32 waitMode);
MMPA_FUNC_VISIBILITY VOID mmCloseNamedPipe(mmPipeHandle namedPipe[]);

MMPA_FUNC_VISIBILITY INT32 mmCreatePipe(mmPipeHandle pipeHandle[], CHAR *pipeName[], UINT32 pipeCount, INT32 waitMode);
MMPA_FUNC_VISIBILITY INT32 mmOpenPipe(mmPipeHandle pipeHandle[], CHAR *pipeName[], UINT32 pipeCount, INT32 waitMode);
MMPA_FUNC_VISIBILITY VOID mmClosePipe(mmPipeHandle pipeHandle[], UINT32 pipeCount);

// Poll related interface
MMPA_FUNC_VISIBILITY mmCompletionHandle mmCreateCompletionPort();
MMPA_FUNC_VISIBILITY VOID mmCloseCompletionPort(mmCompletionHandle handle);
MMPA_FUNC_VISIBILITY INT32 mmPoll(mmPollfd *fds,
                                  INT32 fdCount,
                                  INT32 timeout,
                                  mmCompletionHandle handleIOCP,
                                  pmmPollData polledData,
                                  mmPollBack pollBack);
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

MMPA_FUNC_VISIBILITY INT32 mmMsgSnd(mmMsgid msqid, VOID *buf, INT32 bufLen, INT32 msgFlag);

MMPA_FUNC_VISIBILITY INT32 mmMsgRcv(mmMsgid msqid, VOID *buf, INT32 bufLen, INT32 msgFlag);

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
MMPA_FUNC_VISIBILITY INT32 mmGetOpt(INT32 argc, CHAR *const *argv, const CHAR *opts);
MMPA_FUNC_VISIBILITY INT32 mmGetOptLong(INT32 argc,
                                        CHAR *const *argv,
                                        const CHAR *opts,
                                        const mmStructOption *longOpts,
                                        INT32 *longIndex);

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
MMPA_FUNC_VISIBILITY INT32 mmThreadKill(mmThread id);
MMPA_FUNC_VISIBILITY INT32 mmWaitPid(mmProcess pid, INT32 *status, INT32 options);

MMPA_FUNC_VISIBILITY INT32 mmGetCwd(CHAR *buffer, INT32 maxLen);
MMPA_FUNC_VISIBILITY INT32 mmGetEnv(const CHAR *name, CHAR *value, UINT32 len);
MMPA_FUNC_VISIBILITY INT32 mmSetEnv(const CHAR *name, const CHAR *value, INT32 overwrite);
MMPA_FUNC_VISIBILITY CHAR *mmStrTokR(CHAR *str, const CHAR *delim, CHAR **saveptr);
MMPA_FUNC_VISIBILITY CHAR *mmDirName(CHAR *path);
MMPA_FUNC_VISIBILITY CHAR *mmBaseName(CHAR *path);
MMPA_FUNC_VISIBILITY INT32 mmGetDiskFreeSpace(const CHAR *path, mmDiskSize *diskSize);

/*
 * Function: set the thread name created by mmcreatetask
 * Input: pstThreadHandle: thread ID
 *  name: thread name, the actual length of name must be < MMPA_THREADNAME_SIZE
 * The input parameter error returns EN_INVALID_PARAM, the execution success returns EN_OK, and the
 * execution failure returns EN_ERROR
 */
MMPA_FUNC_VISIBILITY INT32 mmSetThreadName(mmThread *threadHandle, const CHAR *name);

/*
 * Function: get thread name
 * Input: pstThreadHandle: thread ID
 *      size: Cache length of thread name
 *  name:User allocated cache for thread name, Cache length must be >= MMPA_THREADNAME_SIZE
 * The input parameter error returns EN_INVALID_PARAM, the execution success returns EN_OK, and the
 * execution failure returns EN_ERROR
 */
MMPA_FUNC_VISIBILITY INT32 mmGetThreadName(mmThread *threadHandle, CHAR *name, INT32 size);
/*
 * Function:Set the thread name of the currently executing thread - call inside the thread body
 * Input:name:Thread name to be set
 * The input parameter error returns EN_INVALID_PARAM, the execution success returns EN_OK, and the
 * execution failure returns EN_ERROR
 */
MMPA_FUNC_VISIBILITY INT32 mmSetCurrentThreadName(const CHAR *name);
/*
 * Function:Get the thread name of the currently executing thread - in body call
 * Input:name:The name of the thread to get, and the cache is allocated by the user，size>=MMPA_THREADNAME_SIZE
 * The input parameter error returns EN_INVALID_PARAM, the execution success returns EN_OK, and the
 * execution failure returns EN_ERROR
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
MMPA_FUNC_VISIBILITY INT32 mmCreateProcess(const CHAR *fileName,
                                           const mmArgvEnv *env,
                                           const CHAR *stdoutRedirectFile,
                                           mmProcess *id);

MMPA_FUNC_VISIBILITY INT32 mmCreateTaskWithThreadAttr(mmThread *threadHandle,
                                                      const mmUserBlock_t *funcBlock,
                                                      const mmThreadAttr *threadAttr);
MMPA_FUNC_VISIBILITY mmFileHandle mmShmOpen(const CHAR *name, INT32 oflag, mmMode_t mode);
MMPA_FUNC_VISIBILITY INT32 mmShmUnlink(const CHAR *name);
MMPA_FUNC_VISIBILITY VOID *mmMmap(mmFd_t fd, mmSize_t size, mmOfft_t offset, mmFd_t *extra, INT32 prot, INT32 flags);
MMPA_FUNC_VISIBILITY INT32 mmMunMap(VOID *data, mmSize_t size, mmFd_t *extra);

MMPA_FUNC_VISIBILITY mmSize mmGetPageSize();
MMPA_FUNC_VISIBILITY VOID *mmAlignMalloc(mmSize mallocSize, mmSize alignSize);
MMPA_FUNC_VISIBILITY VOID mmAlignFree(VOID *addr);
#define MMPA_DLL_API

#ifdef __cplusplus
#if __cplusplus
}
#endif /* __cpluscplus */
#endif // __cpluscplus

#endif // MMPA_LINUX_MMPA_LINUX_H_
