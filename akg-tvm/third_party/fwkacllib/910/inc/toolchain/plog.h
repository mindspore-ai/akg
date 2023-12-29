/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef PLOG_H_
#define PLOG_H_

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#ifndef LINUX
#define LINUX 0
#endif // LINUX

#ifndef WIN
#define WIN 1
#endif

#ifndef OS_TYPE
#define OS_TYPE 0
#endif // OS_TYPE

#if (OS_TYPE == LINUX)
#define DLL_EXPORT __attribute__((visibility("default")))
#else
#define DLL_EXPORT _declspec(dllexport)
#endif

/**
 * @ingroup plog
 * @brief DlogReportInitialize: init log in service process before all device setting.
 * @return: 0: SUCCEED, others: FAILED
 */
DLL_EXPORT int DlogReportInitialize(void);

/**
 * @ingroup plog
 * @brief DlogReportFinalize: release log resource in service process after all device reset.
 * @return: 0: SUCCEED, others: FAILED
 */
DLL_EXPORT int DlogReportFinalize(void);

/**
 * @ingroup     : plog
 * @brief       : create thread to recv log from device
 * @param[in]   : devId         device id
 * @param[in]   : mode          use macro LOG_SAVE_MODE_XXX in slog.h
 * @return      : 0: SUCCEED, others: FAILED
 */
DLL_EXPORT int DlogReportStart(int devId, int mode);

/**
 * @ingroup     : plog
 * @brief       : stop recv thread
 * @param[in]   : devId         device id
 */
DLL_EXPORT void DlogReportStop(int devId);


#ifdef __cplusplus
}
#endif // __cplusplus
#endif // PLOG_H_
