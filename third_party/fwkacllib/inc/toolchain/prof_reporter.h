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

#ifndef MSPROF_ENGINE_PROF_REPORTER_H_
#define MSPROF_ENGINE_PROF_REPORTER_H_
#ifndef OS_TYPE
#define OS_TYPE 0
#endif // OS_TYPE

#if (OS_TYPE != LINUX)
#define MSVP_PROF_API __declspec(dllexport)
#else
#define MSVP_PROF_API __attribute__((visibility("default")))
#endif

#include "prof_callback.h"

/**
 * @file prof_reporter.h
 * @defgroup reporter the reporter group
 * This is the reporter group
 */
namespace Msprof {
namespace Engine {
/**
 * @ingroup reporter
 * @brief class Reporter
 *  the Reporter class .used to send data to profiling
 */
class MSVP_PROF_API Reporter {
public:
    virtual ~Reporter() {}

public:
    /**
     * @ingroup reporter
     * @name  : Report
     * @brief : API of libmsprof, report data to libmsprof, it's a non-blocking function \n
                The data will be firstly appended to cache, if the cache is full, data will be ignored
    * @param data [IN] const ReporterData * the data send to libmsporf
    * @retval PROFILING_SUCCESS 0 (success)
    * @retval PROFILING_FAILED -1 (failed)
    *
    * @par depend:
    * @li libmsprof
    * @li prof_reporter.h
    * @since c60
    * @see Flush
    */
    virtual int Report(const ReporterData *data) = 0;

    /**
     * @ingroup reporter
     * @name  : Flush
     * @brief : API of libmsprof, notify libmsprof send data over, it's a blocking function \n
                The all datas of cache will be write to file or send to host
    * @retval PROFILING_SUCCESS 0 (success)
    * @retval PROFILING_FAILED -1 (failed)
    *
    * @par depend:
    * @li libmsprof
    * @li prof_reporter.h
    * @since c60
    * @see ProfMgrStop
    */
    virtual int Flush() = 0;

    virtual uint32_t GetReportDataMaxLen() = 0;
};

}  // namespace Engine
}  // namespace Msprof

#endif  // MSPROF_ENGINE_PROF_REPORTER_H_
