/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
 * Description: handle perf data
 * Author: xp
 * Create: 2019-10-13
 */

#ifndef MSPROF_ENGINE_PROF_REPORTER_H
#define MSPROF_ENGINE_PROF_REPORTER_H

#if (defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER))
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

#endif  // MSPROF_ENGINE_PROF_REPORTER_H
