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

#ifndef MSPROF_ENGINE_PROF_MGR_CORE_H_
#define MSPROF_ENGINE_PROF_MGR_CORE_H_
#ifndef OS_TYPE
#define OS_TYPE 0
#endif // OS_TYPE

#if (OS_TYPE != LINUX)
#define MSVP_PROF_API __declspec(dllexport)
#else
#define MSVP_PROF_API __attribute__((visibility("default")))
#endif


#include <string>
#include <vector>

/**
 * @file prof_mgr_core.h
 * @brief : struct ProfMgrCfg
 */
struct ProfMgrCfg {
  std::string startCfg; /**< start cfg. json format */
};

/**
 * @name  : ProfMgrConf
 * @brief : struct ProfMgrConf for example [{"ai_core_events":"0xa"}].the vector size means Number of iterations
 */
struct ProfMgrConf {
  std::vector<std::string> conf; /**< for op trace.Ge call this api to get each iteration profiling cfg.json format.*/
};

/**
 * @name  : ProfMgrStartUP
 * @brief : start Profiling task
 * @param cfg [IN]ProfMgrCfg cfg : config of start_up profiling
 * @retval void * (success)
 * @retval nullptr (failed)
 *
 * @par depend:
 * @li libmsprof
 * @li prof_mgr_core.h
 * @since c60
 * @see ProfMgrStop
 */
MSVP_PROF_API void *ProfMgrStartUp(const ProfMgrCfg *cfg);

/**
 * @name  : ProfMgrStop
 * @brief : stop Profiling task
 * @param handle [in] void * handle return by ProfMgrStartUP
 * @retval PROFILING_SUCCESS 0 (success)
 * @retval PROFILING_FAILED -1 (failed)
 *
 * @par depend:
 * @li libmsprof
 * @li prof_mgr_core.h
 * @since c60
 * @see ProfMgrStartUp
 */
MSVP_PROF_API int ProfMgrStop(void *handle);

/**
 * @name  : ProfMgrGetConf
 * @brief : get profiler events conf
 * @param conf [OUT]ProfMgrConf * return by ProfMgrGetConf
 * @retval PROFILING_SUCCESS 0 (success)
 * @retval PROFILING_FAILED -1 (failed)
 * @par depend:
 * @li libmsprof
 * @li prof_mgr_core.h
 * @since c60
 * @see ProfMgrStartUp
 */
MSVP_PROF_API int ProfMgrGetConf(const std::string &aicoreMetricsType, ProfMgrConf *conf);

#endif  // MSPROF_ENGINE_PROF_MGR_CORE_H_