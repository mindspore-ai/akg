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

#ifndef MSPROF_ENGINE_PROF_ENGINE_H_
#define MSPROF_ENGINE_PROF_ENGINE_H_
#define MSVP_PROF_API __attribute__((visibility("default")))

#include <map>
#include <string>
#include "prof_reporter.h"

/**
 * @file prof_engine.h
 * @defgroup ModuleJobConfig the ModuleJobConfig group
 * This is the ModuleJobConfig group
 */
namespace Msprof {
namespace Engine {
/**
 * @ingroup ModuleJobConfig
 * @brief struct ModuleJobConfig
 * record config info
 */
struct ModuleJobConfig {
  std::map<std::string, std::string> switches; /**< key is the config name, value is the config value(on or off) */
};

/**
 * @defgroup PluginIntf the pluginInf group
 * This is the pluginInf group
 */

/**
 * @ingroup PluginIntf
 * @brief class PluginIntf
 */
class MSVP_PROF_API PluginIntf {
 public:
  virtual ~PluginIntf() {}

 public:
  /**
   * @ingroup PluginIntf
   * @name  : Init
   * @brief : API of user plugin, libmsporf call this API to send a Reporter to user plugin
   * @par description :
   *  API of user plugin, libmsporf call this API to send a Reporter to user plugin.
   * @param reporter [IN] const Reporter* the Reporter from libmsprof
   * @retval PROFILING_SUCCESS 0 (success)
   * @retval PROFILING_FAILED -1 (failed)
   *
   * @par depend:
   * @li libmsprof
   * @li prof_engine.h
   * @since c60
   * @see UnInit
   */
  virtual int Init(const Reporter *reporter) = 0;

  /**
   * @ingroup PluginIntf
   * @name  : OnNewConfig
   * @brief : API of user plugin, libmsprof call this API to send config info to user plugin \n
              If the user plugin needn't config, no need to redefine this function
   * @param config [IN] const ModuleJobConfig * the config from libmsprof
   * @retval PROFILING_SUCCESS 0 (success)
   * @retval PROFILING_FAILED -1 (failed)
   *
   * @par depend:
   * @li libmsprof
   * @li prof_engine.h
   * @since c60
   * @see Init | UnInit
   */
  virtual int OnNewConfig(const ModuleJobConfig *config) { return 0; }

  /**
   * @ingroup PluginIntf
   * @name  : UnInit
   * @brief : API of user plugin, libmsprof call this API to notify plugin stop to send data
   * @retval PROFILING_SUCCESS 0 (success)
   * @retval PROFILING_FAILED -1 (failed)
   *
   * @par depend:
   * @li libmsprof
   * @li prof_engine.h
   * @since c60
   * @see Init
   */
  virtual int UnInit() = 0;
};

/**
 *  @defgroup EngineIntf  the EngineIntf group
 *  This is the EngineIntf group
 */

/**
 *  @ingroup EngineIntf
 *  @brief class EngineIntf
 */
class MSVP_PROF_API EngineIntf {
 public:
  virtual ~EngineIntf() {}

 public:
  /**
   * @ingroup EngineIntf
   * @name  : CreatePlugin
   * @brief : API of user engine, libmsporf call this API to get a plugin
   * @retval PluginIntf * The pointer of the new plugin
   *
   * @par depend:
   * @li libmsprof
   * @li prof_engine.h
   * @since c60
   * @see ReleasePlugin
   */
  virtual PluginIntf *CreatePlugin() = 0;

  /**
   * @ingroup EngineIntf
   * @name  : ReleasePlugin
   * @brief : API of user engine, libmsprof call this API to release a plugin
   * @param plugin [IN] PluginIntf * the plugin to release
   * @retval PROFILING_SUCCESS 0 (success)
   * @retval PROFILING_FAILED -1 (failed)
   *
   * @par depend:
   * @li libmsprof
   * @li prof_engine.h
   * @since c60
   * @see CreatePlugin
   */
  virtual int ReleasePlugin(PluginIntf *plugin) = 0;
};

/**
 *  @defgroup EngineMgr  the EngineMgr group
 *  This is the EngineMgr group
 */

/**
 * @ingroup EngineMgr
 * @name  : RegisterEngine
 * @brief : API of libmsprof, register an engine with a name
 * @param module [IN] const std::string the name of plugin
 * @param engine [IN] const EngineIntf* the plugin
 * @retval PROFILING_SUCCESS 0 (success)
 * @retval PROFILING_FAILED -1 (failed)
 *
 * @par depend:
 * @li libmsprof
 * @li prof_engine.h
 * @since c60
 */
MSVP_PROF_API int RegisterEngine(const std::string &module, const EngineIntf *engine);

/**
 * @ingroup EngineMgr
 * @name  : Init
 * @brief : API of libmsprof, init an engine with a name
 * @param module [IN] const std::string  the name of plugin
 * @param module [IN] const EngineIntf*  the plugin
 * @retval PROFILING_SUCCESS 0 (success)
 * @retval PROFILING_FAILED -1 (failed)
 *
 * @par depend:
 * @li libmsprof
 * @li prof_engine.h
 * @since c60
 * @see UnInit
 */
MSVP_PROF_API int Init(const std::string &module, const EngineIntf *engine);

/**
 * @ingroup EngineMgr
 * @name  : Init
 * @brief : API of libmsprof, uninit an engine with a name
 * @param module [IN] const std::string the name of plugin
 * @retval PROFILING_SUCCESS 0 (success)
 * @retval PROFILING_FAILED -1 (failed)
 *
 * @par depend:
 * @li libmsprof
 * @li prof_engine.h
 * @since c60
 * @see Init
 */
MSVP_PROF_API int UnInit(const std::string &module);
}  // namespace Engine
}  // namespace Msprof

#endif  // MSPROF_ENGINE_PROF_ENGINE_H_