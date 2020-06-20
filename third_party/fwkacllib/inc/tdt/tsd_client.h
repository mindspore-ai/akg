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

#ifndef TDT_HOST_INNER_INC_TSD_CLIENT_H_
#define TDT_HOST_INNER_INC_TSD_CLIENT_H_

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include "tdt/status.h"
#include "tdt/data_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/**
* @ingroup Open
* @brief Used for the Framework process to communicate with the TSDDaemon process,
* and notify TSD to complete the initialization of other processes
*
* @par Function
* Used for the Framework process to communicate with the TSDDaemon process,
* and notify TSD to complete the initialization of other processes
*
* @param phyDeviceId [IN] type #unsigned int. Physical device ID
* @param rankSize [IN] type #unsigned int. The rankSize of the training.
* The default value is 1. When rankSize is greater than 1,
* HCCP will be pulled to perform set communication related operations.
* @retval TDT_OK Success
* @retval OtherValues Failure
*
* @par Dependency
* @li libtsdclient.so: Library to which the interface belongs.
* @li tsd_client.h: Header file where the interface declaration is located.
* @li data_common.h: Header file where 'TDT_StatusT' defined
*/
TDT_StatusT TsdOpen(const uint32_t phyDeviceId, const uint32_t rankSize);

/**
* @ingroup Close
* @brief notify TSDClient close resource
*
* @par Function
* notify TSDClient close resource
*
* @param NA
* @retval TDT_OK Success
* @retval OtherValues Failure
*
* @par Dependency
* @li libtsdclient.so: Library to which the interface belongs.
* @li tsd_client.h: Header file where the interface declaration is located.
* @li data_common.h: Header file where 'TDT_StatusT' defined
*/
TDT_StatusT TsdClose(const uint32_t phyDeviceId);

/**
* @ingroup CreateCmdParameterObj
* @brief creat tsdclient func parameter obj.
*
* @par Function
* creat tsdclient func parameter obj.
*
* @param type [IN] type tdt::TsdCmdType, tsd func type.
* @param cmdParameterObj [IN] type void *, func parameter obj.
* @retval TDT_OK Success
* @retval TDT_INTERFACE_NOT_SUPPORT
*
* @par Dependency
* @li libtsdclient.so: Library to which the interface belongs.
* @li data_common.h: Header file where tdt::TsdCmdType and tdt::InputItem defined.
* @li status.h: Header file where 'TDT_StatusT' defined
*/
TDT_StatusT CreateCmdParameterObj(tdt::TsdCmdType type, void **cmdParameterObj);

/**
* @ingroup SetCmdParameterObjAttribute
* @brief set cmdParameterObj input value.
*
* @par Function
* set cmdParameterObj input value.
*
* @param type [IN] type tdt::TsdCmdType, tsd func type.
* @param cmdParameterObj [IN] type void *, func parameter obj.
* @param itemType [IN] type tdt::InputItem, func input type.
* @param valuePtr [IN] type const void *, input value.
* @param valueLength [IN] type int, input value length.
* @retval TDT_OK Success
* @retval TDT_INTERFACE_NOT_SUPPORT
*
* @par Dependency
* @li libtsdclient.so: Library to which the interface belongs.
* @li data_common.h: Header file where tdt::TsdCmdType and tdt::InputItem defined.
* @li status.h: Header file where 'TDT_StatusT' defined
*/
TDT_StatusT SetCmdParameterObjAttribute(tdt::TsdCmdType type, void *cmdParameterObj, tdt::InputItem itemType, const void *valuePtr, int valueLength);

/**
* @ingroup GetCmdParameterObjAttribute
* @brief set cmdParameterObj input value.
*
* @par Function
* set cmdParameterObj input value.
*
* @param type [IN] type tdt::TsdCmdType, tsd func type.
* @param cmdParameterObj [IN] type void *, func parameter obj.
* @param itemType [IN] type tdt::InputItem, func input type.
* @param valuePtr [IN] type const void *, input value.
* @param valueLength [IN] type int, input value length.
* @retval TDT_OK Success
* @retval TDT_INTERFACE_NOT_SUPPORT
*
* @par Dependency
* @li libtsdclient.so: Library to which the interface belongs.
* @li data_common.h: Header file where tdt::TsdCmdType and tdt::InputItem defined.
* @li status.h: Header file where 'TDT_StatusT' defined
*/
TDT_StatusT GetCmdParameterObjAttribute(tdt::TsdCmdType type, void *cmdParameterObj, tdt::InputItem itemType, void *valuePtr, int &valueLength);

/**
* @ingroup TsdClientCmd
* @brief creat tsdclient func parameter obj.
*
* @par Function
* creat tsdclient func parameter obj.
*
* @param type [IN] type tdt::TsdCmdType, tsd func type.
* @param cmdParameterObj [IN] type void *, func parameter obj.
* @retval TDT_OK Success
* @retval TDT_INTERFACE_NOT_SUPPORT
*
* @par Dependency
* @li libtsdclient.so: Library to which the interface belongs.
* @li data_common.h: Header file where tdt::TsdCmdType and tdt::InputItem defined.
* @li status.h: Header file where 'TDT_StatusT' defined
*/
TDT_StatusT TsdClientCmd(tdt::TsdCmdType cmd, void *cmdParameterObj);

namespace tdt {
/**
* @ingroup  RANK_SIZE_DEFAULT_VALUE。
* The default value of Rank size is 1 by default.
* It does not pull up HCCP to perform set communication related operations.
*
*/
constexpr uint32_t RANK_SIZE_DEFAULT_VALUE = 1;

class TsdClient {
 public:
  /**
  * @ingroup GetInstance
  * @brief Get TsdClient instance
  *
  * @par Function
  * Get TsdClient instance
  *
  * @param NA
  * @retval TsdClient TsdClient instance
  *
  * @par Dependency
  * @li libtsdclient.so: Library to which the interface belongs.
  * @li tsd_client.h: Header file where the interface declaration is located.
  */
  static TsdClient *GetInstance();

   /**
   * @ingroup ~TsdClient
   * @brief TsdClient destructor
   *
   * @par Function
   * TsdClient destructor
   *
   * @param NA
   * @retval NA
   *
   * @par Dependency
   * @li libtsdclient.so: Library to which the interface belongs.
   * @li tsd_client.h: Header file where the interface declaration is located.
   */
  ~TsdClient();

  /**
  * @ingroup Open
  * @brief Used for the Framework process to communicate with the TSDDaemon process,
  * and notify TSD to complete the initialization of other processes
  *
  * @par Function
  * Used for the Framework process to communicate with the TSDDaemon process,
  * and notify TSD to complete the initialization of other processes
  *
  * @param phyDeviceId [IN] type #unsigned int. Physical device ID
  * @param rankSize [IN] type #unsigned int. The rankSize of the training.
  * The default value is 1. When rankSize is greater than 1,
  * HCCP will be pulled to perform set communication related operations.
  * @retval TDT_OK Success
  * @retval OtherValues Failure
  *
  * @par Dependency
  * @li libtsdclient.so: Library to which the interface belongs.
  * @li tsd_client.h: Header file where the interface declaration is located.
  * @li data_common.h: Header file where 'TDT_StatusT' defined
  */
  TDT_StatusT Open(const uint32_t phyDeviceId, const uint32_t rankSize = RANK_SIZE_DEFAULT_VALUE);

  TDT_StatusT Close();

 private:
  TsdClient();
  TsdClient(const TsdClient &) = delete;
  TsdClient(TsdClient &&) = delete;
  TsdClient &operator=(const TsdClient &) = delete;
  TsdClient &operator=(TsdClient &&) = delete;

  uint32_t rankSize_;
};
}  // namespace tdt
#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // TDT_HOST_INNER_INC_TSD_CLIENT_H_
