/**
* @file data_common.h
*
* Copyright (C) Huawei Technologies Co., Ltd. 2019-2020. All Rights Reserved.
*
* This program is used to data structure
*/

#ifndef HOST_INNER_INC_DATA_COMMON_H_
#define HOST_INNER_INC_DATA_COMMON_H_
#include <string>

namespace tdt {
#ifndef TDT_DATA_TYPE
#define TDT_DATA_TYPE

/**
 * @ingroup  Tdt data.
 *
 * Tdt data type.
 */
enum TdtDataType {
  TDT_IMAGE_LABEL = 0, /**< Image label*/
  TDT_TFRECORD,        /**< TF Record*/
  TDT_DATA_LABEL,      /**< Data label*/
  TDT_END_OF_SEQUENCE, /**< End of Sequence*/
  TDT_TENSOR,          /**< Tensor*/
  TDT_ABNORMAL,        /**< ABNORMAL*/
  TDT_DATATYPE_MAX     /**< Max*/
};
#endif

/**
 * @ingroup  Tdt data.
 *
 * Tdt push data between host and device.
 */
struct TdtDataItem {
  TdtDataType dataType_;          /**< Input data type*/
  uint64_t label_;                /**< Input data label*/
  uint64_t dataLen_;              /**< Input data type length*/
  uint64_t realDataLen_;          /**< Real Input data type length*/
  std::string tensorShape_;       /**< Tensor shape*/
  std::string tensorType_;        /**< Tensor type*/
  uint32_t cnt_;                  /**< Data  count*/
  uint32_t currentCnt_;           /**< Data  current count*/
  uint64_t index_;                /**< Data  inde*/
  std::string tensorName_;        /**< Tensor  name*/
  uint64_t md5ValueHead_;         /**< Data  md5*/
  uint64_t md5ValueTail_;         /**< Data  md5*/
  std::shared_ptr<void> dataPtr_; /**< Data  pointer*/
  std::string headMD5_;           /**< MD5 header, 8byte*/
  std::string tailMD5_;           /**< MD5 tail, 8byte*/
};

/**
 * @ingroup  Tdt data.
 *
 * Tdt push data for queuedataset ort mind-data.
 */
struct DataItem {
  TdtDataType dataType_;          /**< Input data type*/
  std::string tensorName_;        /**< Tensor  name*/
  std::string tensorShape_;       /**< Tensor shape*/
  std::string tensorType_;        /**< Tensor type*/
  uint64_t dataLen_;              /**< Input data type length*/
  std::shared_ptr<void> dataPtr_; /**< Data  pointer*/
};

/**
 * @ingroup  Tsdclient.
 *
 * tsdclient func type;
 */
enum TsdCmdType {
    TSDCLOSE = 0,
    TSDOPEN = 1
};

/**
 * @ingroup  Tsdclient.
 *
 * tsdclient func input value object.
 */
enum InputItem {
    OPEN_DEVICEID = 0,
    OPEN_RANKSIZE,
    CLOSE_DEVICEID
};

}  // namespace tdt
#endif  // HOST_INNER_INC_DATA_COMMON_H_
