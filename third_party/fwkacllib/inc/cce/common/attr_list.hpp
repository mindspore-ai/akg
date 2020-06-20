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

#ifndef ATTR_LIST_HPP__
#define ATTR_LIST_HPP__

#include "catch.hpp"

/**
 * @ingroup util
 * @brief frame  Error Value
 */
#define ATTR_SUCCESS (0)
#define ATTR_ERROR_NULL_POINT (1)
#define ATTR_ERROR_ALREADY_EXIST (2)
#define ATTR_ERROR_NOT_EXIST (3)
#define ATTR_ERROR_BUFFER_NOT_ENOUGH (4)
#define ATTR_ERROR_BAD_PARAM (5)
#define ATTR_ERROR_ALLOC_FAIL (6)
#define ATTR_ERROR_FREE_FAIL (7)
#define ATTR_ERROR_RESERVED (8)

struct AttrListPrivate;
/**
 * @ingroup util
 * @brief attribute list
 */
class AttrList {
 public:
  AttrList();
  AttrList(uint32_t initLen);
  ~AttrList();
  AttrList(const AttrList &rhs) = delete;
  AttrList &operator=(const AttrList &rhs);

 public:
  /**
   * @ingroup util
   * @brief add paras
   * @param [in] attrId   attribute id
   * @param [in] attrLen   length of attribute
   * @param [in] attrValue   point to attribute
   * @return ccStatus_t
   */
  uint32_t Add(uint32_t attrId, uint32_t attrLen, const void *attrValue);

  /**
   * @ingroup util
   * @brief read paras
   * @param [in] attrId   attribute id
   * @param [in] attrLen   point to length of attribute
   * @param [in] attrValue   reference of point to attribute
   * @return ccStatus_t
   */
  uint32_t Get(uint32_t attrId, uint32_t &attrLen, const void *&attr_value) const;

  /**
   * @ingroup util
   * @brief get the length of attribute list
   * @return length of attribute
   */
  uint32_t Length() const;

 private:
  AttrListPrivate *impl_;
  uint32_t initLen_;
  uint32_t Init();
};
#endif  // ATTR_LIST_HPP__
