/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef CONTRIB_CCE_PARM_CCECONF_H_
#define CONTRIB_CCE_PARM_CCECONF_H_

#include <map>
#include <string>
#include "contrib/cce_parm/cceparam.h"

namespace akg {
namespace cceconf {
class CceConf {
 public:
  CceConf();
  ~CceConf() { release(); }

  /* !
   * get the CceConf single instance
   */
  static CceConf *getInstance();

  /* !
   * base on the section and key, get value
   */
  int getBufferValue(const std::string &section, const std::string &key) const;

  /* !
   * base on the key, get value
   */
  int getBufferValue(const std::string &key) const;

  /* !
   * base on the section and key, get value
   */
  std::string getCompilerValue(const std::string &section, const std::string &key) const;

  /* !
   * base on the key, get value
   */
  std::string getCompilerValue(const std::string &key) const;

  /* !
   * base on the section and key, get value
   */
  std::string getIntrinsicValue(const std::string &section, const std::string &key) const;

  /* !
   * base on the section and key, get value
   */
  int getCoreValue(const std::string &section, const std::string &key) const;

  /* !
   * base on the key, get value
   */
  int getCoreValue(const std::string &key) const;

  /* !
   * set the debug_switch_ value
   */
  void setDebugSwitch(bool sw);

  /* !
   * get the debug_switch_ value
   */
  bool getDebugSwitch() const;

  /* !
   * get the product name
   */
  std::string getProductName() const;

  /* !
   * set the section name
   */
  void setSection(const std::string &section) { this->section_ = section; }

  /* !
   * get the section name
   */
  std::string getSection() const;

 private:
  /* !
   * release the resource
   */
  void release() noexcept;

  std::string getProductName(const std::string &section) const;

  /* !
   * single instance
   */
  static CceConf instance;

  /* !
   * the cce product parameters
   */
  std::map<std::string, CceParam> kvs_;

  /* !
   * get the product version
   */
  std::string section_;

  /* !
   * withch for check status special register
   */
  bool debug_switch_;
};
}  // namespace cceconf
}  // namespace akg

#endif  // CONTRIB_CCE_PARM_CCECONF_H_
