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
#ifndef CONTRIB_CCE_PARM_CCEPARAM_H_
#define CONTRIB_CCE_PARM_CCEPARAM_H_

#include <map>
#include <string>

namespace akg {
namespace cceconf {
class CceParam {
 public:
  CceParam();
  CceParam(const std::string &key, const std::map<std::string, int> &buffers,
           const std::map<std::string, std::string> &compiler, const std::map<std::string, std::string> &intrinsic,
           const std::map<std::string, int> &core);
  ~CceParam();

  std::string key_;
  /* !
   * base on the section and kye, get value
   */
  int getBufferValue(const std::string &key);
  std::string getCompilerValue(const std::string &key);
  std::string getIntrinsicValue(const std::string &key);
  int getCoreValue(const std::string &key);

 private:
  /* !
   * the cce product parameters of buffers size
   */
  std::map<std::string, int> buffers_;

  /* !
   * the cce product parameters of compiler params
   */
  std::map<std::string, std::string> compiler_;

  /* !
   * the cce product parameters of intrinsic
   */
  std::map<std::string, std::string> intrinsic_;

  /* !
   * the cce product parameters of core number
   */
  std::map<std::string, int> core_;
};
}  // namespace cceconf
}  // namespace akg
#endif  // CONTRIB_CCE_PARM_CCEPARAM_H_
