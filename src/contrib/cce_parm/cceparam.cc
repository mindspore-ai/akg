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
#include <map>
#include <string>
#include "contrib/cce_parm/cceparam.h"

using std::map;
using std::string;

namespace akg {
namespace cceconf {
CceParam::CceParam(const string &key, const map<string, int> &buffers, const map<string, string> &compiler,
                   const map<string, string> &intrinsic, const map<string, int> &core) {
  key_ = key;
  buffers_ = buffers;
  compiler_ = compiler;
  intrinsic_ = intrinsic;
  core_ = core;
}
CceParam::CceParam() {}
CceParam::~CceParam() {
  buffers_.clear();
  compiler_.clear();
  intrinsic_.clear();
  core_.clear();
}

/*!
 * base on the key, get buffer value
 */
int CceParam::getBufferValue(const string &key) {
  auto it = buffers_.find(key);
  if (it != buffers_.end()) {
    return it->second;
  }
  return 0;
}

/*!
 * base on the key, get compiler value
 */
string CceParam::getCompilerValue(const string &key) {
  auto it = compiler_.find(key);
  if (it != compiler_.end()) {
    return it->second;
  }
  return "";
}

/*!
 * base on the key, get intrinsic value
 */
string CceParam::getIntrinsicValue(const string &key) {
  auto it = intrinsic_.find(key);
  if (it != intrinsic_.end()) {
    return it->second;
  }
  return "";
}

/*!
 * base on the key, get intrinsic value
 */
int CceParam::getCoreValue(const string &key) {
  auto it = core_.find(key);
  if (it != core_.end()) {
    return it->second;
  }
  return 0;
}
}  // namespace cceconf
}  // namespace akg
