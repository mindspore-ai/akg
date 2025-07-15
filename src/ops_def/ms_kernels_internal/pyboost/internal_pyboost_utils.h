/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MS_CUSTOM_OPS_INTERNAL_PYBOOST_UTILS_H_
#define MS_CUSTOM_OPS_INTERNAL_PYBOOST_UTILS_H_

#include "internal_helper.h"
#include "internal_tiling_cache.h"
#include "kernel/ascend/acl_ir/op_api_cache.h"
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ms_custom_ops {
void GatherOpHash(const mindspore::tensor::TensorPtr &);
void GatherOpHash(const std::optional<mindspore::tensor::TensorPtr> &);
void GatherOpHash(const std::vector<mindspore::tensor::TensorPtr> &);
void GatherOpHash(const std::vector<int64_t> &);

template <typename T> void GatherOpHash(const T &value) {
  MemcpyToBuf(&value, sizeof(T));
}

 template <typename T> void GatherOpHash(std::optional<T> value) {
   if (value.has_value()) {
     GatherOpHash(value.value());
   }
 }

void GatherOpHash(const std::string &);
void GatherOpHash(const std::optional<string> &);

void GatherOpHash(const ScalarPtr &);
void GatherOpHash(const std::optional<ScalarPtr> &);

void GatherOpHash(const TypePtr &);
void GatherOpHash(const std::optional<TypePtr> &);

template <typename T> void GatherOpHash(const std::vector<T> &values) {
  MemcpyToBuf(reinterpret_cast<const void *>(values.data()),
              values.size() * sizeof(T));
}

 void GatherOpHash();

 template <typename T, typename... Args>
 void GatherOpHash(const T &arg, const Args &...args) {
   GatherOpHash(arg);
   GatherOpHash(args...);
}

template <typename... Args>
uint64_t CalcInternalOpApiHash(const std::string &arg, const Args &... args) {
  g_hash_offset = 0;
  GatherOpHash(arg, args...);
  return calc_hash_id();
}

void GatherTilingHash(const mindspore::tensor::TensorPtr &);
void GatherTilingHash(const std::optional<mindspore::tensor::TensorPtr> &);
void GatherTilingHash(const std::vector<mindspore::tensor::TensorPtr> &);
void GatherTilingHash(const std::vector<int64_t> &);

template <typename T> void GatherTilingHash(const T &value) {
  GatherOpHash(value);
}

 void GatherTilingHash();

 template <typename T, typename... Args>
 void GatherTilingHash(const T &arg, const Args &...args) {
   GatherTilingHash(arg);
   GatherTilingHash(args...);
}

template <typename... Args>
uint64_t CalcInternalOpTilingHash(const std::string &arg,
                                  const Args &... args) {
  GatherTilingHash(arg, args...);
  return calc_hash_id();
}

template <typename D, typename S>
void ConvertVectorDtype(std::vector<D> *dst_vec,
                        const std::vector<S> &src_vec) {
  dst_vec->clear();
  for (const auto &item : src_vec) {
    dst_vec->emplace_back(static_cast<D>(item));
  }
}

template <typename T> ValuePtr ConvertValue(const std::optional<T> &t) {
  if (t.has_value()) {
    return t.value();
  }
  return mindspore::kNone;
}

template <typename T> ValuePtr ConvertValue(const T &t) { return t; }

} // namespace ms_custom_ops
#endif // MS_CUSTOM_OPS_INTERNAL_PYBOOST_UTILS_H_
