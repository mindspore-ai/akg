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
#ifndef MS_CUSTOM_OPS_INTERNAL_HELPER_H_
#define MS_CUSTOM_OPS_INTERNAL_HELPER_H_

#include "include/api/format.h"
#include "include/internal.h"
#include "ir/anf.h"
#include "ir/dtype/type_id.h"
#include "kernel/ascend/visible.h"
#include "mindapi/base/shape_vector.h"
#include <memory>
#include <string>
#include <unordered_map>

using namespace mindspore;
namespace ms_custom_ops {
inline internal::ShapeInfo TransInternalShape(const ShapeVector &shape) {
  if (shape.size() != 0) {
    return shape;
  }
  internal::ShapeInfo internal_shape{1};
  return internal_shape;
}

bool CheckDefaultSupportFormat(const std::string &format);

internal::DataType TransInternalDataType(TypeId ms_type);

internal::TensorFormat TransInternalFormat(Format format);

class InternalNameMapper {
public:
  InternalNameMapper() = default;
  ~InternalNameMapper() = default;

  static InternalNameMapper &GetInstance();

  inline std::string GetInternalName(const std::string &ms_name) const {
    auto iter = ms_to_internal_mapper_.find(ms_name);
    if (iter == ms_to_internal_mapper_.end()) {
      return "";
    }

    return iter->second;
  }

  inline void Insert(const std::string &ms_name,
                     const std::string &internal_name) {
    ms_to_internal_mapper_[ms_name] = internal_name;
  }

private:
  std::unordered_map<std::string, std::string> ms_to_internal_mapper_;
};

class InternalNameRegistrar {
public:
  InternalNameRegistrar(const std::string &ms_name,
                        const std::string &internal_name) {
    InternalNameMapper::GetInstance().Insert(ms_name, internal_name);
  }
  ~InternalNameRegistrar() = default;
};
} // namespace ms_custom_ops
#endif // MS_CUSTOM_OPS_INTERNAL_HELPER_H_
