/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include <dmlc/logging.h>

#include "poly/tiling/hermes/tensor.h"

namespace akg {
namespace ir {
namespace poly {
Tensor::Tensor(const std::vector<int> &shape, DataType datatype, const std::string &format)
    : shape_{shape}, datatype_{datatype}, format_{format}, name_{} {}

Tensor::Tensor(const std::vector<int> &shape, const std::string &datatype, const std::string &format)
    : shape_{shape}, datatype_{DataTypeFromString(datatype)}, format_{format}, name_{} {}

int Tensor::GetDataTypeCoef() const {
  auto datatype_coef = datatype_coef_map_.find(datatype_);
  if (datatype_coef != datatype_coef_map_.end()) {
    return datatype_coef->second;
  }
  LOG(FATAL) << "[Tensor::getDataTypeCoef] This data type is not taken into account yet";
  return 0;
}

int Tensor::GetShapeProduct() {
  int product = 1;
  for (auto iter = this->shape_.begin(); iter != this->shape_.end(); iter++) {
    product *= *iter;
  }
  return product;
}

bool Tensor::operator<(const Tensor &other_tensor) const { return false; }

std::string Tensor::ToString() const {
  auto datatype_string = datatype_string_map_.find(datatype_);
  if (datatype_string != datatype_string_map_.end()) {
    return datatype_string->second;
  }
  LOG(FATAL) << "[Tensor::ToString] This data type is not taken into account yet";
  return "";
}

bool Tensor::IsScalar() { return ((shape_.size() == 1) && (shape_[0] == 1)); }

Tensor::DataType Tensor::GetDataTypeFromTVM(const air::Type &tvm_dtype) {
  if (tvm_dtype.is_float16()) {
    return Tensor::DataType::Float16;
  }
  if (tvm_dtype.is_float() && tvm_dtype.bits() == kThirtyTwoBits) {
    return Tensor::DataType::Float32;
  }
  if (tvm_dtype.is_float() && tvm_dtype.bits() == kSixtyFourBits) {
    return Tensor::DataType::Float64;
  }
  if (tvm_dtype.is_int() && tvm_dtype.bits() == kEightBits) {
    return Tensor::DataType::Int8;
  }
  if (tvm_dtype.is_int() && tvm_dtype.bits() == kSixteenBits) {
    return Tensor::DataType::Int16;
  }
  if (tvm_dtype.is_int() && tvm_dtype.bits() == kThirtyTwoBits) {
    return Tensor::DataType::Int32;
  }
  if (tvm_dtype.is_int() && tvm_dtype.bits() == kSixtyFourBits) {
    return Tensor::DataType::Int64;
  }
  if (tvm_dtype.is_uint() && tvm_dtype.bits() == kEightBits) {
    return Tensor::DataType::UInt8;
  }
  if (tvm_dtype.is_uint() && tvm_dtype.bits() == kSixteenBits) {
    return Tensor::DataType::UInt16;
  }
  if (tvm_dtype.is_uint() && tvm_dtype.bits() == kThirtyTwoBits) {
    return Tensor::DataType::UInt32;
  }
  if (tvm_dtype.is_uint() && tvm_dtype.bits() == kSixtyFourBits) {
    return Tensor::DataType::UInt64;
  }
  if (tvm_dtype.is_bool() && tvm_dtype.bits() == kOneBit) {
    return Tensor::DataType::Bool8;
  }
  if (tvm_dtype.is_bool() && tvm_dtype.bits() == kEightBits) {
    return Tensor::DataType::Bool8;
  }
  LOG(FATAL) << "[GetDataTypeFromTVM] The data type is not taken into account yet";
  return Tensor::DataType::Float16;
}

Tensor::DataType DataTypeFromBytes(const int bytes) {
  Tensor tensor;
  auto datatype_coef = tensor.bytes_datatype_map_.find(bytes);
  if (datatype_coef != tensor.bytes_datatype_map_.end()) {
    return datatype_coef->second;
  }
  LOG(FATAL) << "The number of bytes " << bytes << " is not taken into account yet";
  return Tensor::DataType::Float32;
}

Tensor::DataType DataTypeFromString(const std::string &datatype) {
  Tensor tensor;
  auto datatype_coef = tensor.string_datatype_map_.find(datatype);
  if (datatype_coef != tensor.string_datatype_map_.end()) {
    return datatype_coef->second;
  }
  LOG(FATAL) << "[datatypeFromString] The data type " + datatype + " is not taken into account yet";
  return Tensor::DataType::Float16;
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
