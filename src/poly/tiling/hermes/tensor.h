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
#ifndef POLY_TILING_HERMES_TENSOR_H_
#define POLY_TILING_HERMES_TENSOR_H_

#include <tvm/dtype.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace akg {
namespace ir {
namespace poly {
class Tensor {
 public:
  enum class DataType {
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Bool1,
    Bool8
  };
  std::unordered_map<std::string, Tensor::DataType> string_datatype_map_{
    {"float64", Tensor::DataType::Float64}, {"float32", Tensor::DataType::Float32},
    {"float16", Tensor::DataType::Float16}, {"int64", Tensor::DataType::Int64},
    {"int32", Tensor::DataType::Int32},     {"int16", Tensor::DataType::Int16},
    {"int8", Tensor::DataType::Int8},       {"uint64", Tensor::DataType::UInt64},
    {"uint32", Tensor::DataType::UInt32},   {"uint16", Tensor::DataType::UInt16},
    {"uint8", Tensor::DataType::UInt8},     {"bool", Tensor::DataType::Bool1},
    {"bool8", Tensor::DataType::Bool8}};

  Tensor();
  Tensor(const std::vector<int64_t> &, DataType, const std::string &);
  Tensor(const std::vector<int64_t> &, const std::string &, const std::string &);

  int GetDataTypeCoef() const;
  int64_t GetShapeProduct();
  bool operator<(const Tensor &) const;

  std::string ToString() const;
  bool IsScalar();
  static Tensor::DataType GetDataTypeFromTVM(const air::Type &tvm_dtype);

  std::vector<int64_t> shape_;
  DataType datatype_;
  std::string format_;
  std::string name_;

  static const int kOneBytePerVal = 1;
  static const int kTwoBytesPerVal = 2;
  static const int kFourBytesPerVal = 4;
  static const int kEightBytesPerVal = 8;

 private:
  std::unordered_map<DataType, int> datatype_coef_map_{
    {DataType::Float64, kEightBytesPerVal}, {DataType::Float32, kFourBytesPerVal}, {DataType::Float16, kTwoBytesPerVal},
    {DataType::Int64, kEightBytesPerVal},   {DataType::Int32, kFourBytesPerVal},   {DataType::Int16, kTwoBytesPerVal},
    {DataType::Int8, kOneBytePerVal},       {DataType::UInt64, kEightBytesPerVal}, {DataType::UInt32, kFourBytesPerVal},
    {DataType::UInt16, kTwoBytesPerVal},    {DataType::UInt8, kOneBytePerVal},     {DataType::Bool1, kOneBytePerVal},
    {DataType::Bool8, kFourBytesPerVal}};
  std::unordered_map<DataType, std::string> datatype_string_map_{
    {DataType::Float16, "float16"}, {DataType::Float32, "float32"}, {DataType::Float64, "float64"},
    {DataType::Int8, "int8"},       {DataType::Int16, "int16"},     {DataType::Int32, "int32"},
    {DataType::Int64, "int64"},     {DataType::UInt8, "uint8"},     {DataType::UInt16, "uint16"},
    {DataType::UInt32, "uint32"},   {DataType::UInt64, "uint64"},   {DataType::Bool1, "bool"},
    {DataType::Bool8, "bool8"}};
  std::unordered_map<int, DataType> float_bits_datatype_map_{{kThirtyTwoBits, DataType::Float32},
                                                             {kSixtyFourBits, DataType::Float64}};
  std::unordered_map<int, DataType> int_bits_datatype_map_{{kEightBits, DataType::Int8},
                                                           {kSixteenBits, DataType::Int16},
                                                           {kThirtyTwoBits, DataType::Int32},
                                                           {kSixtyFourBits, DataType::Int64}};
  std::unordered_map<int, DataType> uint_bits_datatype_map_{{kEightBits, DataType::UInt8},
                                                            {kSixteenBits, DataType::UInt16},
                                                            {kThirtyTwoBits, DataType::UInt32},
                                                            {kSixtyFourBits, DataType::UInt64}};
  std::unordered_map<int, DataType> bool_bits_datatype_map_{{kOneBit, DataType::Bool1}, {kEightBits, DataType::Bool8}};

  static const int kOneBit = 1;
  static const int kEightBits = 8;
  static const int kSixteenBits = 16;
  static const int kThirtyTwoBits = 32;
  static const int kSixtyFourBits = 64;
};

Tensor::DataType DataTypeFromString(const std::string &datatype);
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_TILING_HERMES_TENSOR_H_
