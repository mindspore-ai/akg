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
#ifndef SWFT_PYBIND_H
#define SWFT_PYBIND_H

#include <string>
#include <memory>
#include <vector>
#include <iostream>
#include <sstream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

enum DataType : int;
enum FormatType : int;
enum MemType : int;
enum AscendPosition : int;

class Value;
using ValuePtr = std::shared_ptr<Value>;
class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;
template <typename T>
class Scalar;
template <typename T>
using ScalarPtr = std::shared_ptr<Scalar<T>>;

class Value {
 public:
  Value();
  Value(const std::string &dtype);
  Value(DataType dtype);
  void updateName(const std::string &name);
  void updatePosition(int idx, int func_postion);
  void updateFuncPosition(const std::map<int, int> &func_postion);
  std::string getName() const;
  int getPosition(int idx) const;
  std::map<int, int> getFuncPosition() const;
  DataType getDataType() const;
  float getDataTypeSize() const;
  std::string getDataTypeStr() const;
  void setDataType(DataType dtype);
  void setDataType(const std::string &dtype);
  int getId() const;
  void setId(int id);
  virtual ValuePtr defaultCopy();
  const virtual std::string getType() const;
  const virtual std::string getClassType() const;
  virtual std::string toString() const;
  virtual MemType getMemType() const;
  virtual std::string getValueStr(bool useTile = false) const;
  virtual bool isWorkspace() const;
  virtual void setWorkspaceAttr(bool ws);
  static int unique_id;

 private:
  int id_;
  std::string name_{};
  DataType dtype_;
  std::map<int, int> func_postion_;
};

class Tensor : public Value {
 public:
  Tensor(const std::string &mem_type, const bool multi_core);
  Tensor(const std::string &mem_type, const std::string &dtype, const py::list &shape, const std::string &format,
         const bool multi_core);
  Tensor(MemType mem_type, DataType dtype, const std::vector<Scalar<int>> &shape, FormatType format, bool multi_core);
  explicit Tensor(py::array &numpy_array, const std::string &format, const bool multi_core);
  ValuePtr defaultCopy() override;
  const std::string getType() const override;
  std::string toString() const override;
  std::string getValueStr(bool useTile = false) const override;
  bool isWorkspace() const override;
  void setWorkspaceAttr(bool ws) override;

  void updateParam(const std::string &dtype, const py::list &shape, const std::string &format, const bool multi_core);

  int getShapeSize() const;
  const std::string getClassType() const override;
  std::vector<Scalar<int>> getShape() const;
  const std::vector<int> getStaticShape() const;
  Scalar<int> getShape(int idx) const;
  int getStaticShape(int idx) const;
  void setShape(const std::vector<Scalar<int>> &shape);
  void setShape(int idx, const Scalar<int> &value);
  int getStaticSize() const;
  Scalar<int> getSize() const;
  Scalar<int> getByteSize() const;
  int getMaxByteSize() const;
  int getRoundUpByteSize() const;
  const bool isMultiCore() const;
  bool hasBatch() const;
  int getAddr() const;
  void setAddr(int addr);
  FormatType getFormat() const;
  const std::string getFormatStr() const;
  MemType getMemType() const override;
  const std::string getMemTypeStr() const;
  const std::string getDTypePy() const;
  const std::string getMemTypePy() const;
  const std::string getFormatPy() const;
  const std::string getAscendName() const;
  bool hasCpuAddr() const;
  bool hasNpuAddr() const;
  void syncDeviceToHost();
  void syncHostToDevice();
  const py::array asNumpy();
  void *getHostDataPtr();
  void *getDeviceDataPtr();
  void setAscendPos(AscendPosition pos);
  AscendPosition getAscendPos();

 private:
  std::string mem_type_str_;
  std::string dtype_str_;
  std::vector<Scalar<int>> shape_;
  std::string format_str_;
  bool multi_core_;
  std::shared_ptr<uint8_t> host_data_ptr_;
  std::shared_ptr<uint8_t> dev_data_ptr_;
  bool host_initialized_ = false;
  bool device_initialized_ = false;
  MemType mem_type_;
  FormatType format_;
  int addr_ = -1;
  bool is_workspace_ = false;
  AscendPosition ascendPos_;
};

template <typename T>
class Scalar : public Value {
 public:
  Scalar(const std::string &dtype);
  Scalar(const std::string &dtype, const T value);
  Scalar(DataType dtype);
  Scalar(DataType dtype, const T value);
  ValuePtr defaultCopy() override;
  const std::string getClassType() const override;
  MemType getMemType() const override;
  T getValue() const;
  bool hasValue() const;
  void updateValue(T value);
  int getTile() const;
  bool hasTile() const;
  void setTile(const int tile);
  bool isFloatType() const;
  bool isIntType() const;
  bool isUIntType() const;
  std::string getValueStr(bool useTile = false) const override;
  const std::string getType() const override;
  std::string toString() const override;
  std::string getDtypeStr() const;
  Scalar<T> deepcopy(const py::dict &memo) const;
  Scalar<T> operator+(const Scalar<T> &other) const;
  Scalar<T> operator+(T other) const;
  template <typename U>
  friend Scalar<U> operator+(U other, const Scalar<U> &scalar);
  Scalar<T> operator*(const Scalar<T> &other) const;
  Scalar<T> operator*(T other) const;
  template <typename U>
  friend Scalar<U> operator*(U other, const Scalar<U> &scalar);
  Scalar<T> operator-(const Scalar<T> &other) const;
  Scalar<T> operator-(T other) const;
  template <typename U>
  friend Scalar<U> operator-(U other, const Scalar<U> &scalar);
  Scalar<T> operator/(const Scalar<T> &other) const;
  Scalar<T> operator/(T other) const;
  template <typename U>
  friend Scalar<U> operator/(U other, const Scalar<U> &scalar);
  Scalar<int> operator%(const Scalar<int> &other) const;
  Scalar<int> operator%(int other) const;
  friend Scalar<int> operator%(int other, const Scalar<int> &scalar);
  Scalar<uint64_t> operator<<(const Scalar<int> &other) const;
  Scalar<uint64_t> operator<<(int other) const;
  friend Scalar<uint64_t> operator<<(uint64_t other, const Scalar<int> &scalar);
  bool operator==(const Scalar<T> &other) const;
  bool operator==(T other) const;
  template <typename U>
  friend bool operator==(U other, const Scalar<U> &scalar);
  bool operator!=(const Scalar<T> &other) const;
  bool operator!=(T other) const;
  template <typename U>
  friend bool operator!=(U other, const Scalar<U> &scalar);
  bool operator<=(const Scalar<T> &other) const;
  bool operator<=(T other) const;
  template <typename U>
  friend bool operator<=(U other, const Scalar<U> &scalar);
  bool operator>=(const Scalar<T> &other) const;
  bool operator>=(T other) const;
  template <typename U>
  friend bool operator>=(U other, const Scalar<U> &scalar);
  bool operator<(const Scalar<T> &other) const;
  bool operator<(T other) const;
  template <typename U>
  friend bool operator<(U other, const Scalar<U> &scalar);
  bool operator>(const Scalar<T> &other) const;
  bool operator>(T other) const;
  template <typename U>
  friend bool operator>(U other, const Scalar<U> &scalar);
  Scalar<T> &operator=(T other);
  Scalar<T> &operator*=(const Scalar<T> &other);
  Scalar<T> &operator*=(T other);
  Scalar<T> &operator+=(const Scalar<T> &other);
  Scalar<T> &operator+=(T other);
  Scalar<T> &operator-=(const Scalar<T> &other);
  Scalar<T> &operator-=(T other);
  Scalar<T> &operator/=(const Scalar<T> &other);
  Scalar<T> &operator/=(T other);
  T value_;
  int tile_;
  std::string dtype_str_;

 private:
  bool has_value_ = false;
  bool has_tile_ = false;
};

class NPUSession {
 public:
  NPUSession(const int device_id);
  void *getStream();
  const int getCurrentDevice();
  void syncStream();
  ~NPUSession();

 private:
  int device_id_;
  void *stream_;
};

std::ostream &operator<<(std::ostream &os, const Value &value);
extern void newSubKernel(const int core_num, const std::string &kernel_name);
extern void newSyncSubKernel(const int core_num, const std::string &kernel_name, const Tensor &sync_value);
extern void setContext(const std::string &chip_type, const std::string &code_type = "ASCENDC");
extern std::string getContext();
extern void pushToList(int index, const std::string &name, const std::vector<Tensor> &input_tensor,
                       const std::vector<Scalar<double>> &input_scalar, const std::vector<Tensor> &output_tensor,
                       const std::vector<Scalar<double>> &output_scalar, const std::vector<std::string> &key,
                       const std::vector<std::vector<float>> &value);
extern bool compileKernel(const std::string &file_loc, const std::string &name = "", bool hard_sync = false,
                          int kernel_idx = -1);
extern const bool canFitMemory(const int size, const std::string &memtype);
#endif