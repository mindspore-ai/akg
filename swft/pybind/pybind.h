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
#include <pybind11/stl.h>

namespace py = pybind11;

enum DataType :int;
enum FormatType :int;
enum MemType :int;

class Value;
using ValuePtr = std::shared_ptr<Value>;
class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;
class Scalar;
using ScalarPtr = std::shared_ptr<Scalar>;

class Value {
public:
    Value();
    Value(const std::string &dtype);
    Value(DataType dtype);
    void updateName(const std::string &name);
    void updatePosition(int idx, int func_postion);
    std::string getName() const;
    int getPosition(int idx) const;
    DataType getDataType() const;
    float getDataTypeSize() const;
    std::string getDataTypeStr() const;
    void setDataType(DataType dtype);
    void setDataType(const std::string &dtype);
    int getId() const;
    void setId(int id);
    virtual ValuePtr defaultCopy();
    const virtual std::string getType() const;
    virtual std::string toString() const;
    virtual MemType getMemType() const;
    virtual std::string getValueStr() const;
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
    Tensor(MemType mem_type, DataType dtype, const std::vector<int> &shape, FormatType format, bool multi_core);

    ValuePtr defaultCopy() override;
    const std::string getType() const override;
    std::string toString() const override;
    std::string getValueStr() const override;
    bool isWorkspace() const override;
    void setWorkspaceAttr(bool ws) override;

    void updateParam(const std::string &dtype, const py::list &shape, const std::string &format, const bool multi_core);

    int getShapeSize() const;
    const std::vector<int> getShape() const;
    const std::vector<int> getShapePy() const;
    int getShape(int idx) const;
    void setShape(const std::vector<int> &shape);
    void setShape(int idx, int value);
    int getSize() const;
    int getByteSize() const;
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
private:
    std::string mem_type_str_;
    std::string dtype_str_;
    std::vector<int> shape_;
    std::string format_str_;
    bool multi_core_;
    MemType mem_type_;
    FormatType format_;
    int addr_ = -1;
    bool is_workspace_ = false;
};

class Scalar : public Value {
public:
    Scalar(const std::string &dtype);
    Scalar(const std::string &dtype, const float value);
    Scalar(DataType dtype);
    Scalar(DataType dtype, const float value);
    ValuePtr defaultCopy() override;
    MemType getMemType() const override;
    float getValue() const;
    bool hasValue() const;
    bool isFloatType() const;
    std::string getValueStr() const override;
    const std::string getType() const override;
    std::string toString() const override;
    void updateValue(float value);
    std::string getDtypeStr() const;
    Scalar operator+(const Scalar &other) const;
    Scalar operator+(float other) const;
    Scalar operator*(const Scalar &other) const;
    Scalar operator*(float other) const;
    Scalar operator-(const Scalar &other) const;
    Scalar operator-(float other) const;
    Scalar operator/(const Scalar &other) const;
    Scalar operator/(float other) const;
    float value_;
    std::string dtype_str_;
private:
    bool has_value_ = false;
};

std::ostream &operator<<(std::ostream &os, const Value &value);
extern void newSubKernel(const int core_num, const std::string &kernel_name);
extern void newSyncSubKernel(const int core_num, const std::string &kernel_name, const Tensor &sync_value);
extern void setContext(const std::string &chip_type);
extern std::string getContext();
extern void pushToList(int index, const std::string &name, const std::vector<Tensor> &input_tensor,
                const std::vector<Scalar> &input_scalar, const std::vector<Tensor> &output_tensor,
                const std::vector<Scalar> &output_scalar, const std::vector<std::string> &key,
                const std::vector<std::vector<float>> &value);
extern bool compileKernel(const std::string &file_loc, const std::string &name = "", bool hard_sync = false);
extern const bool canFitMemory(const int size, const std::string &memtype);
#endif