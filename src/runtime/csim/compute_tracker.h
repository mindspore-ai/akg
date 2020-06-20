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

#ifndef RUNTIME_CSIM_COMPUTE_TRACKER_H_
#define RUNTIME_CSIM_COMPUTE_TRACKER_H_

#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>
#include <cmath>
#include "half_float.h"

using Hash = double;

using ImmValue = double;

class MemoryTrackingInfo {
 public:
  bool is_defined_;
  ImmValue real_value_;
  Hash hash_;
  int produced_line_;

  MemoryTrackingInfo();
  MemoryTrackingInfo(const ImmValue real_value, const Hash hash, const int produced_line);
  ~MemoryTrackingInfo() {}

  static MemoryTrackingInfo CreateImm(ImmValue real_value);

  friend std::ostream &operator<<(std::ostream &os, const MemoryTrackingInfo &mem_element);
};

class CallTrackingInfo;

class Buffer {
 private:
  std::vector<MemoryTrackingInfo> data_;
  std::string name_;
  std::vector<size_t> shape_;

  void InitializeData();
  MemoryTrackingInfo &AccessAt(const std::vector<size_t> &indexes);
  MemoryTrackingInfo &AccessAt(size_t index);
  MemoryTrackingInfo *GetPointerAt(const std::vector<size_t> &indexes);

  friend std::ostream &operator<<(std::ostream &os, const Buffer &buffer);
  friend class CallTrackingInfo;

 public:
  Buffer(const std::string &buffer_name, const std::vector<size_t> &buffer_shape);
  Buffer(const char *buffer_name, size_t buffer_dimension, const size_t *buffer_shape);
  ~Buffer() {}

  CallTrackingInfo operator[](size_t index);

  const std::string &GetName();
};

class CallTrackingInfo {
 private:
  // if buffer is nullptr, then it is an immediate number (expression)
  // if buffer is a valid reference, then it is a call with args
  Buffer *buffer_;
  std::vector<size_t> args_;  // only valid when buffer != nullptr
  MemoryTrackingInfo value_;  // only valid when buffer == nullptr

  const MemoryTrackingInfo &Dereference() const;
  const MemoryTrackingInfo *DereferencePtr() const;
  CallTrackingInfo(ImmValue real_value, Hash hash);

  template <typename lambda>
  CallTrackingInfo GenericBinaryOperator(const CallTrackingInfo &rhs, const std::string &op_name, lambda op_func) const;

  template <typename lambda_op, typename lambda_hash>
  CallTrackingInfo GenericBinaryOperator(const CallTrackingInfo &rhs, const std::string &op_name, lambda_op op_func,
                                         lambda_hash hash_func) const;

  template <typename lambda_op, typename lambda_hash>
  CallTrackingInfo GenericBinaryOperatorUnchecked(const CallTrackingInfo &rhs, const std::string &op_name,
                                                  lambda_op op_func, lambda_hash hash_func) const;

  template <typename lambda>
  CallTrackingInfo &GenericBinarySelfUpdateOperator(const CallTrackingInfo &rhs, const std::string &op_name,
                                                    lambda op_func);

  template <typename lambda>
  CallTrackingInfo GenericUnaryOperator(const std::string &op_name, lambda op_func) const;

  template <typename lambda>
  CallTrackingInfo &GenericUnarySelfUpdateOperator(const std::string &op_name, lambda op_func);

  friend std::ostream &operator<<(std::ostream &os, const CallTrackingInfo &call);

 public:
  CallTrackingInfo();
  // implicit type promotion from Buffer, half and float to CallTrackingInfo
  // we cannot use const Buffer& here because it will modify the contents of Buffer
  explicit CallTrackingInfo(Buffer &buf);
  explicit CallTrackingInfo(const half &half_immediate);
  explicit CallTrackingInfo(double float_immediate);
  ~CallTrackingInfo() {}

  explicit operator bool() const;
  explicit operator double() const;
  ImmValue GetValue() const;

  CallTrackingInfo &operator=(const CallTrackingInfo &rhs);     // lhs in assignment
  CallTrackingInfo &operator[](size_t index);                   // multi-dimensional array
  CallTrackingInfo &operator[](const CallTrackingInfo &index);  // tensor of tensor

  CallTrackingInfo operator+(const CallTrackingInfo &rhs) const;
  CallTrackingInfo operator-(const CallTrackingInfo &rhs) const;
  CallTrackingInfo operator*(const CallTrackingInfo &rhs) const;
  CallTrackingInfo operator/(const CallTrackingInfo &rhs) const;
  CallTrackingInfo operator%(const CallTrackingInfo &rhs) const;
  CallTrackingInfo operator^(const CallTrackingInfo &rhs) const;
  CallTrackingInfo operator&(const CallTrackingInfo &rhs) const;
  CallTrackingInfo operator|(const CallTrackingInfo &rhs) const;
  CallTrackingInfo operator<(const CallTrackingInfo &rhs) const;
  CallTrackingInfo operator>(const CallTrackingInfo &rhs) const;
  CallTrackingInfo operator<=(const CallTrackingInfo &rhs) const;
  CallTrackingInfo operator>=(const CallTrackingInfo &rhs) const;
  CallTrackingInfo operator==(const CallTrackingInfo &rhs) const;
  CallTrackingInfo operator!=(const CallTrackingInfo &rhs) const;

  CallTrackingInfo &operator+=(const CallTrackingInfo &rhs);
  CallTrackingInfo &operator-=(const CallTrackingInfo &rhs);
  CallTrackingInfo &operator*=(const CallTrackingInfo &rhs);
  CallTrackingInfo &operator/=(const CallTrackingInfo &rhs);
  CallTrackingInfo &operator%=(const CallTrackingInfo &rhs);
  CallTrackingInfo &operator^=(const CallTrackingInfo &rhs);
  CallTrackingInfo &operator&=(const CallTrackingInfo &rhs);
  CallTrackingInfo &operator|=(const CallTrackingInfo &rhs);

  CallTrackingInfo operator~() const;

  CallTrackingInfo &operator++();
  CallTrackingInfo &operator--();
  CallTrackingInfo operator++(int);
  CallTrackingInfo operator--(int);
};

class CallTrackingInfo16 : public CallTrackingInfo {
 private:
  CallTrackingInfo padding_[1];

 public:
  CallTrackingInfo16() : CallTrackingInfo() {}
  explicit CallTrackingInfo16(const CallTrackingInfo &to_copy) : CallTrackingInfo(to_copy) {}
  explicit CallTrackingInfo16(const half &half_immediate) : CallTrackingInfo(half_immediate) {}
  explicit CallTrackingInfo16(float float_immediate) : CallTrackingInfo(float_immediate) {}
  ~CallTrackingInfo16() {}
  CallTrackingInfo16 &operator=(const CallTrackingInfo16 &rhs) {
    CallTrackingInfo::operator=(rhs);
    return *this;
  }
};

class CallTrackingInfo32 : public CallTrackingInfo {
 private:
  CallTrackingInfo padding_[3];

 public:
  CallTrackingInfo32() : CallTrackingInfo() {}
  explicit CallTrackingInfo32(const CallTrackingInfo &to_copy) : CallTrackingInfo(to_copy) {}
  explicit CallTrackingInfo32(const half &half_immediate) : CallTrackingInfo(half_immediate) {}
  explicit CallTrackingInfo32(float float_immediate) : CallTrackingInfo(float_immediate) {}
  ~CallTrackingInfo32() {}
  CallTrackingInfo32 &operator=(const CallTrackingInfo32 &rhs) {
    CallTrackingInfo::operator=(rhs);
    return *this;
  }
};

class CallTrackingInfoI8 : public CallTrackingInfo {
 public:
  CallTrackingInfoI8() : CallTrackingInfo() {}
  explicit CallTrackingInfoI8(const CallTrackingInfo &to_copy) : CallTrackingInfo(to_copy) {}
  explicit CallTrackingInfoI8(const half &half_immediate) : CallTrackingInfo(half_immediate) {}
  explicit CallTrackingInfoI8(float float_immediate) : CallTrackingInfo(float_immediate) {}
  ~CallTrackingInfoI8() {}
  CallTrackingInfoI8 &operator=(const CallTrackingInfoI8 &rhs) {
    CallTrackingInfo::operator=(rhs);
    return *this;
  }
};

class CallTrackingInfoI16 : public CallTrackingInfo {
 private:
  CallTrackingInfo padding_[1];

 public:
  CallTrackingInfoI16() : CallTrackingInfo() {}
  explicit CallTrackingInfoI16(const CallTrackingInfo &to_copy) : CallTrackingInfo(to_copy) {}
  explicit CallTrackingInfoI16(const half &half_immediate) : CallTrackingInfo(half_immediate) {}
  explicit CallTrackingInfoI16(float float_immediate) : CallTrackingInfo(float_immediate) {}
  ~CallTrackingInfoI16() {}
  CallTrackingInfoI16 &operator=(const CallTrackingInfoI16 &rhs) {
    CallTrackingInfo::operator=(rhs);
    return *this;
  }
};

class CallTrackingInfoI32 : public CallTrackingInfo {
 private:
  CallTrackingInfo padding_[3];

 public:
  CallTrackingInfoI32() : CallTrackingInfo() {}
  explicit CallTrackingInfoI32(const CallTrackingInfo &to_copy) : CallTrackingInfo(to_copy) {}
  explicit CallTrackingInfoI32(const half &half_immediate) : CallTrackingInfo(half_immediate) {}
  explicit CallTrackingInfoI32(float float_immediate) : CallTrackingInfo(float_immediate) {}
  ~CallTrackingInfoI32() {}
  CallTrackingInfoI32 &operator=(const CallTrackingInfoI32 &rhs) {
    CallTrackingInfo::operator=(rhs);
    return *this;
  }
};

class CallTrackingInfoI64 : public CallTrackingInfo {
 private:
  CallTrackingInfo padding_[7];

 public:
  CallTrackingInfoI64() : CallTrackingInfo() {}
  explicit CallTrackingInfoI64(const CallTrackingInfo &to_copy) : CallTrackingInfo(to_copy) {}
  explicit CallTrackingInfoI64(const half &half_immediate) : CallTrackingInfo(half_immediate) {}
  explicit CallTrackingInfoI64(float float_immediate) : CallTrackingInfo(float_immediate) {}
  ~CallTrackingInfoI64() {}
  CallTrackingInfoI64 &operator=(const CallTrackingInfoI64 &rhs) {
    CallTrackingInfo::operator=(rhs);
    return *this;
  }
};

class CallTrackingInfoU8 : public CallTrackingInfo {
 public:
  CallTrackingInfoU8() : CallTrackingInfo() {}
  explicit CallTrackingInfoU8(const CallTrackingInfo &to_copy) : CallTrackingInfo(to_copy) {}
  explicit CallTrackingInfoU8(const half &half_immediate) : CallTrackingInfo(half_immediate) {}
  explicit CallTrackingInfoU8(float float_immediate) : CallTrackingInfo(float_immediate) {}
  ~CallTrackingInfoU8() {}
  CallTrackingInfoU8 &operator=(const CallTrackingInfoU8 &rhs) {
    CallTrackingInfo::operator=(rhs);
    return *this;
  }
};

class CallTrackingInfoU16 : public CallTrackingInfo {
 private:
  CallTrackingInfo padding_[1];

 public:
  CallTrackingInfoU16() : CallTrackingInfo() {}
  explicit CallTrackingInfoU16(const CallTrackingInfo &to_copy) : CallTrackingInfo(to_copy) {}
  explicit CallTrackingInfoU16(const half &half_immediate) : CallTrackingInfo(half_immediate) {}
  explicit CallTrackingInfoU16(float float_immediate) : CallTrackingInfo(float_immediate) {}
  ~CallTrackingInfoU16() {}
  CallTrackingInfoU16 &operator=(const CallTrackingInfoU16 &rhs) {
    CallTrackingInfo::operator=(rhs);
    return *this;
  }
};

class CallTrackingInfoU32 : public CallTrackingInfo {
 private:
  CallTrackingInfo padding_[3];

 public:
  CallTrackingInfoU32() : CallTrackingInfo() {}
  explicit CallTrackingInfoU32(const CallTrackingInfo &to_copy) : CallTrackingInfo(to_copy) {}
  explicit CallTrackingInfoU32(const half &half_immediate) : CallTrackingInfo(half_immediate) {}
  explicit CallTrackingInfoU32(float float_immediate) : CallTrackingInfo(float_immediate) {}
  ~CallTrackingInfoU32() {}
  CallTrackingInfoU32 &operator=(const CallTrackingInfoU32 &rhs) {
    CallTrackingInfo::operator=(rhs);
    return *this;
  }
};

class CallTrackingInfoU64 : public CallTrackingInfo {
 private:
  CallTrackingInfo padding_[7];

 public:
  CallTrackingInfoU64() : CallTrackingInfo() {}
  explicit CallTrackingInfoU64(const CallTrackingInfo &to_copy) : CallTrackingInfo(to_copy) {}
  explicit CallTrackingInfoU64(const half &half_immediate) : CallTrackingInfo(half_immediate) {}
  explicit CallTrackingInfoU64(float float_immediate) : CallTrackingInfo(float_immediate) {}
  ~CallTrackingInfoU64() {}
  CallTrackingInfoU64 &operator=(const CallTrackingInfoU64 &rhs) {
    CallTrackingInfo::operator=(rhs);
    return *this;
  }
};

class TrackedIterator {
 private:
  uint64_t value_;
  std::string name_;

 public:
  TrackedIterator() {}
  TrackedIterator(const std::string &name, uint64_t initial_value);
  ~TrackedIterator();
  void Init(const std::string &name, uint64_t initial_value);

  uint64_t operator=(uint64_t imm);
  uint64_t operator+=(uint64_t imm);
  uint64_t operator-=(uint64_t imm);
  uint64_t operator++();
  uint64_t operator--();
  uint64_t operator++(int);
  uint64_t operator--(int);
  operator uint64_t() const;

  uint64_t GetValue() const;
  std::string GetName() const;
};

void RecordLineNumber(int line_number);
void RecordFileName(const char *source_file);

#define RECORD_LINE() RecordLineNumber(__LINE__)
#define RECORD_FILE() RecordFileName(__FILE__)

void BeginRecord();
void BeginCompare();
void LaunchKernel();
void SanityCheck();
void FinalCheck();

const char *GetRecordOrCompareMode();
int GetLineNumber();
const char *GetSourceFile();

void RecordMemRegion(const std::string &name, const CallTrackingInfo *begin, const CallTrackingInfo *end);
void RecordMemRegion(const std::string &name, const CallTrackingInfo *begin, size_t size);

void PrintTrackedIterators(std::ostream &os);

class ReportFail {
 public:
  ReportFail(const char *file, const int line, const char *assertion) {
    std::cerr << file << ":" << line << ": Assertion \"" << assertion << "\" failed on " << GetRecordOrCompareMode()
              << " mode "
              << "\"" << GetSourceFile() << "\" Line " << GetLineNumber() << ": ";
  }

  std::ostream &GetStream() {
    return std::cerr;
  }

  ~ReportFail() {
    std::cerr << std::endl << "Iterators: ";
    PrintTrackedIterators(std::cerr);
    std::cerr << std::endl << std::endl;
    abort();
  }
};

class PrintLog {
 public:
  PrintLog(const char *file, const int line) {
    std::cerr << file << ":" << line << ": " << GetRecordOrCompareMode() << " mode "
              << "\"" << GetSourceFile() << "\" Line " << GetLineNumber() << ": ";
  }

  std::ostream &GetStream() {
    return std::cerr;
  }

  ~PrintLog() {
    std::cerr << " # Iterators: ";
    PrintTrackedIterators(std::cerr);
    std::cerr << std::endl;
  }
};

#define CHECK(x) \
  if (!(x)) ReportFail(__FILE__, __LINE__, (#x)).GetStream()
#define LOG(x) PrintLog(__FILE__, __LINE__).GetStream() << (x)
#define INFO "[INFO] "
#define WARNING "[WARNING] "
#define ERROR "[ERROR] "

#define CHECK_EQ(x, y) CHECK((x) == (y))
#define CHECK_NE(x, y) CHECK((x) != (y))
#define CHECK_LE(x, y) CHECK((x) <= (y))
#define CHECK_LT(x, y) CHECK((x) <  (y))
#define CHECK_GE(x, y) CHECK((x) >= (y))
#define CHECK_GT(x, y) CHECK((x) >  (y))

/*
 * Why we need options to turn on and off assign and compute checks:
 *
 * 1. in CCE, some operations may reference undefined values, for example,
 *    (1) DMA copy padding elements (float16 has one padding element, and DMA copy will access the padding element),
 *    (2) vadds in concat cover,
 *    (3) vector compare (vcmp) instructions
 *
 * 2. in CCE, some operations may produce invalid hashes, for example,
 *    vector compare instructions that reference undefined values, but it is a memory garbage
 *    that is not overwritten by the last instruction.
 */

void DisableUndefinedAssignCheck();
void EnableUndefinedAssignCheck();
void RestoreUndefinedAssignCheck();

void DisableUndefinedComputeCheck();
void EnableUndefinedComputeCheck();
void RestoreUndefinedComputeCheck();

#endif  // RUNTIME_CSIM_COMPUTE_TRACKER_H_
