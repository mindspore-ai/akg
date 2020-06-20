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
#include "compute_tracker.h"

/*
 * if you find assertion failure in compute checker and
 * cannot find the immediate error from the error message,
 * you can change COMPUTE_TRACKER_DEBUG to true, and it will
 * print out every assignment and computation.
 */
#define COMPUTE_TRACKER_DEBUG false

#define STRING_TO_KEY 256

static bool g_is_record_mode = false;
static bool g_is_compare_mode = false;
static int g_line_number = 0;
static const char *g_source_file = nullptr;
static bool g_kernel_launched = false;
static size_t g_input_count = 0;
static std::unordered_set<Hash> g_record_traces;
static std::unordered_set<Hash> g_compare_traces;

struct MemRegion {
  std::string name;
  const CallTrackingInfo *begin;
  const CallTrackingInfo *end;
};
static std::vector<MemRegion> g_mem_regions;

static bool g_undefined_assign_check_enabled = true;
static bool g_undefined_compute_check_enabled = true;
static std::vector<bool> g_undefined_assign_check_enabled_stack;
static std::vector<bool> g_undefined_compute_check_enabled_stack;

static std::vector<const TrackedIterator *> g_tracked_iterators;

void RecordLineNumber(int line_number) { g_line_number = line_number; }

int GetLineNumber() { return g_line_number; }

void RecordFileName(const char *source_file) { g_source_file = source_file; }

const char *GetSourceFile() { return g_source_file != nullptr ? g_source_file : "(unknown)"; }

static inline bool IsInRecordMode() { return g_is_record_mode; }

static inline bool IsInCompareMode() { return g_is_compare_mode; }

const char *GetRecordOrCompareMode() {
  if (IsInRecordMode()) {
    return "record";
  }
  if (IsInCompareMode()) {
    return "compare";
  }
  return "";
}

static inline void RecordTrace(Hash hash) { g_record_traces.insert(hash); }

// return valid or invalid
static inline bool CompareTrace(Hash hash) {
  if (g_record_traces.count(hash)) {
    g_compare_traces.insert(hash);
    return true;
  }
  return false;
}

static inline bool TraceComputation(Hash hash) {
  if (IsInRecordMode()) {
    RecordTrace(hash);
    return true;
  } else if (IsInCompareMode()) {
    return CompareTrace(hash);
  }
  return true;
}

static void InitRecordOrCompare() {
  g_input_count = 0;
  g_kernel_launched = false;
  g_mem_regions.clear();
  EnableUndefinedAssignCheck();
  EnableUndefinedComputeCheck();
}

void BeginRecord() {
  g_is_record_mode = true;
  g_is_compare_mode = false;
  g_record_traces.clear();
  InitRecordOrCompare();
}

void BeginCompare() {
  g_is_record_mode = false;
  g_is_compare_mode = true;
  g_compare_traces.clear();
  InitRecordOrCompare();
}

void LaunchKernel() { g_kernel_launched = true; }

static inline size_t IncrementInputCount() { return ++g_input_count; }

static inline bool IsKernelLaunched() { return g_kernel_launched; }

void RecordMemRegion(const std::string &name, const CallTrackingInfo *begin, const CallTrackingInfo *end) {
  MemRegion region = {name, begin, end};
  g_mem_regions.emplace_back(region);
}

void RecordMemRegion(const std::string &name, const CallTrackingInfo *begin, size_t size) {
  MemRegion region = {name, begin, begin + size};
  g_mem_regions.emplace_back(region);
}

static const MemRegion *FindMemRegion(const CallTrackingInfo *ptr) {
  for (const MemRegion &region : g_mem_regions) {
    if (region.begin <= ptr && ptr < region.end) {
      return &region;
    }
  }
  return nullptr;
}

static bool IsInMemRegion(const CallTrackingInfo *ptr) { return (FindMemRegion(ptr) != nullptr); }

static std::string GetMemRegionName(const CallTrackingInfo *ptr) {
  const MemRegion *region = FindMemRegion(ptr);
  if (region == nullptr) {
    return "unknown";
  }
  return region->name;
}

static size_t GetMemRegionOffset(const CallTrackingInfo *ptr) {
  const MemRegion *region = FindMemRegion(ptr);
  CHECK(region != nullptr);
  CHECK_EQ(((intptr_t)ptr - (intptr_t)region->begin) % sizeof(CallTrackingInfo), 0);
  return ptr - region->begin;
}

static void PrintMemRegionInfo(std::ostream &os, const CallTrackingInfo *ptr) {
  if (IsInMemRegion(ptr)) {
    os << " at " << GetMemRegionName(ptr) << "[" << GetMemRegionOffset(ptr) << "]";
  } else {
    os << " at " << ptr;
  }
}

void DisableUndefinedAssignCheck() {
  g_undefined_assign_check_enabled_stack.emplace_back(g_undefined_assign_check_enabled);
  g_undefined_assign_check_enabled = false;
}

void EnableUndefinedAssignCheck() {
  g_undefined_assign_check_enabled_stack.emplace_back(g_undefined_assign_check_enabled);
  g_undefined_assign_check_enabled = true;
}

void RestoreUndefinedAssignCheck() {
  g_undefined_assign_check_enabled = g_undefined_assign_check_enabled_stack.back();
  g_undefined_assign_check_enabled_stack.pop_back();
}

void RestoreUndefinedComputeCheck() {
  g_undefined_compute_check_enabled = g_undefined_compute_check_enabled_stack.back();
  g_undefined_compute_check_enabled_stack.pop_back();
}

void DisableUndefinedComputeCheck() {
  g_undefined_compute_check_enabled_stack.emplace_back(g_undefined_compute_check_enabled);
  g_undefined_compute_check_enabled = false;
}

void EnableUndefinedComputeCheck() {
  g_undefined_compute_check_enabled_stack.emplace_back(g_undefined_compute_check_enabled);
  g_undefined_compute_check_enabled = true;
}

static inline bool IsUndefinedAssignCheckEnabled() { return g_undefined_assign_check_enabled; }

static inline bool IsUndefinedComputeCheckEnabled() { return g_undefined_compute_check_enabled; }

void PrintTrackedIterators(std::ostream &os) {
  bool is_first = true;
  for (auto it : g_tracked_iterators) {
    if (is_first) {
      is_first = false;
    } else {
      os << ", ";
    }
    os << it->GetName() << "=" << it->GetValue();
  }
  if (is_first) {
    os << "None";
  }
}

static uint64_t StrToKey(const char *str) {
  uint64_t op_key = 0;
  while (*str != '\0') {
    op_key = op_key * STRING_TO_KEY + *str++;
  }
  return op_key;
}

static Hash HashBinaryOp(const char *op, Hash lhs, Hash rhs) {
  uint64_t op_key = StrToKey(op);
  Hash ordered_lhs_rhs = std::hash<double>{}(lhs)-std::hash<double>{}(rhs);
  return std::hash<uint64_t>{}(op_key) + ordered_lhs_rhs;
}

static Hash HashUnaryOp(const char *op, Hash operand) {
  uint64_t op_key = StrToKey(op);
  return std::hash<uint64_t>{}(op_key) + std::hash<double>{}(operand);
}

MemoryTrackingInfo::MemoryTrackingInfo() {
  is_defined_ = false;
  real_value_ = 0;
  hash_ = 0;
  produced_line_ = 0;
}

MemoryTrackingInfo::MemoryTrackingInfo(const ImmValue real_value, const Hash hash, const int produced_line) {
  is_defined_ = true;
  real_value_ = real_value;
  hash_ = hash;
  produced_line_ = produced_line;
}

// convert NaN to zero because NaN is an invalid key of std::unordered_set
static inline Hash NanToZero(const Hash &hash) {
  if (std::isnan(hash)) {
    return 0;
  }
  return hash;
}

static inline Hash CreateHash(const Hash &hash) {
  if (IsKernelLaunched()) {
    return NanToZero(hash);
  } else {
    Hash new_hash = NanToZero(hash) + static_cast<double>(1) / IncrementInputCount();
    CHECK(TraceComputation(new_hash)) << "found different input between record and compare: non-existing hash "
                                      << new_hash;
    return new_hash;
  }
}

MemoryTrackingInfo MemoryTrackingInfo::CreateImm(ImmValue real_value) {
  return MemoryTrackingInfo(real_value, CreateHash(real_value), GetLineNumber());
}

std::ostream &operator<<(std::ostream &os, const MemoryTrackingInfo &mem_element) {
  if (mem_element.is_defined_) {
    os << "MemoryTrackingInfo(is_defined=" << mem_element.is_defined_ << ", real_value=" << mem_element.real_value_
       << ", hash=" << mem_element.hash_ << ", produced_line=" << mem_element.produced_line_ << ")";
  } else {
    os << "undefined";
  }
  return os;
}

Buffer::Buffer(const std::string &buffer_name, const std::vector<size_t> &buffer_shape)
    : name_(buffer_name), shape_(buffer_shape) {
  InitializeData();
}

Buffer::Buffer(const char *buffer_name, size_t buffer_dimension, const size_t *buffer_shape) : name_(buffer_name) {
  for (size_t dim = 0; dim < buffer_dimension; dim++) {
    shape_.emplace_back(buffer_shape[dim]);
  }
  InitializeData();
}

void Buffer::InitializeData() {
  size_t buffer_size = 1;
  for (size_t dim = 0; dim < shape_.size(); dim++) {
    CHECK_GT(shape_[dim], 0) << "shape of buffer must be a positive integer: " << *this;
    buffer_size *= shape_[dim];
  }
  data_.reserve(buffer_size);
  for (size_t i = 0; i < buffer_size; i++) {
    data_.emplace_back(MemoryTrackingInfo());
  }
}

std::ostream &operator<<(std::ostream &os, const Buffer &buffer) {
  os << "Buffer(name=" << buffer.name_ << ", shape=[";
  for (size_t dim = 0; dim < buffer.shape_.size(); dim++) {
    os << buffer.shape_[dim];
    if (dim < buffer.shape_.size() - 1) {
      os << ", ";
    }
  }
  os << "])";
  return os;
}

CallTrackingInfo Buffer::operator[](size_t index) {
  CallTrackingInfo tracking_info(*this);
  return tracking_info[index];
}

const std::string &Buffer::GetName() { return name_; }

MemoryTrackingInfo &Buffer::AccessAt(const size_t index) {
  CHECK_LT(index, data_.size()) << "buffer access overflow: size " << data_.size() << ", want to access address "
                                << index;
  return data_[index];
}

static std::string VectorToStr(const std::vector<size_t> &indexes) {
  std::string str = "";
  for (auto index : indexes) {
    str += "[" + std::to_string(index) + "]";
  }
  return str;
}

MemoryTrackingInfo &Buffer::AccessAt(const std::vector<size_t> &indexes) {
  size_t index = 0;
  CHECK_EQ(indexes.size(), shape_.size()) << "call index and tensor shape must have equal dimensions";
  for (size_t i = 0; i < shape_.size(); i++) {
    CHECK(indexes[i] < shape_[i]) << "index " << indexes[i] << " out of range " << shape_[i];
    index *= shape_[i];
    index += indexes[i];
  }
  CHECK_LT(index, data_.size()) << "buffer access overflow: size " << data_.size() << ", want to access address "
                                << VectorToStr(indexes);
  return data_[index];
}

MemoryTrackingInfo *Buffer::GetPointerAt(const std::vector<size_t> &indexes) {
  size_t index = 0;
  if (indexes.size() != shape_.size()) {
    return nullptr;
  }
  for (size_t i = 0; i < shape_.size(); i++) {
    if (indexes[i] >= shape_[i]) {
      return nullptr;
    }
    index *= shape_[i];
    index += indexes[i];
  }
  if (index >= data_.size()) {
    return nullptr;
  }
  return &data_[index];
}

CallTrackingInfo::CallTrackingInfo() { buffer_ = nullptr; }

CallTrackingInfo::CallTrackingInfo(Buffer &buf) { buffer_ = &buf; }

CallTrackingInfo::CallTrackingInfo(const half &half_immediate) : value_(MemoryTrackingInfo::CreateImm(half_immediate)) {
  buffer_ = nullptr;
}

CallTrackingInfo::CallTrackingInfo(double float_immediate) : value_(MemoryTrackingInfo::CreateImm(float_immediate)) {
  buffer_ = nullptr;
}

ImmValue CallTrackingInfo::GetValue() const {
  return buffer_ == nullptr ? value_.real_value_ : buffer_->AccessAt(args_).real_value_;
}

CallTrackingInfo::operator bool() const { return GetValue() > 0.5; }

CallTrackingInfo::operator double() const { return GetValue(); }

// lhs in assignment
CallTrackingInfo &CallTrackingInfo::operator=(const CallTrackingInfo &rhs) {
  const MemoryTrackingInfo &rhs_value = rhs.Dereference();

  if (IsUndefinedAssignCheckEnabled()) {
    CHECK(rhs_value.is_defined_) << "cannot assign undefined right-hand side " << rhs << " to left-hand side " << *this;
  }

  if (buffer_ != nullptr) {
    buffer_->AccessAt(args_) = rhs_value;
  } else {
    value_ = rhs_value;
  }

#if COMPUTE_TRACKER_DEBUG
  LOG << *this << " = " << rhs;
#endif
  return *this;
}

// multi-dimensional array
CallTrackingInfo &CallTrackingInfo::operator[](size_t index) {
  CHECK(buffer_ != nullptr) << "cannot call operator [] on immediate: " << *this;
  args_.emplace_back(index);
  CHECK_LE(args_.size(), buffer_->shape_.size())
    << "buffer access has more dimensions than defined: access " << *this << ", buffer " << buffer_;
  return *this;
}

// tensor of tensor
CallTrackingInfo &CallTrackingInfo::operator[](const CallTrackingInfo &index) {
  return CallTrackingInfo::operator[](index.GetValue());
}

const MemoryTrackingInfo &CallTrackingInfo::Dereference() const {
  return buffer_ == nullptr ? value_ : buffer_->AccessAt(args_);
}

const MemoryTrackingInfo *CallTrackingInfo::DereferencePtr() const {
  return buffer_ == nullptr ? &value_ : buffer_->GetPointerAt(args_);
}

CallTrackingInfo::CallTrackingInfo(ImmValue real_value, Hash hash)
    : value_(MemoryTrackingInfo(real_value, hash, GetLineNumber())) {
  buffer_ = nullptr;
}

template <typename lambda_op, typename lambda_hash>
CallTrackingInfo CallTrackingInfo::GenericBinaryOperator(const CallTrackingInfo &rhs, const std::string &op_name,
                                                         lambda_op op_func, lambda_hash hash_func) const {
  const MemoryTrackingInfo &lhs_val = Dereference();
  const MemoryTrackingInfo &rhs_val = rhs.Dereference();

  if (IsUndefinedAssignCheckEnabled()) {
    CHECK(lhs_val.is_defined_) << "left-hand side " << *this << " of binary op \"" << op_name
                               << "\" refers to an undefined value";
    CHECK(rhs_val.is_defined_) << "right-hand side " << rhs << " of binary op \"" << op_name
                               << "\" refers to an undefined value";
  }

  ImmValue result_val = op_func(lhs_val.real_value_, rhs_val.real_value_);

#ifdef COMPARE_STRICT
  Hash hash = HashBinaryOp(op_name, lhs_val.hash_, rhs_val.hash_);
#else
  Hash hash = hash_func(lhs_val.hash_, rhs_val.hash_);
#endif

  CallTrackingInfo result(result_val, hash);
  result.value_.is_defined_ = lhs_val.is_defined_ && rhs_val.is_defined_;

  if (result.value_.is_defined_) {
    bool is_valid_computation = TraceComputation(hash);
    if (IsUndefinedComputeCheckEnabled()) {
      CHECK(is_valid_computation) << "found different computation between record and compare: " << *this << " "
                                  << op_name << " " << rhs << std::endl
                                  << "left-hand side produced on Line " << lhs_val.produced_line_ << ", "
                                  << "right-hand side produced on Line " << rhs_val.produced_line_;
    }
  }

#if COMPUTE_TRACKER_DEBUG
  LOG << result << " = " << *this << " " << op_name << " " << rhs;
#endif
  return result;
}

template <typename lambda_op, typename lambda_hash>
CallTrackingInfo CallTrackingInfo::GenericBinaryOperatorUnchecked(const CallTrackingInfo &rhs,
                                                                  const std::string &op_name, lambda_op op_func,
                                                                  lambda_hash hash_func) const {
  DisableUndefinedComputeCheck();
  auto result = GenericBinaryOperator(rhs, op_name, op_func, hash_func);
  RestoreUndefinedComputeCheck();
  return result;
}

template <typename lambda>
CallTrackingInfo CallTrackingInfo::GenericBinaryOperator(const CallTrackingInfo &rhs, const std::string &op_name,
                                                         lambda op_func) const {
  return GenericBinaryOperator(rhs, op_name, op_func, op_func);
}

CallTrackingInfo CallTrackingInfo::operator+(const CallTrackingInfo &rhs) const {
  return GenericBinaryOperator(rhs, "+", [](Hash a, Hash b) -> Hash { return a + b; });
}

CallTrackingInfo CallTrackingInfo::operator-(const CallTrackingInfo &rhs) const {
  return GenericBinaryOperator(rhs, "-", [](Hash a, Hash b) -> Hash { return a - b; });
}

CallTrackingInfo CallTrackingInfo::operator*(const CallTrackingInfo &rhs) const {
  return GenericBinaryOperator(rhs, "*", [](Hash a, Hash b) -> Hash { return a * b; });
}

CallTrackingInfo CallTrackingInfo::operator/(const CallTrackingInfo &rhs) const {
  return GenericBinaryOperator(rhs, "/", [](Hash a, Hash b) -> Hash { return a / b; });
}

CallTrackingInfo CallTrackingInfo::operator%(const CallTrackingInfo &rhs) const {
  return GenericBinaryOperator(rhs, "%", [](Hash a, Hash b) -> Hash { return (int64_t)a % (int64_t)b; });
}

CallTrackingInfo CallTrackingInfo::operator^(const CallTrackingInfo &rhs) const {
  return GenericBinaryOperator(rhs, "^", [](Hash a, Hash b) -> Hash { return (int64_t)a ^ (int64_t)b; });
}

CallTrackingInfo CallTrackingInfo::operator&(const CallTrackingInfo &rhs) const {
  return GenericBinaryOperator(rhs, "&", [](Hash a, Hash b) -> Hash { return (int64_t)a & (int64_t)b; });
}

CallTrackingInfo CallTrackingInfo::operator|(const CallTrackingInfo &rhs) const {
  return GenericBinaryOperator(rhs, "|", [](Hash a, Hash b) -> Hash { return (int64_t)a | (int64_t)b; });
}

CallTrackingInfo CallTrackingInfo::operator<(const CallTrackingInfo &rhs) const {
  return GenericBinaryOperatorUnchecked(
    rhs, "<", [](ImmValue a, ImmValue b) -> ImmValue { return a < b; }, [](Hash a, Hash b) -> Hash { return a - b; });
}

CallTrackingInfo CallTrackingInfo::operator>(const CallTrackingInfo &rhs) const {
  return GenericBinaryOperatorUnchecked(
    rhs, ">", [](ImmValue a, ImmValue b) -> ImmValue { return a > b; }, [](Hash a, Hash b) -> Hash { return b - a; });
}

CallTrackingInfo CallTrackingInfo::operator<=(const CallTrackingInfo &rhs) const {
  return GenericBinaryOperatorUnchecked(
    rhs, "<=", [](ImmValue a, ImmValue b) -> ImmValue { return a <= b; },
    [](Hash a, Hash b) -> Hash { return a - b + 1; });
}

CallTrackingInfo CallTrackingInfo::operator>=(const CallTrackingInfo &rhs) const {
  return GenericBinaryOperatorUnchecked(
    rhs, ">=", [](ImmValue a, ImmValue b) -> ImmValue { return a >= b; },
    [](Hash a, Hash b) -> Hash { return b - a + 1; });
}

CallTrackingInfo CallTrackingInfo::operator==(const CallTrackingInfo &rhs) const {
  return GenericBinaryOperatorUnchecked(
    rhs, "==", [](ImmValue a, ImmValue b) -> ImmValue { return a == b; },
    [](Hash a, Hash b) -> Hash { return (a - b) * 2; });
}

CallTrackingInfo CallTrackingInfo::operator!=(const CallTrackingInfo &rhs) const {
  return GenericBinaryOperatorUnchecked(
    rhs, "!=", [](ImmValue a, ImmValue b) -> ImmValue { return a != b; },
    [](Hash a, Hash b) -> Hash { return (b - a) * 2; });
}

template <typename lambda>
CallTrackingInfo &CallTrackingInfo::GenericBinarySelfUpdateOperator(const CallTrackingInfo &rhs,
                                                                    const std::string &op_name, lambda op_func) {
  CHECK(buffer_ != nullptr) << "left-hand side of self-update operator must not be an immediate: " << this << " "
                            << op_name << " " << rhs;
  CallTrackingInfo tmp_result = GenericBinaryOperator(rhs, op_name, op_func);
  buffer_->AccessAt(args_) = tmp_result.value_;
  this->value_ = tmp_result.value_;
  return *this;
}

CallTrackingInfo &CallTrackingInfo::operator+=(const CallTrackingInfo &rhs) {
  return GenericBinarySelfUpdateOperator(rhs, "+", [](Hash a, Hash b) -> Hash { return a + b; });
}

CallTrackingInfo &CallTrackingInfo::operator-=(const CallTrackingInfo &rhs) {
  return GenericBinarySelfUpdateOperator(rhs, "-", [](Hash a, Hash b) -> Hash { return a - b; });
}

CallTrackingInfo &CallTrackingInfo::operator*=(const CallTrackingInfo &rhs) {
  return GenericBinarySelfUpdateOperator(rhs, "*", [](Hash a, Hash b) -> Hash { return a * b; });
}

CallTrackingInfo &CallTrackingInfo::operator/=(const CallTrackingInfo &rhs) {
  return GenericBinarySelfUpdateOperator(rhs, "/", [](Hash a, Hash b) -> Hash { return a / b; });
}

CallTrackingInfo &CallTrackingInfo::operator%=(const CallTrackingInfo &rhs) {
  return GenericBinarySelfUpdateOperator(rhs, "%", [](Hash a, Hash b) -> Hash { return (int64_t)a % (int64_t)b; });
}

CallTrackingInfo &CallTrackingInfo::operator^=(const CallTrackingInfo &rhs) {
  return GenericBinarySelfUpdateOperator(rhs, "^", [](Hash a, Hash b) -> Hash { return (int64_t)a ^ (int64_t)b; });
}

CallTrackingInfo &CallTrackingInfo::operator&=(const CallTrackingInfo &rhs) {
  return GenericBinarySelfUpdateOperator(rhs, "&", [](Hash a, Hash b) -> Hash { return (int64_t)a & (int64_t)b; });
}

CallTrackingInfo &CallTrackingInfo::operator|=(const CallTrackingInfo &rhs) {
  return GenericBinarySelfUpdateOperator(rhs, "|", [](Hash a, Hash b) -> Hash { return (int64_t)a | (int64_t)b; });
}

CallTrackingInfo CallTrackingInfo::operator~() const {
  return GenericUnaryOperator("~", [](Hash a) -> Hash { return ~(int64_t)a; });
}

template <typename lambda>
CallTrackingInfo CallTrackingInfo::GenericUnaryOperator(const std::string &op_name, lambda op_func) const {
  const MemoryTrackingInfo &lhs_val = Dereference();

  if (IsUndefinedAssignCheckEnabled()) {
    CHECK(lhs_val.is_defined_) << "left-hand side " << *this << " of " << op_name << " refers to undefined value";
  }

  ImmValue result_val = op_func(lhs_val.real_value_);

#ifdef COMPARE_STRICT
  Hash hash = HashUnaryOp(op_name, lhs_val.hash_);
#else
  Hash hash = op_func(lhs_val.hash_);
#endif

  CallTrackingInfo result(result_val, hash);
  result.value_.is_defined_ = lhs_val.is_defined_;

  if (result.value_.is_defined_) {
    bool is_valid_computation = TraceComputation(hash);
    if (IsUndefinedComputeCheckEnabled()) {
      CHECK(is_valid_computation) << "found different computation between record and compare: " << op_name << " "
                                  << *this << std::endl
                                  << "operand produced on Line " << lhs_val.produced_line_;
    }
  }

  return result;
}

CallTrackingInfo &CallTrackingInfo::operator++() {
  return GenericUnarySelfUpdateOperator("++", [](Hash a) -> Hash { return a + 1; });
}

CallTrackingInfo &CallTrackingInfo::operator--() {
  return GenericUnarySelfUpdateOperator("--", [](Hash a) -> Hash { return a - 1; });
}

CallTrackingInfo CallTrackingInfo::operator++(int) {
  CallTrackingInfo old_value(*this);
  GenericUnarySelfUpdateOperator("++", [](Hash a) -> Hash { return a + 1; });
  return old_value;
}

CallTrackingInfo CallTrackingInfo::operator--(int) {
  CallTrackingInfo old_value(*this);
  GenericUnarySelfUpdateOperator("--", [](Hash a) -> Hash { return a - 1; });
  return old_value;
}

template <typename lambda>
CallTrackingInfo &CallTrackingInfo::GenericUnarySelfUpdateOperator(const std::string &op_name, lambda op_func) {
  CHECK(buffer_ != nullptr) << "cannot update immediate: " << op_name << " " << this;
  CallTrackingInfo tmp_result = GenericUnaryOperator(op_name, op_func);
  buffer_->AccessAt(args_) = tmp_result.value_;
  this->value_ = tmp_result.value_;
  return *this;
}

std::ostream &operator<<(std::ostream &os, const CallTrackingInfo &call) {
  os << "CallTrackingInfo(";
  if (call.buffer_) {
    os << "Buffer(" << call.buffer_->GetName() << ")"
       << ", args=" << VectorToStr(call.args_) << ", value=";
    const MemoryTrackingInfo *value = call.DereferencePtr();
    if (value != nullptr) {
      os << *value;
    } else {
      os << "undefined";
    }
  } else {
    os << "imm_value=" << call.value_;
  }
  PrintMemRegionInfo(os, &call);
  os << ")";
  return os;
}

static inline bool IsPowerOfTwo(size_t x) { return x > 0 && (x & (x - 1)) == 0; }

void SanityCheck() {
  CHECK(IsPowerOfTwo(sizeof(CallTrackingInfo)));
  CHECK_EQ(sizeof(CallTrackingInfoI8), sizeof(CallTrackingInfo));
  CHECK_EQ(sizeof(CallTrackingInfoU8), sizeof(CallTrackingInfo));
  CHECK_EQ(sizeof(CallTrackingInfo16), 2 * sizeof(CallTrackingInfo));
  CHECK_EQ(sizeof(CallTrackingInfoI16), 2 * sizeof(CallTrackingInfo));
  CHECK_EQ(sizeof(CallTrackingInfoU16), 2 * sizeof(CallTrackingInfo));
  CHECK_EQ(sizeof(CallTrackingInfo32), 4 * sizeof(CallTrackingInfo));
  CHECK_EQ(sizeof(CallTrackingInfoI32), 4 * sizeof(CallTrackingInfo));
  CHECK_EQ(sizeof(CallTrackingInfoU32), 4 * sizeof(CallTrackingInfo));
  CHECK_EQ(sizeof(CallTrackingInfoI64), 8 * sizeof(CallTrackingInfo));
  CHECK_EQ(sizeof(CallTrackingInfoU64), 8 * sizeof(CallTrackingInfo));
}

void FinalCheck() {
  if (g_record_traces.size() == g_compare_traces.size()) {
    return;
  }

#if COMPUTE_TRACKER_DEBUG
  std::cerr << "WARNING: compare mode execution misses some execution from record mode, "
            << "compare mode: " << g_compare_traces.size() << " computations, "
            << "record mode: " << g_record_traces.size() << " computations" << std::endl;
  for (auto compare_hash : g_compare_traces) {
    g_record_traces.erase(compare_hash);
  }
  std::cerr << "missing hashes:";
  for (auto record_hash : g_record_traces) {
    std::cerr << " " << record_hash;
  }
  std::cerr << std::endl;
#endif
}

TrackedIterator::TrackedIterator(const std::string &input_name, uint64_t initial_value)
    : value_(initial_value), name_(input_name) {
  g_tracked_iterators.emplace_back(this);
}

void TrackedIterator::Init(const std::string &input_name, uint64_t initial_value) {
  value_ = initial_value;
  name_ = input_name;
  g_tracked_iterators.emplace_back(this);
}

static void RemoveTrackedIterator(const TrackedIterator *iter) {
  int num_iters = g_tracked_iterators.size();
  for (int i = 0; i < num_iters;) {
    if (g_tracked_iterators[i] == iter) {
      for (int j = i + 1; j < num_iters; j++) {
        g_tracked_iterators[j - 1] = g_tracked_iterators[j];
      }
      --num_iters;
      g_tracked_iterators.pop_back();
    } else {
      ++i;
    }
  }
}

TrackedIterator::~TrackedIterator() {
  if (g_tracked_iterators.back() == this) {
    g_tracked_iterators.pop_back();
  } else {  // unlikely: non-matching iterator scopes, possibly in exception handling
    RemoveTrackedIterator(this);
  }
}

uint64_t TrackedIterator::operator=(uint64_t imm) {
  value_ = imm;
  return value_;
}

uint64_t TrackedIterator::operator+=(uint64_t imm) {
  value_ += imm;
  return value_;
}

uint64_t TrackedIterator::operator-=(uint64_t imm) {
  value_ -= imm;
  return value_;
}

uint64_t TrackedIterator::operator++() { return ++value_; }

uint64_t TrackedIterator::operator--() { return --value_; }

uint64_t TrackedIterator::operator++(int) { return value_++; }

uint64_t TrackedIterator::operator--(int) { return value_--; }

TrackedIterator::operator uint64_t() const { return value_; }

uint64_t TrackedIterator::GetValue() const { return value_; }

std::string TrackedIterator::GetName() const { return name_; }
