/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "light_cp.h"

namespace LightCP {
STATUS operator|(const STATUS& left, const STATUS& right) {
  if (isFail(left) || isFail(right)) {
    return STATUS::FAIL;
  }
  if (isSuccess(left) || isSuccess(right)) {
    return STATUS::SUCCESS;
  }
  return STATUS::UNKNOWN;
}

STATUS& operator|=(STATUS& left, const STATUS& right) {
  return (left = (isKnown(left) ? left | right : right));
}

std::ostream& operator<<(std::ostream& out, const STATUS& s) {
  return out << (isFail(s) ? "FAIL" : (isSuccess(s) ? "SUCCESS" : "UNKNOWN"));
}

Range::Range()
    : lb_(std::numeric_limits<int>::min()),
      ub_(std::numeric_limits<int>::max()) {}
Range::Range(int lb, int ub) : lb_(lb), ub_(ub) {}
Range::Range(const Range& rng) : lb_(rng.lb_), ub_(rng.ub_) {}
Range& Range::operator=(const Range& rng){
  lb_ = rng.GetLB();
  ub_ = rng.GetUB();
  return *this;
}

void Range::SetBound(int lb, int ub) {
  lb_ = lb;
  ub_ = ub;
}
void Range::SetBound(const Range& r) {
  lb_ = r.lb_;
  ub_ = r.ub_;
}
// This will not lead to overflow since we return an int64_t
// and the range is defined over 32 bits.
// if the range is changed to int64_t cases needs to be checked in
// order to avoid overflows
int64_t Range::Size() const {
  // empty case
  if (ub_ < lb_) {
    return 0;
  }
  return static_cast<int64_t>(ub_) - static_cast<int64_t>(lb_) + 1;
}

std::ostream& operator<<(std::ostream& out, const Range& v) {
  return out << "[" << v.lb_ << "," << v.ub_ << "]";
}

Range operator+(const Range& lhs, const Range& rhs) {
  return {lhs.lb_ + rhs.lb_, lhs.ub_ + rhs.ub_};
}

Range operator-(const Range& lhs, const Range& rhs) {
  return {lhs.lb_ - rhs.ub_, lhs.ub_ - rhs.lb_};
}

int Mult(int x, int y) {
  int max = std::numeric_limits<int>::max();
  int min = std::numeric_limits<int>::min();
  if (x > 0 && y > 0 && x > max / y) {
    return max;
  }
  if (x < 0 && y > 0 && x < min / y) {
    return min;
  }
  if (x > 0 && y < 0 && y < min / x) {
    return min;
  }
  if (x < 0 && y < 0 && (x <= min || y <= min || -x > max / -y)) {
    return min;
  }
  return x * y;
}

Range Mult(const int& a, const int& b, const int& c, const int& d) {
  if (a >= 0 && c >= 0) {
    assert(b >= 0 && d >= 0);
    return {Mult(a, c), Mult(b, d)};
  } else {
    auto ac = Mult(a, c);
    auto ad = Mult(a, d);
    auto bc = Mult(b, c);
    auto bd = Mult(b, d);
    auto min = std::min({ac, ad, bc, bd});
    auto max = std::max({ac, ad, bc, bd});
    return {min, max};
  }
}

// Compute the min and max of the result
// in interval arithmetics
// [a,b] x [c,d] = [min(ac, ad, bc, bd), max(ac, ad, bc, bd)]
Range operator*(const Range& lhs, const Range& rhs) {
  return Mult(lhs.lb_, lhs.ub_, rhs.lb_, rhs.ub_);
}

Range Div(const int& a, const int& b, const int& c, const int& d) {
  int ac = a / c;
  int ad = a / d;
  int bc = b / c;
  int bd = b / d;
  int min = std::min({ac, ad, bc, bd});
  int max = std::max({ac, ad, bc, bd});
  return {min, max};
}

Range DivLight(const int& a, const int& b, const int& c, const int& d) {
  return {std::min({a / d, b / d}), std::max({a / c, b / c})};
}

// Note that it does not do anything if both numerator
// and denominator are 0 since it is used also for inverse
// multiplication
Range operator/(const Range& num, const Range& den) {
  if (IsStrictlyNegative(den) && IsStrictlyNegative(num)) {
    return DivLight(num.GetLB(), num.GetUB(), den.GetLB(), den.GetUB());
  } else if (IsStrictlyPositive(den) && IsStrictlyPositive(num)) {
    return DivLight(num.GetLB(), num.GetUB(), den.GetLB(), den.GetUB());
  } else if (IsStrictlyPositive(den) || IsStrictlyNegative(den)) {
    assert(den.GetLB() > 0 || den.GetUB() < 0);
    return Div(num.GetLB(), num.GetUB(), den.GetLB(), den.GetUB());
  } else if (den.GetUB() == 0 && den.GetLB() == 0 &&
             (IsStrictlyPositive(num) || IsStrictlyNegative(num))) {
    return {1, 0};
  } else {
    return {std::numeric_limits<int>::min(), std::numeric_limits<int>::max()};
  }
}

bool operator==(const Range& lhs, const Range& rhs) {
  return (lhs.lb_ == rhs.lb_) && (lhs.ub_ == rhs.ub_);
}

// RangeDomain
RangeDomain::RangeDomain(RangeDomain const& other) : RangeDomain(static_cast<Range>(other)) {}
RangeDomain::RangeDomain(RangeDomain&& other) : RangeDomain(static_cast<Range>(other)) {}
RangeDomain& RangeDomain::operator=(RangeDomain const& other) {
  UpdateBound(other.GetLB(), other.GetUB());
  return *this;
}
RangeDomain& RangeDomain::operator=(RangeDomain&& other) {
  UpdateBound(other.GetLB(), other.GetUB());
  return *this;
}

STATUS RangeDomain::UpdateLB(int newMin) {
  if (newMin > lb_) {
    lb_ = newMin;
    if (ub_ < lb_) {
      return STATUS::FAIL;
    }
    return STATUS::SUCCESS;
  }
  return STATUS::UNKNOWN;
}

STATUS RangeDomain::UpdateUB(int newMax) {
  if (newMax < ub_) {
    ub_ = newMax;
    if (ub_ < lb_) {
      return STATUS::FAIL;
    }
    return STATUS::SUCCESS;
  }
  return STATUS::UNKNOWN;
}

STATUS RangeDomain::UpdateBound(int newMin, int newMax) {
  STATUS rv = UpdateLB(newMin);
  rv |= UpdateUB(newMax);
  return rv;
}

STATUS RangeDomain::Assign(int v) {
  if (Contains(v)) {
    ub_ = lb_ = v;
    return STATUS::SUCCESS;
  }
  return STATUS::FAIL;
}

// helper

int Incr(const int& v) {
  return (v == std::numeric_limits<int>::max()) ? v : v + 1;
}

int Decr(const int& v) {
  return (v == std::numeric_limits<int>::min()) ? v : v - 1;
}

std::ostream& operator<<(std::ostream& out, const Variable& v) {
  return out << v.GetName() << " " << v.dom_;
}

// Variables
Variable::Variable(Solver* cp, std::string name, int lb, int ub)
    : id_(-1), name_(name), dom_(lb, ub), solver_(cp) {
  solver_->RegisterVar(this);
}

Variable::Variable(const Variable& other)
  : Variable(other.GetSolver(), other.GetName(), other.GetLB(), other.GetUB()) {}

Variable::Variable(Variable&& other)
  : Variable(other.GetSolver(), other.GetName(), other.GetLB(), other.GetUB()) {}

Variable& Variable::operator=(const Variable& other) {
  UpdateBound(other.GetLB(), other.GetUB());
  return *this;
}

Variable& Variable::operator=(Variable&& other) {
  UpdateBound(other.GetLB(), other.GetUB());
  return *this;
}

void Variable::Notify() {
  for (auto c : observers_) { solver_->Schedule(c); }
}

STATUS Variable::UpdateLB(int newMin) {
  auto status = dom_.UpdateLB(newMin);
  if (isSuccess(status)) { Notify(); }
  return status;
}

STATUS Variable::UpdateUB(int newMax) {
  auto status = dom_.UpdateUB(newMax);
  if (isSuccess(status)) { Notify(); }
  return status;
}

STATUS Variable::UpdateBound(int newMin, int newMax) {
  auto status = dom_.UpdateBound(newMin, newMax);
  if (isSuccess(status)) { Notify(); }
  return status;
}

STATUS Variable::UpdateBound(const Range& r) {
  auto status = dom_.UpdateBound(r);
  if (isSuccess(status)) { Notify(); }
  return status;
}

STATUS Variable::Assign(int v) {
  auto status = dom_.Assign(v);
  if (isSuccess(status)) { Notify(); }
  return status;
}

std::string Variable::GetName() const noexcept {
  if (name_ == "") {
    return "X_" + std::to_string(id_);
  }
  return name_;
}

IntVar operator+(IntVar x, IntVar y) {
  Range rng(x.GetRange() + y.GetRange());
  auto cp = x.GetSolver();
  auto& storage = cp->GetStorage();
  auto rv = storage.MakeIntVar(cp, rng);
  cp->Add(storage.MakePlus(*rv, x, y));
  return *rv;
}
IntVar operator+(IntVar x, int c) {
  auto cp = x.GetSolver();
  auto y = cp->GetStorage().MakeIntConstant(cp, c);
  return x + y;
}
IntVar operator+(int c, IntVar x) {
  auto cp = x.GetSolver();
  auto y = cp->GetStorage().MakeIntConstant(cp, c);
  return y + x;
}
IntVar operator+(IntVar x, IntVarPtr y) { return x + *y; }
IntVar operator+(IntVarPtr x, IntVar y) { return *x + y; }

IntVar operator-(IntVar x, IntVar y) {
  Range rng(x.GetRange() - y.GetRange());
  auto cp = x.GetSolver();
  auto rv = cp->GetStorage().MakeIntVar(cp, rng);
  cp->Add(cp->GetStorage().MakeMinus(*rv, x, y));
  return *rv;
}
IntVar operator-(IntVar x, int c) {
  auto cp = x.GetSolver();
  auto y = cp->GetStorage().MakeIntConstant(cp, c);
  return x - y;
}
IntVar operator-(int c, IntVar x) {
  auto cp = x.GetSolver();
  auto y = cp->GetStorage().MakeIntConstant(cp, c);
  return y - x;
}
IntVar operator-(IntVar x, IntVarPtr y) { return x - *y; }
IntVar operator-(IntVarPtr x, IntVar y) { return *x - y; }

IntVar operator*(IntVar x, IntVar y) {
  Range rng(x.GetRange() * y.GetRange());
  auto cp = x.GetSolver();
  auto rv = cp->GetStorage().MakeIntVar(cp, rng);
  cp->Add(cp->GetStorage().MakeMult(*rv, x, y));
  return *rv;
}
IntVar operator*(IntVar x, int c) {
  auto cp = x.GetSolver();
  auto y = cp->GetStorage().MakeIntConstant(cp, c);
  return x * y;
}
IntVar operator*(int c, IntVar x) {
  auto cp = x.GetSolver();
  auto y = cp->GetStorage().MakeIntConstant(cp, c);
  return y * x;
}
IntVar operator*(IntVar x, IntVarPtr y) { return x * *y; }
IntVar operator*(IntVarPtr x, IntVar y) { return *x * y; }

IntVar operator/(IntVar x, IntVar y) {
  Range rng(x.GetRange() / y.GetRange());
  auto cp = x.GetSolver();
  auto rv = cp->GetStorage().MakeIntVar(cp, rng);
  cp->Add(cp->GetStorage().MakeDiv(*rv, x, y));
  return *rv;
}
IntVar operator/(IntVar x, int c) {
  auto cp = x.GetSolver();
  auto y = cp->GetStorage().MakeIntConstant(cp, c);
  return x / y;
}
IntVar operator/(int c, IntVar x) {
  auto cp = x.GetSolver();
  auto y = cp->GetStorage().MakeIntConstant(cp, c);
  return y / x;
}
IntVar operator/(IntVar x, IntVarPtr y) { return x / *y; }
IntVar operator/(IntVarPtr x, IntVar y) { return *x / y; }

IntVar operator%(IntVar x, int N) {
  auto cp = x.GetSolver();
  auto rv = cp->GetStorage().MakeIntVar(cp, -N, N);
  cp->Add(cp->GetStorage().MakeModulo(*rv, x, N));
  return *rv;
}

ConstraintPtr operator==(IntVar x, int c) {
  auto cp = x.GetSolver();
  auto y = cp->GetStorage().MakeIntConstant(cp, c);
  return cp->GetStorage().MakeEqual(x, *y);
}
ConstraintPtr operator==(int c, IntVar x) {
  auto& storage = x.GetSolver()->GetStorage();
  auto y = storage.MakeIntConstant(x.GetSolver(), c);
  return storage.MakeEqual(*y, x);
}
ConstraintPtr operator==(IntVar x, IntVar y) {
  return x.GetSolver()->GetStorage().MakeEqual(x, y);
}

ConstraintPtr operator!=(IntVar x, int c) {
  return x.GetSolver()->GetStorage().MakeNotEqual(x, c);
}
ConstraintPtr operator!=(int c, IntVar x) {
  return x.GetSolver()->GetStorage().MakeNotEqual(x, c);
}

ConstraintPtr operator<=(IntVar x, IntVar y) {
  return x.GetSolver()->GetStorage().MakeLessThanEqual(x, y);
}
ConstraintPtr operator<=(IntVar x, int c) {
  auto cp = x.GetSolver();
  auto y = cp->GetStorage().MakeIntConstant(cp, c);
  return x <= *y;
}
ConstraintPtr operator<=(int c, IntVar x) {
  auto cp = x.GetSolver();
  auto y = cp->GetStorage().MakeIntConstant(cp, c);
  return *y <= x;
}

ConstraintPtr operator>=(IntVar x, IntVar y) {
  return x.GetSolver()->GetStorage().MakeGreaterThanEqual(x, y);
}
ConstraintPtr operator>=(IntVar x, int c) {
  auto cp = x.GetSolver();
  auto y = cp->GetStorage().MakeIntConstant(cp, c);
  return x >= *y;
}
ConstraintPtr operator>=(int c, IntVar x) {
  auto cp = x.GetSolver();
  auto y = cp->GetStorage().MakeIntConstant(cp, c);
  return *y >= x;
}

ConstraintPtr operator<(IntVar x, IntVar y) {
  return x.GetSolver()->GetStorage().MakeLessThan(x, y);
}
ConstraintPtr operator<(IntVar x, int c) {
  auto cp = x.GetSolver();
  auto y = cp->GetStorage().MakeIntConstant(cp, c);
  return x < *y;
}
ConstraintPtr operator<(int c, IntVar x) {
  auto cp = x.GetSolver();
  auto y = cp->GetStorage().MakeIntConstant(cp, c);
  return *y < x;
}

ConstraintPtr operator>(IntVar x, IntVar y) {
  return x.GetSolver()->GetStorage().MakeGreaterThan(x, y);
}
ConstraintPtr operator>(IntVar x, int c) {
  auto cp = x.GetSolver();
  auto y = cp->GetStorage().MakeIntConstant(cp, c);
  return x >= *y;
}
ConstraintPtr operator>(int c, IntVar x) {
  auto cp = x.GetSolver();
  auto y = cp->GetStorage().MakeIntConstant(cp, c);
  return *y > x;
}

Objective::Objective(IntVarPtr obj, Direction dir)
    : obj_(obj),
      dir_(dir),
      value_((dir == Direction::MIN) ? INT_LIMIT::max() : INT_LIMIT::min()),
      todo_(nullptr) {
  todo_ = obj_->GetSolver()->DoOnSolution([this] { value_ = obj_->Value(); });
}

STATUS Objective::Post() {
  obj_->AddObserver(this);
  return Propagate();
}

STATUS Objective::Propagate() {
  if (obj_->Empty()) { 
    return STATUS::FAIL; 
  }
  if (dir_ == Direction::MAX) {
    if (isFail(obj_->UpdateLB(value_ + 1))) { 
      return STATUS::FAIL;
    }
  } else {
    if (isFail(obj_->UpdateUB(value_ - 1))) { 
      return STATUS::FAIL;
    }
  };

  return STATUS::SUCCESS;
}

int Objective::Value() { return value_; }

IntVarPtr Objective::Subject() { return obj_; }

void Objective::Print() {
  if (dir_ == Direction::MAX) {
    std::cout << "Maximize(";
  } else {
    std::cout << "Minimize(";
  }
  std::cout << *obj_ << ")\n";
}

void Objective::Deactivate() {
  Constraint::Deactivate();
  todo_->Disable();
}

void Objective::Activate() {
  Constraint::Activate();
  todo_->Enable();
}

BinaryWithConst::BinaryWithConst(IntVarPtr x, int c)
    : Constraint(), x_(x), c_(c) {}
BinaryWithConst::BinaryWithConst(IntVar x, int c) : BinaryWithConst(&x, c) {}

STATUS BinaryWithConst::Post() {
  if (!x_->IsAssigned()) {
    x_->AddObserver(this);
  } 
  return Propagate();
}

BinaryConstraint::BinaryConstraint(IntVarPtr x, IntVarPtr y)
    : Constraint(), x_(x), y_(y) {}
BinaryConstraint::BinaryConstraint(IntVar x, IntVar y)
    : BinaryConstraint(&x, &y) {}

STATUS BinaryConstraint::Post() {
  if (!x_->IsAssigned()) {
    x_->AddObserver(this);
  }
  if (!y_->IsAssigned()) {
    y_->AddObserver(this);
  }
  return Propagate();
}

TernaryConstraint::TernaryConstraint(IntVarPtr z, IntVarPtr x, IntVarPtr y)
    : Constraint(), x_(x), y_(y), z_(z) {}
TernaryConstraint::TernaryConstraint(IntVar z, IntVar x, IntVar y)
    : TernaryConstraint(&z, &x, &y) {}

STATUS TernaryConstraint::Post() {
  if (!z_->IsAssigned()) { z_->AddObserver(this); }
  if (!x_->IsAssigned()) { x_->AddObserver(this); }
  if (!y_->IsAssigned()) { y_->AddObserver(this); }
  return Propagate();
}

GThan::GThan(IntVarPtr x, IntVarPtr y) : BinaryConstraint(x, y) {}
GThan::GThan(IntVar x, IntVar y) : BinaryConstraint(x, y) {}

STATUS GThan::Propagate() {
  if (x_->IsAssigned() && y_->IsAssigned()) {
    if (x_->Value() <= y_->Value()) { return STATUS::FAIL; }
  } else {
    if (isFail(x_->UpdateLB(y_->GetLB() + 1))) { return STATUS::FAIL; }
    if (isFail(y_->UpdateUB(x_->GetUB() - 1))) { return STATUS::FAIL; }
  }
  return STATUS::SUCCESS;
}

void GThan::Print() { std::cout << *x_ << " > " << *y_ << "\n"; }

LThan::LThan(IntVarPtr x, IntVarPtr y) : BinaryConstraint(x, y) {}
LThan::LThan(IntVar x, IntVar y) : BinaryConstraint(x, y) {}
STATUS LThan::Propagate() {
  if (x_->IsAssigned() && y_->IsAssigned()) {
    if (x_->Value() >= y_->Value()) { return STATUS::FAIL; }
  } else {
    if (isFail(y_->UpdateLB(x_->GetLB() + 1))) { return STATUS::FAIL; }
    if (isFail(x_->UpdateUB(y_->GetUB() - 1))) { return STATUS::FAIL; }
  }
  return STATUS::SUCCESS;
}
void LThan::Print() { std::cout << *x_ << " < " << *y_ << "\n"; }

GTEqual::GTEqual(IntVarPtr x, IntVarPtr y) : BinaryConstraint(x, y) {}
GTEqual::GTEqual(IntVar x, IntVar y) : BinaryConstraint(x, y) {}

STATUS GTEqual::Propagate() {
  if (x_->IsAssigned() && y_->IsAssigned()) {
    if (x_->Value() < y_->Value()) { return STATUS::FAIL; }
  } else {
    if (isFail(x_->UpdateLB(y_->GetLB()))) { return STATUS::FAIL; }
    if (isFail(y_->UpdateUB(x_->GetUB()))) { return STATUS::FAIL; }
  }
  return STATUS::SUCCESS;
}

void GTEqual::Print() { std::cout << *x_ << " >= " << *y_ << "\n"; }

LTEqual::LTEqual(IntVarPtr x, IntVarPtr y) : BinaryConstraint(x, y) {}
LTEqual::LTEqual(IntVar x, IntVar y) : BinaryConstraint(x, y) {}

STATUS LTEqual::Propagate() {
  if (x_->IsAssigned() && y_->IsAssigned()) {
    if (x_->Value() > y_->Value()) { return STATUS::FAIL; }
  } else {
    if (isFail(y_->UpdateLB(x_->GetLB()))) { return STATUS::FAIL; }
    if (isFail(x_->UpdateUB(y_->GetUB()))) { return STATUS::FAIL; }
  }
  return STATUS::SUCCESS;
}
void LTEqual::Print() { std::cout << *x_ << " <= " << *y_ << "\n"; }

NotEqualC::NotEqualC(IntVarPtr x, int c) : BinaryWithConst(x, c) {}
NotEqualC::NotEqualC(IntVar x, int c) : BinaryWithConst(x, c) {}

STATUS NotEqualC::Propagate() {
  if (x_->IsAssigned()) {
    if (x_->Value() == c_) { return STATUS::FAIL; }
  } else if (x_->GetLB() == c_) {
    if (isFail(x_->UpdateLB(c_ + 1))) { return STATUS::FAIL; }
  } else if (x_->GetUB() == c_) {
    if (isFail(x_->UpdateLB(c_ - 1))) { return STATUS::FAIL; }
  }
  return STATUS::SUCCESS;
}

void NotEqualC::Print() { std::cout << *x_ << " != " << c_ << "\n"; }

Equal::Equal(IntVarPtr x, IntVarPtr y) : BinaryConstraint(x, y) {}
Equal::Equal(IntVar x, IntVar y) : BinaryConstraint(x, y) {}

STATUS Equal::Propagate() {
  if (x_->IsAssigned() && y_->IsAssigned()) {
    if (x_->Value() != y_->Value()) { return STATUS::FAIL; }
  } else {
    if (isFail(x_->UpdateBound(y_->GetLB(), y_->GetUB())) ||
        isFail(y_->UpdateBound(x_->GetLB(), x_->GetUB()))) {
      return STATUS::FAIL;
    }
  }
  return STATUS::SUCCESS;
}

void Equal::Print() { std::cout << *x_ << " == " << *y_ << "\n"; }

// z = x + y
STATUS TernaryPlus::Propagate() {
  STATUS status;
  bool hasChanged = true;
  while (hasChanged) {
    hasChanged = false;
    status = z_->UpdateBound(x_->GetRange() + y_->GetRange());
    if (isKnown(status)) {
      if (isFail(status)) { return STATUS::FAIL; }
      hasChanged = true;
    }

    status = x_->UpdateBound(z_->GetRange() - y_->GetRange());
    if (isKnown(status)) {
      if (isFail(status)) { return STATUS::FAIL; }
      hasChanged = true;
    }

    status = y_->UpdateBound(z_->GetRange() - x_->GetRange());
    if (isKnown(status)) {
      if (isFail(status)) { return STATUS::FAIL; }
      hasChanged = true;
    }
  }
  return STATUS::SUCCESS;
}

void TernaryPlus::Print() {
  std::cout << *z_ << "=" << *x_ << "+" << *y_ << "\n";
}

// z = x - y
STATUS TernarySub::Propagate() {
  STATUS status;
  bool hasChanged = true;
  while (hasChanged) {
    hasChanged = false;
    status = z_->UpdateBound(x_->GetRange() - y_->GetRange());
    if (isKnown(status)) {
      if (isFail(status)) { return STATUS::FAIL; }
      hasChanged = true;
    }

    status = x_->UpdateBound(z_->GetRange() + y_->GetRange());
    if (isKnown(status)) {
      if (isFail(status)) { return STATUS::FAIL; }
      hasChanged = true;
    }

    status = y_->UpdateBound(x_->GetRange() - z_->GetRange());
    if (isKnown(status)) {
      if (isFail(status)) { return STATUS::FAIL; }
      hasChanged = true;
    }
  }
  return STATUS::SUCCESS;
}

void TernarySub::Print() {
  std::cout << *z_ << " = " << *x_ << " - " << *y_ << "\n";
}

// z = x * y
STATUS TernaryMult::Propagate() {
  STATUS status;
  bool hasChanged = true;
  while (hasChanged) {
    hasChanged = false;
    status = z_->UpdateBound(x_->GetRange() * y_->GetRange());
    if (isKnown(status)) {
      if (isFail(status)) { return STATUS::FAIL; }
      hasChanged = true;
    }

    status = x_->UpdateBound(z_->GetRange() / y_->GetRange());
    if (isKnown(status)) {
      if (isFail(status)) { return STATUS::FAIL; }
      hasChanged = true;
    }

    status = y_->UpdateBound(z_->GetRange() / x_->GetRange());
    if (isKnown(status)) {
      if (isFail(status)) { return STATUS::FAIL; }
      hasChanged = true;
    }
  }
  return STATUS::SUCCESS;
}

void TernaryMult::Print() {
  std::cout << *z_ << " = " << *x_ << " x " << *y_ << "\n";
}

// z = x * y
STATUS TernaryPositiveMult::Propagate() {
  STATUS status;
  bool hasChanged = true;
  while (hasChanged) {
    hasChanged = false;
    status = z_->UpdateBound(x_->GetLB() * y_->GetLB(), x_->GetUB() * y_->GetUB());
    if (isKnown(status)) {
      if (isFail(status)) { return STATUS::FAIL; }
      hasChanged = true;
    }

    status = x_->UpdateBound(z_->GetLB() / y_->GetUB(), z_->GetUB() / y_->GetLB());
    if (isKnown(status)) {
      if (isFail(status)) { return STATUS::FAIL; }
      hasChanged = true;
    }

    status = y_->UpdateBound(z_->GetLB() / x_->GetUB(), z_->GetUB() / x_->GetLB());
    if (isKnown(status)) {
      if (isFail(status)) { return STATUS::FAIL; }
      hasChanged = true;
    }
  }
  return STATUS::SUCCESS;
}

void TernaryPositiveMult::Print() {
  std::cout << *z_ << " = " << *x_ << " x " << *y_ << "\n";
}

// z = x / y
STATUS TernaryDiv::Propagate() {
  Range z;
  STATUS status;
  bool hasChanged = true;
  while (hasChanged) {
    hasChanged = false;

    if (x_->IsAssignedTo(0) && y_->IsAssignedTo(0)) { return STATUS::FAIL; }

    status = z_->UpdateBound(x_->GetRange() / y_->GetRange());
    if (isKnown(status)) {
      if (isFail(status)) { return STATUS::FAIL; }
      hasChanged = true;
    }
    z.SetBound(Decr(z_->GetLB()), Incr(z_->GetUB()));
    // ex [5,5] = [64,64] / [12,12] is ok since 64/12 = 5.3 but we cannot use
    // 5*12 to filter, we should filter using [5-1*12, 5+1*12]

    status = x_->UpdateBound(y_->GetRange() * z);
    if (isKnown(status)) {
      if (isFail(status)) { return STATUS::FAIL; }
      hasChanged = true;
    }
    // ex [8,8]= [256,256]/[30,120] need to be carefull otherwise it will fix
    // [30,120] to 32
    status = y_->UpdateBound(x_->GetRange() / z);
    if (isKnown(status)) {
      if (isFail(status)) { return STATUS::FAIL; }
      hasChanged = true;
    }
  }

  return STATUS::SUCCESS;
}

void TernaryDiv::Print() {
  std::cout << *z_ << " = " << *x_ << " / " << *y_ << "\n";
}

// z = x / y
STATUS TernaryPositiveDiv::Propagate() {
  Range z;
  STATUS status;
  bool hasChanged = true;
  while (hasChanged) {
    hasChanged = false;

    status = z_->UpdateBound(x_->GetLB() / y_->GetUB(), x_->GetUB() / y_->GetLB());
    // status = z_->UpdateBound(x_->GetRange() / y_->GetRange());
    if (isKnown(status)) {
      if (isFail(status)) { return STATUS::FAIL; }
      hasChanged = true;
    }
    z.SetBound(Decr(z_->GetLB()), Incr(z_->GetUB()));
    // ex [5,5] = [64,64] / [12,12] is ok since 64/12 = 5.3 but we cannot use
    // 5*12 to filter, we should filter using [5-1*12, 5+1*12]

    status = x_->UpdateBound(y_->GetLB() * z.GetLB(), y_->GetUB() * z.GetUB());
    // status = x_->UpdateBound(y_->GetRange() * z);
    if (isKnown(status)) {
      if (isFail(status)) { return STATUS::FAIL; }
      hasChanged = true;
    }
    // ex [8,8]= [256,256]/[30,120] need to be carefull otherwise it will fix
    // [30,120] to 32
    if (z.GetLB() > 0) {
      status = y_->UpdateBound(x_->GetLB() / z.GetUB(), x_->GetUB() / z.GetLB());
      // status = y_->UpdateBound(x_->GetRange() / z);
      if (isKnown(status)) {
        if (isFail(status)) { return STATUS::FAIL; }
        hasChanged = true;
      }
    }
  }
  return STATUS::SUCCESS;
}

void TernaryPositiveDiv::Print() {
  std::cout << *z_ << " = " << *x_ << " / " << *y_ << "\n";
}

// y = x % N

Modulo::Modulo(IntVarPtr y, IntVarPtr x, int N)
    : BinaryConstraint(x, y), N_(N) {}
Modulo::Modulo(IntVar y, IntVar x, int N) : BinaryConstraint(x, y), N_(N) {}

STATUS Modulo::Propagate() {
  if (x_->GetLB() >= 0) {
    if (isFail(y_->UpdateLB(0))) { return STATUS::FAIL; }
  }
  if (x_->GetUB() <= 0) {
    if (isFail(y_->UpdateUB(0))) { return STATUS::FAIL; }
  }
  if (x_->IsAssigned()) {
    if (isFail(y_->Assign(x_->Value() % N_))) { return STATUS::FAIL; }
  } else if (y_->IsAssigned()) {
    int Y = y_->Value();
    if (x_->GetLB() % N_ != Y) {
      for (auto nm = x_->GetLB() + 1, stop = x_->GetUB(); nm <= stop; nm++) {
        if (Y == (nm % N_)) {
          if (x_->UpdateLB(nm) != STATUS::SUCCESS) { return STATUS::FAIL; }
          break;
        }
      }
    }
    if (x_->GetUB() % N_ != Y) {
      for (auto nm = x_->GetUB() - 1, stop = x_->GetLB(); nm >= stop; nm--) {
        if (Y == (nm % N_)) {
          if (x_->UpdateUB(nm) != STATUS::SUCCESS) { return STATUS::FAIL; }
          break;
        }
      }
    }
  }
  return STATUS::SUCCESS;
}

void Modulo::Print() {
  std::cout << *y_ << " = " << *x_ << " % " << N_ << "\n";
}

Solver::Solver() : objective_(nullptr), search_(new Bisection()), init_(false) {
  constexpr int NB_VARS_RESERVED = 256;
  constexpr int NB_CTRS_RESERVED = 256;
  constexpr int TRAIL_RESERVED = 32;
  variables_.reserve(NB_VARS_RESERVED);
  constraints_.reserve(NB_CTRS_RESERVED);
  trail_.reserve(TRAIL_RESERVED);
}

Solver::~Solver() {
  if (objective_) { delete objective_; }
  delete search_;
}

void Solver::RegisterVar(IntVarPtr v) {
  if (v) {
    v->SetId(variables_.size());
    variables_.push_back(v);
  }
}

bool Solver::Post(ConstraintPtr c) {
  if (c == nullptr) { return false; }

  if (c->IsActive()) {
    if (c->Post() == STATUS::FAIL) {
      std::cerr << "Constraint Failed" << '\n';
      return false;
    }
  }
  return true;
}

void Solver::Schedule(ConstraintPtr c) {
  if (!c->InQueue() && c->IsActive()) {
    queue_.emplace(c);
    c->SetInQueue(true);
  }
}

bool Solver::FixPoint(bool enforce) {
  if (enforce) {
    for (auto c : constraints_) { Schedule(c); }
  }
  return FixPoint();
}

// Compute the fix point
bool Solver::FixPoint() {
  ConstraintPtr curr = nullptr;
  while (!queue_.empty()) {
    curr = queue_.front();
    if (curr->Propagate() == STATUS::FAIL) {
      Flush();
      return false;
    }
    queue_.pop();
    curr->SetInQueue(false);
  }
  return true;
}

void Solver::SaveInitialDomains() {
  starting_point_.reserve(variables_.size());
  for (auto v : variables_) {
    starting_point_.push_back(v->dom_);
  }
}

void Solver::Save() {
  trail_.push_back({});
  auto& back = trail_.back();
  back.reserve(variables_.size());
  for (auto v : variables_) {
    back.push_back(v->dom_);
  }
}

void Solver::Restore(const State& checkpoint) {
  for (IntVarPtr v : variables_) {
    v->dom_.lb_ = checkpoint[v->id_].lb_;
    v->dom_.ub_ = checkpoint[v->id_].ub_;
  }
}

void Solver::Restore() {
  assert(trail_.size() > 0);
  Restore(trail_.back());
  trail_.pop_back();
}

void Solver::Reset(bool save) {
  Flush();
  trail_.clear();
  RestoreInitialDomains();
  solutions_.clear();
  todo_.clear();
  if (save) { Save(); }
}

void Solver::Flush() {
  // flush
  while (!queue_.empty()) {
    queue_.front()->SetInQueue(false);
    queue_.pop();
  }
  assert(queue_.empty());
}

void Solver::PrintVariables() {
  for (auto v : variables_) {
    std::cout << (*v) << std::endl;
  }
}

void Solver::PrintSolution() {
  assert(NbSol() > 0);
  for (auto v : variables_) {
    std::cout << v->GetName() << " = " << v->Value() << std::endl;
  }
}

void Solver::Print() {
  std::cout << "----------------------\n";
  std::cout << "Model with " << variables_.size() << " #vars and "
            << constraints_.size() << " # constraints\n";
  for (auto c : constraints_) {
    c->Print();
  }
  std::cout << "----------------------\n";
}

std::shared_ptr<FunctionWrapper> Solver::DoOnSolution(
    std::function<void(void)> l) {
  todo_.emplace_back(std::make_shared<FunctionWrapper>(l, true));
  return todo_.back();
}

void Solver::OnSolution() {
  SaveSolution();
  for (auto& f : todo_) {
    (*f)();
  }
}

bool Solver::SolveImpl() {
  for (;;) {
    if (Depth() == 0) { return false; }
    Restore();
    if (FixPoint(true)) {
      auto x = search_->NextVariable(variables_);
      if (x) {
        assert(!x->Empty());
        Range copy(x->dom_);
        auto splits = search_->Split(copy);
        for (auto& r : splits) {
          x->dom_.SetBound(r);
          Save();
        }
      } else {
        OnSolution();
        // std::cout << "Solution found\n";
        return true;
      }
    }
  }
  assert(false);
  return false;
}

void Solver::SaveSolution() {
  solutions_.push_back({});
  auto& sol = solutions_.back();
  sol.reserve(variables_.size());
  for (auto v : variables_) { sol.push_back(v->Value()); }
}

bool Solver::Solve() {
  if (!init_) {
    // just in case we want to reset the solver
    // and solve again
    SaveInitialDomains();
    for (auto c : constraints_) {
      if (!Post(c)) { return false; }
    }
    Save();
    init_ = true;
  }
  return SolveImpl();
}

int Solver::Optimize(std::function<void(void)> todo, bool all) {
  if (objective_) {
    while (Solve()) {}
    if (all && NbSol() > 0) {
      int best = objective_->Value();
      Reset(false);
      // deactivate objective propagator
      objective_->Deactivate();
      objective_->Subject()->Assign(best);
      Save();
      while (Solve()) {
        todo();
      }
    }
    return objective_->Value();
  } else {
    return Solve();
  }
}

void Solver::Minimize(Variable& obj) {
  objective_ = new Objective(&obj, Direction::MIN);
  Add(objective_);
}

void Solver::Maximize(Variable& obj) {
  objective_ = new Objective(&obj);
  Add(objective_);
}

IntVarPtr Solver::GetObjective() const {
  assert(objective_ != nullptr);
  return objective_->Subject();
}
}  // namespace LightCP
