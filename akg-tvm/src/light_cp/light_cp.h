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
#ifndef LIGHTCP_HPP
#define LIGHTCP_HPP

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace LightCP {
// #######################################
// # STATUS
// #######################################
/**
 * [hzi] we could also add flags and directly apply
 * them in operators to speed up.
 *
 */
/**
 * The status is:
 * - FAIL if a fail occurs (consequence of an empty domain)
 * - SUCCESS if the filtering/update of the bound was successful, i.e., a
 * reduction of the lb or ub was done. Consequently it leads to adding now
 * constraints to the queue.
 * - UNKNOWN neither a success, nor a fail occurs. Consequently, no change is
 * made on the queue of propagation.
 **/
enum class STATUS { SUCCESS = 0, FAIL = 1, UNKNOWN = 2 };

inline bool isKnown(const STATUS& s) { return !(static_cast<int>(s) >> 1); }
inline bool isUnKnown(const STATUS& s) { return (static_cast<int>(s) >> 1); }
inline bool isFail(const STATUS& s) { return (static_cast<int>(s) & 1); }
inline bool isSuccess(const STATUS& s) { return !(static_cast<int>(s)); }

/**
 * @brief Overloading of the operator | (or) between status in order to compute
 * updateLB or updateUB nicely
 *
 * @param left
 * @param right
 * @return STATUS
 */
STATUS operator|(const STATUS& left, const STATUS& right);
/**
 * @brief Overloading of the operator |= (or with assignment) between status in
 * order to compute update |= updateLB; followed by an updateUB nicely
 *
 * @param left
 * @param right
 * @return STATUS&
 */
STATUS& operator|=(STATUS& left, const STATUS& right);

std::ostream& operator<<(std::ostream& out, const STATUS& s);

// #######################################
// # RANGE
// #######################################

/**
 * @brief This class represents intervals, i.e., [lb,ub], where lb and ub are
 * integers. The interval is said empty if lb > ub. This interval relies on int
 * (32 bits). To make it work with 64 bits int, "int" needs to be changed to
 * "int64_t". Consequently the size method need to be changed.
 */
class Range {
  // the goal of this friend class is to restore the directly lb and ub
  friend class Solver;

 protected:
  /**
   * @brief The lower bound of the domain
   *
   */
  int lb_;
  /**
   * @brief The upper bound of the domain
   *
   */
  int ub_;

 public:
  /**
   * @brief Default constructor.
   * Since no bound are specified it creates the domain [INT_MIN, INT_MAX]
   *
   */
  Range();
  /**
   * @brief Construct a new range [lb,ub]
   *
   * @param lb the lower bound of the interval
   * @param ub the upper bound of the interval
   */
  Range(int lb, int ub);
  /**
   * @brief Copy constructor
   *
   * @param rng the range to copy
   */
  Range(const Range& rng);
  /**
   * @brief Assignment operator
   */
  Range& operator=(const Range& rng);
  /**
   * @brief Set both lower and upper bound
   * No check is made i.e., if lb > ub no fail is raised, the range is just
   * empty.
   *
   * @param lb the lower bound
   * @param ub the upper bound
   */
  void SetBound(int lb, int ub);
  /**
   * @brief Set both lower and upper bound of the interval using @p r lower
   * bound and upper bound No check is made i.e., if lb > ub no fail is raised,
   * the range is just empty.
   *
   * @param r the range from which new values are fetch
   */
  void SetBound(const Range& r);

  /**
   * @brief Returns the size of the interval. A 64 bits int is returned in order
   * to get exact size when having [INT_MIN, INT_MAX] If the interval is [a,a]
   * then the size is 1. If the interval is empty the size is 0.
   * @return int64_t the size of the interval
   */
  int64_t Size() const;

  /**
   * @brief Returns the lower bound of the interval
   *
   * @return int the lower bound
   */
  inline int GetLB() const { return lb_; };

  /**
   * @brief Returns the upper bound of the interval
   *
   * @return int the upper bound
   */
  inline int GetUB() const { return ub_; };

  /**
   * @brief Checks if the interval is empty. An interval is said empty if lb >
   * ub
   *
   * @return true the interval is empty
   * @return false the interval is not empty
   */
  inline bool Empty() const { return ub_ < lb_; };

  /**
   * @brief Checks if @p v is in the interval
   *
   * @param v the value to check
   * @return true the value @p v is in the interval
   * @return false the value @p v is not in the interval
   */
  inline bool Contains(int v) const { return v <= ub_ && v >= lb_; };

  /**
   * @brief Helper method to check if the range is strictly positive, i.e., all
   * values of the range are positive.
   * For instance [2,5] is strictly positive.
   * [-2,5] is not strictly positive.
   *
   * @return true all values of the range are positive
   * @return false the range is not strictly positive (i.e., it exists some
   * values that are not strictly positive)
   */
  inline friend bool IsStrictlyPositive(const Range& r) { return r.lb_ > 0; }

  /**
   * @brief Helper method to check if the range is strictly negative, i.e., all
   * values of the range are negative.
   * For instance [-5,-2] is strictly negative.
   * [-2,5] is not strictly positive.
   *
   * @return true all values of the range are negative
   * @return false the range is not strictly negative (i.e., it exists some
   * values that are not strictly negative)
   */
  inline friend bool IsStrictlyNegative(const Range& r) { return r.ub_ < 0; }

  /**
   * @brief Pretty print of a range
   */
  friend std::ostream& operator<<(std::ostream& out, const Range& v);

  /**
   * @brief Method that adds two ranges
   *
   * @param lhs the left and side range
   * @param rhs the right and side range
   * @return Range the resulting range
   */
  friend Range operator+(const Range& lhs, const Range& rhs);

  /**
   * @brief Method that substracts two ranges
   *
   * @param lhs the left and side range
   * @param rhs the right and side range
   * @return Range the resulting range
   */
  friend Range operator-(const Range& lhs, const Range& rhs);

  /**
   * @brief Method that multiplies two ranges
   *
   * @param lhs the left and side range
   * @param rhs the right and side range
   * @return Range the resulting range
   */
  friend Range operator*(const Range& lhs, const Range& rhs);

  /**
   * @brief Method that divides two ranges
   *
   * @param num the left and side range
   * @param den the right and side range
   * @return Range the resulting range
   */
  friend Range operator/(const Range& num, const Range& den);

  /**
   * @brief Method that checks the equality between two ranges
   *
   * @param lhs the left and side range
   * @param rhs the right and side range
   * @return true if both range are the sames, false otherwise
   */
  friend bool operator==(const Range& lhs, const Range& rhs);
};

/**
 * @brief This class represents a domain based on interval
 *
 */
class RangeDomain : public Range {
 public:
  /**
   * @brief Construct a new domain with [lb, ub]
   *
   * @param lb the lower bound
   * @param ub the upper bound
   */
  RangeDomain(int lb, int ub) : Range(lb, ub) {}

  /**
   * @brief Construct a new domain based on the range @p rng
   *
   * @param rng the range defining the domain
   */
  explicit RangeDomain(const Range& rng) : Range(rng) {}

  RangeDomain(RangeDomain const& other);
  RangeDomain(RangeDomain&& other);
  RangeDomain& operator=(RangeDomain const& other);
  RangeDomain& operator=(RangeDomain && other);

  /**
   * @brief Updates the lower bound only if the new lower bound is greater than
   * the current one. If the domain become empty a STATUS::FAIL is returned. If
   * the domain stays valid a STATUS::SUCCESS is returned. If no change was
   * performed on the domain a STATUS::SUSPEND is returned.
   *
   * @return STATUS
   */
  STATUS UpdateLB(int newMin);

  /**
   * @brief Updates the upper bound only if the new upper bound is lower than
   * the current one. If the domain become empty a STATUS::FAIL is returned. If
   * the domain stays valid a STATUS::SUCCESS is returned. If no change was
   * performed on the domain a STATUS::SUSPEND is returned.
   *
   * @return STATUS
   */
  STATUS UpdateUB(int newMax);

  /**
   * @brief Updates the both lower and upper bound only if they are "improving"
   * the current one. It calls both RangeDomain::UpdateLB and
   * RangeDomain::UpdateUB.
   *
   * So, if the domain become empty a STATUS::FAIL is returned. If
   * the domain stays valid a STATUS::SUCCESS is returned. If no change was
   * performed on the domain a STATUS::SUSPEND is returned.
   *
   * @return STATUS
   */
  STATUS UpdateBound(int newMin, int newMax);

  /**
   * @brief Same as @see RangeDomain::UpdateBound(int,int)
   *
   * @return STATUS
   */
  inline STATUS UpdateBound(const Range& rng) {
    return UpdateBound(rng.GetLB(), rng.GetUB());
  };

  /**
   * @brief Assign the upper bound and the lower to @p v iff v is belong to it.
   *
   * If the domain become empty a STATUS::FAIL is returned. If
   * the domain stays valid a STATUS::SUCCESS is returned. If no change was
   * performed on the domain a STATUS::SUSPEND is returned.
   *
   * @return STATUS
   */
  STATUS Assign(int v);
};

/**
 * @brief Method that increments a value safely (i.e., no overflow is done).
 * Note that adding one to INT_MAX will return INT_MAX. Here we use INT_MAX as
 * +"infinity"
 *
 * @return int @p v + 1 if v is not INT_MAX, INT_MAX otherwise
 */
int Incr(const int& v);

/**
 * @brief Method that decrements a value safely (i.e., no underflow is done).
 * Note that substracting one to INT_MIN will return INT_MIN. Here we use
 * INT_MIN as -"infinity"
 *
 * @return int @p v - 1 if v is not INT_MIN, INT_MIN otherwise
 */
int Decr(const int& v);

// #######################################
// # FunctionWrapper
// #######################################

/**
 * @brief A wrapper over function that offers enabling/disabling of the function
 *
 */
class FunctionWrapper {
 private:
  /**
   * @brief The function that is runned
   *
   */
  std::function<void(void)> todo_;
  /**
   * @brief A boolean that says if the function is enabled or not
   *
   */
  bool enabled_;

 public:
  /**
   * @brief Construct a function wrapper of the lambda l
   *
   * @param l the function
   * @param enabled the boolean that says if the function is enabled or not
   */
  FunctionWrapper(std::function<void(void)>& l, bool enabled)
      : todo_(l), enabled_(enabled) {}

  /**
   * @brief Returns true if the function is enabled, false otherwise
   *
   * @return true the function is enabled
   * @return false the function is disabled
   */
  bool isEnabled() { return enabled_; }
  /**
   * @brief Disables the function
   *
   */
  void Disable() { enabled_ = false; }
  /**
   * @brief Enables the function
   *
   */
  void Enable() { enabled_ = true; }
  /**
   * @brief Runs the functions
   *
   */
  void operator()() {
    if (enabled_) { todo_(); }
  }
};

// #######################################
// # VARIABLE
// #######################################

class Constraint;
class Solver;
/**
 * @brief Class that represents an integer variable.
 * The associated domain is an interval of int. If int 64 is required then
 * replacing int by int64_t here and in the domain are enough to get it work.
 *
 */
class Variable {
  friend class Solver;

 private:
  /**
   * @brief The unique id of the variable.
   *
   */
  int id_;
  /**
   * @brief The name of the variable. If no name is given, then the pretty print
   * of the variable will be X_${id_} where ${id_} corresponds to the id of the
   * variable
   *
   */
  std::string name_;
  /**
   * @brief The domain of the variable
   *
   */
  RangeDomain dom_;
  /**
   * @brief The list of constraints that observes the variable change.
   * Each time a Variable::Notify is called all variable are notified by a
   * change of the variable domain.
   * @see Variable::Notify
   */
  std::vector<Constraint*> observers_;
  /**
   * @brief The solver that has register the variable
   *
   * @see Solver::RegisterVar
   *
   */
  Solver* solver_;

 public:
  Variable() = delete;
  Variable(const Variable& other);
  Variable(Variable&& other);
  Variable& operator=(const Variable& other);
  Variable& operator=(Variable&& other);

  Variable(Solver* cp, std::string name, int lb, int ub);
  Variable(Solver& cp, std::string name, int lb, int ub)
      : Variable(&cp, name, lb, ub){};
  Variable(Solver* cp, int lb, int ub) : Variable(cp, "", lb, ub){};
  Variable(Solver& cp, int lb, int ub) : Variable(&cp, "", lb, ub){};
  explicit Variable(Solver* cp)
      : Variable(cp, std::numeric_limits<int>::min(),
                 std::numeric_limits<int>::max()){};
  explicit Variable(Solver& cp)
      : Variable(cp, std::numeric_limits<int>::min(),
                 std::numeric_limits<int>::max()){};
  Variable(Solver* cp, int v) : Variable(cp, std::to_string(v), v, v){};
  Variable(Solver& cp, int v) : Variable(&cp, v){};
  Variable(Solver* cp, std::string name) : Variable(cp) { name_ = name; };
  Variable(Solver& cp, std::string name) : Variable(&cp, name){};

  /**
   * @brief Returns the id of the variable
   *
   * @return int
   */
  inline int GetId() const noexcept { return id_; };

  /**
   * @brief Set the id of the variable, it is called by the solver (see the
   * constructor of the variable)
   *
   */
  inline void SetId(int id) { id_ = id; };

  /**
   * @brief Returns the solver associated with the variable
   *
   * @return Solver*
   */
  inline Solver* GetSolver() const { return solver_; };

  /**
   * @brief Notify all listeners that the variable was modified
   *
   */
  void Notify();
  /**
   * @brief Register a new constraint as a listener of all changes that occurs
   * to the variable.
   *
   */
  inline void AddObserver(Constraint* ctr) { observers_.push_back(ctr); };

  /**
   * @brief Updates the lower bound with @p newMin if newMin is greater than the current
   * lower bound.
   *
   * @return true if the lower bound was updated
   * @return false no update is performed
   */
  STATUS UpdateLB(int newMin);
  /**
   * @brief Updates the upper bound with @p newMax if newMax is less than the current
   * upper bound.
   *
   * @return true if the upper bound was updated
   * @return false no update is performed
   */
  STATUS UpdateUB(int newMax);
  /**
   * @brief Updates both the lower and the upper bound if the update leads to an
   * "improvement"
   *
   * @param newMin the new lower bound
   * @param newMax the new upper bound
   * @return true the domain has changed
   * @return false the domain has not changed
   */
  STATUS UpdateBound(int newMin, int newMax);
  /**
   * @brief Updates both the lower and the upper bound if the update leads to an
   * "improvement"
   *
   * @param r the range that corresponds to the change
   * @return true the domain has changed
   * @return false the domain has not changed
   */
  STATUS UpdateBound(const Range& r);

  /**
   * @brief Updates both the lower and the upper bound to the value @param v if
   * it is contain in the current domain
   *
   * @param v the value of the assignment
   * @return true the domain has changed
   * @return false the domain has not changed
   */
  STATUS Assign(int v);

  /**
   * @brief Checks if the variable is assigned
   *
   * @return true the variable is assigned
   * @return false the variable is not assigned
   */
  inline bool IsAssigned() const { return dom_.Size() == 1; };

  /**
   * @brief Returns true if the variable is assigned to @p v, false otherwise
   *
   * @param v the value to check
   * @return true the variable is assigned to @p v
   * @return false the variable is not assigned to @p v
   */
  inline bool IsAssignedTo(int v) const {
    return IsAssigned() && dom_.GetLB() == v;
  }

  /**
   * @brief Returns true if the domain is empty (i.e., ub < lb)
   *
   * @return true the domain is empty
   * @return false the domain is not empty
   */
  inline bool Empty() const { return dom_.Empty(); };

  /**
   * @brief Returns the value of the variable, if it is bound (assigned), or it
   * raise an exception otherwise
   *
   * @return int
   */
  int Value() const {
    assert(IsAssigned());
    return dom_.GetLB();
  };

  /**
   * @brief Returns the size of the domain of the variable.
   * The size is an int64_t in order to compute exactly the size when the domain
   * is [INT_MIN, INT_MAX]
   *
   * @return int64_t
   */
  inline int64_t Size() const { return dom_.Size(); };

  /**
   * @brief Returns the lower bound of the domain of the variable
   *
   * @return int
   */
  inline int GetLB() const { return dom_.GetLB(); };

  /**
   * @brief Returns the upper bound of the domain of the variable
   *
   * @return int
   */
  inline int GetUB() const { return dom_.GetUB(); };

  /**
   * @brief Returns a copy of the domain as a range
   *
   * @return Range
   */
  inline Range GetRange() const { return static_cast<Range>(dom_); };

  /**
   * @brief Returns the name of the variable.
   * If no name is given when calling the constructor, then it will return
   * X_${id_}, where ${id_} is the value of the id of the variable
   *
   * @return std::string the name of the variable
   */
  std::string GetName() const noexcept;

  friend std::ostream& operator<<(std::ostream& out, const Variable& v);
};

using IntVarPtr = Variable*;
using VecIntVar = std::vector<IntVarPtr>;
using IntVar = Variable&;

// #######################################
// # SEARCH
// #######################################

/**
 * @brief Abstract class that defines a search strategy.
 * It provides both method to define a variable, and a value selector.
 *
 */
class Search {
 public:
  virtual ~Search() = default;
  /**
   * @brief The main method to control/define the variable selector strategy
   * It defines which variable is selected next. A nullptr is return when the
   * selection is done.
   *
   * @param vars the scope of the variable
   * @return IntVarPtr The next variable to select. It should return nullptr if
   * the selection is done.
   */

  virtual IntVarPtr NextVariable(VecIntVar& vars) = 0;
  /**
   * @brief It defines the splitting strategy (value selection strategy)
   * Note that the solver uses a DFS search and not BFS search.
   *
   * @param domain the initial domain
   * @return std::vector<Range> the vector of alternatives.
   */
  virtual std::vector<Range> Split(Range& domain) = 0;
};

/**
 * @brief the bisection search strategy uses:
 * - a lexicographic order as variable selector strategy
 * - a splitting of the domain in two as a value selector strategy
 *
 */
class Bisection : public Search {
 private:
  /**
   * @brief Computes the left and side sub-domain.
   * For instance if @p v is [INT_MIN, INT_MAX], the split left return [INT_MIN,
   * -1]
   *
   * @param v initial domain
   * @return Range output domain
   */
  Range SplitLeft(Range& v) {
    double mid = v.Size() / 2.f;
    int shift = std::ceil(mid);
    // left and side sub-domain
    int newMax = v.GetUB() - shift;
    return {v.GetLB(), newMax};
  }

  /**
   * @brief Computes the right and side sub-domain.
   * For instance if @p v is [INT_MIN, INT_MAX], the split left return [0,
   * INT_MAX]
   *
   * @param v initial domain
   * @return Range output domain
   */
  Range SplitRight(Range& v) {
    double mid = v.Size() / 2.f;
    int shift = std::floor(mid);
    // right and side sub-domain
    int newMin = v.GetLB() + shift;
    return {newMin, v.GetUB()};
  }

 public:
  Bisection() {}

  /**
   * @brief Returns the next variable. A lexicographic order is used
   *
   * @param vars
   * @return IntVarPtr
   */
  IntVarPtr NextVariable(VecIntVar& vars) override {
    for (auto& v : vars) {
      if (!v->IsAssigned()) { return v; }
    }
    return nullptr;
  }

  /**
   * @brief Compute two sub-domains [lb, mid], [mid, ub]
   *
   * @param domain initial domain
   * @return std::vector<Range> alternatives
   */
  std::vector<Range> Split(Range& domain) override {
    std::vector<Range> rv;
    rv.push_back(SplitLeft(domain));
    rv.push_back(SplitRight(domain));
#ifndef NDEBUG
    const size_t expected_size = 2;
    assert(rv.size() == expected_size);
    assert(rv[0].GetUB() + 1 == rv[1].GetLB());
#endif
    return rv;
  }
};

// #######################################
// # Constraint
// #######################################

/**
 * @brief Abstract class that represents a constraint
 *
 */
class Constraint {
 private:
  /**
   * @brief A boolean value to know if the constraint is already in the queue or
   * not
   *
   */
  bool in_queue_;
  /**
   * @brief A boolean value to know if the constraint is active or not.
   *
   */
  bool is_active;

 public:
  /**
   * @brief Constructor
   *
   */
  Constraint() : in_queue_(false), is_active(true) {}
  Constraint(const Constraint&) = delete;
  Constraint(Constraint&&) = delete;
  Constraint& operator=(const Constraint&) = delete;
  Constraint& operator=(Constraint&&) = delete;
  /**
   * @brief Destroy the Constraint
   *
   */
  virtual ~Constraint() = default;
  /**
   * @brief Method that post the constraint.
   * Usual behaviour of this method is to call propagate and to listen to even
   * related to variables of its scope.
   *
   */
  virtual STATUS Post() = 0;
  /**
   * @brief the filtering method.
   * If the constraint is register as a listener of a variable, then each time
   * this variable is modified the constraint will be pushed to the queue of the
   * solver and the propagate method will be called. Whatever the event, it is
   * always propagate which is called.
   */
  virtual STATUS Propagate() = 0;
  /**
   * @brief Returns true if the constraint is in the queue, false otherwise
   *
   * @return true the constraint is in the queue
   * @return false the constraint is not in the queue
   */
  bool InQueue() const noexcept { return in_queue_; }
  /**
   * @brief Set in_queue_ flag to @p v
   *
   * @param v the new value of in_queue_
   */
  void SetInQueue(bool v) { in_queue_ = v; }
  /**
   * @brief Returns true if the constraint is active, false otherwise
   *
   * @return true the constraint is active
   * @return false the constraint is not active
   */
  bool IsActive() const noexcept { return is_active; }
  /**
   * @brief Deactivates the constraint
   *
   */
  void Deactivate() { is_active = false; }
  /**
   * @brief Activates the constraint
   *
   */
  void Activate() { is_active = true; }
  /**
   * @brief A method that print the constraint.
   *
   */
  virtual void Print() {}
};

using ConstraintPtr = Constraint*;

// #######################################
// # EXPR
// #######################################

/**
 * @brief Overloading of the operator + between two references of variables.
 * X + Y
 *
 * @return IntVar the result of the + operator over two variables
 */
IntVar operator+(IntVar x, IntVar y);
/**
 * @brief Overloading of the operator + between one variable and one constant.
 * X + C
 *
 * @return IntVar the result of the + operator over one variable and one
 * constant
 */
IntVar operator+(IntVar x, int c);
/**
 * @brief Overloading of the operator + between one constant and one variable
 * C + X
 *
 * @return IntVar the result of the + operator over one constant and one
 * variable
 */
IntVar operator+(int c, IntVar x);
/**
 * @brief Overloading of the operator + between two variables (one reference,
 * and one pointer). X + Y
 *
 * @return IntVar the result of the + operator over two variables (one
 * reference, and one pointer)
 */
IntVar operator+(IntVar x, IntVarPtr y);
/**
 * @brief Overloading of the operator + between two variables (one pointer, and
 * one reference). X + Y
 *
 * @return IntVar the result of the + operator over two variables (one pointer,
 * and one reference)
 */
IntVar operator+(IntVarPtr x, IntVar y);

/**
 * @brief Overloading of the operator - between two references of variables.
 * X - Y
 *
 * @return IntVar the result of the - operator over two variables
 */
IntVar operator-(IntVar x, IntVar y);
/**
 * @brief Overloading of the operator - between one variable and one constant.
 * X - C
 *
 * @return IntVar the result of the - operator over one variable and one
 * constant
 */
IntVar operator-(IntVar x, int c);
/**
 * @brief Overloading of the operator - between one constant and one variable
 * C - X
 *
 * @return IntVar the result of the - operator over one constant and one
 * variable
 */
IntVar operator-(int c, IntVar x);
/**
 * @brief Overloading of the operator - between two variables (one reference,
 * and one pointer). X - Y
 *
 * @return IntVar the result of the - operator over two variables (one
 * reference, and one pointer)
 */
IntVar operator-(IntVar x, IntVarPtr y);
/**
 * @brief Overloading of the operator - between two variables (one pointer, and
 * one reference). X - Y
 *
 * @return IntVar the result of the - operator over two variables (one pointer,
 * and one reference)
 */
IntVar operator-(IntVarPtr x, IntVar y);

/**
 * @brief Overloading of the operator * between two references of variables.
 * X * Y
 *
 * @return IntVar the result of the * operator over two variables
 */
IntVar operator*(IntVar x, IntVar y);
/**
 * @brief Overloading of the operator * between one variable and one constant.
 * X * C
 *
 * @return IntVar the result of the * operator over one variable and one
 * constant
 */
IntVar operator*(IntVar x, int c);
/**
 * @brief Overloading of the operator * between one constant and one variable
 * C * X
 *
 * @return IntVar the result of the * operator over one constant and one
 * variable
 */
IntVar operator*(int c, IntVar x);
/**
 * @brief Overloading of the operator * between two variables (one reference,
 * and one pointer). X * Y
 *
 * @return IntVar the result of the * operator over two variables (one
 * reference, and one pointer)
 */
IntVar operator*(IntVar x, IntVarPtr y);
/**
 * @brief Overloading of the operator * between two variables (one pointer, and
 * one reference). X * Y
 *
 * @return IntVar the result of the * operator over two variables (one pointer,
 * and one reference)
 */
IntVar operator*(IntVarPtr x, IntVar y);

/**
 * @brief Overloading of the operator / between two references of variables.
 * X / Y
 *
 * @return IntVar the result of the / operator over two variables
 */
IntVar operator/(IntVar x, IntVar y);
/**
 * @brief Overloading of the operator / between one variable and one constant.
 * X / C
 *
 * @return IntVar the result of the / operator over one variable and one
 * constant
 */
IntVar operator/(IntVar x, int c);
/**
 * @brief Overloading of the operator / between one constant and one variable
 * C / X
 *
 * @return IntVar the result of the / operator over one constant and one
 * variable
 */
IntVar operator/(int c, IntVar x);
/**
 * @brief Overloading of the operator / between two variables (one reference,
 * and one pointer). X / Y
 *
 * @return IntVar the result of the / operator over two variables (one
 * reference, and one pointer)
 */
IntVar operator/(IntVar x, IntVarPtr y);
/**
 * @brief Overloading of the operator / between two variables (one pointer, and
 * one reference). X / Y
 *
 * @return IntVar the result of the / operator over two variables (one pointer,
 * and one reference)
 */
IntVar operator/(IntVarPtr x, IntVar y);

/**
 * @brief Overloading of the operator % between one variable (reference) with a
 * constant. X % N
 *
 * @return IntVar the result of the % operator over one variable (reference)
 * with a constant.
 */
IntVar operator%(IntVar x, int N);

/**
 * @brief Overloading of the operator == between a reference of variable and a
 * constant. X == C
 *
 * @return ConstraintPtr The equality constraint of x and c
 */
ConstraintPtr operator==(IntVar x, int c);
/**
 * @brief Overloading of the operator == between a constant and a reference of
 * variable. C == X
 *
 * @return ConstraintPtr The equality constraint of c and x
 */
ConstraintPtr operator==(int c, IntVar x);
/**
 * @brief Overloading of the operator == between two references of variables.
 * X == Y
 *
 * @return ConstraintPtr The equality constraint of x and y
 */
ConstraintPtr operator==(IntVar x, IntVar y);

/**
 * @brief Overloading of the operator != between a reference of variable and a
 * constant. X != C
 *
 * @return ConstraintPtr The disequality constraint of x and c
 */
ConstraintPtr operator!=(IntVar x, int c);
/**
 * @brief Overloading of the operator != between a constant and a reference of
 * variable. C != X
 *
 * @return ConstraintPtr The disequality constraint of c and x
 */
ConstraintPtr operator!=(int c, IntVar x);

/**
 * @brief Overloading of the operator <= between a reference of variable and a
 * constant. X <= C
 *
 * @return ConstraintPtr The inequality constraint x <= c
 */
ConstraintPtr operator<=(IntVar x, int c);
/**
 * @brief Overloading of the operator <= between a constant and a reference of
 * variable. C <= X
 *
 * @return ConstraintPtr The inequality constraint c <= x
 */
ConstraintPtr operator<=(int c, IntVar x);
/**
 * @brief Overloading of the operator <= between two references of variables.
 * X <= Y
 *
 * @return ConstraintPtr The inequality constraint x <= y
 */
ConstraintPtr operator<=(IntVar x, IntVar y);

/**
 * @brief Overloading of the operator >= between a reference of variable and a
 * constant. X >= C
 *
 * @return ConstraintPtr The inequality constraint x >= c
 */
ConstraintPtr operator>=(IntVar x, int c);
/**
 * @brief Overloading of the operator >= between a constant and a reference of
 * variable. C >= X
 *
 * @return ConstraintPtr The inequality constraint c >= x
 */
ConstraintPtr operator>=(int c, IntVar x);
/**
 * @brief Overloading of the operator >= between two references of variables.
 * X >= Y
 *
 * @return ConstraintPtr The inequality constraint x >= y
 */
ConstraintPtr operator>=(IntVar x, IntVar y);

/**
 * @brief Overloading of the operator < between a reference of variable and a
 * constant. X < C
 *
 * @return ConstraintPtr The inequality constraint x < c
 */
ConstraintPtr operator<(IntVar x, int c);
/**
 * @brief Overloading of the operator < between a constant and a reference of
 * variable. C < X
 *
 * @return ConstraintPtr The inequality constraint c < x
 */
ConstraintPtr operator<(int c, IntVar x);
/**
 * @brief Overloading of the operator < between two references of variables.
 * X < Y
 *
 * @return ConstraintPtr The inequality constraint x < y
 */
ConstraintPtr operator<(IntVar x, IntVar y);

/**
 * @brief Overloading of the operator > between a reference of variable and a
 * constant. X > C
 *
 * @return ConstraintPtr The inequality constraint x > c
 */
ConstraintPtr operator>(IntVar x, int c);
/**
 * @brief Overloading of the operator > between a constant and a reference of
 * variable. C > X
 *
 * @return ConstraintPtr The inequality constraint c > x
 */
ConstraintPtr operator>(int c, IntVar x);
/**
 * @brief Overloading of the operator > between two references of variables.
 * X > Y
 *
 * @return ConstraintPtr The inequality constraint x > y
 */
ConstraintPtr operator>(IntVar x, IntVar y);

// #######################################
// # Objective
// #######################################

/**
 * @brief The direction of the objective constraint.
 * If it is MAX it means that we are maximazing the objective.
 * Otherwise we are minimizing it.
 *
 */
enum class Direction { MIN, MAX };

/**
 * @brief Class that models an objective constraint.
 *
 */
class Objective : public Constraint {
 private:
  using INT_LIMIT = std::numeric_limits<int>;
  /**
   * @brief the variable subject to the optimization
   *
   */
  IntVarPtr obj_;
  /**
   * @brief The direction of the optimization
   *
   */
  Direction dir_;
  /**
   * @brief The current best value.
   *
   */
  int value_;
  /**
   * @brief A function to apply at each solution.
   * It is a function wrapper in order to deactivate it if the constraint is
   * deactivated.
   *
   */
  std::shared_ptr<FunctionWrapper> todo_;

 public:
  /**
   * @brief Construct a new objective constraint
   *
   * @param obj the variable subject to the optimization
   * @param dir the direction of the optimization
   */
  explicit Objective(IntVarPtr obj, Direction dir = Direction::MAX);

  /**
   * @brief Returns the current best value
   *
   * @return int
   */
  int Value();
  /**
   * @brief Returns the variable subject to the optimization
   *
   * @return IntVarPtr
   */
  IntVarPtr Subject();

  /**
   * @brief The post of the constraint
   *
   */
  STATUS Post() override;
  /**
   * @brief The propagation of the constraint
   *
   */
  STATUS Propagate() override;

  /**
   * @brief The pretty print of the constraint
   *
   */
  void Print() override;

  /**
   * @brief Method to deactivate the constraint when a reset is done by the
   * solver for instance.
   *
   */
  void Deactivate();
  /**
   * @brief Method to activate the constraint when a reset is done by the solver
   * for instance.
   *
   */
  void Activate();
};

// #######################################
// # Basic Constraints
// #######################################

/**
 * @brief Abstract constraint that automatically registers the constraint to the
 * main variable. It represents all constraint of the form x OP c
 */
class BinaryWithConst : public Constraint {
 protected:
  IntVarPtr x_;
  int c_;

 public:
  BinaryWithConst(IntVarPtr x, int c);
  BinaryWithConst(IntVar x, int c);

  STATUS Post() override;
};

/**
 * @brief Abstract binary constraint that automatically registers the constraint
 * to both variables. It represents all constraint of the form x OP y
 */
class BinaryConstraint : public Constraint {
 protected:
  IntVarPtr x_;
  IntVarPtr y_;

 public:
  BinaryConstraint(IntVarPtr x, IntVarPtr y);
  BinaryConstraint(IntVar x, IntVar y);

  STATUS Post() override;
};

/**
 * @brief Abstract ternary constraint that automatically register the constraint
 * to the three variables. It represents all constraint of the form x OP y OP z
 */
class TernaryConstraint : public Constraint {
 protected:
  IntVarPtr x_;
  IntVarPtr y_;
  IntVarPtr z_;

 public:
  TernaryConstraint(IntVarPtr z, IntVarPtr x, IntVarPtr y);
  TernaryConstraint(IntVar z, IntVar x, IntVar y);

  STATUS Post() override;
};

// #######################################
// # Comparator
// #######################################

/**
 * @brief The constraint Greater Than, i.e., x > y
 *
 */
class GThan : public BinaryConstraint {
 public:
  /**
   * @brief Construct a new Greater Than constraint using pointers
   * x > y
   *
   * @param x the variable of the left
   * @param y the variable of the right
   */
  GThan(IntVarPtr x, IntVarPtr y);
  /**
   * @brief Construct a new Greater Than constraint using reference
   * x > y
   * @param x
   * @param y
   */
  GThan(IntVar x, IntVar y);

  /**
   * @brief The method that filters the constraint
   *
   */
  STATUS Propagate() override;
  /**
   * @brief The pretty print of the method
   *
   */
  void Print() override;
};

/**
 * @brief The constraint Less Than, i.e., x < y
 *
 */
class LThan : public BinaryConstraint {
 public:
  /**
   * @brief Construct a new less than constraint using pointers
   * x < y
   *
   * @param x the left and side variable
   * @param y the right and side variable
   */
  LThan(IntVarPtr x, IntVarPtr y);
  /**
   * @brief Construct a new less than constraint using references
   * x < y
   *
   * @param x the left and side variable
   * @param y the right and side variable
   */
  LThan(IntVar x, IntVar y);

  /**
   * @brief The method that filters the constraint
   *
   */
  STATUS Propagate() override;
  /**
   * @brief The pretty print of the method
   *
   */
  void Print() override;
};

/**
 * @brief The constraint Greater or equal, i.e., x >= y
 *
 */
class GTEqual : public BinaryConstraint {
 public:
  /**
   * @brief Construct a new greater or equal constraint using pointers
   * x >= y
   *
   * @param x the variable of the left and side
   * @param y the variable of the right and side
   */
  GTEqual(IntVarPtr x, IntVarPtr y);
  /**
   * @brief Construct a new greater or equal constraint using pointers
   * x >= y
   *
   * @param x the variable of the left and side
   * @param y the variable of the right and side
   */
  GTEqual(IntVar x, IntVar y);

  /**
   * @brief The method that filters the constraint
   *
   */
  STATUS Propagate() override;
  /**
   * @brief The pretty print of the method
   *
   */
  void Print() override;
};

/**
 * @brief The constraint Less or equal, i.e., x < y
 *
 */
class LTEqual : public BinaryConstraint {
 public:
  /**
   * @brief Construct a new less or equal constraint using pointers
   * x <= y
   *
   * @param x the variable of the left and side
   * @param y the variable of the right and side
   */
  LTEqual(IntVarPtr x, IntVarPtr y);
  /**
   * @brief Construct a new less or equal constraint using references
   * x <= y
   *
   * @param x the variable of the left and side
   * @param y the variable of the right and side
   */
  LTEqual(IntVar x, IntVar y);

  /**
   * @brief The method that filters the constraint
   *
   */
  STATUS Propagate() override;
  /**
   * @brief The pretty print of the method
   *
   */
  void Print() override;
};

/**
 * @brief The constraint not equal to a constant, i.e., x != c
 *
 */
class NotEqualC : public BinaryWithConst {
 public:
  /**
   * @brief Construct a new not equal to constant constraint using pointer
   * x != c
   *
   * @param x the variable
   * @param c the constant
   */
  NotEqualC(IntVarPtr x, int c);
  /**
   * @brief Construct a new not equal to constant constraint using reference
   * x != c
   *
   * @param x the variable
   * @param c the constant
   */
  NotEqualC(IntVar x, int c);

  /**
   * @brief The method that filters the constraint
   *
   */
  STATUS Propagate() override;
  /**
   * @brief The pretty print of the method
   *
   */
  void Print() override;
};

/**
 * @brief The constraint equal to, i.e., x == y
 *
 */
class Equal : public BinaryConstraint {
 public:
  /**
   * @brief Construct a new equal constraint using pointers
   * x == y
   *
   * @param x the variable of the left
   * @param y the variable of the right
   */
  Equal(IntVarPtr x, IntVarPtr y);
  /**
   * @brief Construct a new equal constraint using references
   * x == y
   *
   * @param x the variable of the left
   * @param y the variable of the right
   */
  Equal(IntVar x, IntVar y);

  /**
   * @brief The method that filters the constraint
   *
   */
  STATUS Propagate() override;
  /**
   * @brief The pretty print of the method
   *
   */
  void Print() override;
};

// #######################################
// # Arithmetic
// #######################################

//  Following propagators use copies for interval
//  operations, if performances are not enough removing
//  those copies may improve them.

/**
 * @brief Constraint z = x + y using BC
 *
 */
class TernaryPlus : public TernaryConstraint {
 public:
  /**
   * @brief Construct a new z = x + y constraint using pointers
   *
   * @param z the result variable
   * @param x the first operand
   * @param y the second operand
   */
  TernaryPlus(IntVarPtr z, IntVarPtr x, IntVarPtr y)
      : TernaryConstraint(z, x, y){};
  /**
   * @brief Construct a new z = x + y constraint using references
   *
   * @param z the result variable
   * @param x the first operand
   * @param y the second operand
   */
  TernaryPlus(IntVar z, IntVar x, IntVar y) : TernaryConstraint(z, x, y){};

  /**
   * @brief The filtering method
   *
   */
  STATUS Propagate() override;
  /**
   * @brief The pretty print of the constraint
   *
   */
  void Print() override;
};

/**
 * @brief Constraint z = x - y using BC
 *
 */
class TernarySub : public TernaryConstraint {
 public:
  /**
   * @brief Construct a new z = x - y constraint using pointers
   *
   * @param z the result variable
   * @param x the first operand
   * @param y the second operand
   */
  TernarySub(IntVarPtr z, IntVarPtr x, IntVarPtr y)
      : TernaryConstraint(z, x, y){};
  /**
   * @brief Construct a new z = x - y constraint using references
   *
   * @param z the result variable
   * @param x the first operand
   * @param y the second operand
   */
  TernarySub(IntVar z, IntVar x, IntVar y) : TernaryConstraint(z, x, y){};

  /**
   * @brief The filtering method
   *
   */
  STATUS Propagate() override;
  /**
   * @brief The pretty print of the constraint
   *
   */
  void Print() override;
};

/**
 * @brief Constraint z = x * y using BC
 *
 */
class TernaryMult : public TernaryConstraint {
 public:
  /**
   * @brief Construct a new z = x * y constraint using pointers
   *
   * @param z the result variable
   * @param x the first operand
   * @param y the second operand
   */
  TernaryMult(IntVarPtr z, IntVarPtr x, IntVarPtr y)
      : TernaryConstraint(z, x, y){};

  /**
   * @brief Construct a new z = x * y constraint using references
   *
   * @param z the result variable
   * @param x the first operand
   * @param y the second operand
   */
  TernaryMult(IntVar z, IntVar x, IntVar y) : TernaryConstraint(z, x, y){};

  /**
   * @brief Returns the resulting variable
   *
   * @return IntVar
   */
  inline IntVar Result() { return *z_; };

  /**
   * @brief The filtering method
   *
   */
  STATUS Propagate() override;
  /**
   * @brief The pretty print of the constraint
   *
   */
  void Print() override;
};

/**
 * @brief Constraint z = x * y using BC with x and y strictly positive.
 *
 */
class TernaryPositiveMult : public TernaryConstraint {
 public:
  /**
   * @brief Construct a new z = x * y constraint using pointers
   *
   * @param z the result variable
   * @param x the first operand
   * @param y the second operand
   */
  TernaryPositiveMult(IntVarPtr z, IntVarPtr x, IntVarPtr y)
      : TernaryConstraint(z, x, y){};

  /**
   * @brief Construct a new z = x * y constraint using references
   *
   * @param z the result variable
   * @param x the first operand
   * @param y the second operand
   */
  TernaryPositiveMult(IntVar z, IntVar x, IntVar y) : TernaryConstraint(z, x, y){};

  /**
   * @brief Returns the resulting variable
   *
   * @return IntVar
   */
  inline IntVar Result() { return *z_; };

  /**
   * @brief The filtering method
   *
   */
  STATUS Propagate() override;
  /**
   * @brief The pretty print of the constraint
   *
   */
  void Print() override;
};

/**
 * @brief Constraint z = x / y using BC
 *
 */
class TernaryDiv : public TernaryConstraint {
 public:
  /**
   * @brief Construct a new z = x / y constraint using pointers
   *
   * @param z the result variable
   * @param x the first operand
   * @param y the second operand
   */
  TernaryDiv(IntVarPtr z, IntVarPtr x, IntVarPtr y)
      : TernaryConstraint(z, x, y){};
  /**
   * @brief Construct a new z = x / y constraint using references
   *
   * @param z the result variable
   * @param x the first operand
   * @param y the second operand
   */
  TernaryDiv(IntVar z, IntVar x, IntVar y) : TernaryConstraint(z, x, y){};

  /**
   * @brief Returns the resulting variable
   *
   * @return IntVar
   */
  inline IntVar Result() { return *z_; };

  /**
   * @brief The filtering method
   *
   */
  STATUS Propagate() override;
  /**
   * @brief The pretty print of the constraint
   *
   */
  void Print() override;
};

/**
 * @brief Constraint z = x / y using BC with x and y strictly positive.
 *
 */
class TernaryPositiveDiv : public TernaryConstraint {
 public:
  /**
   * @brief Construct a new z = x / y constraint using pointers
   *
   * @param z the result variable
   * @param x the first operand
   * @param y the second operand
   */
  TernaryPositiveDiv(IntVarPtr z, IntVarPtr x, IntVarPtr y)
      : TernaryConstraint(z, x, y){};
  /**
   * @brief Construct a new z = x / y constraint using references
   *
   * @param z the result variable
   * @param x the first operand
   * @param y the second operand
   */
  TernaryPositiveDiv(IntVar z, IntVar x, IntVar y) : TernaryConstraint(z, x, y){};

  /**
   * @brief Returns the resulting variable
   *
   * @return IntVar
   */
  inline IntVar Result() { return *z_; };

  /**
   * @brief The filtering method
   *
   */
  STATUS Propagate() override;
  /**
   * @brief The pretty print of the constraint
   *
   */
  void Print() override;
};

/**
 * @brief Constraint y = x % N using BC
 *
 */
class Modulo : public BinaryConstraint {
 private:
  /**
   * @brief The constant used in the modulo
   *
   */
  int N_;

 public:
  /**
   * @brief Construct a new y = x % N constraint using pointers
   *
   * @param y the result variable
   * @param x the main operand
   * @param N the value of the modulo
   */
  Modulo(IntVarPtr y, IntVarPtr x, int N);
  /**
   * @brief Construct a new y = x % N constraint using references
   *
   * @param y the result variable
   * @param x the main operand
   * @param N the value of the modulo
   */
  Modulo(IntVar y, IntVar x, int N);

  /**
   * @brief The filtering method
   *
   */
  STATUS Propagate() override;
  /**
   * @brief The pretty print of the constraint
   *
   */
  void Print() override;
};

// #######################################
// # Factory
// #######################################

/**
 * @brief The class factory is used as storage and provides an interface to
 * create variables and constraints
 *
 */
class Factory {
 private:
  /**
   * @brief The storage of variables
   *
   */
  std::vector<Variable> variables_;
  /**
   * @brief The storage of constraints
   *
   */
  std::vector<std::unique_ptr<Constraint>> constraints_;
  /**
   * @brief A map from int constant to variable constant
   * It helps retrieving the unique variable that represents a constant.
   *
   */
  std::unordered_map<int, IntVarPtr> constants_;

 public:
  /**
   * @brief Construct a new factory
   *
   */
  Factory() {
    constexpr int NB_VARS_RESERVED = 256;
    constexpr int NB_CTRS_RESERVED = 256;
    variables_.reserve(NB_VARS_RESERVED);
    constraints_.reserve(NB_CTRS_RESERVED);
  }
  Factory(const Factory&) = delete;
  Factory& operator=(const Factory&) = delete;

  /**
   * @brief A method to creates a constant variable.
   *
   * @param cp the solver
   * @param c the constant
   * @return IntVarPtr the variable that represents the constant
   */
  IntVarPtr MakeIntConstant(Solver* cp, int c) {
    auto rv = constants_.find(c);
    if (rv == constants_.end()) {
      variables_.emplace_back(cp, c);
      constants_[c] = &(variables_.back());
      return constants_[c];
    }
    return rv->second;
  }

  /**
   * @brief The method that creates an integer variable.
   *
   * @param cp the solver
   * @param name the name of the variable
   * @param lb the lower bound of the variable
   * @param ub the upper bound of the variable
   * @return IntVarPtr the resulting variable
   */
  IntVarPtr MakeIntVar(Solver* cp, std::string name, int lb, int ub) {
    if (lb == ub) {
      return MakeIntConstant(cp, lb);
    }
    variables_.emplace_back(cp, name, lb, ub);
    return &(variables_.back());
  }

  /**
   * @brief Method that creates an integer variable
   *
   * @param cp the solver
   * @param lb the lower bound of the variable
   * @param ub the upper bound of the variable
   * @return IntVarPtr the resulting variable
   */
  IntVarPtr MakeIntVar(Solver* cp, int lb, int ub) {
    return MakeIntVar(cp, "", lb, ub);
  }

  /**
   * @brief Method that creates an integer variable
   *
   * @param cp the solver
   * @param rng the domain of the variable
   * @return IntVarPtr the resulting variable
   */
  IntVarPtr MakeIntVar(Solver* cp, Range rng) {
    return MakeIntVar(cp, "", rng.GetLB(), rng.GetUB());
  }
  /**
   * @brief Method that creates an integer variable
   *
   * @param cp the solver
   * @return IntVarPtr the resulting variable
   */
  IntVarPtr MakeIntVar(Solver* cp) {
    return MakeIntVar(cp, std::numeric_limits<int>::min(),
                      std::numeric_limits<int>::max());
  }
  /**
   * @brief Method that creates a plus constraint
   * z = x + y
   * @param z the resulting variable
   * @param x the first operand
   * @param y the second operand
   * @return ConstraintPtr the plus constraint
   */
  ConstraintPtr MakePlus(IntVar z, IntVar x, IntVar y) {
    constraints_.emplace_back(std::make_unique<TernaryPlus>(z, x, y));
    return constraints_.back().get();
  }
  /**
   * @brief Method that creates a substraction constraint
   * z = x - y
   * @param z the resulting variable
   * @param x the first operand
   * @param y the second operand
   * @return ConstraintPtr the substraction constraint
   */
  ConstraintPtr MakeMinus(IntVar z, IntVar x, IntVar y) {
    constraints_.emplace_back(std::make_unique<TernarySub>(z, x, y));
    return constraints_.back().get();
  }
  /**
   * @brief Method that creates a multiplication constraint
   * z = x * y
   * @param z the resulting variable
   * @param x the first operand
   * @param y the second operand
   * @return ConstraintPtr the mult constraint
   */
  ConstraintPtr MakeMult(IntVar z, IntVar x, IntVar y) {
    if (x.GetLB() > 0 && y.GetLB()> 0) {
      constraints_.emplace_back(std::make_unique<TernaryPositiveMult>(z, x, y));
    } else {
      constraints_.emplace_back(std::make_unique<TernaryMult>(z, x, y));
    }
    return constraints_.back().get();
  }
  /**
   * @brief Method that creates a division constraint
   * z = x / y
   * @param z the resulting variable
   * @param x the numerator
   * @param y the denominator
   * @return ConstraintPtr the division constraint
   */
  ConstraintPtr MakeDiv(IntVar z, IntVar x, IntVar y) {
    if (x.GetLB() > 0 && y.GetLB() > 0) {
      constraints_.emplace_back(std::make_unique<TernaryPositiveDiv>(z, x, y));
    } else {
      constraints_.emplace_back(std::make_unique<TernaryDiv>(z, x, y));
    }
    return constraints_.back().get();
  }
  /**
   * @brief Method that creates a modulo constraint
   * z = x % N
   * @param z the resulting variable
   * @param y the main operand
   * @param N the value of the modulo
   * @return ConstraintPtr the Modulo constraint
   */
  ConstraintPtr MakeModulo(IntVar z, IntVar x, int N) {
    constraints_.emplace_back(std::make_unique<Modulo>(z, x, N));
    return constraints_.back().get();
  }
  /**
   * @brief Method that creates an inequality constraint
   * x < y
   * @param x the left and side variable
   * @param y the right and side variable
   * @return ConstraintPtr the inequality < constraint
   */
  ConstraintPtr MakeLessThan(IntVar x, IntVar y) {
    constraints_.emplace_back(std::make_unique<LThan>(x, y));
    return constraints_.back().get();
  }

  /**
   * @brief Method that creates an inequality constraint
   * x > y
   * @param x the left and side variable
   * @param y the right and side variable
   * @return ConstraintPtr the inequality > constraint
   */
  ConstraintPtr MakeGreaterThan(IntVar x, IntVar y) {
    constraints_.emplace_back(std::make_unique<GThan>(x, y));
    return constraints_.back().get();
  }

  /**
   * @brief Method that creates an inequality constraint
   * x <= y
   * @param x the left and side variable
   * @param y the right and side variable
   * @return ConstraintPtr the inequality <= constraint
   */
  ConstraintPtr MakeLessThanEqual(IntVar x, IntVar y) {
    constraints_.emplace_back(std::make_unique<LTEqual>(x, y));
    return constraints_.back().get();
  }

  /**
   * @brief Method that creates an inequality constraint
   * x >= y
   * @param x the left and side variable
   * @param y the right and side variable
   * @return ConstraintPtr the inequality >= constraint
   */
  ConstraintPtr MakeGreaterThanEqual(IntVar x, IntVar y) {
    constraints_.emplace_back(std::make_unique<GTEqual>(x, y));
    return constraints_.back().get();
  }

  /**
   * @brief Method that creates an equality constraint
   * x == y
   * @param x the left and side variable
   * @param y the right and side variable
   * @return ConstraintPtr the equality == constraint
   */
  ConstraintPtr MakeEqual(IntVar x, IntVar y) {
    constraints_.emplace_back(std::make_unique<Equal>(x, y));
    return constraints_.back().get();
  }

  /**
   * @brief Method that creates a disequality constraint
   * x != c
   * @param x the variable
   * @param c the constant
   * @return ConstraintPtr the disequality 1= constraint
   */
  ConstraintPtr MakeNotEqual(IntVar x, int c) {
    constraints_.emplace_back(std::make_unique<NotEqualC>(x, c));
    return constraints_.back().get();
  }
};

// #######################################
// # SOLVER
// #######################################

/**
 * @brief This class represents the solver. As most solvers,
 * it provides methods to solve, optimize, add constraint,
 * save, restore...
 * This solver tries to be as simple as possible. For instance,
 * the trail is just a vector of intervals. Consequently, the save
 * process save a copy of interval of each variable.
 * And obviously a restore will restore the last interval for each
 * variable.
 * Also, this solver has method to explicitly propagate all constraint
 * until it reach it's fixpoint.
 * For the search it relies on a search componenent witch return all
 * alternatives which are pushed on the trail during a save (see the solve
 * procedure to get more informations).
 *
 */
class Solver {
 private:
  using Solution = std::vector<int>;
  using State = std::vector<Range>;
  /**
   * @brief the vector of constraints of the problem
   * they are used to compute the fixpoint
   *
   */
  std::vector<ConstraintPtr> constraints_;
  /**
   * @brief the vector of variable of the problem.
   * They are used mainly for to save the state of the solver.
   *
   */
  std::vector<IntVarPtr> variables_;
  /**
   * @brief The trail.
   * It's just a vector of state.
   * A state is an alias for a vector of intervals.
   * Each time a save is performed, all intervals
   * of each variables are stored in a vector which match to
   * a state. This state is pushed back into the trail.
   *
   * Obviously a restore will set interval of each variable
   * according to the last state.
   *
   */
  std::vector<State> trail_;
  /**
   * @brief The initial state.
   * A state is an alias for a vector of intervals.
   * The initial state is a snapshot of the initial
   * domain of each variable. Dedicated methods are
   * define to save and restore it.
   *
   */
  State starting_point_;
  /**
   * @brief The queue of propagation.
   * This queue is used to propagated variable until fixpoint.
   *
   */
  std::queue<ConstraintPtr> queue_;
  /**
   * @brief A vector of solution.
   * A solution is just a ordered vector (the same order as the variable
   * vector is used) of each value for each variable.
   *
   */
  std::vector<Solution> solutions_;
  /**
   * @brief Objective is an objective constraint.
   * It can be nullptr if it is a satisfaction problem.
   * If it is an optimization problem it set as a Objective
   * constraint.
   *
   */
  Objective* objective_;
  /**
   * @brief The search used to explore the space of solution.
   * It is used to get all alternatives of splitting.
   * A save is performed after each alternative.
   *
   */
  Search* search_;
  /**
   * @brief A vector of function Wrapper.
   * A function wrapper is a wrapper over a function that
   * extends function capabilities as enabling, disabling
   * function.
   *
   */
  std::vector<std::shared_ptr<FunctionWrapper>> todo_;
  /**
   * @brief A boolean value that identifies if an init was performed or not.
   * At initialization a set of operation is done as saving initial domains.
   * @see Solver::solve for more details.
   *
   */
  bool init_;
  /**
   * @brief A storage of variables, and constraints. It is mainly used to store
   * new variables and constraint when relying on operator overloading.
   *
   */
  Factory storage_;

  /**
   * @brief Main solving process that propagates constraints and perform the
   * search until a solution is found, or that the problem is proved unsat.
   *
   * @return true a solution is found
   * @return false no solution
   */
  bool SolveImpl();
  /**
   * @brief flush the queue.
   * I.e., it removes every constraint in the queue.
   * The queue is empty after this operation.
   *
   */
  void Flush();

  /**
   * @brief Post the constraint c. Please note that Solver::Post and Solver::Add
   * does not do the same thing. The add, push the constraint in the solver
   * whereas, the post actually filter values of its scope, and register the
   * constraint as an observer of each variable of its scope.
   *
   * @param c the constraint that is posted
   * @return true the post succeed
   * @return false the post failed
   */
  bool Post(ConstraintPtr c);
  /**
   * @brief compute the fixpoint, i.e., all constraint of the queue are
   * propagate until the queue become empty.
   *
   * @return true the fix point is reached
   * @return false a fail has occurred, so no fixpoint can be reached
   */
  bool FixPoint();
  /**
   * @brief Set the current state as initial domains.
   *
   */
  void SaveInitialDomains();
  /**
   * @brief Save the current state. At the moment, a copy of all range are
   * performed.
   *
   */
  void Save();
  /**
   * @brief Restore a particular state
   *
   * @param checkpoint the state we want to restore
   */
  void Restore(const State& checkpoint);
  /**
   * @brief Restore the last saved state if there is one, otherwise an assert
   * will fail.
   * @see Solver::Save
   *
   */
  void Restore();

  /**
   * @brief Restore the domain as it was at the start of the solving. A call to
   * Solver::SaveInitialDomains is needed to actually solve a state. If no save
   * is done, nothing will be done. The save is done by default when calling
   * Solver::Solve for the first time.
   */
  inline void RestoreInitialDomains() { Restore(starting_point_); };

 public:
  /**
   * @brief Constructor of a solver
   *
   */
  Solver();
  /**
   * @brief Destroy the Solver object
   *
   */
  ~Solver();

  /**
   * @brief Returns the storage
   *
   * @return Factory& the storage factory
   */
  inline Factory& GetStorage() { return storage_; };

  /**
   * @brief Used to register a variable to the solver.
   * It is done automatically by the variable by default (in its constructor)
   * After this operation, the variables vector is extended by the variable @p v
   *
   * @param v the variable to register
   */
  void RegisterVar(IntVarPtr v);

  /**
   * @brief Method to add a constraint to the solver
   * After this operation, the constraints vector is extended by the variable @p
   * c
   *
   * @param c the constraint to add.
   */
  inline void Add(ConstraintPtr c) { constraints_.push_back(c); };

  /**
   * @brief Method to schedule a constraint (i.e., it pushes the constraint into
   * the queue if it is not already in).s
   *
   * @param c the constraint to schedule.
   */
  void Schedule(ConstraintPtr c);
  /**
   * @brief computes the fixpoint by propagating all constraints that are in the
   * queue. If @p enforce is true, all constraint are propagated once before to
   * propagate the queue.
   *
   * @param enforce if true, all constraint are propagated once before to
   * propagate the constraint of the queue.
   * @return true a fixpoint is reached
   * @return false a fail has occurred
   */
  bool FixPoint(bool enforce);
  /**
   * @brief Reset the solver, if @p save is true, it perform a save after the
   * reset, no save is performed otherwise. For instance if you want to redo
   * twice the same solving, a save should be performed. If you want to do
   * something based on the first solve without creating 2 solvers (as for the
   * method Solver::Optimize), you should not save.
   *
   * @param save to specify if a save of the domain should be performed after
   * the reset or not.
   */
  void Reset(bool save = false);

  /**
   * @brief Print all constraint of the model
   *
   */
  void Print();
  /**
   * @brief Print all variables
   *
   */
  void PrintVariables();
  /**
   * @brief Print all variables and their current value assuming we are
   * currently on a solution.
   *
   */
  void PrintSolution();

  /**
   * @brief To add a new method that will be performed at each solution based on
   * the lambda
   *
   * @param l the process we want to do at each solution
   * @return std::shared_ptr<FunctionWrapper>
   */
  std::shared_ptr<FunctionWrapper> DoOnSolution(std::function<void(void)> l);
  /**
   * @brief Performs everything that is needed on solution. More precisely it
   * save the solution and run all lambdas added using Solver::DoOnSolution
   *
   */
  void OnSolution();

  /**
   * @brief Save the state has a solution
   *
   */
  void SaveSolution();

  /**
   * @brief Returns the set of saved solution
   *
   * @return std::vector<Solution>
   */
  std::vector<Solution> Solutions() { return solutions_; };

  /**
   * @brief Returns the current number of solutions found
   *
   * @return int the number of solutions
   */
  inline int NbSol() { return solutions_.size(); };

  /**
   * @brief Returns the current depth of the solver (i.e., the size of the
   * trail)
   *
   * @return int the current depth
   */
  inline int Depth() const noexcept { return trail_.size(); };

  /**
   * @brief Solve until it find the next solution or if it prove that there is
   * none. Note that it uses a DFS search but is should be easy to change it to
   * BFS.
   *
   * @return true if the next solution is found
   * @return false no solution is found
   */
  bool Solve();
  /**
   * @brief if no objective is defined it behaves as Solver::Solve().
   * If an objective is defined, this method computes the best objective. If @p
   * all is false then it stops. Otherwise a reset of the solver is done, the
   * objective constraint deactivated, and the objective value set to the best
   * we found. Then we computes all solution if @p all is true.
   * @p todo specify a lambda that is ran at each solution (useful to print
   * solution for instance). Also if you just need one solution you should
   * prefer Solver::Solve instead.
   *
   * @param todo a lambda that is ran at each solution
   * @param all if true all best solutions are computed, otherwise it returns
   * the best objective.
   * @return int the value of the best objective.
   */
  int Optimize(
      std::function<void(void)> todo = [] {}, bool all = false);

  /**
   * @brief if no objective is defined it behaves as Solver::Solve().
   * If an objective is defined, this method computes the best objective. If @p
   * all is false then it stops. Otherwise a reset of the solver is done, the
   * objective constraint deactivated, and the objective value set to the best
   * we found. Then we computes all solution if @p all is true.  Also if you
   * just need one solution you should prefer Solver::Solve instead.
   *
   * @param all if true all best solutions are computed, otherwise it returns
   * the best objective.
   * @return int the value of best objective.
   */
  int Optimize(bool all = false) {
    return Optimize([] {}, all);
  };

  /**
   * @brief Creates and adds an Objective constraint with min as direction.
   * To get a solution Solver::Solve, or Solver::Optimize still need to be
   * called.
   *
   * @param obj
   */
  void Minimize(Variable& obj);

  /**
   * @brief Creates and adds an Objective constraint with max as direction.
   * To get a solution Solver::Solve, or Solver::Optimize still need to be
   * called.
   *
   * @param obj
   */
  void Maximize(Variable& obj);

  /**
   * @brief Returns the objective variable
   *
   * @return IntVarPtr the objective variable
   */
  IntVarPtr GetObjective() const;
};

/**
 * @brief Class to monitor the time.
 */
class TimeMonitor {
 public:
  using Time = std::chrono::time_point<std::chrono::steady_clock>;
  /**
   * @brief The start time. It is set either when creating the object, or when
   * calling TimeMonitor::CheckPoint
   *
   */
  Time start_;
  /**
   * @brief Construct a new time monitor
   *
   */
  TimeMonitor() : start_(std::chrono::steady_clock::now()) {}

  /**
   * @brief Returns the current time
   *
   * @return Time
   */
  Time Now() { return std::chrono::steady_clock::now(); }

  /**
   * @brief Set the starting point to the current time
   *
   */
  void CheckPoint() { start_ = Now(); }

  /**
   * @brief Computes the elapsed duration between the start time and now.
   * Note that the start is not changed.
   *
   * @return int
   */
  int ElapsedDuration() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Now() - start_)
        .count();
  }
};
}  // namespace LightCP
#endif /* LIGHTCP_HPP */
