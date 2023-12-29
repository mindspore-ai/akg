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
#ifndef POLY_ISL_H_
#define POLY_ISL_H_

#include "isl/cpp.h"
namespace isl {
struct IslIdIslHash {
  size_t operator()(const isl::id &id) const { return isl_id_get_hash(id.get()); }
};

template <typename E, typename L>
struct ListIter {
  ListIter(L &list, int pos) : list_(list), pos_(pos) {}
  bool operator!=(const ListIter &other) { return pos_ != other.pos_; }
  ListIter &operator++() {
    pos_++;
    return *this;
  }
  E operator*() { return list_.get_at(pos_); }

 private:
  L &list_;
  int pos_;
};

template <typename L>
auto begin(L &list) -> ListIter<decltype(list.get_at(0)), L> {
  return ListIter<decltype(list.get_at(0)), L>(list, 0);
}

template <typename L>
auto end(L &list) -> ListIter<decltype(list.get_at(0)), L> {
  return ListIter<decltype(list.get_at(0)), L>(list, list.size());
}

template <typename T>
inline T operator+(const T a, const T b) {
  return a.add(b);
}

template <typename T>
inline T operator-(const T a, const T b) {
  return a.sub(b);
}

// isl::val
inline isl::val operator+(const isl::val &v, int64_t i) { return v.add(isl::val(v.ctx(), i)); }
inline isl::val operator+(int64_t i, const isl::val &v) { return v + i; }
inline isl::val operator-(const isl::val &v, int64_t i) { return v.sub(isl::val(v.ctx(), i)); }
inline isl::val operator-(int64_t i, const isl::val &v) { return isl::val(v.ctx(), i).sub(v); }
inline isl::val operator*(const isl::val &l, const isl::val &r) { return l.mul(r); }
inline isl::val operator*(const isl::val &v, int64_t i) { return v.mul(isl::val(v.ctx(), i)); }
inline isl::val operator*(int64_t i, const isl::val &v) { return v * i; }
inline bool operator==(const isl::val &v, int64_t i) { return v.eq(isl::val(v.ctx(), i)); }
inline bool operator==(int64_t i, const isl::val &v) { return v == i; }
inline bool operator==(const isl::val &v1, const isl::val &v2) { return v1.eq(v2); }
inline bool operator!=(const isl::val &v, int64_t i) { return !(v == i); }
inline bool operator!=(int64_t i, const isl::val &v) { return !(v == i); }
inline bool operator!=(const isl::val &v1, const isl::val &v2) { return !(v1 == v2); }
inline bool operator<(const isl::val &l, const isl::val &r) { return l.lt(r); }
inline bool operator<=(const isl::val &l, const isl::val &r) { return l.le(r); }
inline bool operator>(const isl::val &l, const isl::val &r) { return l.gt(r); }
inline bool operator>=(const isl::val &l, const isl::val &r) { return l.ge(r); }

// isl::aff
inline isl::aff operator+(const isl::aff &a, const isl::val &v) {
  return a.add(isl::aff(isl::local_space(a.get_space().domain()), v));
}
inline isl::aff operator+(int i, const isl::aff &a) { return a + isl::val(a.ctx(), i); }
inline isl::aff operator+(const isl::val &v, const isl::aff &a) { return a + v; }
inline isl::aff operator+(const isl::aff &a, int i) { return i + a; }
inline isl::aff operator-(const isl::aff &a, int i) { return a + (-i); }
inline isl::aff operator-(int i, const isl::aff &a) { return (a + (-i)).neg(); }
inline isl::aff operator*(const isl::val &v, const isl::aff &a) { return a.scale(v); }
inline isl::aff operator*(const isl::aff &a, const isl::val &v) { return v * a; }
inline isl::aff operator*(int i, const isl::aff &a) { return a * (isl::val(a.ctx(), i)); }
inline isl::aff operator*(const isl::aff &a, int i) { return i * a; }
inline isl::aff operator/(const isl::aff &a, int i) { return a.scale_down(isl::val(a.ctx(), i)); }

inline isl::set operator>=(const isl::aff &a, const isl::val &v) {
  auto b = isl::aff(isl::local_space(a.get_space().domain()), v);
  return a.ge_set(b);
}

inline isl::set operator>=(const isl::aff &a, int i) {
  auto ctx = a.ctx();
  return a >= isl::val(ctx, i);
}
inline isl::set operator>(const isl::aff &a, int i) { return a >= (i + 1); }

inline isl::set operator<=(const isl::aff &a, const isl::val &v) { return a.neg() >= v.neg(); }
inline isl::set operator<(const isl::aff &a, const isl::val &v) { return a <= v - 1; }
inline isl::set operator<=(const isl::aff &a, int i) { return a.neg() >= -i; }
inline isl::set operator<(const isl::aff &a, int i) { return a <= i - 1; }
inline isl::set operator&(const isl::set &s1, const isl::set &s2) { return s1.intersect(s2); }

inline isl::map operator>(const isl::aff &a, const isl::aff &b) {
  auto pw_a = isl::pw_aff(a);
  auto pw_b = isl::pw_aff(b);
  return pw_a.gt_map(pw_b);
}
inline isl::map operator<(const isl::aff &a, const isl::aff &b) {
  auto pw_a = isl::pw_aff(a);
  auto pw_b = isl::pw_aff(b);
  return pw_a.lt_map(pw_b);
}
inline isl::map operator<=(const isl::aff &a, const isl::aff &b) { return a < b + 1; }
inline isl::map operator&(const isl::map &m1, const isl::map &m2) { return m1.intersect(m2); }
inline bool operator==(const isl::id &id1, const isl::id &id2) { return id1.get() == id2.get(); }
inline bool operator!=(const isl::id &id1, const isl::id &id2) { return id1.get() != id2.get(); }
}  // namespace isl

#endif  // POLY_ISL_H_
