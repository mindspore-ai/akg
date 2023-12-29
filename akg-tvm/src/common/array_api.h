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

#ifndef COMMON_ARRAY_API_H_
#define COMMON_ARRAY_API_H_

#include <tvm.h>
#include <tvm/ir.h>
#include "pass/utils.h"

namespace akg {
inline size_t ConvertNegativeIdxToPositive(size_t size, int index) {
  size_t idx = 0;
  int64_t ind = index;
  if (index < 0) {
    idx = size + ind;
  } else {
    idx = ind;
  }

  return idx;
}

/// Get index of item in array
/// \tparam T
/// \param array
/// \param elem
/// \param index
/// \return
template <typename T>
bool GetIndexOfElement(const Array<T> &array, const T &elem, size_t &index) {
  for (size_t i = 0; i < array.size(); ++i) {
    const auto item = array[i];
    if (Equal(elem, item)) {
      index = i;
      return true;
    }
  }

  return false;
}

/// Check is item in array
/// \tparam T
/// \param array
/// \param elem
/// \return
template <typename T>
bool IsInArray(const Array<T> &array, const T &elem) {
  for (size_t i = 0; i < array.size(); ++i) {
    const auto item = array[i];
    if (Equal(elem, item)) {
      return true;
    }
  }

  return false;
}

/// Get item at index, index could be negative, in that case, it will count from end
/// \tparam T
/// \param array
/// \param index
/// \return
template <typename T>
T GetItem(const Array<T> &array, int index) {
  CHECK(!array.empty()) << "array is empty!";

  size_t idx = ConvertNegativeIdxToPositive(array.size(), index);
  if (idx >= array.size()) {
    LOG(FATAL) << "idx " << idx << " is invalid!";
  }

  return array[idx];
}

/// Get item at index, index must >= 0
/// \tparam T
/// \param array
/// \param index
/// \return
template <typename T>
T GetItem(const Array<T> &array, size_t index) {
  CHECK(!array.empty()) << "array is empty!";

  if (index >= array.size()) {
    LOG(FATAL) << "idx " << index << " is invalid!";
  }

  return array[index];
}

/// Set item to index, index could be negative, in that case, it will count from end
/// \tparam T
/// \param array
/// \param index
/// \param item
template <typename T>
void SetItem(Array<T> &array, int index, const T &item) {
  CHECK(!array.empty()) << "array is empty!";

  size_t idx = ConvertNegativeIdxToPositive(array.size(), index);
  if (idx >= array.size()) {
    LOG(FATAL) << "idx " << idx << " is invalid!";
  }

  array.Set(idx, item);
}

/// Get sub-array of array started from begin
/// \tparam T
/// \param array
/// \param begin
/// \param length
/// \return
template <typename T>
Array<T> GetRange(const Array<T> &array, int begin, size_t length) {
  Array<T> result;
  size_t idx = ConvertNegativeIdxToPositive(array.size(), begin);
  if (length + idx > array.size()) {
    LOG(FATAL) << "begin index error";
  }

  for (size_t i = idx; i < idx + length; ++i) {
    result.push_back(array[i]);
  }

  return result;
}

/// Remove item at index, index could be negative, in that case, it will count form end
/// \tparam T
/// \param array
/// \param index
/// \return
template <typename T>
Array<T> RemoveItemAtIndex(const Array<T> &array, int index) {
  Array<T> result;
  size_t idx = ConvertNegativeIdxToPositive(array.size(), index);
  CHECK(idx < array.size()) << "Remove index error: " << idx << " while array size is " << array.size();

  for (size_t i = 0; i < array.size(); ++i) {
    if (i == idx) {
      continue;
    }
    result.push_back(array[i]);
  }

  return result;
}

/// Remove item at index
/// \tparam T
/// \param array
/// \param index
/// \return
template <typename T>
Array<T> RemoveItemAtIndex(const Array<T> &array, size_t index) {
  Array<T> result;
  CHECK(index < array.size()) << "Remove index error: " << index << " while array size is " << array.size();

  for (size_t i = 0; i < array.size(); ++i) {
    if (i == index) {
      continue;
    }
    result.push_back(array[i]);
  }

  return result;
}

/// Is item in array a equal to item in array b at same index
/// \tparam T
/// \param a
/// \param b
/// \param index
/// \param compareValue
/// \return
template <typename T>
bool IsTwoItemEqual(const Array<T> &a, const Array<T> &b, int index, bool compareValue = false) {
  if (a.empty() || b.empty()) {
    return false;
  }

  size_t idxA = ConvertNegativeIdxToPositive(a.size(), index);
  size_t idxB = ConvertNegativeIdxToPositive(b.size(), index);

  if (idxA >= a.size()) {
    LOG(FATAL) << "idxA " << idxA << " is invalid.";
  }
  if (idxB >= b.size()) {
    LOG(FATAL) << "idxB " << idxB << " is invalid.";
  }

  if (compareValue) {
    return ir::GetIntConst(a[idxA]) == ir::GetIntConst(b[idxB]);
  }

  return Equal(a[idxA], b[idxB]);
}

/// intersect two array
/// \tparam T
/// \param a array a
/// \param b array b
/// \return intersect result
template <typename T>
Array<T> IntersectionArray(const Array<T> &a, const Array<T> &b) {
  Array<T> result;
  for (size_t i = 0; i < a.size(); ++i) {
    for (size_t j = 0; j < b.size(); ++j) {
      if (Equal(a[i], b[j])) {
        result.push_back(b[j]);
      }
    }
  }

  return result;
}

/// Reverse array
/// \tparam T
/// \param array
/// \return
template <typename T>
Array<T> Reverse(const Array<T> &array) {
  Array<T> result;
  if (array.size() == 0) {
    return result;
  }

  for (int i = static_cast<int>(array.size()) - 1; i >= 0; --i) {
    result.push_back(array[i]);
  }

  return result;
}

/// Insert item to array
/// \tparam T
/// \param array
/// \param index
/// \param item
template <typename T>
void Insert(Array<T> &array, size_t index, const T &item) {
  CHECK(index <= array.size());

  array.push_back(item);
  for (size_t i = array.size() - 1; i > index; i--) {
    array.Set(i, array[i - 1]);
  }
  array.Set(index, item);
}

/// Merge two array to one
/// \tparam T
/// \param a
/// \param b
/// \return
template <typename T>
Array<T> MergeTwo(const Array<T> &a, const Array<T> &b) {
  Array<T> result;
  for (auto item : a) {
    result.push_back(item);
  }
  for (auto item : b) {
    result.push_back(item);
  }

  return result;
}

/// Check is two arrays same
/// \tparam T
/// \param a
/// \param b
/// \param elementWise
/// \return
template <typename T>
bool IsSame(const Array<T> &a, const Array<T> &b, bool elementWise = true) {
  if (a.size() != b.size()) {
    return false;
  }

  if (elementWise) {
    for (size_t i = 0; i < a.size(); ++i) {
      if (!Equal(a[i], b[i])) {
        return false;
      }
    }
  } else {
    for (auto i : b) {
      if (!IsInArray(a, i)) {
        return false;
      }
    }
  }

  return true;
}
}  // namespace akg
#endif  // COMMON_ARRAY_API_H_
