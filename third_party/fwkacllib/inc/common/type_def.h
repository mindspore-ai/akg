/**
* Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*
* Description:interface.
* Create: 2021-12-21
*/
#ifndef AICPU_TYPE_DEF_H
#define AICPU_TYPE_DEF_H

#include <cstdint>
#include <cstddef>
#ifndef char_t
typedef char char_t;
#endif

#ifndef float32_t
typedef float float32_t;
#endif

#ifndef float64_t
typedef double float64_t;
#endif

inline uint64_t PtrToValue(const void *ptr)
{
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
}

inline void *ValueToPtr(const uint64_t value)
{
    return reinterpret_cast<void *>(static_cast<uintptr_t>(value));
}

template<typename TI, typename TO>
inline TO *PtrToPtr(TI *ptr)
{
    return reinterpret_cast<TO *>(ptr);
}

template<typename T>
inline T *PtrAdd(T * const ptr, const size_t maxIdx, const size_t idx)
{
    if ((ptr != nullptr) && (idx < maxIdx)) {
        return reinterpret_cast<T *>(ptr + idx);
    }
    return nullptr;
}
#endif  // AICPU_TYPE_DEF_H
