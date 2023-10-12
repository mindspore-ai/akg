/**
 * @file aoe_errcodes.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.\n
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n
 *
 */
#ifndef AOE_EXTERNAL_AOE_ERRCODES_H
#define AOE_EXTERNAL_AOE_ERRCODES_H

#include <cstdint>

namespace Aoe {
using AoeStatus = int32_t;

constexpr AoeStatus AOE_FAILURE                                      = -1;   // 通用AOE错误
constexpr AoeStatus AOE_SUCCESS                                      = 0;    // 成功
constexpr AoeStatus AOE_ERROR_MEMORY_ALLOC                           = 1;    // 内存分配失败
constexpr AoeStatus AOE_ERROR_INVALID_PARAM                          = 2;    // 非法参数
constexpr AoeStatus AOE_ERROR_UNINITIALIZED                          = 3;    // API未初始化
constexpr AoeStatus AOE_ERROR_REPEAT_INITIALIZE                      = 4;    // API重复初始化
constexpr AoeStatus AOE_ERROR_INVALID_SESSION                        = 5;    // 非法的sessionID
constexpr AoeStatus AOE_ERROR_BUSY                                   = 6;    // 多线程调用阻塞
constexpr AoeStatus AOE_ERROR_INVALID_GRAPH                          = 7;    // 非法的调优图
constexpr AoeStatus AOE_ERROR_NON_OPTIMIZE_GRAPH                     = 8;    // 无调优图
constexpr AoeStatus AOE_ERROR_DYNAMIC_GRAPH                          = 9;    // 调优业务不支持动态图
constexpr AoeStatus AOE_ERROR_DYNAMIC_SHAPE_RANGE                    = 10;   // 动态shape（opat）
constexpr AoeStatus AOE_ERROR_INVALID_OPTION_ATTR                    = 11;   // option属性错误
constexpr AoeStatus QTEST_LIB_OPEN_FAILED                            = 12;   // 打开lib失败
constexpr AoeStatus QTEST_LIB_INVALID_FUNC                           = 13;   // 无效函数

constexpr AoeStatus AOE_ERROR_TUNING_ERROR                           = 100;  // 其他调优错误
} // namespace Aoe
#endif // AOE_EXTERNAL_AOE_ERRCODES_H