/**
 * @file trace.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 */

#ifndef AOE_TRACE_H
#define AOE_TRACE_H
#include <string>

namespace Aoe {
class Trace {
public:
    explicit Trace(const std::string &name)
        : start_(true), stop_(false), name_(name)
    {
        Start();
    }
    explicit Trace(const std::string &name, const std::string &pidTag, const std::string &tidTag)
        : start_(true), stop_(false), name_(name), pidTag_(pidTag), tidTag_(tidTag)
    {
        Start();
    }
    explicit Trace(const std::string &name, bool start) : start_(start), stop_(false), name_(name)
    {
        Start();
    }
    ~Trace()
    {
        Stop();
    }
    static void StartDumpTraceToPath(const std::string &recordPath);
    void Start();
    void Stop();
private:
    bool start_;
    bool stop_;
    std::string name_;
    std::string pidTag_;
    std::string tidTag_;
};
}
#endif
