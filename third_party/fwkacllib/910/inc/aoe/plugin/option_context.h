/**
 * @file option_context.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef AOE_PLUGIN_AOE_OPTION_CONTEXT_H
#define AOE_PLUGIN_AOE_OPTION_CONTEXT_H

#include <string>
#include <unordered_map>
#include <set>
#include "any_option.h"

namespace Aoe {
class OptionContext {
public:
    OptionContext() = default;

    ~OptionContext() = default;

    AnyOption &operator[](const std::string &name)
    {
        return options_[name];
    }

    bool Has(const std::string &name) const
    {
        return options_.find(name) != options_.cend();
    }

    void Delete(const std::string &name)
    {
        (void) options_.erase(name);
    }

    size_t Size() const
    {
        return options_.size();
    }

    std::set<std::string> AllNames() const;

    template <typename T> T TryGet(const std::string &name, const T defaultValue) const;

    template <typename T> bool Get(const std::string &name, T &retValue) const;

    template <typename T> bool Set(const std::string &name, const T &value);

private:
    std::unordered_map<std::string, AnyOption> options_;
};

template <typename T> T OptionContext::TryGet(const std::string &name, const T defaultValue) const
{
    T retValue;
    bool got = Get<T>(name, retValue);
    return got ? retValue : defaultValue;
}

template <typename T> bool OptionContext::Get(const std::string &name, T &retValue) const
{
    const auto it = options_.find(name);
    if (it == options_.cend()) {
        return false;
    }

    auto opt = it->second;
    T *tp = opt.Get<T>();
    if (tp == nullptr) {
        return false;
    }
    retValue = *tp;
    return true;
}

template <typename T> bool OptionContext::Set(const std::string &name, const T &value)
{
    const auto it = options_.find(name);
    if (it != options_.cend()) {
        return false;
    }

    options_[name] = AnyOption(value);
    return true;
}
} // namespace Aoe
#endif // AOE_PLUGIN_AOE_OPTION_CONTEXT_H