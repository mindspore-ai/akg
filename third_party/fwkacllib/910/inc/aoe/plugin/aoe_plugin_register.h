/**
 * @file aoe_plugin_register.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef AOE_PLUGIN_AOE_PLUGIN_REGISTER_H
#define AOE_PLUGIN_AOE_PLUGIN_REGISTER_H

#include <functional>
#include <string>

namespace Aoe {
class AoePlugin;
using PluginCreator = std::function<AoePlugin* ()>;

class AoePluginRegister {
public:
    explicit AoePluginRegister(const std::string &jobType, const std::string &pluginName, const PluginCreator &creator);

    ~AoePluginRegister() = default;

    template <typename P> static AoePlugin* DefaultCreator()
    {
        static_assert(std::is_base_of<AoePlugin, P>::value, "Plugin type must derived from Aoe::AoePlugin");
        return new (std::nothrow) P;
    }
};
} // namespace Aoe

#define CONCAT_STR_IMPL(s1, s2) s1##s2
#define CONCAT_STR(s1, s2) CONCAT_STR_IMPL(s1, s2)

#ifdef __COUNTER__
#define PLUGIN_ANONYMOUS_VARIABLE(var) CONCAT_STR(var, __COUNTER__)
#else
#define PLUGIN_ANONYMOUS_VARIABLE(var) CONCAT_STR(var, __LINE__)
#endif

#define AOE_REGISTER_PLUGIN(pluginName, jobType, clazzType)                                 \
    static Aoe::AoePluginRegister PLUGIN_ANONYMOUS_VARIABLE(g_plugin##clazzType)            \
        __attribute__((unused))(jobType, pluginName, Aoe::AoePluginRegister::DefaultCreator<clazzType>)


#endif // AOE_PLUGIN_AOE_PLUGIN_REGISTER_H