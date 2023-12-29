/**
 * @file aoe_plugin.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef AOE_PLUGIN_AOE_PLUGIN_H
#define AOE_PLUGIN_AOE_PLUGIN_H

#include "aoe_types.h"
#include "plugin/option_context.h"
#include "plugin/aoe_plugin_register.h"

namespace Aoe {
class AoePlugin {
public:
    AoePlugin() = default;
    virtual ~AoePlugin() = default;

    /* *
     * @brief       : Initialize aoe plugin
     * @return      : == AOE_SUCCESS : success, != AOE_SUCCESS : failed
     */
    virtual AoeStatus Initialize() = 0;

    /* *
     * @brief       : Finalize aoe plugin
     * @return      : == AOE_SUCCESS : success, != AOE_SUCCESS : failed
     */
    virtual AoeStatus Finalize() = 0;

    /* *
     * @brief       :  Main process function
     * @param [in]  : OptionContext ctx
     * @return      : == AOE_SUCCESS : success, != AOE_SUCCESS : failed
     */
    virtual AoeStatus Process(const OptionContext &ctx) = 0;
};
using AoePluginPtr = std::shared_ptr<AoePlugin>;
} // namespace Aoe
#endif // AOE_PLUGIN_AOE_PLUGIN_H