/**
 * @file aoe.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.\n
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n
 *
 */

#ifndef AOE_EXTERNAL_AOE_H
#define AOE_EXTERNAL_AOE_H

#include <map>
#include "ge/ge_api.h"
#include "graph/ascend_string.h"
#include "external/aoe_errcodes.h"

namespace Aoe {
/**
 * @brief       : initialize aoe tuning api
 * @param [in]  : std::map<ge::AscendString, ge::AscendString> &globalOptions          global options
 * @return      : == AOE_SUCCESS : success, != AOE_SUCCESS : failed
 */
extern "C" AoeStatus AoeInitialize(const std::map<ge::AscendString, ge::AscendString> &globalOptions);

/**
 * @brief       : finalize aoe tuning api
 * @return      : == AOE_SUCCESS : success, != AOE_SUCCESS : failed
 */
extern "C" AoeStatus AoeFinalize();

/**
 * @brief       : create aoe session
 * @param [out] : uint64_t sessionId                                                    session id
 * @return      : == AOE_SUCCESS : success, != AOE_SUCCESS : failed
 */
extern "C" AoeStatus AoeCreateSession(uint64_t &sessionId);

/**
 * @brief       : destroy aoe session
 * @param [in]  : uint64_t sessionId                                      session id
 * @return      : == AOE_SUCCESS : success, != AOE_SUCCESS : failed
 */
extern "C" AoeStatus AoeDestroySession(uint64_t sessionId);

/**
 * @brief       : set ge session for session id
 * @param [in]  : uint64_t sessionId                                       session id
 * @param [in]  : ge::Session *geSession                                   ge session handle
 * @return      : == AOE_SUCCESS : success, != AOE_SUCCESS : failed
 */
extern "C" AoeStatus AoeSetGeSession(uint64_t sessionId, ge::Session *geSession);

/**
 * @brief       : set tuning graphs for session id
 * @param [in]  : uint64_t sessionId                                       session id
 * @param [in]  : ge::Graph &tuningGraph                                   tuning graph
 * @return      : == AOE_SUCCESS : success, != AOE_SUCCESS : failed
 */
extern "C" AoeStatus AoeSetTuningGraph(uint64_t sessionId, const ge::Graph &tuningGraph);

/**
 * @brief       : set input of tuning graph for session id
 * @param [in]  : uint64_t sessionId                                       session id
 * @param [in]  : std::vector<ge::Tensor> &inputs                          input tensor
 * @return      : == AOE_SUCCESS : success, != AOE_SUCCESS : failed
 */
extern "C" AoeStatus AoeSetTuningGraphInput(uint64_t sessionId, const std::vector<ge::Tensor> &input);

/**
 * @brief       : tuning graph
 * @param [in]  : uint64_t sessionId                                                session id
 * @param [in]  : std::map<ge::AscendString, ge::AscendString> &tuningOptions       tuning options
 * @return      : == AOE_SUCCESS : success, != AOE_SUCCESS : failed
 */
extern "C" AoeStatus AoeTuningGraph(uint64_t sessionId,
    const std::map<ge::AscendString, ge::AscendString> &tuningOptions);
} // namespace Aoe
#endif // AOE_EXTERNAL_AOE_H
