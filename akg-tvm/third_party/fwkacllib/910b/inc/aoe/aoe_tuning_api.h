/**
 * @file aoe_tuning_api.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2023. All rights reserved.\n
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n
 *
 */

#ifndef AOE_TUNING_API_H
#define AOE_TUNING_API_H

#include <memory>
#include "aoe_types.h"
#include "ge/ge_api.h"
#include "external/aoe.h"

namespace Aoe {
// this set for global option key
const std::set<ge::AscendString> GLOBAL_OPTION_SET = {
    ge::AscendString(WORK_PATH),
    ge::AscendString(SERVER_IP),
    ge::AscendString(SERVER_PORT),
    ge::AscendString(TUNING_PARALLEL_NUM),
    ge::AscendString(DEVICE),
    ge::AscendString(CORE_TYPE),
    ge::AscendString(BUFFER_OPTIMIZE),
    ge::AscendString(ENABLE_COMPRESS_WEIGHT),
    ge::AscendString(COMPRESS_WEIGHT_CONF),
    ge::AscendString(PRECISION_MODE),
    ge::AscendString(DISABLE_REUSE_MEMORY),
    ge::AscendString(ENABLE_SINGLE_STREAM),
    ge::AscendString(AICORE_NUM),
    ge::AscendString(FUSION_SWITCH_FILE),
    ge::AscendString(ENABLE_SMALL_CHANNEL),
    ge::AscendString(OP_SELECT_IMPL_MODE),
    ge::AscendString(OPTYPELIST_FOR_IMPLMODE),
    ge::AscendString(ENABLE_SCOPE_FUSION_PASSES),
    ge::AscendString(OP_DEBUG_LEVEL),
    ge::AscendString(VIRTUAL_TYPE),
    ge::AscendString(SPARSITY),
    ge::AscendString(MODIFY_MIXLIST),
    ge::AscendString(CUSTOMIZE_DTYPES),
    ge::AscendString(FRAMEWORK),
    ge::AscendString(JOB_TYPE),
    ge::AscendString(RUN_LOOP),
    ge::AscendString(RESOURCE_CONFIG_PATH),

    ge::AscendString(SOC_VERSION),
    ge::AscendString(TUNE_DEVICE_IDS),
    ge::AscendString(EXEC_DISABLE_REUSED_MEMORY),
    ge::AscendString(AUTO_TUNE_MODE),
    ge::AscendString(OP_COMPILER_CACHE_MODE),
    ge::AscendString(OP_COMPILER_CACHE_DIR),
    ge::AscendString(DEBUG_DIR),
    ge::AscendString(EXTERNAL_WEIGHT),
    ge::AscendString(DETERMINISTIC),
    ge::AscendString(OPTION_HOST_ENV_OS),
    ge::AscendString(OPTION_HOST_ENV_CPU),
    ge::AscendString(HOST_ENV_OS),
    ge::AscendString(HOST_ENV_CPU),
    ge::AscendString(COMPRESSION_OPTIMIZE_CONF),
    ge::AscendString(OPTION_GRAPH_RUN_MODE),
    ge::AscendString(OP_TUNE_MODE),
    ge::AscendString(SOC_VER),
    ge::AscendString(OPTION_SCREEN_PRINT_MODE),
};

// this set for tuning option key
const std::set<ge::AscendString> TUNING_OPTION_SET = {
    ge::AscendString(INPUT_FORMAT),
    ge::AscendString(INPUT_SHAPE),
    ge::AscendString(INPUT_SHAPE_RANGE),
    ge::AscendString(OP_NAME_MAP),
    ge::AscendString(DYNAMIC_BATCH_SIZE),
    ge::AscendString(DYNAMIC_IMAGE_SIZE),
    ge::AscendString(DYNAMIC_DIMS),
    ge::AscendString(PRECISION_MODE),
    ge::AscendString(OUTPUT_TYPE),
    ge::AscendString(OUT_NODES),
    ge::AscendString(INPUT_FP16_NODES),
    ge::AscendString(LOG_LEVEL),
    ge::AscendString(OP_DEBUG_LEVEL),
    ge::AscendString(INSERT_OP_FILE),
    ge::AscendString(GE_INPUT_SHAPE_RANGE),
    ge::AscendString(OUTPUT),
    ge::AscendString(RELOAD),
    ge::AscendString(TUNING_NAME),
    ge::AscendString(FRAMEWORK),
    ge::AscendString(MODEL_PATH),
    ge::AscendString(TUNE_OPS_FILE),
    ge::AscendString(RECOMPUTE),
    ge::AscendString(AOE_CONFIG_FILE),
    ge::AscendString(OP_PRECISION_MODE),
    ge::AscendString(KEEP_DTYPE),
    ge::AscendString(SINGLE_OP),
    ge::AscendString(TUNE_OPTIMIZATION_LEVEL),
    ge::AscendString(FEATURE_DEEPER_OPAT),
    ge::AscendString(FEATURE_NONHOMO_SPLIT),
    ge::AscendString(FEATURE_INNER_AXIS_CUT),
    ge::AscendString(FEATURE_OP_FORMAT),
    ge::AscendString(OUT_FILE_NAME),
    ge::AscendString(HOST_ENV_OS),
    ge::AscendString(HOST_ENV_CPU),
    ge::AscendString(EXEC_DISABLE_REUSED_MEMORY),
    ge::AscendString(AUTO_TUNE_MODE),
    ge::AscendString(OP_COMPILER_CACHE_MODE),
    ge::AscendString(OP_COMPILER_CACHE_DIR),
    ge::AscendString(DEBUG_DIR),
    ge::AscendString(MDL_BANK_PATH),
    ge::AscendString(OP_BANK_PATH),
    ge::AscendString(MODIFY_MIXLIST),
    ge::AscendString(SHAPE_GENERALIZED_BUILD_MODE),
    ge::AscendString(OP_DEBUG_CONFIG),
    ge::AscendString(EXTERNAL_WEIGHT),
    ge::AscendString(EXCLUDE_ENGINES),
    ge::AscendString(OP_TUNE_MODE),
    ge::AscendString(OP_TUNE_KERNEL_PATH),
    ge::AscendString(OP_TUNE_KERNEL_NAME),
    ge::AscendString(IMPL_MODE),
    ge::AscendString(DETERMINISTIC),
    ge::AscendString(OPTION_SCREEN_PRINT_MODE),
};

/**
 * @brief       : set depend graphs for session id
 * @param [in]  : uint64_t sessionId                                       session id
 * @param [in]  : std::vector<ge::Graph> &dependGraphs                     depend graphs
 * @return      : == AOE_SUCCESS : success, != AOE_SUCCESS : failed
 */
extern "C" AoeStatus AoeSetDependGraphs(uint64_t sessionId, const std::vector<ge::Graph> &dependGraphs);

/**
 * @brief       : set inputs of depend graphs for session id
 * @param [in]  : uint64_t sessionId                                       session id
 * @param [in]  : std::vector<std::vector<ge::Tensor>> &inputs             depend input tensor
 * @return      : == AOE_SUCCESS : success, != AOE_SUCCESS : failed
 */
extern "C" AoeStatus AoeSetDependGraphsInputs(uint64_t sessionId, const std::vector<std::vector<ge::Tensor>> &inputs);
}  // namespace Aoe

#endif
