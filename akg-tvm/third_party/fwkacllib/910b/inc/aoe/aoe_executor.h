/**
 * @file aoe_executor.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2023. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

/** @defgroup aoe executor */
#ifndef AOE_EXECUTOR_H
#define AOE_EXECUTOR_H
#include <map>
#include <limits>
#include <string>
#include <future>
#include "graph/graph.h"
#include "graph/tuning_utils.h"
#include "aoe_types.h"

namespace Aoe {
using ExecuteSession = uint64_t;
using ExecuteBufferData = struct AoeDataInfo;
using ExecuteResult = struct RunnerResult;
using ExecuteOnlineResult = struct RunnerRunResult;

enum class CompileType {
    COMPILE_TYPE_BASIC = 0,
    COMPILE_TYPE_SPLIT,
    COMPILE_TYPE_TUNING,
    COMPILE_TYPE_TUNED,
    COMPILE_TYPE_NORMAL
};

enum class ExecutorType {
    EXE_DEFAULT                     = 0,                                   // follow initialize executor type
    EXE_OFFLINE_COMPILE             = 1 << 0,                              // support offline compiler
    EXE_OFFLINE_RUN                 = 1 << 1,                              // support offline runner
    EXE_ONLINE_COMPILE              = 1 << 2,                              // support online compiler
    EXE_ONLINE_RUN                  = 1 << 3,                              // support online runner
    EXE_REMOTE_COMPILE              = 1 << 4,                              // use remote compiler
    EXE_REMOTE_RUN                  = 1 << 5,                              // use remote runner
    EXE_REMOTE_COMPILE_RUN          = 1 << 6,                              // remote support build and run together
    EXE_QTEST_RUN                   = 1 << 7,                              // support qtest runner
};

struct ExecuteInitConfig {
    uint32_t executorType = static_cast<uint32_t>(ExecutorType::EXE_DEFAULT);  // executor type
    std::map<std::string, std::string> options;                                // init option
};

struct OfflineRunParam {
    bool isProf = true;
    uint32_t loop;
    uint64_t takeTimeUpperBound = std::numeric_limits<uint64_t>::max();
    std::string dynamicType;
    std::string dynamicValue;
    std::string failedFilePath;
    std::string buildRunKey;
    std::vector<ExecuteBufferData> input;
    std::vector<ExecuteBufferData> output;
    ExecuteResult costTime;
    RunMode runMode = RunMode::DEFAULT_RUN_MODE;
};

struct OnlineRunParam {
    bool isProf = true;
    uint32_t loop;
    uint64_t takeTimeUpperBound = std::numeric_limits<uint64_t>::max();
    ge::Session *session = nullptr;
    std::map<std::string, std::string> options;
    std::vector<std::vector<ge::Tensor>> inputs;
    std::vector<std::vector<ge::Tensor>> outputs;
    std::vector<ge::Graph> dependGraph;
    std::vector<ExecuteOnlineResult> costTime;
    RunMode runMode = RunMode::DEFAULT_RUN_MODE;
};

constexpr uint32_t EXE_OFFLINE_EXECUTOR            = static_cast<uint32_t>(ExecutorType::EXE_OFFLINE_COMPILE) |
                                                     static_cast<uint32_t>(ExecutorType::EXE_OFFLINE_RUN);
constexpr uint32_t EXE_ONLINE_EXECUTOR             = static_cast<uint32_t>(ExecutorType::EXE_ONLINE_COMPILE) |
                                                     static_cast<uint32_t>(ExecutorType::EXE_ONLINE_RUN);
constexpr uint32_t EXE_REMOTE_EXECUTOR             = static_cast<uint32_t>(ExecutorType::EXE_REMOTE_COMPILE) |
                                                     static_cast<uint32_t>(ExecutorType::EXE_REMOTE_RUN);
constexpr uint32_t EXE_REMOTE_COMPILE_RUN_EXECUTOR = EXE_REMOTE_EXECUTOR |
                                                     static_cast<uint32_t>(ExecutorType::EXE_REMOTE_COMPILE_RUN);
constexpr uint32_t EXE_QTEST_EXECUTOR              = static_cast<uint32_t>(ExecutorType::EXE_QTEST_RUN);

const std::map<CompileType, std::map<std::string, std::string>> BUILD_MAP = {
    {CompileType::COMPILE_TYPE_BASIC, {
        {ge::BUILD_MODE, ge::BUILD_MODE_BASELINE}}
    },
    {CompileType::COMPILE_TYPE_NORMAL, {
        {ge::BUILD_MODE, ge::BUILD_MODE_NORMAL}}
    },
    {CompileType::COMPILE_TYPE_SPLIT, {
        {ge::BUILD_MODE, ge::BUILD_MODE_TUNING},
        {ge::BUILD_STEP, ge::BUILD_STEP_BEFORE_UB_MATCH}}
    },
    {CompileType::COMPILE_TYPE_TUNING, {
        {ge::BUILD_MODE, ge::BUILD_MODE_TUNING},
        {ge::BUILD_STEP, ge::BUILD_STEP_AFTER_UB_MATCH}}
    },
    {CompileType::COMPILE_TYPE_TUNED, {
        {ge::BUILD_MODE, ge::BUILD_MODE_TUNING},
        {ge::BUILD_STEP, ge::BUILD_STEP_AFTER_MERGE}}
    },
};

// this set for executor init option
const std::set<std::string> EXECUTOR_INIT_OPTION_SET = {
    SERVER_IP,
    SERVER_PORT,
    DEVICE,
    CORE_TYPE,
    BUFFER_OPTIMIZE,
    ENABLE_COMPRESS_WEIGHT,
    COMPRESS_WEIGHT_CONF,
    PRECISION_MODE,
    DISABLE_REUSE_MEMORY,
    ENABLE_SINGLE_STREAM,
    SOC_VER,
    AICORE_NUM,
    FUSION_SWITCH_FILE,
    ENABLE_SMALL_CHANNEL,
    OP_SELECT_IMPL_MODE,
    OPTYPELIST_FOR_IMPLMODE,
    ENABLE_SCOPE_FUSION_PASSES,
    OP_DEBUG_LEVEL,
    VIRTUAL_TYPE,
    SPARSITY,
    MODIFY_MIXLIST,
    CUSTOMIZE_DTYPES,
    FRAMEWORK,
    COMPRESSION_OPTIMIZE_CONF,

    QTEST_SOC_VERSION,
    SOC_VERSION,
    TUNE_DEVICE_IDS,
    EXEC_DISABLE_REUSED_MEMORY,
    AUTO_TUNE_MODE,
    OP_COMPILER_CACHE_MODE,
    OP_COMPILER_CACHE_DIR,
    EXTERNAL_WEIGHT,
    DETERMINISTIC,
    OPTION_HOST_ENV_OS,
    OPTION_HOST_ENV_CPU,
    HOST_ENV_OS,
    HOST_ENV_CPU,
    DEBUG_DIR,
    OP_TUNE_MODE,
    OPTION_SCREEN_PRINT_MODE,
};

const std::set<std::string> EXECUTOR_COMPILE_OPTION_SET = {
    INPUT_FORMAT,
    INPUT_SHAPE,
    INPUT_SHAPE_RANGE,
    OP_NAME_MAP,
    DYNAMIC_BATCH_SIZE,
    DYNAMIC_IMAGE_SIZE,
    DYNAMIC_DIMS,
    PRECISION_MODE,
    OUTPUT_TYPE,
    INPUT_FP16_NODES,
    OUT_NODES,
    LOG_LEVEL,
    OP_DEBUG_LEVEL,
    INSERT_OP_FILE,
    DISABLE_REUSE_MEMORY,
    GE_INPUT_SHAPE_RANGE,
    OPAT_BUILD_RUN_KEY,
    ge::BUILD_MODE,
    ge::BUILD_STEP,
    ge::TUNING_PATH,
    ge::ir_option::OP_BANK_UPDATE,
    OP_PRECISION_MODE,
    QTEST_SOC_VERSION,
    HOST_ENV_OS,
    HOST_ENV_CPU,

    EXEC_DISABLE_REUSED_MEMORY,
    AUTO_TUNE_MODE,
    OP_COMPILER_CACHE_MODE,
    OP_COMPILER_CACHE_DIR,
    DEBUG_DIR,
    MDL_BANK_PATH,
    OP_BANK_PATH,
    MODIFY_MIXLIST,
    SHAPE_GENERALIZED_BUILD_MODE,
    OP_DEBUG_CONFIG,
    EXTERNAL_WEIGHT,
    EXCLUDE_ENGINES,
    ge::OPTION_GRAPH_RUN_MODE,
    DETERMINISTIC,
    IMPL_MODE,
};

/**
 * @ingroup     aoe executor
 * @brief       aoe executor initialize
 * @param  [in] ExecuteInitConfig &config  config
 * @return      success == AOE_SUCCESS; failed != AOE_SUCCESS
 */
extern "C" AoeStatus AoeExecutorInitialize(ExecuteInitConfig &config);

/**
 * @ingroup    aoe executor
 * @brief      aoe executor finalize
 * @return     success == AOE_SUCCESS; failed != AOE_SUCCESS
 */
extern "C" AoeStatus AoeExecutorFinalize();

/**
 * @ingroup     aoe executor
 * @brief       aoe executor create session
 * @param [out] ExecuteSession &session  session
 * @return      success == AOE_SUCCESS; failed != AOE_SUCCESS
 */
extern "C" AoeStatus AoeExecutorCreateSession(ExecuteSession &session);

/**
 * @ingroup     aoe executor
 * @brief       aoe executor destroy session
 * @param  [in] ExecuteSession &session  session
 * @return      success == AOE_SUCCESS; failed != AOE_SUCCESS
 */
extern "C" AoeStatus AoeExecutorDestroySession(const ExecuteSession &session);

/**
 * @ingroup     aoe executor
 * @brief       aoe executor save model
 * @param  [in] const ExecuteSession &session   session
 * @param  [in] std::string &omPath             save model path
 * @return      success == AOE_SUCCESS; failed != AOE_SUCCESS
 */
extern "C" AoeStatus AoeExecutorSaveModel(const ExecuteSession &session, const std::string &omPath);

/**
 * @ingroup     aoe executor
 * @brief       aoe executor load model
 * @param  [in] const ExecuteSession &session   session
 * @param  [in] std::string &omPath             load model path
 * @return      success == AOE_SUCCESS; failed != AOE_SUCCESS
 */
extern "C" AoeStatus AoeExecutorLoadModel(const ExecuteSession &session, const std::string &omPath);

/**
 * @ingroup     aoe executor
 * @brief       aoe executor unload model
 * @param  [in] const ExecuteSession &session   session
 * @return      success == AOE_SUCCESS; failed != AOE_SUCCESS
 */
extern "C" AoeStatus AoeExecutorUnloadModel(const ExecuteSession &session);
/**
 * @ingroup     aoe executor
 * @brief       aoe executor compile graph by session
 * @param  [in] const ExecuteSession &session                              session
 * @param  [in] const ge::Graph &graph                                     compile graph
 * @param  [in] const std::map<std::string, std::string> &compileOptions   compile option
 * @return      success == AOE_SUCCESS; failed != AOE_SUCCESS
 */
extern "C" AoeStatus AoeExecutorOffilneCompile(const ExecuteSession &session, const ge::Graph &graph,
    const std::map<std::string, std::string> &compileOptions);

/**
 * @ingroup      aoe executor
 * @brief        aoe executor offline run graph by session
 * @param  [in]  const ExecuteSession &session      session
 * @param  [out] OfflineRunParam &param             run param
 * @return       success == AOE_SUCCESS; failed != AOE_SUCCESS
 */
extern "C" AoeStatus AoeExecutorOfflineRun(const ExecuteSession &session, OfflineRunParam &param);

/**
 * @ingroup     aoe executor
 * @brief       aoe executor offline split graph
 * @param  [in] const ge::Graph &graph                                    graph splited
 * @param  [in] const std::string &subgraphPath                           splited graph path
 * @param  [in] const std::map<std::string, std::string> &splitOptions    splited graph option
 * @return      success == AOE_SUCCESS; failed != AOE_SUCCESS
 */
extern "C" AoeStatus AoeExecutorOfflineSpiltGraph(const ge::Graph &graph, const std::string &subgraphPath,
    const std::map<std::string, std::string> &splitOptions);

/**
 * @ingroup      aoe executor
 * @brief        aoe executor compile run async
 * @param  [in]  const ExecuteSession &session   session
 * @param  [in]  const ge::Graph &graph                                         compile graph
 * @param  [in]  const std::map<std::string, std::string> &compileOptions       compile option
 * @param  [out] OfflineRunParam &param                                         offline param
 * @return      success == AOE_SUCCESS; failed != AOE_SUCCESS
 */
extern "C" std::future<AoeStatus> AoeExecutorOfflineCompileRunAsync(const ExecuteSession &session,
    const ge::Graph &graph, const std::map<std::string, std::string> &compileOptions, OfflineRunParam &param);

/**
 * @ingroup      aoe executor
 * @brief        aoe executor compile run async
 * @param  [in]  const ExecuteSession &session   session
 * @param  [in]  const ge::Graph &graph                                         compile graph
 * @param  [in]  const std::map<std::string, std::string> &compileOptions       compile option
 * @param  [out] OfflineRunParam &param                                         offline param
 * @return      success == AOE_SUCCESS; failed != AOE_SUCCESS
 */
extern "C" std::future<AoeStatus> AoeExecutorQtestCompileRunAsync(const ExecuteSession &session,
    const ge::Graph &graph, const std::map<std::string, std::string> &compileOptions, OfflineRunParam &param);

/**
 * @ingroup      aoe executor
 * @brief        aoe executor online run graph
 * @param  [in]  const ge::Graph &graph             run graph
 * @param  [out] OfflineRunParam &param             run param
 * @return       success == AOE_SUCCESS; failed != AOE_SUCCESS
 */
extern "C" AoeStatus AoeExecutorOnlineRun(const ge::Graph &graph, OnlineRunParam &param);

/**
 * @ingroup      aoe executor
 * @brief        aoe executor qtest run graph
 * @param  [in]  const ge::Graph &graph             run graph
 * @param  [in]  const std::string &graphPath       graph path
 * @param  [out] OfflineRunParam &param               run param
 * @return       success == AOE_SUCCESS; failed != AOE_SUCCESS
 */
extern "C" AoeStatus AoeExecutorQtestRun(const ge::Graph &graph, const std::string &graphPath, OfflineRunParam &param);

/**
 * @ingroup     aoe executor
 * @brief       aoe executor online split graph
 * @param  [in] const ge::Graph &graph                                    graph splited
 * @param  [in] ge::Session *session                                      ge session
 * @param  [in] const std::string &subgraphPath                           splited graph path
 * @param  [in] const std::map<std::string, std::string> &splitOptions    splited graph option
 * @param  [in] const std::vector<ge::Tensor> &inputs                     graph inputs
 * @return      success == AOE_SUCCESS; failed != AOE_SUCCESS
 */
extern "C" AoeStatus AoeExecutorOnlineSpiltGraph(const ge::Graph &graph, ge::Session *session,
    const std::string &subgraphPath, const std::map<std::string, std::string> &splitOptions,
    const std::vector<ge::Tensor> &inputs);
}
#endif
