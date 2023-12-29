/**
 * @file aoe_types.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2023. All rights reserved.\n
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n
 *
 */

#ifndef AOE_TYPES_H
#define AOE_TYPES_H
#include <vector>
#include <limits>
#include <memory>
#include "graph/graph.h"
#include "ge/ge_api.h"
#include "external/aoe_errcodes.h"

namespace Aoe {

// aoe 框架参数 （包括SGAT/OPAT等调优参数）
const char * const WORK_PATH                                 = "work_path";
const char * const SERVER_IP                                 = "ip";
const char * const SERVER_PORT                               = "port";
const char * const TUNING_PARALLEL_NUM                       = "tuning_parallel_num";
const char * const DEVICE                                    = "device";
const char * const JOB_TYPE                                  = "job_type";
const char * const RUN_LOOP                                  = "run_loop";
const char * const OUTPUT                                    = "output";
const char * const FRAMEWORK                                 = "framework";
const char * const MODEL_PATH                                = "model_path";
const char * const TUNE_OPS_FILE                             = "tune_ops_file";
const char * const SINGLE_OP                                 = "singleop";
const char * const OP_TUNE_FILE                              = "op_tune_file";
const char * const OP_TUNE_MODE                              = "op_tune_mode";
const char * const OP_TUNE_KERNEL_PATH                       = "op_tune_kernel_path";
const char * const TUNE_OPTIMIZATION_LEVEL                   = "tune_optimization_level";
const char * const FEATURE_DEEPER_OPAT                       = "Fdeeper_opat";
const char * const FEATURE_NONHOMO_SPLIT                     = "Fnonhomo_split";
const char * const FEATURE_INNER_AXIS_CUT                    = "Finner_axis_cut";
const char * const FEATURE_OP_FORMAT                         = "Fop_format";
const char * const RELOAD                                    = "reload";
const char * const MODEL                                     = "model";
const char * const WEIGHT                                    = "weight";
const char * const HELP                                      = "help";
const char * const PROGRESS                                  = "progress_bar";

// aoe 命令行支持的GE参数 (包括parser)
const char * const SOC_VER                                   = "soc_version"; // AOE命令行不支持，内部通过acl接口获取
const char * const CORE_TYPE                                 = "core_type";
const char * const BUFFER_OPTIMIZE                           = "buffer_optimize";
const char * const ENABLE_COMPRESS_WEIGHT                    = "enable_compress_weight";
const char * const COMPRESS_WEIGHT_CONF                      = "compress_weight_conf";
const char * const PRECISION_MODE                            = "precision_mode";
const char * const DISABLE_REUSE_MEMORY                      = "disable_reuse_memory";
const char * const ENABLE_SINGLE_STREAM                      = "enable_single_stream";
const char * const AICORE_NUM                                = "aicore_num";
const char * const FUSION_SWITCH_FILE                        = "fusion_switch_file";
const char * const ENABLE_SMALL_CHANNEL                      = "enable_small_channel";
const char * const OP_SELECT_IMPL_MODE                       = "op_select_implmode";
const char * const OPTYPELIST_FOR_IMPLMODE                   = "optypelist_for_implmode";
const char * const ENABLE_SCOPE_FUSION_PASSES                = "enable_scope_fusion_passes";
const char * const OP_DEBUG_LEVEL                            = "op_debug_level";
const char * const INPUT_FORMAT                              = "input_format";
const char * const INPUT_SHAPE                               = "input_shape";
const char * const INPUT_SHAPE_RANGE                         = "input_shape_range";
const char * const OP_NAME_MAP                               = "op_name_map";
const char * const DYNAMIC_BATCH_SIZE                        = "dynamic_batch_size";
const char * const DYNAMIC_IMAGE_SIZE                        = "dynamic_image_size";
const char * const DYNAMIC_DIMS                              = "dynamic_dims";
const char * const OUTPUT_TYPE                               = "output_type";
const char * const OUT_NODES                                 = "out_nodes";
const char * const INPUT_FP16_NODES                          = "input_fp16_nodes";
const char * const LOG_LEVEL                                 = "log";
const char * const INSERT_OP_FILE                            = "insert_op_conf";
const char * const VIRTUAL_TYPE                              = "virtual_type";
const char * const COMPRESSION_OPTIMIZE_CONF                 = "compression_optimize_conf";
const char * const SPARSITY                                  = "sparsity";
const char * const OP_PRECISION_MODE                         = "op_precision_mode";
const char * const MODIFY_MIXLIST                            = "modify_mixlist";
const char * const KEEP_DTYPE                                = "keep_dtype"; // 转换成图属性， 不需要传给ACL
const char * const CUSTOMIZE_DTYPES                          = "customize_dtypes";
const char * const HOST_ENV_OS                               = "host_env_os";
const char * const HOST_ENV_CPU                              = "host_env_cpu";
const char * const IS_INPUT_ADJUST_HW_LAYOUT                 = "is_input_adjust_hw_layout"; // parser only
const char * const IS_OUTPUT_ADJUST_HW_LAYOUT                = "is_output_adjust_hw_layout"; // parser only

// tf only
const char * const AOE_CONFIG_FILE                           = "ge.aoe_config_file";

// aoe 内部参数
const char * const JOB_ID                                    = "job_id";
const char * const TUNING_NAME                               = "tuning_name";
const char * const OPAT_BUILD_RUN_KEY                        = "opat_build_run_key"; // opat
const char * const PLUGIN_OPTION_TUNING_GRAPH                = "tuning_graph";
const char * const PLUGIN_OPTION_TUNING_DEPEND_GRAPH         = "tuning_depend_graph";
const char * const PLUGIN_OPTION_GRAPH_INPUTS                = "graph_inputs";
const char * const PLUGIN_OPTION_DEPEND_GRAPH_INPUTS         = "depend_graph_inputs";
const char * const PLUGIN_OPTION_GE_SESSION                  = "ge_session";
const char * const PLUGIN_OPTION_IS_SINGLE_OP                = "is_singleop";
const char * const PLUGIN_OPTION_IS_TF_OFFLINE               = "is_tfoffline";
const char * const PLUGIN_OPTION_SESSION_ID                  = "session_id";
const char * const FEATURE                                   = "feature";
const char * const OUT_FILE_NAME                             = "out_file_name";

// aoe API透传的GE参数
const char * const GE_INPUT_SHAPE_RANGE                      = "ge.exec.dataInputsShapeRange";
const char * const RESOURCE_CONFIG_PATH                      = "ge.resourceConfigPath";
const char * const RECOMPUTE                                 = "ge.recompute";
const char * const AUTO_TUNE_MODE                            = "ge.autoTuneMode";
const char * const DEBUG_DIR                                 = "ge.debugDir";
const char * const DETERMINISTIC                             = "ge.deterministic";
const char * const EXTERNAL_WEIGHT                           = "ge.externalWeight";
const char * const OP_COMPILER_CACHE_DIR                     = "ge.op_compiler_cache_dir";
const char * const OP_COMPILER_CACHE_MODE                    = "ge.op_compiler_cache_mode";
const char * const OP_DEBUG_CONFIG                           = "op_debug_config";
const char * const OPTION_HOST_ENV_CPU                       = "ge.host_env_cpu";
const char * const OPTION_HOST_ENV_OS                        = "ge.host_env_os";
const char * const SHAPE_GENERALIZED_BUILD_MODE              = "ge.shape_generalized_build_mode";
const char * const SOC_VERSION                               = "ge.socVersion";
const char * const EXCLUDE_ENGINES                           = "ge.exec.exclude_engines";
const char * const EXEC_DISABLE_REUSED_MEMORY                = "ge.exec.disableReuseMemory";
const char * const MDL_BANK_PATH                             = "ge.mdl_bank_path";
const char * const OP_BANK_PATH                              = "ge.op_bank_path";
const char * const TUNE_DEVICE_IDS                           = "ge.exec.tuneDeviceIds";
const char * const OPTION_GRAPH_RUN_MODE                     = "ge.graphRunMode";
const char * const OP_TUNE_KERNEL_NAME                       = "_kernelname";
const char * const IMPL_MODE                                 = "ge.opSelectImplmode";
const char * const OPTION_SCREEN_PRINT_MODE                  = "ge.screen_print_mode";


// 其他常量定义
constexpr char NO_DYNAMIC_PARAM[] = "no_dynamic_param";
constexpr char const *QTEST_SOC_VERSION = "QTEST_SOC_VERSION";
constexpr char OP_TUNE_MODE_STATIC_KERNEL[] = "static_kernel";
constexpr char OP_TUNE_MODE_FAST[] = "fast";

const AoeStatus AOE_ERROR_NO_AICORE_GRAPH                           = AOE_ERROR_NON_OPTIMIZE_GRAPH; // for sgat

/* system api error code */
const AoeStatus AOE_ERROR_LIB_ACCESS                                = 201;  // 内部状态码
const AoeStatus AOE_ERROR_LIB_SYMBOL                                = 202;
const AoeStatus AOE_ERROR_MEMORY_OPERATION                          = 203;
const AoeStatus AOE_ERROR_CHECK_MEMORY                              = 204;

/* aoe cmd error code */
const AoeStatus AOE_NO_ERROR                                        = 301;  // --help
const AoeStatus AOE_ERROR_NO_SUPPORT                                = 302;  // not support model
const AoeStatus AOE_ERROR_PARAM_EXCP                                = 303;  // param exception

/* aoe library error code */
const AoeStatus AOE_ERROR_REPEAT_GRAPH                              = 401;
const AoeStatus AOE_ERR_PLUGIN_LIBRARY_LOAD                         = 402;
const AoeStatus AOE_ERR_PLUGIN_LIBRARY_UNLOAD                       = 403;

/* aoe executor compiler return code */
const AoeStatus AOE_ERROR_EXECUTE_COMPILER                          = 2000;

/* aoe executor runner return code */
const AoeStatus AOE_ERROR_EXECUTE_RUNNER                            = 3000;
const AoeStatus AOE_ERROR_NO_SPACE                                  = 3001;
const AoeStatus AOE_ERROR_EXECUTE_FAILED                            = 3002;
const AoeStatus AOE_ERROR_OUT_OF_BOUND                              = 3003;
const AoeStatus AOE_ERROR_SET_DYNAMIC_PARAM_FAILED                  = 3004;

/* profiling return code */
const AoeStatus AOE_ERROR_PROF_TIMEOUT                              = 3202;
const AoeStatus AOE_ERROR_PROF_PARSER                               = 3203;

/* network(nca) return code */
const AoeStatus AOE_ERROR_NET_DOWN                                  = 3300;
const AoeStatus AOE_ERROR_NET_UNREACH                               = 3301;

enum class RunMode : uint32_t {
    DEFAULT_RUN_MODE = 0,
    LOAD_MODEL_PREV_ALLOC_MEM,
    LOAD_MODEL_WITHOUT_CHECK_MEM,
    RUN_MODE_MAX = 100
};

struct AoeDataInfo {
    uint8_t *ptr = nullptr;
    size_t size = 0;
    int64_t *dims = nullptr;
    uint32_t dimNum = 0;
    ge::DataType type = ge::DT_UNDEFINED;
};

struct AoeBufferData {
    std::shared_ptr<uint8_t> data = nullptr;
    uint64_t length;
};

struct RunnerInitConfig {
    // ncs only
    std::vector<uint32_t> devList;
};

struct RunnerConfig {
    bool isProf = true;
    uint32_t loop;
    // Running time above this threshold will end the run prematurely
    uint64_t takeTimeUpperBound = std::numeric_limits<uint64_t>::max();
    // offline only
    std::vector<AoeDataInfo> input;
    std::vector<AoeDataInfo> output;
    std::string modelPath; // run with model file
    AoeDataInfo modelData; // run with model data
    uint32_t modelId; // run with model Id
    // online only
    ge::Session *session = nullptr;
    std::map<std::string, std::string> options;
    std::vector<std::vector<ge::Tensor>> inputs;
    std::vector<std::vector<ge::Tensor>> outputs;
    std::vector<ge::Graph> dependGraph; // run graph (for training)
    RunMode runMode = RunMode::DEFAULT_RUN_MODE;
    std::string dynamicType;
    std::string dynamicValue;
    std::string buildRunKey;
};

struct RunnerOpInfo {
    std::string opName;
    uint64_t opCostTime;
    uint64_t aicoreCostTime;
    // gradient_split only
    std::string modelName;
    std::string opType;
    uint64_t start;
    uint64_t end;
};

struct RunnerModelInfo {
    uint64_t totalCostTime;
};

// online run result
struct RunnerRunResult {
    std::vector<RunnerModelInfo> modelInfo;
    std::vector<RunnerOpInfo> opInfo;
};

// offline run result
struct RunnerResult {
    uint64_t totalCostTime;
    uint64_t profE2ECostTime;
    std::map<std::string, uint64_t> opCostTime;
    std::map<std::string, uint64_t> aicoreCostTime;
};

// tuning result
struct PerformanceMetric {
    uint64_t totalCostTime = 0UL;
    uint64_t totalProfilingCostTime = 0UL;
    std::map<std::string, uint64_t> opCostTime;
    std::map<std::string, uint64_t> aicoreCostTime;
};
}
#endif
