/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
 * Description:
 * Author: huawei
 * Create: 2019-10-15
 */
#ifndef __AICPU_LIB_H__
#define __AICPU_LIB_H__

extern "C" {
#ifndef CCE_CLANG
#define __global__
#define __aicpu__
#define __gm__
#else
#ifndef __aicpu__
#define __aicpu__ [aicpu]
#endif
#endif


/**< print level */
#define PRINT_LEVEL_DEBUG       0
#define PRINT_LEVEL_INFO        1
#define PRINT_LEVEL_WARN        2
#define PRINT_LEVEL_ERROR       3
#define PRINT_LEVEL_MAX         4

/**
    print info to uart
    支持的打印格式: %llx %c %d %u %s %b %x %f %%
    @ fmt           : print info
    @ return        : void
*/
__aicpu__ void printfw(__gm__ const char *fmt, ...);


#define FPRINT_TYPE_LOG         1   /**< send the info by logging channel, print the info to file */
#define FPRINT_TYPE_UART        2   /**< print info to uart */
#define FPRINT_TYPE_DUMP        4   /**< send the info by dump channel, print the info to file */
/**
    print info to uart or file
    支持的打印格式: %llx %c %d %u %s %b %x %f %%
    @ print_type    : print type bitmap(FPRINT_TYPE_LOG/FPRINT_TYPE_UART/FPRINT_TYPE_DUMP)
    @ level         : print level(PRINT_LEVEL_INFO/PRINT_LEVEL_DEBUG/PRINT_LEVEL_ERROR)
    @ fmt           : print info
    @ return        : void
*/
__aicpu__ void fprintfw(int print_type, int level, __gm__ const char *fmt, ...);

/**
    get cpu tick
    @ return        : cpu tick
*/
__aicpu__ unsigned long long aicpu_get_cpu_tick();

/**
    get cpu tick freq
    @ return        : tick freq
*/
__aicpu__ unsigned int aicpu_get_cpu_tick_freq();

/* 函数时间采样类型 */
#define FUNC_TIME_TYPE_START	0
#define FUNC_TIME_TYPE_END		1
/* 函数开始和结束时间采样函数 */
__aicpu__ void aicpu_func_time_cap(__gm__ const char *func_name, unsigned int type, unsigned long long tick);

/* 内存申请函数 */
__aicpu__ void *aicpu_malloc(unsigned int size);
__aicpu__ void aicpu_free(void *ptr);

#define AI_CPU_CORE_NUM_MAX   64
__aicpu__ int aicpu_get_proc_core_list(unsigned int *core_num, unsigned int core_list[AI_CPU_CORE_NUM_MAX]);
}

#endif /* __AICPU_LIB_H__ */

