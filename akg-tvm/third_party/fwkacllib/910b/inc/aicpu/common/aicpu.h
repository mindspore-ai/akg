/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
 * Description:
 * Author: huawei
 * Create: 2019-10-15
 */
#ifndef __AICPU_H__
#define __AICPU_H__

#ifdef __ASSEMBLY__
#define __ULL(x) x
#else
#define __AC__(X, Y) (X##Y)
#define __ULL(x) __AC__(x, ULL)
#endif

#define SYSTEM_MUL_CHIP_CONFIG_BASE __ULL(0x200000000000) /* ��Ƭ��ַ��� */

#ifdef CFG_SOC_PLATFORM_MINI
#define SYSTEM_CONFIG_PLAT_BASE __ULL(0x9fe000) /* 10M - 8K */
#define CONFIG_CPU_MAX_NUM 16                   /* CPU������������Ctrl_CPU */
#endif

#ifdef CFG_SOC_PLATFORM_CLOUD
#define SYSTEM_CONFIG_PLAT_BASE __ULL(0x9fe000) /* 10M - 8K */
#define CONFIG_CPU_MAX_NUM 16                   /* CPU������������Ctrl_CPU */
#endif

#ifdef CFG_SOC_PLATFORM_KIRIN990
#include "global_ddr_map.h"
#ifdef AICPU_HIBENCH_SLT
#include "hitest_ddr_map.h"
#undef HISI_RESERVED_NPU_AI_SERVER_PHYMEM_BASE
#define HISI_RESERVED_NPU_AI_SERVER_PHYMEM_BASE HITEST_AICPU_PHYMEM_BASE
#endif
#define AICPU_BOOT_ADDR (HISI_RESERVED_NPU_AI_SERVER_PHYMEM_BASE)
#define AICPU_ALG_MALLOC_ADDR (HISI_RESERVED_NPU_AI_SERVER_PHYMEM_BASE + 0x100000)
/* AICPU_ALG_MALLOC_ADDR - SYSTEM_CONFIG_ALG_SUB_SIZE */
#define SYSTEM_CONFIG_ALG_SUB_SIZE __ULL(0x2000)
#define SYSTEM_CONFIG_PLAT_BASE (AICPU_ALG_MALLOC_ADDR - SYSTEM_CONFIG_ALG_SUB_SIZE)
#define CONFIG_CPU_MAX_NUM 2 /* AICPU */
#endif

#ifdef CFG_SOC_PLATFORM_KIRIN990_ES
#include "global_ddr_map.h"
#ifdef AICPU_HIBENCH_SLT
#include "hitest_ddr_map.h"
#undef HISI_RESERVED_NPU_AI_SERVER_PHYMEM_BASE
#define HISI_RESERVED_NPU_AI_SERVER_PHYMEM_BASE HITEST_AICPU_PHYMEM_BASE
#endif
#define AICPU_BOOT_ADDR (HISI_RESERVED_NPU_AI_SERVER_PHYMEM_BASE)
#define AICPU_ALG_MALLOC_ADDR (HISI_RESERVED_NPU_AI_SERVER_PHYMEM_BASE + 0x100000)
/* AICPU_ALG_MALLOC_ADDR - SYSTEM_CONFIG_ALG_SUB_SIZE */
#define SYSTEM_CONFIG_ALG_SUB_SIZE __ULL(0x2000)
#define SYSTEM_CONFIG_PLAT_BASE (AICPU_ALG_MALLOC_ADDR - SYSTEM_CONFIG_ALG_SUB_SIZE)
#define CONFIG_CPU_MAX_NUM 2 /* AICPU */
#endif

#ifdef CFG_SOC_PLATFORM_ORLANDO
#include "global_ddr_map.h"
#define HISI_RESERVED_NPU_AI_SERVER_PHYMEM_BASE HISI_RESERVED_NPU_NORMAL_PHYMEM_BASE
#ifdef AICPU_HIBENCH_SLT
#include "hitest_ddr_map.h"
#undef HISI_RESERVED_NPU_AI_SERVER_PHYMEM_BASE
#define HISI_RESERVED_NPU_AI_SERVER_PHYMEM_BASE HITEST_AICPU_PHYMEM_BASE
#endif
#define AICPU_BOOT_ADDR (HISI_RESERVED_NPU_AI_SERVER_PHYMEM_BASE)
#define AICPU_ALG_MALLOC_ADDR (HISI_RESERVED_NPU_AI_SERVER_PHYMEM_BASE + 0x100000)
/* AICPU_ALG_MALLOC_ADDR - SYSTEM_CONFIG_ALG_SUB_SIZE */
#define SYSTEM_CONFIG_ALG_SUB_SIZE __ULL(0x2000)
#define SYSTEM_CONFIG_PLAT_BASE (AICPU_ALG_MALLOC_ADDR - SYSTEM_CONFIG_ALG_SUB_SIZE)
#define CONFIG_CPU_MAX_NUM 2 /* AICPU */
#endif

#ifdef CFG_SOC_PLATFORM_CLOUD_V2
#define DAVINCI_CLOUD_V2
#endif

#ifdef DAVINCI_CLOUD_V2
#ifdef SYSTEM_CONFIG_PLAT_BASE
#undef SYSTEM_CONFIG_PLAT_BASE
#endif

#ifdef CONFIG_CPU_MAX_NUM
#undef CONFIG_CPU_MAX_NUM
#endif
#define SYSTEM_CONFIG_PLAT_BASE __ULL(0x1036EFE000)      /* 0x36E00000 - last 8K */
#define CONFIG_CPU_MAX_NUM      8
#endif

#ifdef DAVINCI_CLOUD_V2_FFTS
#define SYSTEM_CONFIG_PLAT_BASE __ULL(0x1036EFE000)      /* 0x36E00000 - last 8K */
#define CONFIG_CPU_MAX_NUM      8
#endif

#ifdef CFG_SOC_PLATFORM_MINIV3
#define DAVINCI_MINI_V3
#endif

#ifdef DAVINCI_MINI_V3
#ifdef SYSTEM_CONFIG_PLAT_BASE
#undef SYSTEM_CONFIG_PLAT_BASE
#endif

#ifdef CONFIG_CPU_MAX_NUM
#undef CONFIG_CPU_MAX_NUM
#endif
#if defined(CFG_MEMORY_OPTIMIZE) || defined(DAVINCI_MDC_AS31XM1X)
#define SYSTEM_CONFIG_PLAT_BASE __ULL(0xBB3E000) /* 0xBB40000 - 8K */
#else
#define SYSTEM_CONFIG_PLAT_BASE __ULL(0x2223e000) /* 0x22240000 - 8K */
#endif
#define CONFIG_CPU_MAX_NUM 4
#endif

#ifndef SYSTEM_CONFIG_PLAT_BASE
#define SYSTEM_CONFIG_PLAT_BASE __ULL(0x9fe000)
#endif

#ifndef CONFIG_CPU_MAX_NUM
#define CONFIG_CPU_MAX_NUM 16 /* CPU������������Ctrl_CPU */
#endif


/* ϵͳ���� */
#ifdef COMPILE_UT_TEST
extern u64 system_confg_mem;

#define SYSTEM_CONFIG_BASE (system_confg_mem)
#else
#define SYSTEM_CONFIG_BASE SYSTEM_CONFIG_PLAT_BASE
#endif

#ifdef CFG_SOC_PLATFORM_MINIV2
#define SYSTEM_CONFIG_SIZE (4 * 1024) /* ��ȱҳ���ڴ� */
#else
#define SYSTEM_CONFIG_SIZE (8 * 1024) /* ��ȱҳ���ڴ� */
#endif

#define DT_CONFIG_BASE SYSTEM_CONFIG_BASE
#define DEV_DRV_DESC_OFF (0 * 1024)
#define DEV_DRV_DESC_SIZE __ULL(0x400)
#define DEV_DRV_DESC_BASE (DT_CONFIG_BASE + DEV_DRV_DESC_OFF)
#define MACHINE_DESC_OFF (1 * 1024)
#define MACHINE_DESC_SIZE __ULL(0x400)
#define MACHINE_DESC_BASE (DT_CONFIG_BASE + MACHINE_DESC_OFF)
#define DEV_GRP_DESC_OFF (2 * 1024)
#define DEV_GRP_DESC_SIZE __ULL(0xC00)
#define DEV_GRP_DESC_BASE (DT_CONFIG_BASE + DEV_GRP_DESC_OFF)

#define DT_CONFIG_SIZE (5 * 1024)
#define DT_SYSTEM_CONFIG_SIZE SYSTEM_CONFIG_SIZE

/* �����׶�Firmware ״̬��ʶ��ÿ����ռ��4�ֽ� */
#define SYSTEM_FIRMWARE_STATUS (SYSTEM_CONFIG_BASE + (5 * 1024))

/* Firmware �Լ�״̬��ʶ��ÿ����ռ��4�ֽ� */
#define SYSTEM_FIRMWARE_SELFTEST_STATUS (SYSTEM_CONFIG_BASE + (5 * 1024) + 0x100)

/* ȱҳ������control cpuͨѶ�õ�memory */
#define PAGE_MISS_MSG_BASE (SYSTEM_CONFIG_BASE + (7 * 1024))

#define FIRMWARE_SIZE __ULL(128 * 1024) /* firmware�ļ�ռ�ô�С */
#define EL1_SP_SIZE __ULL(12 * 1024)    /* EL1ջ�ռ��С */
#define EL1_PAGE_SIZE __ULL(512 * 1024) /* EL1ҳ��ռ��С */
#ifdef AICPU_LITE_CHIP_TEST
#define EL1_DATA_SIZE __ULL(16 * 1024) /* EL1���ݶοռ��С */
#else
#define EL1_DATA_SIZE __ULL(12 * 1024) /* EL1���ݶοռ��С */
#endif
#define EL0_SP_SIZE __ULL(8 * 1024)         /* EL0ջ�ռ��С */
#define EL0_DATA_SIZE __ULL(20 * 1024)      /* EL0���ݶοռ��С */
#define EL0_CONFIG_SIZE __ULL(4 * 1024)     /* EL0�������ռ��С */
#define DATA_BACKUP_SIZE __ULL(4 * 1024)    /* ��������С�������µ����ݱ��� */
#define ALG_MEM_SIZE __ULL(2 * 1024 * 1024) /* ���Ӷ�̬�ڴ� */

/* ����CPUռ�ÿռ��С */
#define ONE_CPU_TOTAL_SIZE \
    (EL1_SP_SIZE + EL1_PAGE_SIZE + EL1_DATA_SIZE + EL0_SP_SIZE + EL0_DATA_SIZE + EL0_CONFIG_SIZE + DATA_BACKUP_SIZE)

/* ÿ��CPUռ�õĴ�ӡ�ռ䣬��ControlCPUͨѶ */
#ifdef LITE_MEM_LAYOUT
#define PRINT_BUFF_SIZE __ULL(8192)
#else
#define PRINT_BUFF_SIZE __ULL(512)
#endif
#define PRINT_BUFF_ALL_SIZE (CONFIG_CPU_MAX_NUM * PRINT_BUFF_SIZE)

/* firmware��ռ���ڴ��С�����������Ӷ�̬�ڴ� */
#define FIRMWARE_TATOL_MEM_SIZE(cpu_num) (FIRMWARE_SIZE + (ONE_CPU_TOTAL_SIZE * (cpu_num)) + PRINT_BUFF_ALL_SIZE)

/* ��ȡCPU���ݶ�ƫ��,�����������ַ��ƫ�� */
#define EL1_SP_OFFSET(cpu_ofs) (FIRMWARE_SIZE + ((cpu_ofs)*ONE_CPU_TOTAL_SIZE))
#define EL1_DATA_OFFSET(cpu_ofs) (EL1_SP_OFFSET(cpu_ofs) + EL1_SP_SIZE)
#define EL1_PAGE_OFFSET(cpu_ofs) (EL1_DATA_OFFSET(cpu_ofs) + EL1_DATA_SIZE)

#define EL0_SP_OFFSET(cpu_ofs) (EL1_PAGE_OFFSET(cpu_ofs) + EL1_PAGE_SIZE)
#define EL0_DATA_OFFSET(cpu_ofs) (EL0_SP_OFFSET(cpu_ofs) + EL0_SP_SIZE)
#define EL0_CONFIG_OFFSET(cpu_ofs) (EL0_DATA_OFFSET(cpu_ofs) + EL0_DATA_SIZE)

#define DATA_BACKUP_OFFSET(cpu_ofs) (EL0_CONFIG_OFFSET(cpu_ofs) + EL0_CONFIG_SIZE)
#define PRINT_BUFF_OFFSET(cpu_num) (FIRMWARE_SIZE + (ONE_CPU_TOTAL_SIZE * (cpu_num)))

/* ���漸���ڴ�������ַ�̶� */
#define EL0_SP_VADDR 0xffff000042000000
#define EL0_ALG_MEM_VADDR 0xffff000041000000
#ifdef COMPILE_UT_TEST
extern u64 el0_confog_mem;

#define EL0_CONFIG_VADDR el0_confog_mem
#else
#define EL0_CONFIG_VADDR 0xffff000040000000
#endif

/* �ر�Ҫע��:
aicpu_id_base��aicpu_system_config�ṹ���ƫ��
aicpu_system_config�ṹ�����޸ģ�Ҫ��֤������ֵ��ȷ
����л�ʹ����� */
#define AICPU_ID_BASE_OFFSET 8

#define FUNC_NAME_SIZE (48)
#define AI_CPU_MAX_EVENT_NUM (8)
#define TS_MEM_VALID_RANGE_MAX_NUM (10U)

#define SYSTEM_CONFIG_FLAG 0x5a5aa5a55a5aa5a5ULL

#define AOS_KERNEL_FLAG 0
#define AOS_CORE_FLAG 1

#ifndef __ASSEMBLY__
/* ����ʱ��ʱ����Ϣ */
struct firmware_boot_time {
    unsigned long long sec;
    unsigned long long nsec;
};

#define TS_DMA_CHAN_MAX_NUM 8
struct ts_dma_chan_sqcq_desc {
    unsigned long long dma_addr;
    unsigned long long phy_addr;
    unsigned int len;
    unsigned int rsv;
};

struct valid_addr_range {
    unsigned long long start;
    unsigned long long end;
};
/* AICPUϵͳ���� */
struct aicpu_system_config {
    unsigned long long flag;
    unsigned int aicpu_id_base; /* aicpu start id, ע��ǰ���������ɾ��Ҫ�޸�startup.s��ջ��ʼ�� */
    unsigned int aicpu_num;     /* aicpu num */

    unsigned long long ts_boot_addr;        /* ts��������ַ */
    unsigned long long ts_blackbox_base;    /* ts�ĺ�ϻ���ڴ����ַ */
    unsigned long long ts_blackbox_size;    /* ts�ĺ�ϻ��size */
    unsigned long long ts_start_log_base;   /* ts��start log�ڴ����ַ */
    unsigned long long ts_start_log_size;   /* ts��start log size */
    unsigned long long ts_vmcore_base;      /* ts��vmcore�ڴ����ַ */
    unsigned long long ts_vmcore_size;      /* ts��vmcore�ڴ�size */
    unsigned char enable_bbox;              /* ts��vmcoreʹ�ܱ�־ */
    unsigned char ts_host_irq_base;         /* ts host irq base */
    unsigned char ts_host_irq_mode;         /* ts host irq mode 0:36irq 1:4irq */
    unsigned char connect_protocol;         /* ����Ԥ�� */
    unsigned char reserve[4];               /* ����Ԥ�� */
    unsigned long long aicpu_boot_addr;     /* aicpu��������ַ */
    unsigned long long aicpu_blackbox_base; /* aicpu�ĺ�ϻ���ڴ����ַ */
    unsigned long long aicpu_blackbox_size; /* aicpu�ĺ�ϻ���ڴ�size */

    unsigned long long print_buff_base; /* ��ӡ�ռ���ʼ��ַ */

    unsigned int ts_int_start_id;       /* ����TS����ʼ�жϺ� */
    unsigned int ctrl_cpu_int_start_id; /* ����ControlCPU����ʼ�жϺ� */
    unsigned int ipc_cpu_int_start_id;  /* IPC CPU��ʼ�жϺ�,��ӦCPU0 */
    unsigned int ipc_mbx_int_start_id;  /* IPC mailbox��ʼ�жϺ�,��ӦCPU0 */
    unsigned int firmware_bin_size;     /* aicpu_fw.bin��С */
    unsigned int system_cnt_freq;       /* timer frequency */

    unsigned int aicpu_gicr_status; /* aicpu  gicr ״̬  ÿ���� 1 bit  1 ���� 0  ���� */

    /* ���Ӷ�̬�ڴ�,�����ں�kmalloc���β�������̫�࣬���Էֿ����� */
    unsigned int alg_memory_size; /* ÿ��CPU����Ķ�̬�ڴ��С */
    unsigned long long alg_memory_base[CONFIG_CPU_MAX_NUM];

    unsigned int print_init_flag; /* ����ʱ��ֵ1��print_app��⵽��־���ͽ��г�ʼ������ֵΪ0 */
    struct firmware_boot_time boot_time;

    unsigned long long aicore_bitmap;
    unsigned long long aicore_freq;
    unsigned long long vector_core_bitmap;
    unsigned long long vector_core_freq;

    unsigned long long total_size;   /* DT Middle Struct Total Size */
    unsigned long long tzpc_pa_base; /* Add for OneTrack */
    unsigned long long gicd_pa_base;
    unsigned long long gicd_pa_size;
    unsigned long long gicr_pa_size;
    unsigned long long sram_pa_base;
    unsigned long long sram_pa_size;
    unsigned long long ts_aicpu_status_base;
    unsigned int gic_multichip_off;
    unsigned int aicpu_alloc_size;
    unsigned int ts_alloc_size;
    unsigned int ffts_mcu_int_start_id;
    /* add for dma chan info */
    unsigned int chan_id_base;
    unsigned int chan_num;
    unsigned int chan_done_irq_base; /* only support cloud */
    int board_id;
    unsigned int nvme_pf_num;
    struct ts_dma_chan_sqcq_desc sq_desc[TS_DMA_CHAN_MAX_NUM];
    struct ts_dma_chan_sqcq_desc cq_desc[TS_DMA_CHAN_MAX_NUM];

    /* for STL */
    unsigned long long dcache_va;
    unsigned long long dcache_pa;

    unsigned long long sq_pa_base;      /* reserved sq static memory base physical addr */
    unsigned long long sq_pa_size;      /* ts_sq_static_size */
    unsigned int product_num; // 1p 2p 4p 8p
    unsigned int valid_range_num; // valid range numbers for tsch
    struct valid_addr_range addr_ranges[TS_MEM_VALID_RANGE_MAX_NUM];
    unsigned long long vpc_bitmap;  /* reserved for PG of vpc */
    unsigned long long jpegd_bitmap; /* reserved for PG of jpegd */
    unsigned long long mata_bitmap; /* reserved for PG of mata bitmap */
    unsigned int mata_num; /* reserved for PG of mata number */

    unsigned long long stl_va;
    unsigned int stl_ssid;
    unsigned int aos_flag;
    unsigned int stl_test_period;
};

#define SYSTEM_CONFIG_PHY ((struct aicpu_system_config *)SYSTEM_CONFIG_BASE)
#define MULTI_CONFIG_BASE(socket_id) (socket_id * SYSTEM_MUL_CHIP_CONFIG_BASE)
#define SYSTEM_MULTI_CONFIG_BASE(socket_id) (MULTI_CONFIG_BASE(socket_id) + SYSTEM_CONFIG_BASE)
#define SYSTEM_CONFIG_MULTI_PHY(socket_id) \
    ((struct aicpu_system_config *)(uintptr_t)(SYSTEM_MULTI_CONFIG_BASE(socket_id)))
#ifdef COMPILE_UT_TEST
#define SYSTEM_CONFIG_VIR ((struct aicpu_system_config *)(SYSTEM_CONFIG_BASE))
#define SYSTEM_CONFIG_MULTI_VIR(socket_id) ((struct aicpu_system_config *)(SYSTEM_CONFIG_BASE))
#else
#define SYSTEM_CONFIG_VIR ((struct aicpu_system_config *)(0xffff000000000000ULL + SYSTEM_CONFIG_BASE))
#define SYSTEM_CONFIG_MULTI_VIR(socket_id) \
    ((struct aicpu_system_config *)(uintptr_t)(0xffff000000000000ULL + SYSTEM_MULTI_CONFIG_BASE(socket_id)))
#endif

/* ����������ռ�,ռ��1K,ÿ��CPUռsizeof(struct aicpu_cmd_input) */
#define AICPU_CMD_INPUT_BASE(socket_id) (SYSTEM_MULTI_CONFIG_BASE(socket_id) + (6 * 1024))
#define AICPU_CMD_PARA_NUM 5 /* max param num */
#define CMD_INPUT_BASE AICPU_CMD_INPUT_BASE(0)

/* ��������ṹ�� */
struct aicpu_cmd_input {
    unsigned int cmd_id;
    unsigned long long param[AICPU_CMD_PARA_NUM]; /* input param, max equal 5 */
    unsigned long long ret;                       /* return value */
};

/* EL0������Ϣ */
struct aicpu_config_el0 {
    unsigned int cpu_tick_freq;
    unsigned int cpu_id;
    unsigned long long boot_addr;
    void *printfw_ptr;
    void *fprintfw_ptr;
    void *func_time_cap_ptr;
    void *malloc_ptr;
};

#ifdef AICPU_HIBENCH_SLT
typedef int (*PROCPTR)(int arg0, int arg1);

struct aicpu_avs_config {
    volatile int status;   /* return value 0: OK */
    volatile PROCPTR proc; /* function vector entry */
    volatile int argcount;
    volatile int arg0;
    volatile int arg1;
    volatile int arg2;
};

#define AICPU_VECTORS_CONFIG_PHY ((struct aicpu_avs_config *)(SYSTEM_CONFIG_BASE + 724 * 1024))
#define AICPU_VECTORS_CONFIG_PA (SYSTEM_CONFIG_BASE + 724 * 1024)
#define AICPU_CFG_SIZE (8 * 1024)
#endif

typedef enum {
    PROF_CHL_CMD_DISABLE = 0,
    PROF_CHL_CMD_ENABLE = 1,
    PROF_CHL_CMD_MAX,
} t_prof_cmd;

typedef enum {
    PROF_TYPE_TASK = 0,
    PROF_TYPE_SAMPLE = 1,
    PROF_TYPE_FUNC_TIME = 2,
    PROF_TYPE_MAX,
} t_prof_type;

struct hot_spot_function_element {
    unsigned int element_head;      /* �̶�У��ͷ ʹ��ʱ�ж��Ƿ�Ϊ0x6a6a6a6a */
    unsigned int cpu_id;            /* aicpu id�� */
    char func_name[FUNC_NAME_SIZE]; /* �ȵ㺯���� FUNC_NAME_SIZE = 36 change to 64 */
    unsigned long long start_time;  /* �ȵ㺯����ʼʱ�����λns */
    unsigned long long end_time;    /* �ȵ㺯������ʱ�����λns  */
};

struct aicpu_profile_func_element {
    unsigned short tag;
    unsigned short size; /* �ṹ�����tag��size����Ĵ�С */
    unsigned short sub_tag;
    unsigned short stream_id;
    unsigned short task_id;
    unsigned short block_id;
    unsigned short block_num;
    unsigned short cpu_id; /* aicpu id�� */
    char func_name[FUNC_NAME_SIZE];
    unsigned long long entry_time_stamp;
    unsigned long long return_time_stamp;
};
/* profile buff �ṹ */
struct aicpu_profile_element {
    /* �̶�У��ͷ ʹ��ʱ�ж����Ƿ�Ϊ 0x5a5a5a5a */
    unsigned int element_head;
    unsigned int cpu_id; /* aicpu id�� */
    unsigned int counter_num;
    unsigned int reserved;         /* used for .aligment 3 */
    unsigned long long time_stamp; /* ϵͳʱ��� */
    unsigned long long event_count[AI_CPU_MAX_EVENT_NUM];
};

struct aicpu_profile_task_element {
    unsigned short tag;
    unsigned short size; /* �ṹ�����tag��size����Ĵ�С */
    unsigned short sub_tag;
    unsigned short stream_id;
    unsigned short task_id;
    unsigned short block_id;
    unsigned short block_num;
    unsigned short cpu_id; /* aicpu id�� */
    unsigned short counter_num;
    unsigned short reserved[3];
    unsigned long long entry_time_stamp;
    unsigned long long return_time_stamp;
    unsigned long long event_count[AI_CPU_MAX_EVENT_NUM];
};

/* print queue */
#ifdef LITE_MEM_LAYOUT
#define MAXQSIZE (8176)
#else
#define MAXQSIZE (482)
#endif

#define PRINT_FLAG1 (0x5a5a5a5aUL)
#define PRINT_FLAG2 (0x96)
#define PRINT_FLAG3 (0xa5a5a5a5UL)
struct print_queue {
    volatile unsigned int flag1;
    volatile unsigned int front;
    volatile unsigned int rear;
    volatile unsigned char is_full;
    volatile unsigned char flag2;
    volatile unsigned char queuedate[MAXQSIZE];
};

/* chip_id */
#define AICPU_TYPE_CLOUD (1980)
#define AICPU_TYPE_MINI (1910)
#define AICPU_TYPE_KIRIN_990_ES (369005)
#define AICPU_TYPE_KIRIN_990 (369006)
#define AICPU_TYPE_ORLANDO (6280)

#define AICPU_TYPE_FPGA 3
#define AICPU_TYPE_EMU 4
#define AICPU_TYPE_AISC 5
#define AICPU_TYPE_ESL 6

/* AICPU's bbox phy addr and size */
#define AICPU_BBOX_ADDR (SYSTEM_CONFIG_MULTI_PHY->aicpu_blackbox_base)
#define AICPU_BBOX_SIZE (SYSTEM_CONFIG_MULTI_PHY->aicpu_blackbox_size)

#endif

#endif
