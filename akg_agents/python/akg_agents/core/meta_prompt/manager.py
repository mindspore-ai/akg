import logging
import math
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class HardwareProfile:
    name: str = "Generic"
    has_tensor_cores: bool = False

@dataclass
class ParameterSpace:
    """描述一个可调参数及其计算语义。

    该层仅定义参数语义与硬性约束，不直接给出具体候选值。
    后端可依据算子形状/目标平台/资源模型动态生成候选集合。

    constraints 描述参数的硬性约束（如 "2的幂次"、">= 16"），
    后端在生成候选值时必须满足这些约束。

    min_candidates / max_candidates 用于约束后端为该参数生成的候选个数范围，
    便于 Search Agent 控制搜索空间规模。
    """
    name: str
    description: str
    typical_values: List[Union[int, str]] = field(default_factory=list)
    constraints: str = ""  # 硬性约束，如 "2的幂次"、">= 16"、"整数"
    min_candidates: int = 2
    max_candidates: int = 8

@dataclass
class MetaPrompt:
    id: str
    category: str
    description: str
    architectural_intent: str
    implementation_logic: str
    # 语义层：后端无关，仅描述策略目标与约束；为空时回退到 implementation_logic。
    semantic_logic: str = ""
    # 实现层：按 realization mode 分发（如 declarative / explicit）。
    realization_by_mode: Dict[str, str] = field(default_factory=dict)
    # 禁止模式：按 realization mode 给出负向约束，抑制错误代码形态。
    forbidden_patterns_by_mode: Dict[str, List[str]] = field(default_factory=dict)
    implementation_type: str = "Manual"
    incompatible_with: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    parameter_space: List[ParameterSpace] = field(default_factory=list)
    priority: int = 10
    

    

class MetaPromptManager:
    def __init__(self, arch: Optional[str] = None, dsl: Optional[str] = "triton"):
        self.prompts: Dict[str, MetaPrompt] = {}
        self.arch = arch
        self.dsl = dsl
        self.realization_mode = self._resolve_realization_mode()
        self._initialize_curated_space()

    def _resolve_realization_mode(self) -> str:
        """解析当前后端实现模式。

        - declarative: 高级 DSL（如 Triton），主要通过编译期配置表达策略。
        - explicit: 低层实现（如 CUDA/C++/IR），允许显式并行与同步细节。
        """
        dsl_key = (self.dsl or "").lower()
        if dsl_key:
            if dsl_key in {"triton", "triton_cuda", "triton_ascend"} or "triton" in dsl_key:
                return "declarative"
            return "explicit"

        arch_key = (self.arch or "default").lower()
        if arch_key in {"a100", "cuda", "nvidia", "triton"}:
            return "declarative"
        return "explicit"

    def _get_semantic_logic(self, prompt: MetaPrompt) -> str:
        return (prompt.semantic_logic or prompt.implementation_logic).strip()

    def _get_realization_logic(self, prompt: MetaPrompt) -> str:
        if prompt.realization_by_mode:
            if self.realization_mode in prompt.realization_by_mode:
                return prompt.realization_by_mode[self.realization_mode].strip()
            if "default" in prompt.realization_by_mode:
                return prompt.realization_by_mode["default"].strip()
        return prompt.implementation_logic.strip()

    def _get_forbidden_patterns(self, prompt: MetaPrompt) -> List[str]:
        if prompt.forbidden_patterns_by_mode:
            if self.realization_mode in prompt.forbidden_patterns_by_mode:
                return prompt.forbidden_patterns_by_mode[self.realization_mode]
            if "default" in prompt.forbidden_patterns_by_mode:
                return prompt.forbidden_patterns_by_mode["default"]
        return []

    def render_prompt_logic(self, prompt: MetaPrompt) -> str:
        """将策略按“语义层 + 实现层 + 禁止模式”渲染为文本。"""
        semantic_logic = self._get_semantic_logic(prompt)
        realization_logic = self._get_realization_logic(prompt)
        forbidden_patterns = self._get_forbidden_patterns(prompt)

        sections: List[str] = []
        if semantic_logic:
            sections.append(f"[语义层]\n{semantic_logic}")

        if realization_logic and realization_logic != semantic_logic:
            sections.append(f"[实现层:{self.realization_mode}]\n{realization_logic}")

        if forbidden_patterns:
            forbidden_text = "\n".join(f"- {item}" for item in forbidden_patterns)
            sections.append(f"[禁止模式:{self.realization_mode}]\n{forbidden_text}")

        return "\n".join(sections) if sections else "None"

    def _initialize_curated_space(self):
        arch_key = (self.arch or "default").lower()
        # 仅在明确支持声明式 async pipeline 的后端暴露复杂策略，避免生成正确性下降。
        enable_declarative_async_pipeline = arch_key in {"a100", "cuda", "nvidia", "triton"}
        # Ascend 后端不暴露 num_stages/num_warps 类编译期调参项。
        disable_pipeline_stage_tuning = arch_key in {"910b", "ascend", "ascend910b4", "cann"}
        
        print("======================================================")
        print(
            f"MetaPromptManager initialized with arch='{self.arch}', dsl='{self.dsl}', "
            f"realization_mode='{self.realization_mode}': "
            f"enable_declarative_async_pipeline={enable_declarative_async_pipeline}, "
            f"disable_pipeline_stage_tuning={disable_pipeline_stage_tuning}"
        )
        print("======================================================")
        
        # 1. 基础分块 (Tiling)
        # 1.1 矩阵加速单元专用分块
        self.add_prompt(MetaPrompt(
            id="strat_tiling_2d_block_ptr",
            category="data_partition",
            description="基于 2D Block Pointer 的层级分块 (Tensor Core/Cube Optimized)",
            architectural_intent="针对具备矩阵乘加速单元（如 Tensor Core / Cube Unit）的硬件架构。通过层级分块确保数据在片上缓存（Shared Memory / L1 Buffer）中的对齐与布局，以满足硬件对矩阵运算的高带宽与访存约束要求。",
            implementation_logic="""- 采用 M/N/K 层级分块组织计算与访存，确保计算与数据切片粒度一致
- 使用二维块级访存语义加载输入矩阵，避免退化为线性偏移访问
- 缩约轴按固定块深推进，保持流水阶段的稳定吞吐
- 输出采用延迟写回策略，优先在片上累积后统一落盘""",
            semantic_logic="""- 目标: 面向矩阵核心算子构建稳定的 M/N/K 层级分块
- 保证分块粒度、访存切片与计算推进节奏一致，兼顾吞吐与正确性
- 在不改变算子数学语义前提下提升矩阵计算利用率""",
            realization_by_mode={
                "declarative": """- 使用 DSL 的 block pointer / tile 原语表达二维分块加载与写回
- 将 BLOCK_SIZE_M/N/K 作为编译期参数或 autotune 配置
- 依赖编译器生成向量化访存与矩阵核心调度""",
                "explicit": """- 显式实现 tile 索引、片上缓存布局与矩阵核心计算映射
- 显式控制共享存储读写、同步与尾块掩码处理
- 明确累加与延迟写回路径，避免访存竞争""",
            },
            forbidden_patterns_by_mode={
                "declarative": [
                    "禁止用手写指针算术替代 block 级访存原语",
                    "禁止在主计算循环中硬编码底层同步链路",
                ],
                "explicit": [
                    "禁止忽略对齐约束导致未定义访存行为",
                    "禁止无边界检查的尾块读写",
                ],
            },
            implementation_type="Hybrid",
            tags=["gemm", "conv", "heavy_compute", "tensor_core", "cube"],
            incompatible_with=["strat_tiling_general", "strat_mem_layout_coalescing"],
            parameter_space=[
                ParameterSpace("BLOCK_SIZE_M", "行分块高度", constraints="2的幂次，>= 16", min_candidates=4, max_candidates=4),
                ParameterSpace("BLOCK_SIZE_N", "列分块宽度", constraints="2的幂次，>= 16", min_candidates=4, max_candidates=4),
                ParameterSpace("BLOCK_SIZE_K", "收缩维深度", constraints="2的幂次，>= 16", min_candidates=4, max_candidates=4),
            ],
            priority=50,
        ))

        # 1.2 通用分块 (General Tiling)
        self.add_prompt(MetaPrompt(
            id="strat_tiling_general",
            category="data_partition",
            description="通用线性/向量化分块 (General Purpose)",
            architectural_intent="针对逐元素（Elementwise）、归约（Reduction）或非对齐张量。通过线性或多维分块映射任务空间，最大化内存带宽利用率而非计算密度。",
            implementation_logic="""- 采用线性或多维分块映射任务空间，匹配硬件线程/核的并行度
- 以地址连续访问为优先，保证访存合并（Coalescing）或向量化读取
- 对尾块与边界区域使用统一处理策略，确保正确性""",
            semantic_logic="""- 目标: 为通用算子提供鲁棒分块与边界处理框架
- 优先保障地址连续访问与任务映射均衡，提高带宽利用率
- 在形状不规则场景保持可预测性能与正确性""",
            realization_by_mode={
                "declarative": """- 通过 DSL 块映射与掩码语义表达分块与边界处理
- 将 BLOCK_SIZE 作为编译期参数接入 autotune
- 依赖编译器完成向量化和访存合并优化""",
                "explicit": """- 显式实现线性/多维索引映射与尾块处理逻辑
- 显式控制向量加载宽度、步长和对齐策略
- 对非连续布局增加中转重排并给出边界掩码""",
            },
            forbidden_patterns_by_mode={
                "declarative": [
                    "禁止将结构参数改为运行时分支控制内核结构",
                    "禁止为追求局部优化而破坏统一边界处理语义",
                ],
                "explicit": [
                    "禁止省略尾块边界检查",
                    "禁止无对齐保障的激进向量化访存",
                ],
            },
            implementation_type="Hybrid",
            tags=["elementwise", "reduction", "general", "mem_bound"],
            incompatible_with=["strat_tiling_2d_block_ptr"],
            parameter_space=[
                ParameterSpace("BLOCK_SIZE", "通用分块大小", constraints="2的幂次，>= 64", min_candidates=4, max_candidates=4),
            ],
            priority=45,
        ))

        # 2. 并行增强 (Split-K)
        self.add_prompt(MetaPrompt(
            id="strat_parallel_split_k",
            category="parallelism",
            description="归约轴并行 (Split-K)",
            architectural_intent="当算子输出网格不足以填满硬件所有计算单元时，对缩约维进行物理切分。牺牲部分同步/原子操作开销，换取核心占用率（Occupancy）的整体提升。",
            implementation_logic="""- 在缩约轴引入额外并行切分维度，增加可调度任务数
- 各分片独立计算局部部分和，再通过片上归约或原子操作汇总
- 在并行增益与同步开销之间进行联合权衡""",
            semantic_logic="""- 目标: 通过缩约维切分提升任务并行度与硬件占用率
- 保持局部归约与最终汇总的可组合性，避免数值语义偏差
- 在同步/原子开销与吞吐收益之间进行可控权衡""",
            realization_by_mode={
                "declarative": """- 通过 grid 维度和编译期参数表达 SPLIT_K 切分
- 使用高层归约语义或二段式聚合接口表达汇总流程
- 让编译器选择合适的并行归并路径""",
                "explicit": """- 显式实现 K 维切片、局部部分和与最终聚合阶段
- 根据冲突模型选择原子写回或分阶段归并内核
- 显式设置同步点与汇总顺序，保证结果一致性""",
            },
            forbidden_patterns_by_mode={
                "declarative": [
                    "禁止在业务代码中拼接复杂手工归并同步链",
                    "禁止忽略分片结果汇总的确定性语义",
                ],
                "explicit": [
                    "禁止未同步即跨分片读取部分和",
                    "禁止原子竞争过高且无回退策略",
                ],
            },
            implementation_type="Hybrid",
            tags=["reduction", "low_parallelism"],
            incompatible_with=["strat_parallel_exclusive_write", "strat_persistent_kernel"],
            parameter_space=[ParameterSpace("SPLIT_K", "K 轴并行切分数", constraints="正整数，典型 2~16", min_candidates=4, max_candidates=4)],
            priority=40,
        ))

        # 3. 空间独占 (Exclusive Write)
        self.add_prompt(MetaPrompt(
            id="strat_parallel_exclusive_write",
            category="parallelism",
            description="空间维度独占写入 (Zero-Synchronization)",
            architectural_intent="针对卷积或大规模逐点运算。确保每个处理单元独占输出张量的一个独立切片，完全消除原子操作或锁竞争，达到硬件理论带宽峰值。",
            implementation_logic="""- 将输出空间划分为互斥切片，保证写路径无冲突
- 优先保障带宽利用率，适用于输出规模远大于计算开销的算子""",
            semantic_logic="""- 目标: 通过输出切片独占消除写冲突与同步开销
- 维持输出空间覆盖完整且互斥，保证结果可复现
- 适配大输出规模场景下的带宽上限优化""",
            realization_by_mode={
                "declarative": """- 用 program 映射直接声明输出切片所有权
- 保持每个并行单元只写唯一输出区域
- 通过参数化切片粒度平衡并行度与缓存命中""",
                "explicit": """- 显式构造输出切片到线程块/线程组的独占映射
- 显式验证写路径无重叠并避免原子写回
- 对边界切片补充掩码与回写顺序控制""",
            },
            forbidden_patterns_by_mode={
                "declarative": [
                    "禁止在同一输出区域引入多 writer",
                    "禁止为局部便捷加入隐式原子更新",
                ],
                "explicit": [
                    "禁止切片映射重叠导致写冲突",
                    "禁止边界块遗漏导致输出空洞",
                ],
            },
            implementation_type="Hybrid",
            tags=["pointwise", "large_spatial", "conv_spatial"],
            incompatible_with=["strat_parallel_split_k"],
            priority=30,
        ))

        # 4. 软件流水线 (Pipelining / Double Buffering)
        self.add_prompt(MetaPrompt(
            id="strat_pipeline_overlap",
            category="pipeline",
            description="基础访存-计算流水重叠 (Pipelining)",
            architectural_intent="利用硬件的异步执行特性，通过在全局内存与计算单元之间建立缓冲级数，隐藏访存延迟，提升计算单元的执行效率。",
            implementation_logic="""- 启用多级缓冲（如前瞻加载/预取）实现访存与计算的逻辑并行
- 适用于大多数访存受限或计算密集型算子，旨在屏蔽长延迟的全局内存读取
- 在简单的逐点运算中优先考虑单层分块与低级数缓冲""",
            semantic_logic="""- 目标: 建立稳定的访存-计算重叠以隐藏长延迟
- 保持阶段推进顺序一致，确保数据生产消费关系正确
- 在收益与资源占用之间选择合适流水深度""",
            realization_by_mode={
                "declarative": """- 通过 num_stages 等编译期参数声明重叠深度
- 保持主循环简洁，不手动编码缓冲轮转细节
- 依赖编译器自动安排预取与计算交叠""",
                "explicit": """- 显式实现双/多缓冲与阶段推进
- 显式管理 async copy、barrier 与跨阶段依赖
- 显式给出阶段边界与最终写回时序""",
            },
            forbidden_patterns_by_mode={
                "declarative": [
                    "禁止将 declarative 流水线退化为手写缓冲轮转代码",
                    "禁止在业务循环内插入无必要的显式同步链",
                ],
                "explicit": [
                    "禁止跨阶段读写无序导致脏数据消费",
                    "禁止省略同步点导致竞态",
                ],
            },
            implementation_type="Hybrid",
            tags=["heavy_compute", "latency_sensitive"],
            incompatible_with=["strat_async_multi_pipelining"],
            parameter_space=[] if disable_pipeline_stage_tuning else [
                ParameterSpace("num_stages", "流水线级数/缓冲深度", constraints="正整数，典型 2~3", min_candidates=2, max_candidates=2)
            ],
            priority=45,
        ))

        # 5. 访存对齐 (Memory Coalescing)
        self.add_prompt(MetaPrompt(
            id="strat_mem_layout_coalescing",
            category="memory",
            description="连续访存对齐 (Coalesced / Vectorized Access)",
            architectural_intent="确保相邻并行线程访问连续内存地址（Stride=1），将离散请求合并为更大宽度的访存事务（Transaction），最小化 DRAM 访问频次。",
            implementation_logic="""- 优先将线程映射到张量的最内层维度，提升合并效率
- 对非连续布局考虑片上重排，避免跨步读取（Strided Access）""",
            semantic_logic="""- 目标: 以连续对齐访问提升内存事务合并效率
- 优先消除跨步访问导致的带宽浪费
- 在不改变计算语义下优化读写布局""",
            realization_by_mode={
                "declarative": """- 通过高层布局、stride 与向量类型声明连续访存
- 使用掩码访问保证边界安全
- 依赖编译器完成事务合并与向量化选型""",
                "explicit": """- 显式设计线程到地址的映射确保 stride=1
- 必要时显式执行片上转置/重排以提升合并访问
- 显式控制向量宽度、对齐和边界掩码""",
            },
            forbidden_patterns_by_mode={
                "declarative": [
                    "禁止在高层布局已声明后再手写低层地址重排",
                    "禁止忽略边界掩码导致越界访问",
                ],
                "explicit": [
                    "禁止无对齐保证的向量加载/存储",
                    "禁止跨步访问未做中转优化仍强行向量化",
                ],
            },
            implementation_type="Hybrid",
            tags=["mem_bound", "general"],
            incompatible_with=["strat_tiling_2d_block_ptr"],
            priority=20,
        ))

        # 6. 持久化内核 (Persistent Kernel)
        self.add_prompt(MetaPrompt(
            id="strat_persistent_kernel",
            category="parallelism",
            description="持久化线程驻留 (Persistent Kernel)",
            architectural_intent="当任务总网格数不足以填满硬件所有计算单元，或单个任务极短导致启动开销占比过高时。通过启动固定数量的工作单元长期驻留并自行分发任务，消除多次启动开销与尾部气泡。",
            implementation_logic="""- 采用固定规模的驻留工作单元执行循环任务拉取
- 通过统一的任务索引分发机制降低启动延迟
- 适用于小规模数据或延迟敏感型实时计算场景""",
            semantic_logic="""- 目标: 以驻留执行单元复用执行上下文，降低频繁启动开销
- 保持任务拉取与分发的有序性，避免负载失衡
- 在低网格规模场景提升端到端时延表现""",
            realization_by_mode={
                "declarative": """- 通过 program 循环和编译期参数表达驻留处理模型
- 将任务批次大小与拉取步长作为调参维度
- 保持任务分发语义简单，交由编译器优化执行细节""",
                "explicit": """- 显式实现工作队列/索引分发与驻留循环
- 显式控制任务拉取原子操作与退出条件
- 显式处理尾任务与负载不均场景""",
            },
            forbidden_patterns_by_mode={
                "declarative": [
                    "禁止在业务逻辑中引入复杂手写任务调度状态机",
                    "禁止将驻留模型参数化为不可预测的运行时结构分支",
                ],
                "explicit": [
                    "禁止任务索引分配无同步保护",
                    "禁止缺失退出条件导致死循环或饥饿",
                ],
            },
            implementation_type="Hybrid",
            tags=["small_grid", "latency_sensitive"],
            incompatible_with=["strat_parallel_split_k"],
            priority=35,
        ))

        # 7. 栅格化 (Rasterization / Swizzle)
        self.add_prompt(MetaPrompt(
            id="strat_swizzle_rasterization",
            category="scheduling",
            description="L2 缓存局部性优化 (Block Swizzling)",
            architectural_intent="通过重新映射逻辑网格到物理计算单元的映射顺序（如 Tile Grouping），提升相邻处理单元在 L2 Cache 上的数据复用率，降低全局内存带宽压力。",
            implementation_logic="""- 对任务块调度顺序进行重映射以提升邻近访问局部性
- 常用策略包括 Grouping 或特定曲线排布（如 Hilbert）
- 目标是在维持负载均衡的前提下最大化缓存命中率""",
            semantic_logic="""- 目标: 通过网格重排提升缓存局部性并降低带宽压力
- 在局部性提升与负载均衡之间保持稳定折中
- 不改变算子计算语义，仅调整调度访问顺序""",
            realization_by_mode={
                "declarative": """- 用 program id 重映射函数声明 swizzle/grouping 策略
- 将 GROUP_SIZE_M 等参数纳入编译期配置
- 依赖编译器在给定映射下完成执行细化""",
                "explicit": """- 显式实现逻辑块到物理执行块的映射函数
- 显式处理边界 tile、尾块与分组过渡
- 显式验证缓存复用收益与负载均衡""",
            },
            forbidden_patterns_by_mode={
                "declarative": [
                    "禁止硬编码固定 swizzle 路径导致形状泛化失效",
                    "禁止在调度表达中混入与计算语义耦合的分支",
                ],
                "explicit": [
                    "禁止映射函数不连续导致严重负载倾斜",
                    "禁止边界块映射遗漏",
                ],
            },
            implementation_type="Hybrid",
            tags=["scheduling", "cache_opt"],
            parameter_space=[ParameterSpace("GROUP_SIZE_M", "Swizzle 分组大小", constraints="正整数，典型 4~16", min_candidates=3, max_candidates=3)],
            priority=25,
        ))

        # ======================================================================
        # Level2: 融合算子专用策略 (Fused Operator Strategies)
        # ======================================================================

        # 8. Epilogue 融合 (计算 + 逐点后处理)
        self.add_prompt(MetaPrompt(
            id="strat_fusion_epilogue",
            category="fusion",
            description="主计算 + Epilogue 逐点融合 (Register-Level Fusion)",
            architectural_intent="将计算密集型算子（GEMM/Conv）的结果直接在寄存器层面衔接后续的逐点操作，避免中间结果写回全局内存。适用于 Linear+Activation 等常见融合模式。",
            implementation_logic="""- 将主计算输出在片上直接衔接后处理算子，减少中间落盘
- 控制融合深度，平衡访存节省与寄存器压力
- 最终结果统一写回输出以降低冗余事务""",
            semantic_logic="""- 目标: 将主计算结果直接衔接逐点后处理，减少中间张量落盘
- 保持主计算与 epilogue 的数据依赖顺序与数值语义一致
- 在访存收益与寄存器压力之间做可控权衡""",
            realization_by_mode={
                "declarative": """- 在单 kernel 语义内声明主计算与 epilogue 链路
- 将 EPILOGUE_OPS 作为编译期组合配置
- 依赖编译器完成寄存器级衔接与指令调度""",
                "explicit": """- 显式将主计算累加器结果在寄存器内传递给后处理步骤
- 显式管理融合顺序、寄存器占用和最终写回路径
- 对融合链路中的边界/掩码执行显式控制""",
            },
            forbidden_patterns_by_mode={
                "declarative": [
                    "禁止将 epilogue 融合拆成多次中间落盘",
                    "禁止在声明式融合中引入手写底层缓存交换细节",
                ],
                "explicit": [
                    "禁止融合后寄存器生命周期失控导致溢出",
                    "禁止改变 epilogue 算子顺序导致语义偏差",
                ],
            },
            implementation_type="Hybrid",
            tags=["fusion", "epilogue", "gemm", "heavy_compute", "mem_bound"],
            incompatible_with=[],
            parameter_space=[
                ParameterSpace("EPILOGUE_OPS", "Epilogue 变换列表", constraints="算子枚举列表", min_candidates=2, max_candidates=6),
            ],
            priority=55,
        ))

        # 9. 纵向生产-消费者融合
        self.add_prompt(MetaPrompt(
            id="strat_fusion_producer_consumer",
            category="fusion",
            description="纵向生产-消费者缓冲融合 (Vertical Buffer Fusion)",
            architectural_intent="针对存在数据依赖的算子链，将中间张量缓存在片上局部存储（Shared Memory / L1 Buffer）中直接传递，完全消除中间数据的全局内存往返开销。",
            implementation_logic="""- 识别具备直接数据依赖的算子并进行纵向合并
- 中间结果优先通过片上缓冲区传递，减少全局内存寻址
- 在存储开销与链路深度之间进行折中""",
            semantic_logic="""- 目标: 在生产者与消费者间建立片上直通数据链路
- 最小化中间张量全局内存往返并保持依赖顺序正确
- 在融合深度与片上存储压力之间平衡""",
            realization_by_mode={
                "declarative": """- 以单 kernel 数据流语义声明 producer-consumer 链
- 将 INTER_BUFFER_SIZE/FUSION_STAGES 作为编译期调参维度
- 依赖编译器安排中间值驻留与调度""",
                "explicit": """- 显式分配并管理片上中间缓冲区与生命周期
- 显式定义生产、消费阶段边界与同步点
- 显式处理融合链尾部写回与资源回收""",
            },
            forbidden_patterns_by_mode={
                "declarative": [
                    "禁止把可直通链路退化为中间全局落盘",
                    "禁止在声明式融合中引入复杂手写缓冲管理状态机",
                ],
                "explicit": [
                    "禁止中间缓冲生命周期泄漏导致覆盖或脏读",
                    "禁止跨阶段依赖未同步就消费",
                ],
            },
            implementation_type="Hybrid",
            tags=["fusion", "producer_consumer", "mem_bound", "latency_sensitive"],
            incompatible_with=["strat_parallel_split_k"],
            parameter_space=[
                ParameterSpace("INTER_BUFFER_SIZE", "中间缓冲区大小", constraints="受片上存储容量限制", min_candidates=4, max_candidates=4),
                ParameterSpace("FUSION_STAGES", "融合算子链深度", constraints="正整数，>= 2", min_candidates=3, max_candidates=3),
            ],
            priority=50,
        ))

        # 10. 在线归约融合 (Online Reduction)
        self.add_prompt(MetaPrompt(
            id="strat_fusion_online_reduction",
            category="fusion",
            description="在线归约单遍融合 (Online Reduction)",
            architectural_intent="针对需要多次遍历数据的归约算子（如 Softmax/LayerNorm），采用流式统计算法在单次遍历中完成统计与归一化，将带宽需求降低约 50%。",
            implementation_logic="""- 采用单遍扫描算法（如 Online Softmax）代替传统的“统计-写回-归一”流程
- 在分块推进中持续更新局部统计量
- 确保数值稳定性与计算精度的平衡""",
            semantic_logic="""- 目标: 以单遍流式统计完成归约与归一化，减少重复访存
- 在分块推进中维护稳定的在线统计状态
- 保持数值稳定性与精度一致性""",
            realization_by_mode={
                "declarative": """- 使用高层在线归约语义表达统计与归一化链路
- 将 REDUCTION_BLOCK/ONLINE_ALG 作为编译期配置
- 让编译器在单遍框架下优化访存与计算排布""",
                "explicit": """- 显式实现单遍在线统计更新（max/sum 等）与归一化计算
- 显式控制分块归约顺序、稳定项更新和边界掩码
- 显式验证与基线算法的一致性与精度""",
            },
            forbidden_patterns_by_mode={
                "declarative": [
                    "禁止退化回多遍统计-写回-归一流程",
                    "禁止忽略在线算法的数值稳定项",
                ],
                "explicit": [
                    "禁止在线统计更新顺序错误导致数值不稳定",
                    "禁止精度敏感路径缺少校验阈值",
                ],
            },
            implementation_type="Hybrid",
            tags=["fusion", "reduction", "softmax", "layernorm", "mem_bound", "latency_sensitive"],
            incompatible_with=["strat_parallel_split_k"],
            parameter_space=[
                ParameterSpace("REDUCTION_BLOCK", "归约轴分块大小", constraints="2的幂次，>= 64", min_candidates=4, max_candidates=4),
                ParameterSpace("ONLINE_ALG", "在线算法类型", constraints="枚举值", min_candidates=2, max_candidates=2),
            ],
            priority=52,
        ))

        # 11. 横向独立算子融合 (Horizontal Fusion)
        self.add_prompt(MetaPrompt(
            id="strat_fusion_horizontal",
            category="fusion",
            description="横向独立算子批量融合 (Horizontal Kernel Fusion)",
            architectural_intent="将多个相互独立、形状兼容的算子合并到同一次启动中，由不同的计算群组分配处理。旨在消除多次启动开销和硬件流水线空隙，提升整体利用率。",
            implementation_logic="""- 将无依赖的子算子聚合到同一执行序列
- 通过任务分发机制为不同子算子分配独立执行切片
- 严格控制子算子间的资源竞争与一致性""",
            semantic_logic="""- 目标: 将独立算子合并执行以降低多次启动开销
- 保持子算子间无依赖隔离与资源可控竞争
- 提升整体流水线填充度与设备利用率""",
            realization_by_mode={
                "declarative": """- 通过分组 program 映射声明多算子并行执行切片
- 将 N_OPS 及分组策略作为编译期配置
- 依赖编译器完成子任务调度与资源编排""",
                "explicit": """- 显式构建子算子任务表与执行分发机制
- 显式约束各子算子的寄存器/共享存储占用
- 显式处理不同子算子尾块和回写路径""",
            },
            forbidden_patterns_by_mode={
                "declarative": [
                    "禁止将存在依赖的算子错误归入横向融合",
                    "禁止把资源竞争控制留空不设限制",
                ],
                "explicit": [
                    "禁止子算子资源超配导致整体退化",
                    "禁止任务分发无隔离导致写回串扰",
                ],
            },
            implementation_type="Hybrid",
            tags=["fusion", "horizontal", "multi_op", "latency_sensitive"],
            incompatible_with=["strat_fusion_producer_consumer"],
            parameter_space=[
                ParameterSpace("N_OPS", "融合算子数量", constraints="正整数，>= 2", min_candidates=4, max_candidates=4),
            ],
            priority=42,
        ))

        # 12. 硬件加速流水化 (Declarative Asynchronous Pipelining)
        if enable_declarative_async_pipeline:
            self.add_prompt(MetaPrompt(
                id="strat_async_multi_pipelining",
                category="pipeline",
                description="基于框架特性的声明式多级流水化 (Declarative Pipelining)",
                architectural_intent="利用高级编译器或DSL（如 Triton）内置的流水线与并行度优化能力，通过配置参数驱动底层自动生成针对具备矩阵加速单元硬件的异步调度指令，掩盖访存延迟。",
                implementation_logic="""- 通过编译期参数化表达流水线深度与并行发射粒度，以隐藏访存延迟并保持稳定吞吐
- 保持数学语义与并行执行语义解耦，不改变原始算子定义
- 将策略实现交由对应后端编译器或运行时机制完成，避免与业务逻辑耦合""",
                semantic_logic="""- 通过编译期参数化表达流水线深度与并行发射粒度，以隐藏访存延迟并保持稳定吞吐
- 保持数学语义与并行执行语义解耦，不改变原始算子定义
- 将策略实现交由对应后端编译器或运行时机制完成，避免与业务逻辑耦合""",
                realization_by_mode={
                    "declarative": """- 保持最基础的单级循环访存与计算逻辑，不手写 software pipelining
- 仅通过编译期配置（如 PIPELINE_STAGES/num_stages、PARALLEL_UNITS/num_warps）表达流水化策略
- 依赖 DSL 编译器在编译时自动展开多级流水与 async copy""",
                    "explicit": """- 允许显式实现多级缓冲（double/triple buffering）与阶段推进
- 允许显式 async copy、barrier/sync、分阶段提交与消费
- 必须给出边界检查、同步点与写回顺序，避免数据竞争与越界访问""",
                },
                forbidden_patterns_by_mode={
                    "declarative": [
                        "禁止手写环形缓冲数组轮转与手动阶段索引推进",
                        "禁止在业务循环中插入显式同步屏障链模拟软件流水",
                    ],
                    "explicit": [
                        "禁止省略跨阶段同步导致的读写竞态",
                        "禁止无边界检查的异步搬运与向量化写回",
                    ],
                },
                implementation_type="Hybrid",
                tags=["heavy_compute", "latency_sensitive", "advanced_pipeline", "compiler_driven"],
                incompatible_with=["strat_tiling_general", "strat_parallel_split_k", "strat_pipeline_overlap"],
                parameter_space=[
                    ParameterSpace("PARALLEL_UNITS", "框架级并行发射单元数 (映射为 num_warps)", constraints="纯编译期配置参数，不介入业务计算或循环逻辑", min_candidates=4, max_candidates=4),
                    ParameterSpace("PIPELINE_STAGES", "框架缓冲级数 (映射为 num_stages)", constraints="纯编译期配置参数，用于驱动 Autotune，切勿手动轮转分配缓冲", min_candidates=3, max_candidates=3),
                ],
                priority=48,
            ))
    def add_prompt(self, prompt: MetaPrompt):
        """Add a MetaPrompt to the manager"""
        self.prompts[prompt.id] = prompt

    def _select_prompt_ids(self, op_features: Optional[Dict[str, Any]] = None) -> List[str]:
        """基于 op_features 预筛候选策略 ID；若无特征则返回全量空间。"""
        if not op_features:
            return list(self.prompts.keys())

        selected: set = set()

        is_compute_heavy = bool(op_features.get("is_compute_heavy", False))
        is_reduction = bool(op_features.get("is_reduction", False))
        has_tensor_cores = bool(op_features.get("has_tensor_cores", False))
        has_epilogue = bool(op_features.get("has_epilogue", False))
        has_online_reduction = bool(op_features.get("has_online_reduction", False))
        fusion_depth = int(op_features.get("fusion_depth", 1) or 1)
        n_independent_ops = int(op_features.get("n_independent_ops", 1) or 1)
        total_output = int(op_features.get("total_output", 1) or 1)

        # 基础策略：先保证分块策略可选
        if is_compute_heavy and has_tensor_cores:
            selected.add("strat_tiling_2d_block_ptr")
            selected.add("strat_pipeline_overlap")
        else:
            selected.add("strat_tiling_general")
            selected.add("strat_mem_layout_coalescing")

        # 缩约类策略
        if is_reduction:
            selected.add("strat_parallel_split_k")

        # 非缩约且输出规模大，偏向空间独占写路径
        if (not is_reduction) and total_output >= 1_000_000:
            selected.add("strat_parallel_exclusive_write")

        # 小规模输出更可能受 launch/调度开销影响
        if total_output <= 256 * 256:
            selected.add("strat_persistent_kernel")

        # 融合相关策略
        if has_epilogue:
            selected.add("strat_fusion_epilogue")
        if has_online_reduction:
            selected.add("strat_fusion_online_reduction")

        # 安全兜底：未检测到在线归约特征时，禁止该策略进入候选池
        if not has_online_reduction:
            selected.discard("strat_fusion_online_reduction")
        if fusion_depth > 1:
            selected.add("strat_fusion_producer_consumer")
        if n_independent_ops > 1:
            selected.add("strat_fusion_horizontal")

        # 计算密集型常见的缓存/调度收益策略
        if is_compute_heavy:
            selected.add("strat_swizzle_rasterization")

        # 删除不存在的策略 ID
        selected = {pid for pid in selected if pid in self.prompts}

        # 按优先级和互斥关系做收敛，避免把互斥策略同时交给 LLM
        sorted_candidates = sorted(
            selected,
            key=lambda pid: self.prompts[pid].priority,
            reverse=True,
        )
        final_selected: List[str] = []
        final_set: set = set()
        for pid in sorted_candidates:
            incompatible = set(self.prompts[pid].incompatible_with)
            if incompatible & final_set:
                continue
            final_selected.append(pid)
            final_set.add(pid)

        # 兜底：至少给出若干常规策略，避免预筛过窄
        if not final_selected:
            for default_pid in [
                "strat_tiling_general",
                "strat_mem_layout_coalescing",
                "strat_pipeline_overlap",
            ]:
                if default_pid in self.prompts:
                    final_selected.append(default_pid)

        return final_selected

    def get_selected_prompt_ids(self, op_features: Optional[Dict[str, Any]] = None) -> List[str]:
        """对外暴露预筛后的候选 ID，便于日志和调试。"""
        return self._select_prompt_ids(op_features)

    def get_prompt_space_str(self, op_features: Optional[Dict[str, Any]] = None) -> str:
        """将内存中的元提示空间格式化为 Prompt 字符串，供 LLM 选择策略。

        返回内容不包含任何硬件具体信息（硬件无关），仅描述计算语义与优化意图。
        参数层面仅输出语义、约束与候选个数范围，不输出具体候选值。
        """
        res = []
        selected_ids = self._select_prompt_ids(op_features)
        for pid in selected_ids:
            p = self.prompts[pid]
            # 展示参数名、语义描述、约束条件、候选个数范围（不展示具体候选值）
            param_details = []
            for ps in p.parameter_space:
                entry = f"`{ps.name}` — {ps.description}"
                if ps.constraints:
                    entry += f"；约束: {ps.constraints}"
                if ps.min_candidates > 0 and ps.max_candidates >= ps.min_candidates:
                    entry += f"；候选个数: {ps.min_candidates}~{ps.max_candidates}"
                param_details.append(entry)
            params_str = "\n    ".join(param_details) if param_details else "None"

            # 处理不兼容信息
            incompatible_info = f" [互斥策略: {', '.join(p.incompatible_with)}]" if p.incompatible_with else ""

            # 组装单条元提示，严格匹配 search.j2 要求的结构
            meta_str = f"[{p.id}] ({p.category}){incompatible_info} | implementation_type: {p.implementation_type}\n"
            meta_str += f"- **核心意图**: {p.architectural_intent}\n"
            meta_str += f"- **实现逻辑**: \n{self.render_prompt_logic(p)}\n"
            meta_str += f"- **参数列表**: {params_str}"
            res.append(meta_str)

        return "\n\n".join(res)



class MetaPromptSearcher:
    """IR 层策略选择器。

    根据算子特征（包含算子语义特征和硬件能力特征）从 MetaPromptManager 中
    选出适合的策略 ID 列表。

    设计原则：
        - 算子语义特征（形状、计算类型、融合深度）是硬件无关的，直接从 op_meta 提取
        - 硬件能力特征（is_small_grid、has_tensor_cores）由 hw_profile 计算，
          用于判断选择哪些策略更适合目标硬件
        - 策略 ID 列表本身可以随 IR 传播（策略 ID 是硬件无关的符号）
    """

    DEFAULT_PROFILES = {
        "a100":    HardwareProfile(name="NVIDIA A100",   has_tensor_cores=True),
        "ascend910b4":    HardwareProfile(name="Ascend 910B",   has_tensor_cores=False),
        "default": HardwareProfile(name="Generic"),
    }

    def __init__(self, manager: MetaPromptManager, arch: str = "default"):
        self.manager = manager
        self.hw_profile = self.DEFAULT_PROFILES.get(arch.lower(), self.DEFAULT_PROFILES["default"])

    def _extract_op_features(self, op_meta: Dict[str, Any]) -> Dict[str, Any]:
        """从元数据中纯提取特征，适配 extractor_torch.py 的返回结构。
        
        Level1 特征：单算子形状/计算强度/归约类型。
        Level2 特征：融合深度/Epilogue 检测/在线归约/横向独立算子数量。
        """
        # 1. 寻找输出形状 (在 Torch FX 图中，最后一个节点通常是输出)
        out_shape = []
        graph_tensors = op_meta.get("graph_tensors", [])
        
        if graph_tensors:
            # 尝试从最后一个节点获取形状
            out_shape = graph_tensors[-1].get("shape", [])
        
        # fallback: 如果没有图张量，尝试使用第一个输入的形状（仅限简单逐点运算）
        if not out_shape and op_meta.get("inputs"):
            first_input = op_meta["inputs"][0]
            if isinstance(first_input, dict):
                out_shape = first_input.get("shape", [])

        total_output = 1
        for s in out_shape: 
            if isinstance(s, int): total_output *= s

        # 2. 算子语义推导 (计算强度与 Reduction 检测)
        is_compute_heavy = False
        is_reduction = False
        
        compute_ops = ["matmul", "gemm", "conv", "linear", "bmm", "addmm", "einsum", "dot"]
        reduction_ops = ["sum", "mean", "max", "min", "prod", "argmin", "argmax", "norm", "topk"]
        # 注意：计算型算子通常隐含了内积缩约过程，故也包含在 reduction 检测中
        reduction_ops.extend(compute_ops)

        # Level2: Epilogue 与在线归约算子集合
        epilogue_ops = ["relu", "gelu", "silu", "tanh", "sigmoid", "bias", "add",
                        "mul", "scale", "dropout", "clamp", "leaky_relu", "hardswish"]
        online_reduction_ops = ["softmax", "log_softmax", "layer_norm", "rms_norm",
                                "group_norm", "batch_norm", "normalize"]

        # 遍历图节点进行语义识别
        has_epilogue = False
        has_online_reduction = False
        compute_node_count = 0
        epilogue_node_count = 0
        online_reduction_count = 0

        # 统计独立输出分支（用于横向融合检测）
        output_roots: set = set()

        # 记录主计算节点，用于后续判断 Epilogue 是否在其之后
        main_compute_node_indices = []
        for i, node in enumerate(graph_tensors):
            target_str = str(node.get("target", "")).lower()
            
            # 1. 识别主计算节点
            is_main_compute = any(op in target_str for op in compute_ops)
            if is_main_compute:
                is_compute_heavy = True
                compute_node_count += 1
                main_compute_node_indices.append(i)
                output_roots.add(node.get("name", id(node)))

            if any(op in target_str for op in reduction_ops):
                is_reduction = True
            
            # 2. 识别在线归约算子
            if any(op in target_str for op in online_reduction_ops):
                has_online_reduction = True
                online_reduction_count += 1
                is_reduction = True

        # 3. 识别真正的 Epilogue 节点 (必须发生在至少一个主计算节点之后)
        for i, node in enumerate(graph_tensors):
            target_str = str(node.get("target", "")).lower()
            
            # 只有在主计算节点之后出现的逐点算子才算真正的 Epilogue
            is_after_main = any(i > main_idx for main_idx in main_compute_node_indices)
            
            if is_after_main and any(op in target_str for op in epilogue_ops):
                # 排除掉已经是主计算的节点
                is_main = any(op in target_str for op in compute_ops)
                if not is_main:
                    has_epilogue = True
                    epilogue_node_count += 1

        # 3. Level2: 融合深度推断
        # fusion_depth = 计算节点数 + epilogue 节点数 + 在线归约节点数
        # op_meta 中也可以直接提供 fusion_depth 字段（由外部 extractor 填充）
        fusion_depth = op_meta.get("fusion_depth",
                                   compute_node_count + epilogue_node_count + online_reduction_count)
        fusion_depth = max(fusion_depth, 1)  # 至少为 1

        # 横向独立算子数：通过 op_meta["n_independent_ops"] 或独立输出根节点数推断
        n_independent_ops = op_meta.get("n_independent_ops", len(output_roots) if len(output_roots) > 1 else 1)

        # 4. 硬件能力特征：结合 hw_profile 计算，用于判断应选哪些策略
        avg_block_size = 4096  # 默认块大小估算（元素数）
        # is_small_grid = (total_output / avg_block_size) < self.hw_profile.compute_units
        has_tensor_cores = self.hw_profile.has_tensor_cores

        return {
            # Level1 算子语义特征（硬件无关）
            "rank": len(out_shape),
            "is_matrix": len(out_shape) >= 2,
            "is_conv": len(out_shape) >= 4,
            "total_output": total_output,
            "is_compute_heavy": is_compute_heavy,
            "is_reduction": is_reduction,
            # Level2 融合特征
            "fusion_depth": fusion_depth,
            "has_epilogue": has_epilogue,
            "has_online_reduction": has_online_reduction,
            "n_independent_ops": n_independent_ops,
            # 硬件能力特征：由 hw_profile 计算，供策略选择逻辑使用
            # （策略 ID 本身不含硬件信息，硬件特征仅作为选择判断依据）
            # "is_small_grid": is_small_grid,       # Grid 规模不足以填满所有计算单元
            "has_tensor_cores": has_tensor_cores,  # 硬件是否支持 Tensor Core / MMA 加速单元
        }


    def get_strategy_glossary(self, strategy_line: str) -> str:
        """为选中的策略生成术语解释，附加当前硬件的上下文描述。"""
        hw = self.hw_profile
        res = "## 术语解释 (Strategy Glossary)\n"
        res += f"**目标硬件**: {hw.name}\n\n"
        for p_id, p in self.manager.prompts.items():
            if p_id in strategy_line:
                res += f"- **{p_id}**: {self._render_intent(p)}\n"
        return res

    def format_as_directive(
        self,
        selected_prompts: List[MetaPrompt],
        param_table: Optional[Dict[str, Any]] = None,
    ) -> str:
        """将选中的策略渲染为 LLM 设计指令。

        参数层面仅输出语义、约束、候选个数，不直接输出具体取值。

        Args:
            selected_prompts: 选出的 MetaPrompt 列表。
            param_table: 可选，{参数名: 候选值列表} 映射，仅用于统计候选个数。
        """
        param_table = param_table or {}
        hw = self.hw_profile

        res = "### 【架构规约】结构化逻辑规约 (Architecture & Optimization Routes)\n"
        res += f"**当前目标硬件**: {hw.name}\n\n"
        res += "你必须严格按照以下架构组合及其参数空间进行设计。这也是你进行 Reasoning 的基石：\n\n"

        all_params: List[ParameterSpace] = []
        for p in selected_prompts:
            res += f"#### [{p.id}] ({p.category})\n"
            res += f"- **核心意图**: {self._render_intent(p)}\n"
            res += f"- **实现逻辑**:\n{self.manager.render_prompt_logic(p)}\n"
            all_params.extend(p.parameter_space)
            res += "\n"

        if all_params:
            res += "#### 联合建议搜索空间 (Joint Decision Space):\n"
            seen_params: set = set()
            for param in all_params:
                if param.name not in seen_params:
                    seen_params.add(param.name)
                    res += f"  * `{param.name}` — {param.description}"
                    if param.constraints:
                        res += f"；约束: {param.constraints}"

                    if param.name in param_table and isinstance(param_table[param.name], list):
                        actual_count = len(param_table[param.name])
                        res += f"；后端候选个数: {actual_count}"
                    elif param.min_candidates > 0 and param.max_candidates >= param.min_candidates:
                        res += f"；候选个数范围: {param.min_candidates}~{param.max_candidates}"
                    res += "\n"
            res += "\n"

        res += "**设计师任务**: \n"
        res += "1. **架构实现**: 必须完整实现上述组合策略，不得遗漏并行逻辑或访存逻辑。\n"
        res += '2. **精细化参数决策**: 从"联合建议搜索空间"中挑选最适配当前 Shape 的参数范围，由于后续autotune搜索（如缩小 BLOCK_SIZE 的范围，用于autotune搜索）。\n'
        res += "3. **显式标注**: 在代码中使用 `@metaprompt` 标注上述 ID 的实现位置。\n"
        res += "4. **决策透明化**: 在 reasoning 字段给出你如何平衡寄存器和核心占用率的定量逻辑。\n"

        return res

    def _render_intent(self, p: MetaPrompt) -> str:
        """在硬件无关的 architectural_intent 末尾追加当前硬件上下文。"""
        hw = self.hw_profile
        hw_ctx = (
            f" [硬件上下文: {hw.name}, "
            f"TensorCore={'支持' if hw.has_tensor_cores else '不支持'}]"
        )
        return p.architectural_intent + hw_ctx
