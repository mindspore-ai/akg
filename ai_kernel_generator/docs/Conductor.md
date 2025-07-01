# Conductor 设计文档

## 概述
Conductor 是 AI Kernel Generator 中的任务指挥者组件，继承自 `AgentBase`，负责管理和协调整个任务执行流程。它通过检查各个代理的输出结果，决策下一步的执行动作，并在出现错误时进行智能分析和修复指导。

## 核心功能
- **任务流程管理**：协调Designer、Coder、Verifier的执行顺序
- **代码自检验证**：对生成的设计文档和实现代码进行质量检查，判断是否要进行回退修复
- **错误智能分析**：分析测试失败原因并提供修复建议
- **状态跟踪记录**：维护完整的任务执行轨迹
- **重试机制控制**：管理修复重试次数，避免无限循环

## 初始化参数
| 参数名称 | 类型/必选 | 参数说明 |
|---------|---------|---------|
| op_name | str (必选) | 操作名称，标识具体的算子 |
| task_id | str (必选) | 任务唯一标识符 |
| log_dir | str (必选) | 日志存储目录路径 |
| impl_type | str (必选) | 实现类型："triton" 或 "swft" |
| model_config | dict (必选) | LLM模型配置，包含conductor_check和conductor_analyze配置 |

## 执行流程 get_next_action

1. **轨迹分析阶段**
   - 获取上一步执行记录（pre_trace）
   - 根据action_type判断执行路径
   - 更新步骤计数器（step）

2. **决策执行阶段**
   - Designer/Coder完成后：执行self_check()进行代码检查
   - Verifier失败后：执行analyze_error()进行错误分析
   - 成功或达到退出条件：返回EXIT

3. **结果返回**
   - 返回三元组：(下一步动作类型, 解析的代码对象, 修复建议)

## 关键方法说明

### self_check() - 代码自检
- **功能**：检查Designer或Coder输出的代码质量
- **流程**：解析代码 → 检查重试限制 → LLM进行质量评估 → 做出是否进行修复的决策
- **输出**：下一步动作(继续/修复)和建议（如果需要执行修复时）

### analyze_error() - 错误分析  
- **功能**：分析测试失败的根本原因
- **流程**：提取最近匹配的Designer和Coder代码 → LLM分析错误日志 → 定位问题源头（Designer/Coder）
- **输出**：修复目标(Designer/Coder)和具体修复建议

### initialize_check_docs() - 文档初始化
- **功能**：基于trace.base_doc初始化各种检查模板的输入数据
- **支持**：Designer检查、Triton/SWFT Coder检查、错误分析

## 用户自定义扩展

### 扩展概述
参考AIKG项目流程图，**Conductor模块作为调度模块**，居中调控任务走向，通过`trace`中存储的信息来进行决策。用户可以自由地修改/扩展这一模块，**修改的主要入口为`get_next_action()`函数**。用户可以根据自行设计的流程图和判断条件，控制任务走向。

### 默认执行流程
以当前任务典型流程为例：

- `DO_DESIGNER` → `conductor (self_check)` → `FIX_DESIGNER` / `DO_CODER`
- `DO_CODER` → `conductor (self_check)` → `FIX_CODER` / `VERIFY`  
- `VERIFY` → `conductor (error_analyze)` → `FIX_DESIGNER` / `FIX_CODER`

通过Conductor，把各个模块连接起来，**形成一个自检循环**，智能分析任务、生成算子。
