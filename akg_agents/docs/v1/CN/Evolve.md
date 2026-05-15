# AI Kernel Generator 进化脚本使用说明

## 概述

AI Kernel Generator 提供了四个文件，用于自动生成和优化算子：

1. **`evolve.py`** - 核心进化算法实现
2. **`single_evolve_runner.py`** - 单任务进化执行器
3. **`run_batch_evolve.py`** - 批量任务进化执行器
4. **`run_torch_evolve_triton.py`** - PyTorch + Triton 进化示例脚本

相关配置见**`evolve_config.yaml`**配置文件

## 功能说明

### 1. evolve.py - 核心进化算法

evolve.py 文件实现了基于遗传算法的算子进化生成系统，支持两种运行模式：

- **岛屿模式**：通过分布式岛屿模型保持种群多样性，利用精英保留机制确保优秀个体不丢失，通过迁移机制实现岛屿间信息共享，结合分层采样机制确保进化过程的多样性和创新性
- **普通模式**：简化的进化模式，适用于快速测试和简单任务，无需复杂的岛屿管理和精英机制

两种模式都能显著提升算子生成成功率和质量，用户可根据任务复杂度选择合适的模式。

**模式选择**

系统根据参数自动选择运行模式：
- 当 `num_islands > 1` 且 `elite_size > 0` 时，启用**岛屿模式**
- 否则使用**普通模式**

**主要流程**

1. **初始化**
   - 创建存储目录
   - 根据模式配置相应参数
   - 加载meta-prompts

2. **进化循环**
   每轮执行：
   - **迁移**: 岛屿间交换精英个体（仅岛屿模式）
   - **灵感采样**: 
     - 岛屿模式：选择父代，在其所在岛屿内采样，支持精英池选择
     - 普通模式：从全局历史实现中采样
   - **任务执行**: 并行生成和测试多个实现
   - **结果处理**: 收集成功实现，生成sketch描述
   - **精英管理**: 更新精英库，保存最佳实现（仅岛屿模式）

3. **最终处理**
   - 排序所有最佳实现
   - 计算统计信息
   - 返回进化结果

**初始化参数**:
| 参数名称 | 类型/必选 | 参数说明 |
|---------|---------|---------|
| op_name | str (必选) | 算子名称 |
| task_desc | str (必选) | 任务描述 |
| dsl | str (必选) | DSL类型："triton_cuda"、"triton_ascend"、"swft"等 |
| framework | str (必选) | 前端框架："mindspore"、"torch"、"numpy"等 |
| backend | str (必选) | 后端类型："ascend"、"cuda"等 |
| arch | str (必选) | 硬件架构："a100"、"ascend910b4"等 |
| max_rounds | int (必选) | 最大进化轮数 |
| parallel_num | int (必选) | 每轮并行任务数 |
| num_islands | int (必选) | 岛屿数量 |
| migration_interval | int (必选) | 迁移间隔 |
| elite_size | int (必选) | 精英保留数量 |
| parent_selection_prob | float (必选) | 父代选择概率 |
| device_list | List[int] (必选) | 设备列表 |

#### 岛屿模式

岛屿模式是一种分布式进化算法，将进化种群分成多个独立的"岛屿"，每个岛屿独立进化，同时通过迁移机制实现岛屿间的信息互通。通过保持种群多样性、探索不同解空间区域、提供系统鲁棒性，提升算子生成的成功率和质量。

**精英保留与父代选择机制**：为后续轮次提供高质量的父代选择。通过按性能排序维护固定大小的精英池，结合岛屿内选择和精英池选择的概率控制，避免过度依赖单一来源，增加实现的多样性，确保进化过程的稳定性和有效性。   

**配置示例**
```yaml
# 启用岛屿模式
island:
  num_islands: 2              # 创建2个岛屿
  migration_interval: 2       # 每2轮进行一次迁移
  elite_size: 5              # 精英池大小
  parent_selection_prob: 0.7  # 70%概率在当前岛屿选择，30%概率在精英池选择
```

#### 普通模式

普通模式是简化的进化模式，适用于快速测试和简单任务。无需复杂的岛屿管理和精英机制，直接使用全局历史实现进行采样和进化。

**配置示例**
```yaml
# 启用普通模式
island:
  num_islands: 1              # 设置为1禁用岛屿模式
  elite_size: 0              # 设置为0禁用精英机制
```

#### 分层采样机制

分层采样机制从历史实现中智能选择inspiration，为每轮进化提供多样化的参考样本。通过将实现按性能分为好、中、差三个层级，从不同层级采样避免只选择最优个体，结合重复避免机制排除当前轮次已生成的实现，确保进化过程的多样性和创新性。

该机制在岛屿模式和普通模式中都会生效：
- **岛屿模式**：在指定岛屿的历史实现中进行分层采样
- **普通模式**：在全局历史实现中进行分层采样



### 2. single_evolve_runner.py - 单任务执行器

single_evolve_runner.py 是单任务进化执行器，用于执行单个算子的进化生成任务。支持配置文件和命令行参数两种配置方式，提供详细的执行日志和结果统计，是测试和调试单个算子的理想工具。

**使用方法**:
```bash
# 使用默认配置
python single_evolve_runner.py

# 使用指定配置文件
python single_evolve_runner.py evolve_config.yaml
```

### 3. run_batch_evolve.py - 批量执行器

run_batch_evolve.py 是批量任务进化执行器，用于批量执行多个算子的进化生成任务。支持并行执行和动态设备分配，提供详细的执行统计和报告生成，是生产环境中大规模算子生成的高效工具。

**使用方法**:
```bash
# 使用默认配置
python run_batch_evolve.py

# 使用指定配置文件
python run_batch_evolve.py evolve_config.yaml
```

## 使用示例

### 4. PyTorch + Triton 进化示例

`run_torch_evolve_triton.py` 示例展示了如何使用进化系统与 PyTorch 框架和 Triton DSL 进行自动内核生成和优化。

**主要特性**:
- **框架**: PyTorch 与 CUDA 后端
- **DSL**: Triton 用于 GPU 内核生成
- **架构**: A100 GPU 支持
- **进化模式**: 岛屿模式，包含 2 个岛屿和精英保留机制
- **任务**: ReLU 激活函数优化

**使用方法**:
```bash
# 运行 PyTorch 进化示例
python run_torch_evolve_triton.py
```