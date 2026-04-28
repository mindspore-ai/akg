---
name: akg-env-setup
description: >
  akg_agents 环境准备。缓存优先，检查 akg_cli + LLM 连通性，
  未安装则引导安装，采集硬件/Framework/DSL。
  FULL_SETUP 模式额外包含参数确认和运行时依赖安装。
argument-hint: >
  可选：ENV_TYPE（conda/venv）、CONDA_ENV、VENV_PATH、AKG_AGENTS_DIR。
  可选：FULL_SETUP=true（包含参数确认和运行时依赖安装，算子相关 agent 使用）。
---

# akg_agents 环境准备

## ⛔ 核心规则

1. 检查失败后**只有两条路径**：询问用户是否有其他可用环境、询问用户是否同意自动安装。**除此之外禁止任何操作。**
2. 遇到 🛑 必须调用 `question` 工具并等待用户回复，不得跳过或自行决定。
3. 禁止执行本文档未列出的命令。

## 流程总览

```
Step 0  解析 $HOME_DIR → 读取缓存
  ├─ 命中 ──────────────────────────────┬─ FULL_SETUP → Step 5
  │                                     └─ 否则 → 结束
  └─ 未命中 → Step 1  检查 akg_cli + LLM
                  ├─ 通过 → Step 4  采集硬件/Framework/DSL
                  │             ├─ 4a 硬件
                  │             ├─ 4b Framework
                  │             ├─ 4c DSL & 后端推断
                  │             └─ 4d 写入缓存（必须，不可跳过）
                  │                ├─ FULL_SETUP → Step 5 → (Step 6 [可选，仅需要新依赖时]) → 结束
                  │                └─ 否则 → 结束
                  └─ 失败 → Step 2  询问用户（一次收集处理方式+环境信息）
                              ├─ 已有环境 → Step 1（重试）
                              └─ 需要安装 → Step 3 → Step 1（重试）
```

---

## Step 0: 解析 HOME_DIR & 读取环境缓存

**首先**执行以下命令获取 home 目录绝对路径，赋值给 `$HOME_DIR`：

```bash
echo $HOME
```

后续所有路径中的 `$HOME_DIR` 均使用此值。

然后读取 `$HOME_DIR/.akg/check_env.md`：

- **文件存在**：
  - **FULL_SETUP** → **跳到 Step 5**（每次任务都需确认参数）
  - 非 FULL_SETUP → 报告就绪，**流程结束**
- **文件不存在** → 进入 Step 1

---

## Step 1: 执行两项检查

需要先确定 `$ENV_TYPE`、`$CONDA_ENV`/`$VENV_PATH` 和 `$AKG_AGENTS_DIR`。优先使用调用方提供的值，未提供时使用当前已激活的环境。

**检查 1 — akg_cli 可用性**：

conda:
```bash
conda run -n $CONDA_ENV --no-capture-output bash -c \
  "cd $AKG_AGENTS_DIR && source env.sh && which akg_cli"
```

venv:
```bash
bash -c "source $VENV_PATH/bin/activate && cd $AKG_AGENTS_DIR && source env.sh && which akg_cli"
```

**检查 2 — LLM 连通性**：

conda:
```bash
conda run -n $CONDA_ENV --no-capture-output bash -c \
  "cd $AKG_AGENTS_DIR && source env.sh && \
   python tools/v2/use_llm_check/test_run_llm.py"
```

venv:
```bash
bash -c "source $VENV_PATH/bin/activate && cd $AKG_AGENTS_DIR && source env.sh && \
  python tools/v2/use_llm_check/test_run_llm.py"
```

**判定**：

- ✅ 两项全部通过 → **进入 Step 4**
- ❌ 任一失败 → **进入 Step 2**，**禁止**自行尝试其他环境。

---

## Step 2: 🛑 报告失败并询问用户

调用 `question` 工具，**一次调用 3 个问题**，同时收集处理方式和环境信息：

| # | header | question | options |
|---|--------|----------|---------|
| 1 | 处理方式 | akg_agents 在当前环境中不可用。检查结果：<哪项通过、哪项失败及错误信息>。请选择处理方式： | "使用已有环境" / "新建环境安装" |
| 2 | 环境类型 | 选择 Python 环境类型 | "conda" / "venv" |
| 3 | 环境名或路径 | conda 填环境名（如 akg_agents），venv 填绝对路径（如 ~/akg/.venv） | "akg_agents（默认）" |

> 问题 3 用户可选择默认值或通过「自定义输入」填写自己的环境名/路径。

**处理回复**：

- 处理方式 = **使用已有环境** → 用问题 2、3 的回答设置 `$ENV_TYPE`、`$CONDA_ENV`/`$VENV_PATH`，**回到 Step 1** 重试
- 处理方式 = **新建环境安装** → 检测目标环境是否已存在：
  - **不存在** → **进入 Step 3**
  - **已存在** → 再调用一次 `question`：
    > 环境 $CONDA_ENV/$VENV_PATH 已存在，请选择：
    > 1. 复用该环境安装
    > 2. 删除重建
    > 3. 取消
    - 复用 / 重建 → **进入 Step 3**
    - 取消 → **流程结束**

---

## Step 3: 委派 akg-installer

```
task(subagent_type="akg-installer", load_skills=[],
     run_in_background=false,
     description="Install akg_agents",
     prompt="请安装 akg_agents。ENV_TYPE=<类型> CONDA_ENV=<名称>/VENV_PATH=<路径> ENV_ACTION=<reuse|recreate>。<用户原始请求>")
```

安装完成后，重新从 Step 1 开始检查（安装成功 → 检查通过 → 进入 Step 4）。

---

## Step 4: 采集环境信息 & 写入缓存

Step 1 两项检查通过后执行以下步骤。

### 4a. 硬件采集（系统级，无需环境激活）

> ⛔ `nvidia-smi` 和 `npu-smi info` 的输出**禁止截断**（不得添加 `head`/`tail`/`| head -N` 等），必须获取完整输出以准确识别所有设备数量。截断会导致多卡被误认为少卡。

```bash
uname -m
cat /proc/cpuinfo 2>/dev/null | grep "model name" | head -1 || sysctl -n machdep.cpu.brand_string 2>/dev/null
nvidia-smi 2>/dev/null
npu-smi info 2>/dev/null
free -h 2>/dev/null || sysctl hw.memsize 2>/dev/null
```

**采集后处理**：从工具输出中提取芯片型号后，**必须按规范化规则转换**再写入缓存。例如 `npu-smi` 输出的 `910B4` 必须转换为 `ascend910b4`，`nvidia-smi` 输出的 `A100` 必须转换为 `a100`。缓存中的 `arch` 字段必须是规范化后的值。

### 4b. Framework 采集（在环境中执行）

torch 始终采集。torch_npu 仅在 4a 检测到 Ascend NPU 时采集。

| 项目 | 采集条件 | `<CMD>` |
|------|----------|---------|
| torch | 始终 | `python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"` |
| torch_npu | 有 Ascend NPU | `python -c "import torch_npu; print(torch_npu.__version__)"` |

### 4c. DSL 采集 & 后端推断

triton 仅在 4a 检测到 Ascend NPU 或 NVIDIA GPU 时采集：

| 项目 | `<CMD>` |
|------|---------|
| triton | `python -c "import triton; print(triton.__version__)"` |
| g++/clang++ | `g++ --version 2>/dev/null \|\| clang++ --version 2>/dev/null`（系统命令，无需命令模板） |

按检测结果列出**所有**可用后端（一台机器可匹配多行）：

| 条件 | 后端 | DSL | 架构 |
|------|------|-----|------|
| Ascend NPU + torch_npu + triton | ascend | triton_ascend | 从 npu-smi 解析型号 |
| NVIDIA GPU + CUDA + triton | cuda | triton_cuda | 从 nvidia-smi 解析型号 |
| g++/clang++ 可用 | cpu | cpp | `uname -m` |

### ⛔ 4d. 写入 `$HOME_DIR/.akg/check_env.md`

**重要：**此步骤不可跳过。不写入缓存，后续每次启动都会重复完整环境检查。

将 Step 4a–4c 的采集结果按附录格式写入 `$HOME_DIR/.akg/check_env.md`。环境缓存示例：`@references/example_check_env.md`

**写入后立即验证**：

```bash
test -f $HOME/.akg/check_env.md && echo "✅ 缓存已写入" || echo "❌ 缓存写入失败"
```

验证通过后：
- **非 FULL_SETUP** → 报告环境就绪，**流程结束**
- **FULL_SETUP** → **进入 Step 5**

---

## Step 5: 🛑 参数推断与确认（FULL_SETUP 模式）

从 `$HOME_DIR/.akg/check_env.md` 的「可用后端 & DSL & 架构」表中读取所有可用组合，用 `question` 工具请用户选择本次使用的配置：

> 根据您的环境检测，可用配置如下：
>
> 请选择：
> 1. backend=cuda, arch=a100, framework=torch, dsl=triton_cuda
> 2. backend=cpu, arch=x86_64, framework=torch, dsl=cpp
> ...

options 只列出缓存中检测到的组合，不要添加"自行输入"类选项（question 工具自带「Type your own answer」入口，用户需要自定义时会自行使用）。缓存中只有一行时也需确认。

确认后按「算子参数规范化」规则处理（小写、别名转换、连字符→下划线等）。

进入 Step 6。

---

## Step 6: 运行时依赖安装（FULL_SETUP 模式）

如果用户在 Step 5 选择的配置来自 `check_env.md` 中已有的组合，说明依赖已就绪，**跳过本步骤**，直接报告环境就绪并返回参数。

仅当用户通过「Type your own answer」输入了 `check_env.md` 中不存在的配置时，才需要检测并安装缺失依赖（所有 pip 命令通过命令模板在环境中执行）：

**backend = cpu**

| 缺失 | 处理 |
|------|------|
| torch | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| g++/clang++ | 检查可用性，不可用告知用户 |

**backend = cuda**

| 缺失 | 处理 |
|------|------|
| torch | `pip install torch` |
| triton | `pip install triton` |

**backend = ascend**

| 缺失 | aarch64 | x86_64 |
|------|---------|--------|
| torch | `pip install` cpu 版 | 🛑 需用户提供 .whl |
| torch_npu | `pip install` | 🛑 需用户提供 .whl |
| triton-ascend | 🛑 需用户提供 .whl | 🛑 需用户提供 .whl |

> 🛑 标记项必须调用 `question` 工具索要 .whl 路径。
> pip 包名用连字符（如 `triton-ascend`），API 参数用下划线（如 `triton_ascend`）。

**【重要】：**安装完成后，**更新 `$HOME_DIR/.akg/check_env.md`**，重新执行 Step 4b–4c 采集最新 Framework 和 DSL 版本。

报告环境就绪，向调用方返回已确认的 framework/backend/arch/dsl。**流程结束**。

---

## 附录：`$HOME_DIR/.akg/check_env.md` 完整格式

```markdown
# AKG Agents 环境缓存

> 自动生成。删除此文件可强制重新检查。

## 环境配置
- HOME_DIR: <绝对路径>
- ENV_TYPE: <conda 或 venv>
- CONDA_ENV: <环境名>（仅 conda）
- VENV_PATH: <路径>（仅 venv）
- AKG_AGENTS_DIR: <绝对路径>

## 命令模板
将 <CMD> 替换为实际命令：
<根据 ENV_TYPE 和 AKG_AGENTS_DIR 生成的完整命令模板，含激活环境和 source env.sh>

## 验证状态
- akg_cli: ✅
- LLM: ✅ 或 ⚠️（未配置 API Key）

## 硬件
- CPU: <架构和型号>
- GPU/NPU: <型号或无>
- 内存: <大小>

## Framework
- torch: <版本> (CUDA: True/False)
- torch_npu: <版本>（如有）

## DSL
- triton: <版本>（如有）
- g++/clang++: <版本或"未安装">

## 可用后端 & DSL & 架构（可多行）
| 后端 | DSL | 架构 |
|------|-----|------|
| <后端1> | <DSL1> | <架构1> |
| <后端2> | <DSL2> | <架构2> |

```

> 环境缓存仅存储**环境级**信息（硬件、已安装的 Framework和DSL、命令模板）。任务级参数配置（framework/backend/arch/dsl 选择、当次所用的 device id等）不缓存，每次任务由调用方确认。
