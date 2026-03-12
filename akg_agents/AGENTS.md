# AKG Agents 通用规则

本文件包含所有涉及 akg_agents 的 agent / skill 共享的通用规则。

---

## 环境变量约定

以下变量为 prompt 中的逻辑占位符，由 agent 在运行时根据实际检测结果赋值：

| 变量 | 含义 | 示例 |
|------|------|------|
| `$HOME_DIR` | 当前用户 home 目录绝对路径（由 `akg-env-setup` Step 0 通过 `echo $HOME` 解析，**禁止猜测**） | 取决于系统 |
| `$AKG_AGENTS_DIR` | akg_agents 目录（`env.sh`、`setup.py` 所在目录，非仓库根目录） | `$HOME_DIR/akg/akg_agents` |
| `$ENV_TYPE` | Python 环境类型：`conda` 或 `venv` | `conda` |
| `$CONDA_ENV` | conda 环境名（仅 `$ENV_TYPE=conda`） | `akg_agents` |
| `$VENV_PATH` | venv 目录绝对路径（仅 `$ENV_TYPE=venv`） | `$HOME_DIR/akg/akg_agents/.venv` |

---

## 环境缓存（`$HOME_DIR/.akg/check_env.md`）

首次环境检查通过后，**环境级**信息缓存到 `$HOME_DIR/.akg/check_env.md`（硬件、已安装 Framework、命令模板等）。

**使用规则**：
1. 任何 akg_agents 功能启动前，先读取 `$HOME_DIR/.akg/check_env.md`
2. **文件存在** → 直接使用其中的环境配置和命令模板，跳过环境检查
3. **文件不存在** → 执行完整环境检查流程，通过后写入该文件

删除 `$HOME_DIR/.akg/check_env.md` 可强制重新检查环境。

---

## ⛔ 算子参数规范化

用户输入的 framework / backend / arch / dsl 可能不规范。**所有传给 akg_agents API、脚本、akg_cli 的参数值都必须是下方有效值之一，否则会报错。**

在构建命令、生成配置、填写参数**之前**，必须逐项对照下表校验。

### 有效值参照

| 参数 | 有效值 |
|------|--------|
| `framework` | `torch`, `mindspore` |
| `backend` | `cuda`, `ascend`, `cpu` |
| `dsl` | `triton_cuda`, `triton_ascend`, `cpp`, `cuda_c`, `tilelang_cuda`, `ascendc`, `pypto` |
| `arch` | cuda: `a100`, `v100`；ascend: `ascend910b1`~`ascend910b4`, `ascend310p3`；cpu: `x86_64`, `aarch64` |

---

## Shell 命令规则

每条 shell 命令运行在独立 session，环境激活不跨命令持久化。

**conda 环境**（`$ENV_TYPE=conda`）：
```bash
conda run -n $CONDA_ENV --no-capture-output bash -c \
  "cd $AKG_AGENTS_DIR && source env.sh && <CMD>"
```

**venv 环境**（`$ENV_TYPE=venv`）：
```bash
bash -c "source $VENV_PATH/bin/activate && cd $AKG_AGENTS_DIR && source env.sh && <CMD>"
```

> 如果已有 `$HOME_DIR/.akg/check_env.md`，直接使用其中的「命令模板」字段，将 `<CMD>` 替换为实际命令即可。

### 禁止行为

| 行为 | 级别 |
|------|------|
| 裸执行 `pip`/`python`（未激活环境） | ⛔ 致命 |
| 依赖 `conda activate` 或 `source activate` 跨命令持久化 | ⛔ 致命 |
| 不 `source env.sh` 就运行 akg_agents 相关脚本 | ❌ 错误 |
| 用 `echo 'y' \|` 管道代替 `--yes` 标志 | ⛔ 致命 |

---

## 环境检查 Skill

| Skill | 用途 | 加载场景 |
|-------|------|---------|
| `akg-env-setup` | 环境检查 + 采集 + 缓存；FULL_SETUP 模式额外含当次任务的参数确认和运行时依赖安装 | 安装请求（基础模式）；op-optimizer Phase 1（FULL_SETUP 模式） |
