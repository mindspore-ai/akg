# AscendC Memcheck 自动化工作流指南

## 概述

`run_memcheck_pre.sh` 脚本自动化执行 AscendC 内存检测工作流的第 1-3 步：

1. **编译**：使用 sanitizer 选项编译算子
2. **安装**：安装算子包到 `code_base_dir/custom_output_dir` 目录
3. **运行 MemCheck**：执行 mssanitizer memcheck 检测

## 快速开始

### 1. 准备配置文件

从技能目录复制模板到当前工作目录：

```bash
cp scripts/memcheck_input.json.template memcheck_input.json
```

编辑 `memcheck_input.json`，填写必要的参数：

```json
{
  "operator": {
    "name": "your_operator_name"
  },
  "paths": {
    "code_base_dir": "/path/to/your/training/ascendc"
  },
  "testing": {
    "test_script_dir": "/path/to/your/code/training/ascendc/src/.../tests/st",
    "test_script_exe": "pytest test_npu_xxx.py"
  },
  "environment": {
    "device_type": "ascend910b",
    "cann_env": "/path/to/cann"
  },
  "options": {
    "rebuild": true
  }
}
```

### 2. 拷贝并运行脚本

从技能目录拷贝脚本到当前路径并执行：

```bash
cp scripts/run_memcheck_pre.sh .
./run_memcheck_pre.sh
```

### 3. 查看结果

输出结果保存在 `code_base_dir/memcheck_output/` 目录：

```
code_base_dir/memcheck_output/
├── status.txt              # 执行状态摘要
├── build/
│   ├── build.log           # 编译日志
│   ├── build_errors.log    # 编译错误日志
│   └── package_path.txt    # 编译产物路径
├── install/
│   ├── install.log         # 安装日志
│   └── install_errors.log  # 安装错误日志
├── memcheck/
│   ├── memcheck.log        # Memcheck 日志
│   ├── ascendc_memcheck_report_raw.txt  # 原始报告
│   └── mindstudio_sanitizer_log/        # mssanitizer 日志
└── timestamp.txt           # 执行时间戳
```

## 详细说明

### 配置文件参数

| 字段路径 | 说明 | 是否必需 | 默认值 | 示例 |
|---------|------|---------|--------|------|
| `operator.name` | 算子名称 | 是 | - | `"sparse_flash_attention_enhance"` |
| `paths.code_base_dir` | 代码库根目录（包含 build.sh） | 是 | - | `"/home/user/code/training/ascendc"` |
| `testing.test_script_dir` | ST 测试脚本绝对路径 | 否 | - | `"/home/user/code/training/ascendc/.../tests/st"` |
| `testing.test_script_exe` | ST 测试脚本执行方式 | 否 | - | `"pytest test_npu_xxx.py"` |
| `environment.device_type` | NPU 设备类型 | 是 | - | `"ascend910b"` |
| `environment.cann_env` | CANN 环境路径 | 是 | - | `"/home/user/pkg/cann/latest"` |
| `compilation.sanitizer_options` | 编译选项 | 否 | `"-sanitizer;-g"` | `"-sanitizer;-g"` |
| `memcheck.log_level` | 日志级别 | 否 | `"3"` | `"3"` |
| `memcheck.slog_print_to_stdout` | 标准输出开关 | 否 | `"true"` | `"true"` |
| `memcheck.timeout` | 超时时间（秒） | 否 | `600` | `600` |
| `options.rebuild` | 是否重新编译 | 否 | `true` | `true` |
| `installation.load_environment` | 是否加载算子环境 | 否 | `true` | `true` |

### 命令行参数

```bash
./run_memcheck_pre.sh [options]
```

| 参数 | 说明 |
|------|------|
| `-h, --help` | 显示帮助信息 |
| `-c, --config FILE` | 配置文件路径（默认：`./memcheck_input.json`） |
| `--skip-build` | 跳过编译步骤 |
| `--keep-build` | 保留构建目录 |
| `--verbose` | 显示详细输出 |

### 使用示例

#### 示例 1：使用默认配置

```bash
./run_memcheck_pre.sh
```

#### 示例 2：指定配置文件

```bash
./run_memcheck_pre.sh --config /path/to/my_config.json
```

#### 示例 3：跳过编译（已有编译产物）

```bash
./run_memcheck_pre.sh --skip-build
```

#### 示例 4：详细输出模式

```bash
./run_memcheck_pre.sh --verbose
```

#### 示例 5：组合选项

```bash
./run_memcheck_pre.sh \
  --config my_config.json \
  --verbose
```

## 工作流程

### 完整工作流程（3 步）

1. **编译算子**
   - 加载 CANN 环境：`source <cann_env>/bin/setenv.bash`
   - 进入代码目录：`cd <code_base_dir>`
   - 执行编译：`bash build.sh -n <op_name> -c <device> -p <cann_env> --ops-compile-options "<sanitizer_opts>"`
   - 检查编译产物（.run 文件）
   - 保存编译产物路径到 `memcheck_output/build/package_path.txt`

2. **安装算子**
   - 查找编译产物：从 `code_base_dir/output` 目录查找 .run 文件
   - 创建/清理安装目录：`code_base_dir/custom_output_dir/`
   - 执行安装：`./output/<package>.run --install-path=<code_base_dir/custom_output_dir>`
   - 加载算子环境（可选）：`source <code_base_dir>/custom_output_dir/vendors/omni_training_custom_ops/bin/set_env.bash`

3. **运行 MemCheck**
   - 设置日志环境变量：
     - `ASCEND_GLOBAL_LOG_LEVEL=<log_level>`
     - `ASCEND_SLOG_PRINT_TO_STDOUT=<slog_print>`
   - 进入 ST 目录（从 `test_script_dir` 绝对路径提取）
   - 执行：`mssanitizer --tool=memcheck <test_script_exe>`
   - 收集输出和日志文件
   - 生成状态报告：
     - 统计 ERROR 和 WARNING 数量
     - 显示前 5 条错误/警告
     - 检查测试结果

### 跳过编译的工作流程

当使用 `--skip-build` 或配置文件中 `rebuild=false` 时：

1. **跳过编译**：直接使用现有的编译产物
2. **安装算子**：从 `code_base_dir/output` 查找现有算子包并安装到 `code_base_dir/custom_output_dir`
3. **运行 MemCheck**：执行内存检测

## 输出文件说明

### status.txt

执行状态摘要，包含：
- 执行时间
- 配置文件路径
- 各步骤执行状态（成功/失败/跳过）
- 输出文件路径列表

### build/

编译相关日志：
- `build.log`：完整的编译输出
- `build_errors.log`：编译错误信息（如果编译失败）
- `package_path.txt`：编译产物的绝对路径

### install/

安装相关日志：
- `install.log`：完整的安装输出
- `install_errors.log`：安装错误信息（如果安装失败）

### memcheck/

Memcheck 相关文件：
- `memcheck.log`：完整的 memcheck 输出（从 raw_report 复制）
- `ascendc_memcheck_report_raw.txt`：原始 memcheck 报告
- `mindstudio_sanitizer_log/`：mssanitizer 工具生成的详细日志

## 错误处理

脚本会在以下情况退出并返回非零状态码：

1. 配置文件不存在或格式错误
2. CANN 环境脚本不存在
3. 代码目录或编译脚本不存在
4. 编译失败
5. 安装失败
6. 测试目录不存在
7. Memcheck 执行失败（根据情况）

每次失败都会在日志中保存完整的错误信息。

## 常见问题

### Q: 如何更改编译选项？

A: 编辑配置文件中的 `compilation.sanitizer_options` 字段。

```json
{
  "compilation": {
    "sanitizer_options": "-sanitizer;-g"  // 使用完整调试信息
  }
}
```

### Q: 如何只运行 MemCheck，不重新编译？

A: 使用 `--skip-build` 参数或在配置文件中设置 `rebuild=false`。

```bash
./run_memcheck_pre.sh --skip-build
```

或

```json
{
  "options": {
    "rebuild": false
  }
}
```

### Q: 产物保存在哪里？

A: 所有输出保存在 `code_base_dir/memcheck_output/` 目录。算子包安装在 `code_base_dir/custom_output_dir/` 目录。

### Q: 如何查看 Memcheck 结果？

A: 查看 `code_base_dir/memcheck_output/memcheck/ascendc_memcheck_report_raw.txt` 文件。

### Q: 脚本需要什么依赖？

A: 脚本需要：
- Bash 3.2+
- Python 3.6+（用于解析 JSON）
- CANN 环境
- mssanitizer 工具（通常由 CANN 提供）

### Q: 测试脚本为什么使用绝对路径？

A: 使用绝对路径可以避免相对路径计算错误，特别是在不同目录执行脚本时更可靠。配置文件中使用 `test_script_dir` 指定测试目录绝对路径，`test_script_exe` 指定在该目录下执行的命令（通常为 pytest 命令）。

### Q: `memcheck.timeout` 参数有什么用？

A: 此参数在配置文件中定义，用于设置 Memcheck 执行的超时时间（单位：秒），默认 600 秒（10 分钟）。如果执行时间超过设定值，进程可能被终止。

### Q: 安装失败如何处理？

A: 脚本会先清理旧安装目录，然后重新安装。如果失败：
1. 检查 `code_base_dir/memcheck_output/install/install.log` 了解详细错误
2. 检查是否还有残留文件占用目录
3. 手动清理 `code_base_dir/custom_output_dir/` 目录
4. 重新运行脚本

### Q: 如何禁用自动加载算子环境？

A: 在配置文件中设置 `installation.load_environment` 为 `false`：

```json
{
  "installation": {
    "load_environment": false
  }
}
```

## 与手动执行的区别

| 项目 | 手动执行 | 自动化脚本 |
|------|---------|-----------|
| 配置 | 每次手动设置命令行参数 | 通过 JSON 配置文件管理 |
| 编译 | 手动执行 build.sh | 自动执行并检查结果 |
| 安装 | 手动查找安装包并指定路径 | 自动查找并安装到固定路径 |
| 环境 | 手动加载多次 | 根据配置自动加载必要环境 |
| 日志 | 分散保存 | 集中管理到 `code_base_dir/memcheck_output` |
| 错误处理 | 需要手动检查 | 自动检测并报告 |
| 测试脚本路径 | 需要手动计算相对/绝对路径 | 使用绝对路径，自动进入测试目录 |
| 安装路径管理 | 需要手动指定绝对路径 | 固定安装在 `code_base_dir/custom_output_dir/` |
| 结果摘要 | 需要手动统计 | 自动生成 ERROR/WARNING 数量摘要 |

## 后续步骤

脚本完成后，继续执行第 4-7 步：

1. **分析错误输出**：查看 memcheck 报告中的 ERROR 和 WARNING
2. **定位源代码**：根据调用栈定位问题代码
3. **根因分析**：分析错误原因
4. **生成报告**：创建详细的内存检测报告

详细步骤请参考主文档 [`SKILL.md`](./SKILL.md)。

## 技巧和建议

### 1. 使用版本控制管理配置文件

为不同的算子或测试场景创建不同的配置文件：

```bash
cp scripts/memcheck_input.json.template memcheck_input_op1.json
cp scripts/memcheck_input.json.template memcheck_input_op2.json
```

### 2. 定制超时时间

对于大型算子或复杂测试用例，调整 `memcheck.timeout` 参数：

```json
{
  "memcheck": {
    "timeout": 1200  // 20 分钟
  }
}
```

### 3. 调试模式

使用 `--verbose` 查看详细执行过程：

```bash
./run_memcheck_pre.sh --verbose
```

### 4. 批量测试

结合脚本进行多算子测试（创建测试脚本）：

```bash
#!/bin/bash
cp scripts/run_memcheck_pre.sh .

for config in memcheck_input_*.json; do
    ./run_memcheck_pre.sh -c "$config"
done
```

### 5. 环境管理

- 如果不希望脚本自动加载算子环境，设置 `load_environment: false`
- 编译环境（CANN）和算子环境会分别管理，互不干扰

### 6. 清理旧版本

如果不再需要某个版本的算子包：

```bash
# 手动删除指定的 custom_output_dir 目录
rm -rf /path/to/code/base/custom_output_dir

# 或修改 code_base_dir 配置指向不同的代码目录
```

## 目录说明

### 技能目录结构

```
ascendc-crash-debug/
├── scripts/
│   ├── memcheck_input.json.template    # 配置文件模板
│   ├── run_memcheck_pre.sh             # 自动化脚本
│   └── parse_plog.py                   # plog 日志解析脚本
└── references/
    └── memcheck/
        ├── automated_workflow.md       # 本文档 - 自动化工作流指南
        ├── README.md                   # 用户使用指南
        └── mssanitizer_guide.md        # msSanitizer 工具原始文档
```

## 支持和反馈

如遇问题，请检查：

1. 配置文件格式是否正确（JSON 语法）
2. 所有必需参数是否已填写
3. CANN 环境路径是否正确
4. NPU 设备是否可用
5. 日志文件中的错误信息
6. 是否有其他进程占用算子包安装路径
7. 测试脚本目录是否存在
8. 测试脚本名称是否与配置一致

---

**文档版本**: 2.4
**最后更新**: 2026-05-16
**适用版本**: ascendc-crash-debug skill (memcheck 子模块)