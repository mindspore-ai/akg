# mssanitizer 内存检测使用指南

用于检测 AscendC 算子内核中的内存错误（越界读写、非对齐访问、多核踩踏等），基于 `mssanitizer --tool=memcheck`。

**这是 `ascendc-crash-debug` skill 的子功能模块，请通过 `/ascendc-crash-debug` 触发使用。**

## 前置条件

- NPU 设备可用（`npu-smi info` 正常）
- CANN 环境已安装并可用（含 `mssanitizer` 工具）
- 算子代码仓库中包含 `build.sh` 编译脚本
- 已有可运行的 ST 测试用例（pytest）

## 快速开始

### 第 1 步：准备配置文件

将模板拷贝到你的工作目录，并根据实际情况修改：

```bash
cp scripts/memcheck_input.json.template ./memcheck_input.json
```

编辑 `memcheck_input.json`，填写你的算子信息：

```json
{
  "operator": {
    "name": "your_operator_name"
  },
  "paths": {
    "code_base_dir": "/path/to/your/training/ascendc"
  },
  "testing": {
    "test_script_dir": "/path/to/your/.../tests/st",
    "test_script_exe": "pytest test_npu_your_operator.py"
  },
  "environment": {
    "device_type": "ascend910b",
    "cann_env": "/path/to/cann/latest"
  }
}
```

各字段说明：

| 字段 | 必填 | 说明 |
|------|------|------|
| `operator.name` | 是 | 算子名称 |
| `paths.code_base_dir` | 是 | 代码仓根目录（包含 `build.sh`） |
| `testing.test_script_dir` | 是 | ST 测试脚本所在目录（绝对路径） |
| `testing.test_script_exe` | 是 | 测试执行命令，如 `pytest test_npu_xxx.py` |
| `environment.device_type` | 是 | NPU 设备类型，如 `ascend910b` |
| `environment.cann_env` | 是 | CANN 环境路径 |
| `compilation.sanitizer_options` | 否 | 编译选项，默认 `"-sanitizer;-g"` |
| `memcheck.timeout` | 否 | 超时秒数，默认 `600` |
| `options.rebuild` | 否 | 是否重新编译，默认 `true` |

### 第 2 步：拷贝脚本并执行

```bash
cp scripts/run_memcheck_pre.sh .
chmod +x run_memcheck_pre.sh
./run_memcheck_pre.sh
```

脚本会自动完成：编译（带 sanitizer）→ 安装算子包 → 运行 mssanitizer memcheck。

### 第 3 步：查看结果

输出保存在 `<code_base_dir>/memcheck_output/` 下：

```
memcheck_output/
├── status.txt                          # 执行状态摘要
├── build/build.log                     # 编译日志
├── install/install.log                 # 安装日志
└── memcheck/
    ├── ascendc_memcheck_report_raw.txt # memcheck 原始报告（重点看这个）
    └── mindstudio_sanitizer_log/       # mssanitizer 详细日志
```

### 第 4 步：让 Claude 分析

在 Claude Code 中输入 `/ascendc-crash-debug` 并说明需要内存检测，Claude 会自动：
1. 读取 memcheck 报告，提取 ERROR 和 WARNING
2. 根据调用栈定位源代码
3. 分析根因并给出修复建议
4. 生成 `memcheck_detailed_report.md` 详细报告

## 常用选项

```bash
# 跳过编译，直接用已有产物运行 memcheck
./run_memcheck_pre.sh --skip-build

# 指定其他配置文件
./run_memcheck_pre.sh --config my_config.json

# 详细输出
./run_memcheck_pre.sh --verbose
```

## 典型使用场景

- 算子运行崩溃、偶发崩溃无法复现
- 怀疑存在越界读写或内存踩踏导致崩溃
- aic error 或 coredump 堆栈不清晰
- 多核场景下偶发崩溃
- 代码提交前的内存安全检查


## 目录结构

```
ascendc-crash-debug/
├── scripts/
│   ├── memcheck_input.json.template     # 配置文件模板（拷贝到工作目录使用）
│   └── run_memcheck_pre.sh              # 自动化检测脚本
│   └── parse_plog.py                    # plog 日志解析脚本
└── references/
    └── memcheck/
        ├── automated_workflow.md        # 自动化工作流详细说明
        ├── README.md                    # 本文件 - 用户使用指南
        └── mssanitizer_guide.md         # msSanitizer 工具原始文档
```

## 常见问题

**Q: 脚本报找不到 `mssanitizer`？**
确认 CANN 环境已正确加载：`source <cann_env>/bin/setenv.bash`

**Q: 编译失败？**
检查 `memcheck_output/build/build.log`，确认 `code_base_dir` 路径正确且包含 `build.sh`。

**Q: memcheck 超时？**
在配置文件中增大 `memcheck.timeout`（单位秒），大型算子建议设为 1200+。
