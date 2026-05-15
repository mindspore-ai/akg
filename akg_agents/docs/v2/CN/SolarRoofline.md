# Solar Roofline

## 概述

AKG Agents 的 roofline 能力通过 **已安装的 `solar` Python 包** 提供。

运行时约束：
- AKG **不依赖**本地 `SOLAR` 工作树路径
- AKG **不要求**修改或 patch `SOLAR` 仓库源码
- AKG 内部自行维护了接入所需的：
  - SOLBench workload wrapper 生成逻辑
  - Ascend / A100 / V100 的 roofline 架构配置

也就是说，AKG 现在只要求：

```bash
python -c "import solar"
```

能够成功。

## 安装

推荐方式：

```bash
bash download.sh --with_solar
```

或：

```bash
SOLAR_DIR=/path/to/SOLAR bash download.sh --with_solar
SOLAR_REF=<tag|branch|commit> bash download.sh --with_solar
```

该脚本会：
1. 默认 clone 官方 Solar 到 `thirdparty/SOLAR`
2. 可选通过 `SOLAR_DIR` 覆盖目录，通过 `SOLAR_REF` 指定 tag / branch / commit
3. 优先调用 Solar 自带的 `install.sh`
4. 安装 Solar 依赖（含 patched torchview）
5. 将 Solar 以 editable 方式安装到当前 Python 环境
6. 校验 `import solar` 以及核心 API 可用

## 运行时行为

当 profile 开启 roofline 时，AKG 会直接调用 Solar 的 Python API：

- `solar.graph.PyTorchProcessor`
- `solar.einsum.PyTorchToEinsum`
- `solar.analysis.EinsumGraphAnalyzer`
- `solar.perf.EinsumGraphPerfModel`

产物包括：
- `roofline_profile_result.json`
- profile 返回字段：
  - `roofline_time`
  - `roofline_speedup`
  - `roofline`

其中：

```text
roofline_speedup = roofline_time / gen_time
```

`1.0x` 表示生成实现已经达到 roofline；小于 `1.0x` 表示仍低于 roofline 上界。

## 失败降级

如果 Solar 未安装，或 Solar 分析失败：
- correctness / profile 主流程仍继续
- roofline 仅降级为缺失数据
- 不会导致 AKG profile 整体失败
