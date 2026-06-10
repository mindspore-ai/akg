# thirdparty

本目录用于存放按需下载的第三方 benchmark 仓库。

统一通过项目根目录的下载脚本管理：

```bash
bash download.sh --with_kernelbench
bash download.sh --with_multikernelbench
bash download.sh --with_evokernel
bash download.sh --with_sol_execbench
bash download.sh --with_npukernelbench    # NPUKernelBench 数据集（atomgit）
bash download.sh --with_cann_bench
bash download.sh --with_catlass            # ascendc_catlass DSL 用的 C++ 模板库
# 一次性下载全部 benchmark
bash download.sh --with_all_benchmarks
```

这些仓库会被 clone 到当前目录，并自动 checkout 到项目固定的 commit。

## CATLASS（ascendc_catlass DSL 必备）

`thirdparty/catlass` 是 ascendc_catlass 算子生成依赖的 C++ 模板头文件库。它的特殊之处：

- 不是 Python 包（pip install 不到），也不是 benchmark
- 是 cmake 编译时需要 `include/catlass/...` 头文件的硬依赖
- 不同部署可能装在任意路径，历史上要求操作员手动 export `CATLASS_ROOT` 或填 `task.yaml: catlass.root`

clone 到本目录后，[`catlass_paths.resolve_catlass_root`](../python/akg_agents/op/utils/catlass_paths.py) 按 `<akg-root>/thirdparty/catlass` 这条标准路径自动发现，**操作员不再需要任何 catlass 相关配置**。

解析顺序：

1. `task.yaml: catlass.root`（per-task 覆盖，少见）
2. `CATLASS_ROOT` 环境变量（worker 进程级覆盖）
3. `<akg-root>/thirdparty/catlass`（标准位置，本文档推荐）

非 ascendc_catlass DSL（triton / pypto / tilelang / …）完全不需要 catlass，不下载也行。`--with_all_benchmarks` 顺手把它带上（低成本 clone）。
